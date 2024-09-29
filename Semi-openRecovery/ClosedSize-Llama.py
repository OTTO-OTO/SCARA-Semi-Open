import os
from pkg_resources import packaging
import wandb

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from training_utils.model_checkpointing import checkpoint_handler
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
    AutoTokenizer, 
    AutoModel,
    T5Tokenizer, 
    T5ForConditionalGeneration,
    BertTokenizer, 
    BertForMaskedLM,
    BertModel,
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    GPT2Config,
    BartTokenizer, 
    BartForConditionalGeneration,
    XLNetModel, 
    XLNetTokenizer,
    OPTModel,
    OPTForCausalLM,
    OPTConfig
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from training_utils.configs import fsdp_config, train_config, alpaca_dataset
from training_utils.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from training_utils.utils import fsdp_auto_wrap_policy
from training_utils.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from training_utils.utils.dataset_utils_minidatasets import get_preprocessed_dataset

from training_utils.utils.train_utils_llama import (
    train,
    evaluation,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)

def print_configurations(local_rank,config):
    if local_rank == 0:
        # 获取实例属性列表
        attributes = dir(config)
        # 遍历实例属性
        for attribute in attributes:
            # 排除掉一些特殊属性
            if not attribute.startswith("__"):
                value = getattr(config, attribute)
                print(f"{attribute}: {value}")

def main(**kwargs):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # Update the configuration for the training and sharding process
    # 更新数据集
    update_config((train_config, fsdp_config, alpaca_dataset), **kwargs)
    train_config.eta_min=train_config.lr*0.9
    train_config.T_max = int(train_config.data_per_GPU*train_config.num_epochs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    print(train_config.model_name)

    if train_config.record_loss:
        #run_id_list = ['6tyw9271','513aur61']
        wandb.init(
        # set the wandb project where this run will be logged
        project=train_config.project_name,
        group=train_config.group_name,
        # track hyperparameters and run metadata
        config={
        "architecture": train_config.model_name,
        "dataset": train_config.dataset,
        "epochs": train_config.num_epochs,
        "batch_size_training": train_config.batch_size_training,
        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
        "learning_rate": train_config.lr,
        "weight_decay": train_config.weight_decay,
        "gamma": train_config.gamma,
        "seed": train_config.seed,
        "mixed_precision": train_config.mixed_precision,
        "use_fp16": train_config.use_fp16,
        "scheduler": train_config.scheduler,
        "T_max": train_config.T_max,
        "eta_min": train_config.eta_min,
        'wipe_layers': train_config.wipe_layer,
        'layer_start_point':train_config.layer_start_point,
        "pure_bf16": fsdp_config.pure_bf16,
        "fsdp_cpu_offload": fsdp_config.fsdp_cpu_offload,
        "optimizer": fsdp_config.optimizer,
        },
        #id = run_id_list[int(os.environ["RANK"])],
        #resume = 'must'
        )


    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # print configurations
        if local_rank == 0:
            print('='*30,'Training Config','='*30)
            print_configurations(local_rank,train_config)
            print('='*30,'FSDP Config','='*30)
            print_configurations(local_rank,fsdp_config)
            print('='*60)
        print('local_rank',local_rank)

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        # print(torch.cuda.current_device())

    if local_rank == 0 and not os.path.exists(f'./project-llama2/loss_txtfiles/{train_config.loss_store_path}/{train_config.group_name}'):
        os.makedirs(f'./project-llama2/loss_txtfiles/{train_config.loss_store_path}/{train_config.group_name}', exist_ok=True) 
                

    # Load the tokenizer and add special tokens
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_path)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_config.dataset_type = 'train'
     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set {dataset_config.data_path} Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set {dataset_config.eval_file_path} Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
                val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )


    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    eval_dataloader = None
    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
        
    use_cache = False if train_config.enable_fsdp else None      
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        
        if rank == 0:
            if train_config.wipe_layer >= 31:
                llama_config = LlamaConfig.from_pretrained(train_config.model_path)
                llama_config.use_cache = use_cache
                model = LlamaForCausalLM(llama_config)
            else:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )

        else:
            # 得到的仅仅是config
            llama_config = LlamaConfig.from_pretrained(train_config.model_path)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                # Initializing a model from the llama-7b style configuration
                model = LlamaForCausalLM(llama_config)
    else:
        # print('quantization', train_config.quantization)
        
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_path,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
        )        


    # Load the pre-trained model and setup its configuration
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model = model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        print('finished')
        if train_config.load_from_ckpt:
            checkpoint_handler.load_model_sharded(model,rank,train_config) 

        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    # 写入表头
    file_name = f'./project-llama2/loss_txtfiles/{train_config.loss_store_path}/{train_config.group_name}/Training Loss of {train_config.group_name}-{local_rank}.txt'
    with open(file_name, 'a') as file:
            file.write("Training Epoch\t step\t loss\t lr\t training_ppl\n")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    # scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    # T_max = 1000 # Maximum number of iterations.
    if train_config.scheduler=='Cosine':
        scheduler = CosineAnnealingLR(optimizer, train_config.T_max, eta_min=train_config.eta_min, last_epoch=-1, verbose=False)
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
        print('StepLR')

    llama_config = LlamaConfig.from_pretrained(train_config.model_path)
    llama_config.use_cache = use_cache
    layer_container = LlamaForCausalLM(llama_config)
 
    wipe_off = {
        'whole':['self_attn.k_proj','self_attn.q_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj',
        'mlp.up_proj','mlp.down_proj','input_layernorm', 'post_attention_layernorm'],
        '0.25':['self_attn.k_proj'],
        '0.5':['self_attn.k_proj','self_attn.q_proj'],
        '1':['self_attn.k_proj','self_attn.q_proj','self_attn.v_proj','self_attn.o_proj'],
        '50M':['self_attn.k_proj','self_attn.q_proj','self_attn.v_proj'],
        '100M':['self_attn.o_proj','self_attn.q_proj','self_attn.v_proj','mlp.up_proj'],
        '160M':['self_attn.k_proj','self_attn.q_proj','self_attn.v_proj','self_attn.o_proj','mlp.up_proj','mlp.down_proj'],
        '300M':['self_attn.o_proj','self_attn.q_proj','self_attn.v_proj','mlp.up_proj']
        }
    
    state_dict = model.state_dict()
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        layer_container.to(torch.bfloat16)
    new_state_dict = layer_container.state_dict() #torch.load(train_config.empty_model_path)
    for k in range(train_config.target_layer,train_config.target_layer+train_config.wipe_layer):
        if str(train_config.wipe_off_type) == '300M':
            if k == 4:
                for item in wipe_off['whole']:
                    state_dict[f'model.layers.{k}.{item}.weight'].data.copy_(new_state_dict[f'model.layers.{k}.{item}.weight'].data)
            elif k == 5:
                for item in wipe_off[str(train_config.wipe_off_type)]:
                    state_dict[f'model.layers.{k}.{item}.weight'].data.copy_(new_state_dict[f'model.layers.{k}.{item}.weight'].data)
        else:

            print(wipe_off[str(train_config.wipe_off_type)])
            for item in wipe_off[str(train_config.wipe_off_type)]:
                state_dict[f'model.layers.{k}.{item}.weight'].data.copy_(new_state_dict[f'model.layers.{k}.{item}.weight'].data)
    
    model.load_state_dict(state_dict)

    # Start the training process
    if train_config.mode == 0 or train_config.mode == 1:
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            train_config.gradient_accumulation_steps,
            train_config,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
            record=train_config.record_loss,
        )
        if not train_config.enable_fsdp or rank==0:
            [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    else:
        eval_ppl, eval_epoch_loss = evaluation(model, train_config, eval_dataloader, rank, tokenizer)
        if not train_config.enable_fsdp or rank==0:
            print(eval_epoch_loss)
        if train_config.record_loss:
            wandb.log({'Eval Loss':eval_epoch_loss, 'Eval Perplexity': eval_ppl})
    
    if train_config.record_loss:
        wandb.finish() 
            
if __name__ == "__main__":
    fire.Fire(main)
