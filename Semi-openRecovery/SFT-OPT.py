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
from training_utils.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing_opt

from training_utils.utils import fsdp_auto_wrap_policy
from training_utils.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from training_utils.utils.dataset_utils import get_preprocessed_dataset

from training_utils.utils.train_utils_opt import (
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
        attributes = dir(config)
        for attribute in attributes:
            if not attribute.startswith("__"):
                value = getattr(config, attribute)
                print(f"{attribute}: {value}")

def main(**kwargs):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # Update the configuration for the training and sharding process
    
    update_config((train_config, fsdp_config, alpaca_dataset), **kwargs)
    train_config.eta_min=train_config.lr*0.9
    train_config.T_max = int(train_config.data_per_GPU*train_config.num_epochs*2)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    print(train_config.model_name)

    if train_config.record_loss:
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
        }
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

    if local_rank == 0 and not os.path.exists(f'./project-opt/loss_txtfiles/{train_config.loss_store_path}/{train_config.group_name}'):
        os.makedirs(f'./project-opt/loss_txtfiles/{train_config.loss_store_path}/{train_config.group_name}', exist_ok=True)


    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    if not train_config.enable_fsdp or rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")

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
    
    model = OPTForCausalLM.from_pretrained(train_config.model_path, use_cache=use_cache)         
    opt_config = OPTConfig.from_pretrained(train_config.model_path)
    opt_config.use_cache = use_cache
    layer_container = OPTForCausalLM(opt_config)

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
            layer_container = BetterTransformer.transform(layer_container)  
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model = model.to(torch.bfloat16)
        layer_container  =layer_container.to(torch.bfloat16)

    # state_dict = model.state_dict()
    # keys = list(state_dict.keys())
    # kes_la = list(layer_container.state_dict().keys())
    # if local_rank == 0:
    #     with open (f'./{train_config.model_name}-layer-item.txt','w') as f:
    #         f.write('fsdp\n')
    #         for key in keys:
    #             f.write(key)
    #             f.write('\n')
    #         f.write('\n')
    #     with open (f'./layercontainer-layer-item.txt','w') as f:
    #         f.write('layercontainer\n')
    #         for key in kes_la:
    #             f.write(key)
    #             f.write('\n')
    #         f.write('\n')

    state_dict = model.state_dict() 
    new_state_dict = layer_container.state_dict()
    for k in range(train_config.target_layer,train_config.target_layer+train_config.wipe_layer):
        state_dict[f'model.decoder.layers.{k}.self_attn.k_proj.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.k_proj.weight'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.k_proj.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.k_proj.bias'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.v_proj.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.v_proj.weight'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.v_proj.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.v_proj.bias'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.q_proj.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.q_proj.weight'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.q_proj.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.q_proj.bias'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.out_proj.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.out_proj.weight'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn.out_proj.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn.out_proj.bias'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn_layer_norm.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn_layer_norm.weight'].data)
        state_dict[f'model.decoder.layers.{k}.self_attn_layer_norm.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.self_attn_layer_norm.bias'].data)
        state_dict[f'model.decoder.layers.{k}.fc1.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.fc1.weight'].data)
        state_dict[f'model.decoder.layers.{k}.fc1.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.fc1.bias'].data)
        state_dict[f'model.decoder.layers.{k}.fc2.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.fc2.weight'].data)
        state_dict[f'model.decoder.layers.{k}.fc2.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.fc2.bias'].data)
        state_dict[f'model.decoder.layers.{k}.final_layer_norm.weight'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.final_layer_norm.weight'].data)
        state_dict[f'model.decoder.layers.{k}.final_layer_norm.bias'].data.copy_(new_state_dict[f'model.decoder.layers.{k}.final_layer_norm.bias'].data)

    model.load_state_dict(state_dict)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        import functools
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.opt.modeling_opt import OPTDecoderLayer
        opt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            OPTDecoderLayer,
        },)

        model = FSDP(
            model,
            auto_wrap_policy= opt_auto_wrap_policy,
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
            apply_fsdp_checkpointing_opt(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

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
        wandb.finish() 
            
if __name__ == "__main__":
    fire.Fire(main)
