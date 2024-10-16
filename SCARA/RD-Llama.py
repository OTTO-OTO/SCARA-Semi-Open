import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from pkg_resources import packaging
import wandb

import fire
import torch
import time
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
    XLNetModel, 
    XLNetTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from training_utils.utils.memory_utils import MemoryTrace
from tqdm import tqdm
from training_utils.configs import fsdp_config, train_config, alpaca_dataset
from training_utils.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from training_utils.utils import fsdp_auto_wrap_policy
from training_utils.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from training_utils.utils.dataset_utils_minidatasets import get_preprocessed_dataset

from training_utils.utils.train_utils_falcon import (
    train,
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

# this is used to randomly initialize the grey-box model
def privitizeModel(use_cache, rank, wipe_layers, full_priv_state_dict):

    model = LlamaForCausalLM.from_pretrained(
        train_config.model_path,  # pretrained model path
        load_in_8bit=True if train_config.quantization else None,
        device_map="auto" if train_config.quantization else None,
        use_cache=use_cache,
    )
    state_dict = model.state_dict()
    for k in range(0,wipe_layers):
        state_dict[f'model.layers.{k}.self_attn.q_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.self_attn.q_proj.weight'].data)
        state_dict[f'model.layers.{k}.self_attn.k_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.self_attn.k_proj.weight'].data)
        state_dict[f'model.layers.{k}.self_attn.v_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.self_attn.v_proj.weight'].data)
        state_dict[f'model.layers.{k}.self_attn.o_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.self_attn.o_proj.weight'].data) # (linear layer)
        state_dict[f'model.layers.{k}.self_attn.rotary_emb.inv_freq'].data.copy_(full_priv_state_dict[f'model.layers.{k}.self_attn.rotary_emb.inv_freq'].data)
        state_dict[f'model.layers.{k}.mlp.gate_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.mlp.gate_proj.weight'].data)
        state_dict[f'model.layers.{k}.mlp.up_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.mlp.up_proj.weight'].data)  # (feed forward)
        state_dict[f'model.layers.{k}.mlp.down_proj.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.mlp.down_proj.weight'].data)  # (feed forward)
        state_dict[f'model.layers.{k}.input_layernorm.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.input_layernorm.weight'].data)
        state_dict[f'model.layers.{k}.post_attention_layernorm.weight'].data.copy_(full_priv_state_dict[f'model.layers.{k}.post_attention_layernorm.weight'].data)

    model.load_state_dict(state_dict)

    model = wrap_model_fsdp(model, rank)
    
    return model

# this is used for wrap the model with FSDP Wrapper such that the model now can be trained using FSDP
def wrap_model_fsdp(model, rank):
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
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)
    
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
    return model


def main(**kwargs):
    update_config((train_config, fsdp_config, alpaca_dataset), **kwargs)
    train_config.eta_min=train_config.lr*0.9
    train_config.T_max = int(train_config.data_per_GPU*train_config.num_epochs)
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    print(train_config.model_name)

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
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
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
        
    use_cache = False if train_config.enable_fsdp else None      

    llama_config = LlamaConfig.from_pretrained(train_config.model_path)
    llama_config.use_cache = use_cache
    full_priv = LlamaForCausalLM(llama_config)
    # use the full privatization state dict to randomly initialized the grey-box model's private layers
    full_priv_state_dict = full_priv.state_dict()
    # wrap the full privatized model
    full_priv = wrap_model_fsdp(full_priv, rank)
    # calculate the recovery difficulty of full privatization
    full_priv_score = RDCal(full_priv, train_config, train_dataloader, rank)


    # initialize the recovery difficulty of the semi-open model:
    Score= -1

    # load semi-open model
    for wipe_layers in range(1,llama_config.num_hidden_layers):
        clear_gpu_cache(local_rank)

        # privatization starts from the first layer
        model = privitizeModel(use_cache, rank, wipe_layers, full_priv_state_dict)
        start_time = time.time()
        temp_score = RDCal(model, train_config, train_dataloader, rank)
        end_time = time.time()
        print('Calculation Time:', end_time-start_time)
        if temp_score*(1+train_config.tolerance_magnitude) >= full_priv_score:
            print('The privatization layer count is',wipe_layers) 

# Calculating the Recovery Difficulty
def RDCal(model, train_config, eval_dataloader, local_rank):
    
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()

    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    
    return eval_epoch_loss


if __name__ == "__main__":
    fire.Fire(main)
