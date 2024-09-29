# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    project_name = ''
    group_name = ''
    model_name: str="llama"
    model_path: str="meta-llama/Llama-2-7b-chat-hf"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True # wether to run the validation
    batch_size_training: int=32
    gradient_accumulation_steps: int=1
    num_epochs: int=5
    num_workers_dataloader: int=1
    lr: float=2e-5
    weight_decay: float=0.1
    gamma: float= 1
    seed: int=68
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=32
    dataset = "gen_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"  # save peft model
    freeze_layers: bool = False # whether to freeze layers during training
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True # whether to save the checkpoint
    dist_checkpoint_root_folder: str="model_ckpt" # will be used if using FSDP
    dist_checkpoint_root_load_folder: str="model_ckpt"
    dist_checkpoint_folder: str="SFT" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    load_from_ckpt: bool=False
    scheduler: str='StepLR'
    data_per_GPU: int = 400 # the iteration number within an epoch
    T_max: int=1000
    eta_min: float=0
    record_loss: bool=False # whether use wandb to record the training loss
    wipe_layer: int=0 # privatized layer number
    target_layer: int = 0 # the initial layer number when performing privatization
    wipe_all: bool = False
    lambda_step: int = 1
    model_type: str='llama'
    eval_step: int = 10
    loss_json_num: int=0
    start_model: int = 0
    loss_store_path: str='mmlu-alpacamix-loss-record'
    tolerance_magnitude: float=0.2
    
    
    
