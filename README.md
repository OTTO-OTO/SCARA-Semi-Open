# SCARA

Welcome to the repository for "Archilles' Heel in Semi-open LLMs: Hiding Bottom against Recovery Attacks" SCARA stands for **S**elective **C**losed-sourcing Approach **A**gainst **R**ecovery **A**ttack, which is an approach that keeps only a few bottom layer as closed-source to enhance model customizability and preserve resilience to semi-open model recovery attack. This code is adapted from the Llama-recipe repository(https://github.com/meta-llama/llama-recipes).

## Repository Info

This repository contains a total of four folders:

* **Benchmarks**: We have listed the LawEvaluation code we used. Specifically, we use perplexity to evaluate the model's performance.
* **Customization**: During the Customization phase, we utilized the Llama Factory repository. Therefore, we provide the YAML files and corresponding configurations used for training.
* **SCARA**: We provide the code for calculating Recovery Difficulty, which is used to implement the functionality of SCARA.
* **Semi-openRecovery**: We provide the code for implementing Semi-open model Recovery, along with the verification code for the specific transition layer.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Semi-open Model Recovery

1. **Running the EX-Priv Algorithm**: If you want to run the SCARA algorithm, you can use the provided bash script in SCARA folder:

    ```bash
    bash runSCARA-Llama.sh
    ```

2. **Replicating Our Training Experiments**: To replicate our experiments on training models, you can execute the following command in Semi-openRecovery folder:

    ```bash
    torchrun --nnodes 1 --nproc_per_node 4 --master_port=25001 SFT-Llama2.py \
        --enable_fsdp \
        --model_name llama \
        --project_name Llama-Sft-On-MixDataset \
        --group_name Layer0 \
        --model_path meta-llama/Llama-2-7b-chat-hf \
        --dist_checkpoint_root_folder path_to_save_checkpoints \
        --dist_checkpoint_root_load_folder path_to_save_checkpoints \
        --dist_checkpoint_folder exact_folder_name_of_your_checkpoints \
        --target_layer 0 \
        --lr 2e-5 \
        --weight_decay 0.1 \
        --dataset gen_dataset \
        --data_path path_to_your_training_data/validation_data.json \
        --eval_file_path path_to_your_validation_data/validation_data.json \
        --batch_size_training 32 \
        --val_batch_size 32 \
        --num_epochs 5 \
        --pure_bf16 \
        --low_cpu_fsdp False \
        --save_model True \
        --save_optimizer \
        --load_from_ckpt False \
        --scheduler Cosine \
        --record_loss True \
        --data_per_GPU 400 \ 
        --fsdp_cpu_offload False \
        --gamma 1 \
        --wipe_layer 1 \
        --seed 68 \
        --model_type llama \
        --loss_store_path mmlu-alpacamix-loss-record \
        --eval_step 40'
    ```

    Alternatively, you can use the `ModelTraining.py` file we provide. This file contains the execution code for the Llama2-7B model. 

    ```bash
    python ModelTraining.py
    ```

    If you wish to extend this to other models, simply change the target python file name in the command line:

    ```bash
    torchrun --nnodes 1 --nproc_per_node 4 --master_port=25001 SFT-Mistral.py \...
    ```

3. **Additional Configuration Settings**: For more parameter settings, you can find them in the `EX-Priv/training_utils/configs/training.py`.
