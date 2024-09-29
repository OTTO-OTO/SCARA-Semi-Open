#!/bin/bash
LOG_NAME=/LlamaScore.log
# find /root/HHB/ -name $LOG_NAME -exec rm -rf {} \;
rm $LOG_NAME
# export NCCL_P2P_DISABLE=1

screen -L -Logfile $LOG_NAME \
torchrun --nnodes 1 --nproc_per_node 8 SCARA-Llama.py \
  --enable_fsdp \
  --dataset gen_dataset \
  --model_path meta-llama/Llama-2-7b-chat-hf\
  --data_path /path/to/your/datafiles\
  --eval_file_path /path/to/your/datafiles\
  --batch_size_training 60 \ 
  --use_fast_kernels \
  --pure_bf16 \
  --tolerance_magnitude 0.2 \
  --low_cpu_fsdp False\
  --fsdp_cpu_offload


# batch_size_training stands for the total batch size used for identifying the privatization layer count. 
# data_path is the path of your evaluation dataset