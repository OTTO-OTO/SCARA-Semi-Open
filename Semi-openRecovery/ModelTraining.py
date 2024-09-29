import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
import torch
import os


# this is the running method for finetuning Llama model
def LlamaTraining(startLayer, endLayer, step, wipe_layer, worker_name, node_num):

    model_folder = 'project-llama2' # folder name to store your training and evaluation loss
    os.makedirs(f'./{model_folder}/logs', exist_ok=True)
    os.makedirs(f'./{model_folder}/commands', exist_ok=True)

    master_ports = [25000, 25008, 25016, 25024]
    pool_size = int(torch.cuda.device_count()/node_num)
    print(pool_size)
    num_cycle = cycle(range(pool_size))
    commands = []
    seed = [68]
    for sed in seed:
        for layer in range(startLayer, endLayer, step):

            num = next(num_cycle)

            command = f'torchrun --nnodes 1 --nproc_per_node {node_num} --master_port={master_ports[num]} SFT-Llama2.py \
                --enable_fsdp \
                --model_name llama\
                --project_name Llama-Sft-On-MixDataset-seed{sed} \
                --group_name Layer{layer} \
                --model_path meta-llama/Llama-2-7b-chat-hf \
                --dist_checkpoint_root_folder path_to_save_checkpoints \
                --dist_checkpoint_root_load_folder path_to_save_checkpoints \
                --dist_checkpoint_folder eact_folder_name_of_your_checkpoints\
                --target_layer {layer}\
                --lr 2e-5\
                --weight_decay 0.1\
                --dataset gen_dataset \
                --data_path path_to_your_training_data/validation_data.json\
                --eval_file_path path_to_your_validation_data/validation_data.json\
                --batch_size_training {int(128/node_num)} \
                --val_batch_size {int(128/node_num)}\
                --num_epochs 5\
                --pure_bf16\
                --low_cpu_fsdp False\
                --save_model True\
                --save_optimizer\
                --load_from_ckpt False\
                --scheduler Cosine\
                --record_loss True\
                --data_per_GPU 400 \ 
                --fsdp_cpu_offload False\
                --gamma 1\
                --wipe_layer {wipe_layer}\
                --seed {sed}\
                --model_type llama\
                --loss_store_path mmlu-alpacamix-loss-record-{sed}\
                --eval_step 40'
            commands.append(command)

    with open(f'./{model_folder}/commands/commands-{worker_name}.txt', 'a') as f:
        for command in commands:
            f.write(f"command\n\n")


    gpu_pairs=[]
    for i in range(0, torch.cuda.device_count(), node_num):
        temp = []
        for j in range(node_num):
            temp.append(str(i+j))
        gpu_pairs.append(','.join(temp))
    print(gpu_pairs)

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = []
        gpu_cycle = cycle(range(pool_size)) 
        
        for command in commands:
            gpu_pair_num = next(gpu_cycle)
            startIndex = command.find('layer')
            marker = command[startIndex: int(7+startIndex)]
            futures.append(executor.submit(run_command, command, gpu_pairs[gpu_pair_num], model_folder, marker, worker_name))
        
        
        for future in futures:
            future.result()
    print("All commands have been executed.")

# this is the running method for finetuning Llama model under different privatization size
def llama2PrivatizationSize(node_num, worker_name='4090-2'):
    model_folder = 'project-llama2'
    commands = []
    os.makedirs(f'./{model_folder}/logs', exist_ok=True)
    master_ports = [25000, 25008, 25016, 25024, 25009,26007,29001,29008,29162]
    pool_size = int(torch.cuda.device_count()/node_num)
    print(pool_size)
    num_cycle = cycle(range(len(master_ports)))
    wipe_off_types=[
        {
            'wipe_layer': 1,
            'wipe_off_type': '0.25',
            'target_layer':4,
            'model_name':'layer4-k'
        },{
            'wipe_layer': 1,
            'wipe_off_type': '0.5',
            'target_layer':4,
            'model_name':'layer4-kq'
        },
        {
            'wipe_layer': 1,
            'wipe_off_type': '1',
            'target_layer':4,
            'model_name':'layer4-kqvo'
        },
        {
            'wipe_layer': 2,
            'wipe_off_type': 'whole',
            'target_layer':4,
            'model_name':'layer4-5'
        },
        {
            'wipe_layer': 5,
            'wipe_off_type': 'whole',
            'target_layer':2,
            'model_name':'layer2-6'
        },{
            'wipe_layer': 10,
            'wipe_off_type': 'whole',
            'target_layer':0,
            'model_name':'layer0-9'
        },{
            'wipe_layer': 16,
            'wipe_off_type': 'whole',
            'target_layer':0,
            'model_name':'layer0-15'
        },{
            'wipe_layer': 1,
            'wipe_off_type': '50M',
            'target_layer':4,
            'model_name':'layer4-kqv'
        },{
            'wipe_layer': 1,
            'wipe_off_type': '100M',
            'target_layer':4,
            'model_name':'layer4-qvo-up'
        },{
            'wipe_layer': 1,
            'wipe_off_type': '160M',
            'target_layer':4,
            'model_name':'layer4-kqvo-up-down'
        },{
            'wipe_layer': 2,
            'wipe_off_type': '300M',
            'target_layer':4,
            'model_name':'layer4-5qvo-up'
        },{
            'wipe_layer': 3,
            'wipe_off_type': 'whole',
            'target_layer':3,
            'model_name':'layer3-5'
        }
    ]
    seeds = [68]
    for seed in seeds:
        for item in wipe_off_types:
            num = next(num_cycle)

            command = f"torchrun --nnodes 1 --nproc_per_node {node_num} --master_port={master_ports[num]} PrivatizedSizes-Llama.py \
                --enable_fsdp \
                --model_name llama\
                --project_name Llama2-Sft-On-MixDataset-Seed{seed} \
                --group_name {item['model_name']}\
                --model_path meta-llama/Llama-2-7b-chat-hf\
                --dist_checkpoint_root_folder path_to_save_checkpoints \
                --dist_checkpoint_root_load_folder path_to_save_checkpoints \
                --dist_checkpoint_folder eact_folder_name_of_your_checkpoints\
                --target_layer {item['target_layer']}\
                --lr 2e-5\
                --weight_decay 0.1\
                --dataset gen_dataset \
                --data_path path_to_your_training_data/validation_data.json\
                --eval_file_path path_to_your_validation_data/validation_data.json\
                --batch_size_training {int(128/node_num)} \
                --val_batch_size {int(128/node_num)}\
                --num_epochs 5\
                --pure_bf16\
                --low_cpu_fsdp False\
                --save_model True\
                --save_optimizer\
                --load_from_ckpt False\
                --scheduler Cosine\
                --record_loss False\
                --data_per_GPU 400\
                --fsdp_cpu_offload False\
                --gamma 1\
                --wipe_layer {item['wipe_layer']}\
                --seed {seed}\
                --wipe_off_type {item['wipe_off_type']}\
                --model_type llama\
                --loss_store_path Supplement-loss-record-seed{seed}\
                --eval_step 40"
            commands.append(command)
        
    os.makedirs(f'./{model_folder}/commands/', exist_ok=True)
    with open(f'./{model_folder}/commands/commands-{worker_name}.txt', 'w') as f:
        for command in commands:
            f.write(f"{command}\n\n")

    gpu_pairs=['0,1,2,3', '4,5,6,7']
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = []
        gpu_cycle = cycle(range(pool_size)) 
        
        for command in commands:
            gpu_pair_num = next(gpu_cycle)
            startIndex = command.find('dist_checkpoint_folder')
            marker = command[startIndex: int(14+startIndex)]
            futures.append(executor.submit(run_command, command, gpu_pairs[gpu_pair_num], model_folder, marker, worker_name))
        
        for future in futures:
            future.result()
    print("All commands have been executed.")


def run_command(command, gpu, model_folder, marker, worker):
    
    start_time = time.time()
        
    # 记录执行信息和使用的GPU
    with open(f'./{model_folder}/logs/Training-Detai-{worker}.txt', 'a') as f:
        f.write(f"Executing on GPU: {marker}, {worker}\n")
    
    # 执行命令
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    subprocess.run(command, shell=True, executable="/bin/bash", check=True, env=env)
    
    with open(f'./{model_folder}/logs/Training-Detai-{worker}.txt', 'a') as f:
        f.write("Execution completed.\n")
    
    end_time = time.time()
    with open(f'./{model_folder}/logs/Training-Detai-{worker}.txt', 'a') as f:
        f.write(f"Time Consuming: {end_time - start_time}.\n\n")


if __name__ == '__main__':
    LlamaTraining(startLayer=0, endLayer=32, step=2, wipe_layer=1, worker_name='xx', node_num=4)