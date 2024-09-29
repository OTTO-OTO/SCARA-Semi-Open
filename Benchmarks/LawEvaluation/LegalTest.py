# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# 这个实验只针对于偶数层
# 1. 对于layer l,  我们保留layer 1 -- l-1的参数, 抹除掉layer l,  舍弃layer > l的层级进行测试,  
# 2. 使用val set上测试得到的结果计算我们的metric
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from pkg_resources import packaging
import wandb

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
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
    MistralForCausalLM,
    PhiForCausalLM,
    default_data_collator,
    AutoTokenizer, 
    AutoModel,
    XLNetModel, 
    XLNetTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import pandas as pd


def main(output_name,data_path):
    
    model_list = ['/path/to/your/model']
    # []
    import time
    for model_path in model_list:

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.add_special_tokens(
                {
                    "pad_token": "<PAD>",
                }
            )
        # random_initialized_state_dict = 0
        model_path_lower = model_path.lower()
        if 'mistral' in model_path_lower:
            model = MistralForCausalLM.from_pretrained(model_path, device_map='auto')
        elif 'llama' in model_path_lower:
            model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto')
        elif 'phi' in model_path_lower:
            model = PhiForCausalLM.from_pretrained(model_path, device_map='auto')
        start_time = time.time()
        total_acc = 0
        total_questions = 0
        
        for root, dirs, files in os.walk(data_path):
            for dir_name in dirs:
                # 获取子文件夹的完整路径
                sub_dir_path = os.path.join(root, dir_name)
                acc, nums = evaluation_perplextiy(model,tokenizer,data_path = sub_dir_path)
                with open(output_name, 'a') as file:
                    file.write(f"{model_path}\t{acc}\t{nums}\t{dir_name}\n")
                total_questions += nums
                total_acc += acc
        
        with open(output_name, 'a') as file:
            file.write(f"==========================================================================\n")
            file.write(f"{model_path}\t total_acc={total_acc/total_questions}\t {total_acc}\t{total_questions}\n")
            file.write(f"==========================================================================\n")
            file.write(f"==========================================================================\n\n\n")
        print(total_acc, total_questions)
        end_time = time.time()




def evaluation_perplextiy(model,tokenizer,data_path):
    from tqdm import tqdm

    data = pd.read_csv(f'{data_path}/test.tsv', sep='\t')
    with open(f'{data_path}/prompt.txt', 'r') as file:
        blank_prompt = file.read()
    
    model.eval()
    accurate = 0
    total_questions = data.shape[0]
    answer_list = data['answer'].drop_duplicates().tolist()
    with torch.no_grad():
        for index, da in tqdm(data.iterrows(), total=data.shape[0]):  
            perplex = 1000000
            model_choice = 'non'
            for ans in answer_list:
                if 'question' in data.columns.tolist():
                    ques = blank_prompt.format(text=da['text'], question=da['question']) + ans
                else:
                    ques = blank_prompt.format(da['text']) + ans
                # ques = blank_prompt.format(da['text']) + ans
                inputs = tokenizer(ques, return_tensors='pt').to(model.device)  # 使用 tokenizer 编码输入
                output = model(**inputs, labels=inputs['input_ids'])  # 计算损失
                cur_per = torch.exp(output.loss)  # 使用损失的指数作为困惑度
                if cur_per < perplex:
                    perplex = cur_per
                    model_choice = ans

            if model_choice == da['answer']:
                accurate += 1
    return accurate, total_questions


if __name__ == "__main__":
    output_name = 'log.txt'
    data_path = './law_data'
    with open(output_name, 'a') as file:
        file.write(f"DATA PATH:{data_path}\n")
        file.write(f"{'model_path'}\t{'accuracy'}\t{'total_questions'}\n")
    main(output_name, data_path)

