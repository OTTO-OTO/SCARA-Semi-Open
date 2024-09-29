# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "./training_utils/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "./lama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "./training_utils/datasets/alpaca_data.json"
    file_path: str = 'alpaca_data.json'
    dataset_type: str = 'train'
    eval_dataset_length: int = 38790
    eval_all: bool = False

@dataclass
class qa_dataset:
    dataset: str = "qa_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "./training_utils/datasets/qa_dataset.json"
    file_path: str = 'qa_dataset.json'
    dataset_type: str = 'train'
    eval_dataset_length: int = 38790
    eval_all: bool = False
    eval_file_path: str = '/gemini/data-2/MMLU/mmlu-val.json'

@dataclass
class gen_dataset:
    dataset: str = "gen_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/gemini/data-2/mmlu-alpaca-train-51k.json"
    file_path: str = 'mmlu_dataset.json'
    dataset_type: str = 'train'
    eval_dataset_length: int = 38790
    eval_all: bool = False
    eval_file_path: str = '/gemini/data-2/mmlu-alpaca-val.json'
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"