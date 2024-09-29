# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset

PROMPT_DICT = {
    "prompt_choices": (
        # "The following are multiple choice questions (with answers)."
        "The following is a multiple choice question, paired with choices."
        "Answer the question in format: 'Choice:content'.\n\n"
        "### Question:\n{question}\n\n### Choices:\n{choices}\n\n### Answer:"
    ),
    "prompt_no_choices": (
        "Below is a question with no choices. "
        "Write the correct answer that appropriately solve the question.\n\n"
        "### Question:\n{question}\n\n### Answer:"
    ),
}

class QADataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        # self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = json.load(open(dataset_config.data_path))
        else:
            self.ann = json.load(open(dataset_config.eval_file_path))
            # if dataset_config.dataset_type == 'eval':
            #     if dataset_config.eval_all:
            #         self.ann = self.ann
            #     else:
            #         self.ann = self.ann[:dataset_config.eval_dataset_length] # 全部load进去
            # elif dataset_config.dataset_type == 'train':
            #     self.ann = self.ann[:640]# initial 200

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("choices", "") == "":
            prompt = PROMPT_DICT["prompt_no_choices"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_choices"].format_map(ann)
        example = prompt + ann["answer"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)  # 从输入变量完全复刻一个相同的变量，无论怎么改变新变量，原有变量的值都不会受到影响
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0  # ~x = -(x+1)
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
