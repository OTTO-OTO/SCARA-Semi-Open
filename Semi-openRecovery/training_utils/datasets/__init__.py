# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from training_utils.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from training_utils.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from training_utils.datasets.alpaca_dataset_minidatasets import InstructionDataset as get_alpaca_minidataset
from training_utils.datasets.generalDataset import Template as getGen_dataset
from training_utils.datasets.QA_dataset import QADataset as get_QA_dataset
from training_utils.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset