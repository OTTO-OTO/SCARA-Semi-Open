# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, GPT2Block)
# check_fn = lambda submodule: isinstance(submodule, FalconDecoderLayer)


def apply_fsdp_checkpointing_gpt2(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
