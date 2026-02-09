# Copyright (c) 2025, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import torch


@torch.compile()
def unpacking(input_tensor, padding_size, mapping_inv):
    """
    In:  (num_head, pool_size + padding_size, emb_size); bfloat16; contiguous
         (); int64; contiguous; detached
         (num_head * (pool_size + padding_size),); int64; contiguous; detached
    Out: (num_head, pool_size, emb_size); bfloat16; contiguous
    """
    # Define variables
    num_head = input_tensor.size(0)
    pool_size = input_tensor.size(1) - padding_size
    emb_size = input_tensor.size(2)

    # Flatten `input_tensor`
    # (num_head * (pool_size + padding_size), emb_size); bfloat16; contiguous
    input_tensor = input_tensor.view(num_head * (pool_size + padding_size), emb_size)

    # Permute `input_tensor` using `mapping_inv`
    # (num_head * pool_size + num_head * padding_size, emb_size); bfloat16; contiguous
    input_tensor = input_tensor[mapping_inv]

    # Remove padding
    # Note: For example, when j == 0, input_tensor[:-j] becomes empty; Below is a safer approach
    # (num_head * pool_size, emb_size); bfloat16; contiguous
    input_tensor = input_tensor[:num_head * pool_size]

    # Unflatten `input_tensor`
    # (num_head, pool_size, emb_size); bfloat16; contiguous
    input_tensor = input_tensor.view(num_head, pool_size, emb_size)

    # (num_head, pool_size, emb_size); bfloat16; contiguous
    return input_tensor
