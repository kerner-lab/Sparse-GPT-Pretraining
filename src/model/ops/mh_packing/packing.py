# Copyright (c) 2025, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import torch


@torch.compile()
def packing(input_tensor, padding_size, mapping):
    """
    In:  (num_head, pool_size, emb_size); bfloat16; contiguous
         (); int64; contiguous; detached
         (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    Out: (num_head, pool_size + padding_size, emb_size); bfloat16; contiguous
    """
    # Define variables
    num_head, pool_size, emb_size = input_tensor.shape

    # Flatten `input_tensor`
    # (num_head * pool_size, emb_size); bfloat16; contiguous
    input_tensor = input_tensor.view(num_head * pool_size, emb_size)

    # Define `padding_tensor`
    # (num_head * padding_size, emb_size); bfloat16; contiguous
    padding_tensor = torch.zeros((num_head * padding_size, emb_size), dtype=torch.bfloat16, device="cuda")

    # Apply padding to `input_tensor`
    # (num_head * pool_size + num_head * padding_size, emb_size); bfloat16; contiguous
    input_tensor = torch.concat((input_tensor, padding_tensor), dim=0)

    # Permute `input_tensor` using `mapping`
    # (num_head * (pool_size + padding_size), emb_size); bfloat16; contiguous
    input_tensor = input_tensor[mapping]

    # Unflatten `input_tensor`
    # (num_head, pool_size + padding_size, emb_size); bfloat16; contiguous
    input_tensor = input_tensor.view(num_head, pool_size + padding_size, emb_size)

    # (num_head, pool_size + padding_size, emb_size); bfloat16; contiguous
    return input_tensor
