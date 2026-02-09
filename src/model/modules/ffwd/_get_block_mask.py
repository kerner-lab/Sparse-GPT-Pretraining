# Copyright (c) 2025, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import torch
from torch.nn.attention.flex_attention import BlockMask


# Note: Potential improvement: kv_indices could be sparse
# Note: The outputs are int32, as required by Flex Attention
@torch.no_grad()
@torch.compile()
def _get_block_mask(block_level_expert_assign, num_expert, expert_size, block_size):
    """
    In:  (num_head, num_block_q); int64; contiguous; detached
         int
         int
         int
    Out: `block_mask`
    """
    # ----- #
    # Define variables
    # ----- #
    num_head, num_block_q = block_level_expert_assign.shape
    num_block_per_expert = expert_size // block_size
    num_block_k = num_expert * num_block_per_expert
    # ----- #

    # ----- #
    # Define `kv_num_blocks` and `kv_indices`
    # ----- #
    # TODO: Create kv_indices only once
    # Note: They are zeros because we only have full blocks
    # (1, num_head, num_block_q); int32; contiguous
    kv_num_blocks = torch.zeros((1, num_head, num_block_q), dtype=torch.int32, device="cuda")
    # (1, num_head, num_block_q, num_block_k); int32; contiguous
    kv_indices = torch.zeros((1, num_head, num_block_q, num_block_k), dtype=torch.int32, device="cuda")
    # ----- #

    # ----- #
    # Define `full_kv_num_blocks`
    # ----- #
    # (1, num_head, num_block_q); int32; contiguous
    full_kv_num_blocks = num_block_per_expert * torch.ones((1, num_head, num_block_q), dtype=torch.int32, device="cuda")
    # ----- #

    # ----- #
    # Define `full_kv_indices`
    # ----- #
    # Note: We create indices [0, ..., num_block_k - 1], then add expert-specific offsets
    #       The first num_block_per_expert elements always map to the assigned expert's blocks (in-bounds)
    #       Elements beyond num_block_per_expert overflow but are unused downstream
    # (num_block_k,); int32; contiguous
    full_kv_indices = torch.arange(num_block_k, dtype=torch.int32, device="cuda")
    # (1, 1, 1, num_block_k); int32; contiguous
    full_kv_indices = full_kv_indices.view(1, 1, 1, num_block_k)
    # (1, num_head, 1, num_block_k); int32; non-contiguous
    full_kv_indices = full_kv_indices.expand(1, num_head, 1, num_block_k)

    # Apply offsets
    # (1, num_head, num_block_q, 1); int32; contiguous
    offsets = num_block_per_expert * block_level_expert_assign.to(torch.int32).view(1, num_head, num_block_q, 1)
    # (1, num_head, num_block_q, num_block_k); int32; contiguous
    full_kv_indices = full_kv_indices + offsets
    # ----- #

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=None,
    )
