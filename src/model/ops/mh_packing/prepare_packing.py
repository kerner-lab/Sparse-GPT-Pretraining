# Copyright (c) 2025, Chenwei Cui, Benjamin Joseph Herrera, Kerner Lab
# SPDX-License-Identifier: MIT

import torch


@torch.no_grad()
@torch.compile()
def prepare_packing(expert_assign, expert_bincount, block_size):
    """
    In:  (num_head, pool_size); int64; contiguous; detached
         (num_head, num_expert); int64; contiguous; detached
         int
    Out: (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
         (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
         (); int64; contiguous; detached
         (num_head, num_block_q); int64; contiguous; detached
         (num_head, pool_size + padding_size); int64; contiguous; detached
    """
    # ----- #
    # Define variables
    # ----- #
    num_head, pool_size = expert_assign.shape
    num_expert = expert_bincount.size(1)
    # ----- #


    # ----- #
    # Get `padding_needed`
    # ----- #
    # (num_head, num_expert); int64; contiguous; detached
    padding_needed = (-expert_bincount) % block_size
    # ----- #


    # ----- #
    # Make sure `padding_size` matches across the `num_head` dimension
    # ----- #
    # Note: Instead of pad to the last expert, we can potentially use this step to promote load balancing
    # (num_head,); int64; contiguous; detached
    padding_size_all = padding_needed.sum(dim=1, keepdim=False)
    # (); int64; contiguous; detached
    padding_size = padding_size_all.max()
    # ----- #


    # ----- #
    # Update `padding_needed`
    # ----- #
    # Note: The FlexAttention call looks at padding_size_all, not padding_needed
    #       Therefore the last expert does not have to match
    # (num_head, num_expert); int64; contiguous; detached
    padding_needed[:, -1] += padding_size - padding_size_all
    # ----- #


    # ----- #
    # Apply offsets to `expert_assign`
    # ----- #
    # Get offsets
    # (num_head, 1); int64; contiguous; detached
    offsets = num_expert * torch.arange(num_head, device="cuda").view(num_head, 1)
    # Apply offsets to expert_assign
    # (num_head, pool_size); int64; contiguous; detached
    expert_assign = expert_assign + offsets
    # Flatten expert_assign
    # (num_head * pool_size,); int64; contiguous; detached
    expert_assign = expert_assign.view(num_head * pool_size)
    # ----- #


    # ----- #
    # Materialize `padding_tensor`
    # ----- #
    # Consider: Find a more parallelizable `repeat_interleave`
    # (num_head * padding_size,); int64; contiguous; detached
    padding_tensor = torch.arange(num_head * num_expert, device="cuda")
    padding_tensor = torch.repeat_interleave(padding_tensor, padding_needed.view(num_head * num_expert))
    # ----- #


    # ----- #
    # Apply padding to `expert_assign`
    # ----- #
    # (num_head * pool_size + num_head * padding_size); int64; contiguous; detached
    expert_assign = torch.concat((expert_assign, padding_tensor), dim=0)
    # ----- #


    # ----- #
    # Sort `expert_assign` to get `mapping`
    # ----- #
    # Note: stable=True for determinism; stable=False is functionally the same
    # (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    # (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    expert_assign, mapping = torch.sort(expert_assign, stable=False)
    # ----- #

    # ----- #
    # Unflatten `expert_assign` and then remove the offsets
    # ----- #
    # (num_head, pool_size + padding_size); int64; contiguous; detached
    expert_assign = expert_assign.view(num_head, pool_size + padding_size)
    # (num_head, pool_size + padding_size); int64; contiguous; detached
    expert_assign = expert_assign - offsets
    # ----- #


    # ----- #
    # Get `mapping_inv`
    # ----- #
    # (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    mapping_inv = torch.empty_like(mapping)
    # (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    mapping_inv[mapping] = torch.arange(mapping.size(0), device="cuda")
    # ----- #


    # ----- #
    # Get `num_block_q`
    # ----- #
    # Note: Should be provably divisible
    # (); int64; contiguous; detached
    num_block_q = (pool_size + padding_size) // block_size
    # ----- #


    # ----- #
    # Get `block_level_expert_assign`
    # ----- #
    # (num_head, num_block_q); int64; non-contiguous; detached
    block_level_expert_assign = expert_assign.view(num_head, num_block_q, block_size)[..., 0]
    # (num_head, num_block_q); int64; contiguous; detached
    block_level_expert_assign = block_level_expert_assign.contiguous()
    # ----- #


    # (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    # (num_head * pool_size + num_head * padding_size,); int64; contiguous; detached
    # (); int64; contiguous; detached
    # (num_head, num_block_q); int64; contiguous; detached
    # (num_head, pool_size + padding_size); int64; contiguous; detached
    return mapping, mapping_inv, padding_size, block_level_expert_assign, expert_assign
