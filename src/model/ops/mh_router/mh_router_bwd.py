# Copyright (c) 2025, Rockwell Jackson, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    # torch.empty() already returns page-aligned, UC-cached memory
    return torch.empty(size, dtype=torch.int8, device="cuda")

triton.set_allocator(alloc_fn)


@triton.jit
def mh_router_bwd(
    # Input tensors
    X,            # (batch_size, num_token, num_head, head_size); bfloat16; contiguous
    R,            # (num_head, head_size, num_expert); float32; contiguous
    top_idx,      # (batch_size, num_token, num_head, k); int64; contiguous
    d_top_logit,  # (batch_size, num_token, num_head, k); float32; contiguous
    dX,           # (batch_size, num_token, num_head, head_size); bfloat16; contiguous
    dR,           # (num_head, head_size, num_expert); float32; contiguous
    # Strides
    stride_X_B, stride_X_T, stride_X_H, stride_X_D,
    stride_R_H, stride_R_D, stride_R_E,
    stride_top_idx_B, stride_top_idx_T, stride_top_idx_H, stride_top_idx_K,
    stride_d_top_logit_B, stride_d_top_logit_T, stride_d_top_logit_H, stride_d_top_logit_K,
    stride_dX_B, stride_dX_T, stride_dX_H, stride_dX_D,
    stride_dR_H, stride_dR_D, stride_dR_E,
    # Shape information
    head_size: tl.constexpr,
    number_of_experts: tl.constexpr,
    number_of_tokens: tl.constexpr,
    # Compile Time Constants
    K: tl.constexpr,
    # Constants
    BLOCK_N: tl.constexpr,  # Along token Dimension
):
    # Compiler hints
    tl.assume(head_size % 16 == 0)
    tl.assume(number_of_experts % 16 == 0)
    # tl.assume(number_of_tokens % 16 == 0)  # Consider enabling this
    tl.assume(BLOCK_N % 16 == 0)

    # Note: grid coordinates which reflect tensor tile position
    idx_b = tl.program_id(0)  # Batch ID
    idx_t = tl.program_id(1)  # Token Tile ID
    idx_h = tl.program_id(2)  # Head ID

    # Base pointers for this (b, h)
    X = X + idx_b * stride_X_B + idx_h * stride_X_H
    R = R + idx_h * stride_R_H
    top_idx = top_idx + idx_b * stride_top_idx_B + idx_h * stride_top_idx_H
    d_top_logit = d_top_logit + idx_b * stride_d_top_logit_B + idx_h * stride_d_top_logit_H
    dX = dX + idx_b * stride_dX_B + idx_h * stride_dX_H
    dR = dR + idx_h * stride_dR_H

    # Note: `row0` is the starting row index into X for this block
    row0 = idx_t * BLOCK_N
    # Compiler hint
    tl.multiple_of(row0, BLOCK_N)

    # Make the tensor descriptors
    X_desc = tl.make_tensor_descriptor(
        X,
        shape=[number_of_tokens, head_size],
        strides=[stride_X_T, stride_X_D],
        block_shape=[BLOCK_N, head_size],
    )
    dX_desc = tl.make_tensor_descriptor(
        dX,
        shape=[number_of_tokens, head_size],
        strides=[stride_dX_T, stride_dX_D],
        block_shape=[BLOCK_N, head_size],
    )

    # Get the offsets for the row tile
    offsets = row0 + tl.arange(0, BLOCK_N)              # (BLOCK_N,)
    mask = offsets < number_of_tokens                   # (BLOCK_N,)

    # Load the X tile once (float32 for accumulation)
    tile_X = tl.load_tensor_descriptor(X_desc, [row0, 0]).to(tl.float32)  # (BLOCK_N, D)

    # Prepare dims for expert gather / scatter
    offsets_D = tl.arange(0, head_size)                 # (D,)

    # Accumulator for dX across top-k
    dX_accum = tl.zeros([BLOCK_N, head_size], dtype=tl.float32)

    # Run the pipeline K times
    for i in range(K):
        # --- LOADS with correct K stride ---
        # vec_top_idx: (BLOCK_N,) int64 -> safe-indexed and cast for addressing
        vec_top_idx_k = tl.load(
            top_idx + offsets * stride_top_idx_T + i * stride_top_idx_K,
            mask=mask,
            other=0
        ).to(tl.int32)

        # vec_d_top_logit: (BLOCK_N,) float32
        vec_d_top_logit_k = tl.load(
            d_top_logit + offsets * stride_d_top_logit_T + i * stride_d_top_logit_K,
            mask=mask,
            other=0.0
        )

        # For out-of-range rows, force index 0 and contribution 0
        safe_E = tl.where(mask, vec_top_idx_k, 0)       # (BLOCK_N,)
        safe_dy = tl.where(mask, vec_d_top_logit_k, 0.0)

        # --- dR update: atomic add of (X * dy) into (D x E) slice ---
        contrib_R = tile_X * safe_dy[:, None]           # (BLOCK_N, D) float32
        tl.atomic_add(
            dR + safe_E[:, None] * stride_dR_E + offsets_D[None, :] * stride_dR_D,
            contrib_R
        )

        # --- dX accumulation: gather R[:, E] * dy and sum over top-k ---
        tile_R_k = tl.load(
            R + safe_E[:, None] * stride_R_E + offsets_D[None, :] * stride_R_D
        ).to(tl.float32)                                # (BLOCK_N, D)
        dX_accum += tile_R_k * safe_dy[:, None]         # (BLOCK_N, D) float32

    # Store accumulated dX (non-atomic, rows are disjoint across programs)
    tl.store_tensor_descriptor(dX_desc, [row0, 0], dX_accum.to(tl.bfloat16))
