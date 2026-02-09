# Copyright (c) 2025, Rockwell Jackson, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    # torch.empty() already returns page-aligned, UC-cached memory
    return torch.empty(size, dtype=torch.int8, device="cuda")

triton.set_allocator(alloc_fn)


# ---- float32 <-> ordered uint32 key helpers (monotone with fp32 order) ----

@triton.jit
def _get_masks_uint32():
    tm: tl.constexpr = 1 << 31
    fm: tl.constexpr = (1 << 32) - 1
    return tl.full([1], tm, dtype=tl.uint32), tl.full([1], fm, dtype=tl.uint32)

@triton.jit
def _fp32_to_ordered_uint32(x_f32):
    x_u = x_f32.to(tl.uint32, bitcast=True)
    tm, fm = _get_masks_uint32()
    return x_u ^ tl.where((x_u & tm) != 0, fm, tm)

@triton.jit
def _ordered_uint32_to_fp32(x_u32):
    tm, fm = _get_masks_uint32()
    y = x_u32 ^ tl.where((x_u32 & tm) == 0, fm, tm)
    return y.to(tl.float32, bitcast=True)


@triton.jit
def mh_router_fwd(
    # Input tensors
    X,             # (batch_size, num_tokens, num_head, head_size); bfloat16; contiguous
    R,             # (num_head, head_size, number_of_experts); float32; contiguous
    auxfree_bias,  # (num_head, num_expert); float32; contiguous
    top_logit,     # (batch_size, num_tokens, num_head, K); float32; contiguous
    top_idx,       # (batch_size, num_tokens, num_head, K); int64; contiguous
    # Strides
    stride_X_B, stride_X_T, stride_X_H, stride_X_D,
    stride_R_H, stride_R_D, stride_R_E,
    stride_auxfree_bias_H, stride_auxfree_bias_E,
    stride_top_logit_B, stride_top_logit_T, stride_top_logit_H, stride_top_logit_K,
    stride_top_idx_B, stride_top_idx_T, stride_top_idx_H, stride_top_idx_K,
    # Shape information
    head_size: tl.constexpr,  # Head size
    number_of_experts: tl.constexpr,  # Size of experts / hidden layer size
    number_of_tokens: tl.constexpr,
    # Compile Time Flags
    USE_SIGMOID: tl.constexpr,  #Whether or not to apply sigmoid
    # Compile Time Constants
    K: tl.constexpr,  # K for top-k, number of expert logits to return per token
    # Constants
    BLOCK_N: tl.constexpr,  # Along token Dimension
    BLOCK_M: tl.constexpr,  # Along expert dimension
):
    # Compiler hints
    tl.assume(head_size % 16 == 0)
    tl.assume(number_of_experts % 16 == 0)
    tl.assume(BLOCK_N % 16 == 0)
    tl.assume(BLOCK_M % 16 == 0)

    # Note: grid coordinates which reflect tensor tile position
    # Note: `tl.program_id` returns int32
    idx_b = tl.program_id(0)  # Batch ID
    idx_t = tl.program_id(1)  # Token Tile ID
    idx_h = tl.program_id(2)  # Head ID

    # Note: `row0` is the starting row index into X for this block
    row0 = idx_t * BLOCK_N
    tl.multiple_of(row0, BLOCK_N)  # Compiler hint

    # Base pointers for this (b, h)
    X = X + idx_b * stride_X_B + idx_h * stride_X_H
    R = R + idx_h * stride_R_H
    auxfree_bias = auxfree_bias + idx_h * stride_auxfree_bias_H

    # Make the tensor descriptors
    x_desc = tl.make_tensor_descriptor(
        X,
        shape=[number_of_tokens, head_size],
        strides=[stride_X_T, stride_X_D],
        block_shape=[BLOCK_N, head_size],
    )
    r_desc = tl.make_tensor_descriptor(
        R,
        shape=[head_size, number_of_experts],
        strides=[stride_R_D, stride_R_E],
        block_shape=[head_size, BLOCK_M],
    )

    # Note: Upcast `tile_X` to float32 right after load, instead of during `.dot(.)`
    #       (1) Avoid repeated casting; (2) Full FP32 router
    tile_X = tl.load_tensor_descriptor(x_desc, [row0, 0]).to(tl.float32)

    # Bias pointer, streamed across experts
    auxfree_bias_ptr = tl.make_block_ptr(
        base=auxfree_bias,
        shape=(1, number_of_experts),
        strides=(stride_auxfree_bias_H, stride_auxfree_bias_E),
        offsets=(0, 0),
        block_shape=(1, BLOCK_M),
        order=(1, 0),  # row-major
    )

    # Accumulator for streaming top-k. (packed uint64: [value_key: 32 | index:32])
    acc = tl.full([BLOCK_N, K], 0, dtype=tl.uint64)

    # Loop over all the experts
    for row_M in tl.range(0, number_of_experts, BLOCK_M):
        # Compiler hint
        tl.multiple_of(row_M, BLOCK_M)

        # (1, BLOCK_M) bias slice
        tile_auxfree_bias = tl.load(auxfree_bias_ptr)
        auxfree_bias_ptr = tl.advance(auxfree_bias_ptr, (0, BLOCK_M))


        # (D, M) weights -> (N, M) logits
        # Load `tile_R`
        tile_R = tl.load_tensor_descriptor(r_desc, [0, row_M])  # (D, M); float32
        # Dot-product in float32; Re-assign `tile_R` to free-up SRAM
        # Note: `tf32` is not numerically stable; Both `tf32x3` and `ieee` are numerically stable
        tile_R = tl.dot(tile_X, tile_R, input_precision="ieee")  # (N, M); float32
        # Apply sigmoid if desired
        if USE_SIGMOID:
            tile_R = tl.sigmoid(tile_R)
        # Apply aux_free bias, creating the "dirty" logit
        # Note: the auxfree bias value is later subtracted away
        tile_R = tile_R + tile_auxfree_bias

        col_idx = tl.arange(0, BLOCK_M) + row_M
        tile_R_idx = tl.broadcast_to(col_idx[None, :], (BLOCK_N, BLOCK_M))

        # Pack to uint64
        val_key_u32 = _fp32_to_ordered_uint32(tile_R)             # (N, M) uint32
        idx_u32     = tile_R_idx.to(tl.uint32)                    # (N, M) uint32
        packed_chunk = (val_key_u32.to(tl.uint64) << 32) | idx_u32.to(tl.uint64)

        # Top-K
        chunk_topk = tl.topk(packed_chunk, K, dim=1)              # (N, K)
        pair = tl.join(acc, chunk_topk)  # (N, K, 2)
        pair_2 = tl.reshape(pair, (BLOCK_N, 2 * K))  # (N, K * 2)
        acc = tl.topk(pair_2, K, dim=1)  # (N, K)

    # ---- unpack accumulator to values / indices ----
    val_key_u32_final = (acc >> 32).to(tl.uint32)
    idx_u32_final     = (acc & tl.full([1], 0xFFFFFFFF, dtype=tl.uint64)).to(tl.uint32)
    y_values  = _ordered_uint32_to_fp32(val_key_u32_final)        # (N, K) fp32
    y_indices = idx_u32_final.to(tl.int64)                        # (N, K) int64


    # ---- write back to VRAM ----
    offs = tl.arange(0, BLOCK_N)
    rows = row0 + offs
    rowmask = rows < number_of_tokens

    ks = tl.arange(0, K)[None, :]                       # (1, K)
    row_col = rows[:, None]                             # (BLOCK_N, 1)

    top_logit_ptrs = (top_logit +
                      idx_b * stride_top_logit_B +
                      idx_h * stride_top_logit_H +
                      row_col * stride_top_logit_T +
                      ks * stride_top_logit_K)
    top_idx_ptrs   = (top_idx   +
                      idx_b * stride_top_idx_B +
                      idx_h * stride_top_idx_H +
                      row_col * stride_top_idx_T +
                      ks * stride_top_idx_K)

    mask_2d = rowmask[:, None]

    tl.store(top_logit_ptrs, y_values,  mask=mask_2d)
    tl.store(top_idx_ptrs,   y_indices, mask=mask_2d)
