# Copyright (c) 2025, Rockwell Jackson, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import torch
from model.ops.mh_router.mh_router_fwd import mh_router_fwd
from model.ops.mh_router.mh_router_bwd import mh_router_bwd

@torch.compile()
def _remove_auxfree_bias(top_logit, top_idx, auxfree_bias):
    bias_view = auxfree_bias.unsqueeze(0).unsqueeze(0)                  # (1,1,H,E)
    bias_selected = torch.take_along_dim(bias_view, top_idx, dim=3)     # (B,T,H,K)
    top_logit = top_logit - bias_selected                               # (B,T,H,K) now CLEAN
    return top_logit


# TODO: Rename `mh_router` into `io_efficient_router`; `mh_router` is an early project name.
# Note: We use IEEE FP32 for tl.dot; TF32 deteriorates numerical precision; Possibly due to multiple accumulations
# Note: Both FP32 and `tf32x3` are numerically stable in our test
class MHRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, R, auxfree_bias, K=1, USE_SIGMOID=False):
        """
        In:  (batch_size, num_token, num_head, head_size); bfloat16; contiguous
             (num_head, head_size, num_expert); float32; contiguous
             (num_head, num_expert); float32; contiguous
             int bool
        Out: (batch_size, num_token, num_head, num_expert_active); float32; contiguous
             (batch_size, num_token, num_head, num_expert_active); int64; contiguous
        """

        # Get shapes
        batch_size, number_of_tokens, number_of_heads, head_size = X.shape
        _, _, number_of_experts = R.shape

        # Hyperparameters
        if head_size == 64:
            N, M, num_of_warps, num_of_stages = 64, 32, 4, 3
        elif head_size == 128:
            N, M, num_of_warps, num_of_stages = 64, 64, 8, 3
        elif head_size == 256:
            N, M, num_of_warps, num_of_stages = 32, 64, 8, 3
        else:
            raise Exception("Unexpected head_size")

        assert number_of_experts >= M, "We only support num_expert that is larger than M"

        assert X.ndim == 4, f"X must be 4D (B,T,H,D), got {X.shape}"
        assert R.ndim == 3, f"R must be 3D (H,D,E), got {R.shape}"
        assert auxfree_bias.ndim == 2, f"auxfree_bias must be 2D (H, E), got {auxfree_bias.shape}"

        assert X.dtype == torch.bfloat16, f"X is expecting to be passed as bfloat16"
        assert R.dtype == torch.float32, f"R is expecting to be passed as float32"
        assert auxfree_bias.dtype == torch.float32, f"auxfree_bias is expecting to be passed as float32"

        assert head_size == R.shape[1], f"Head Size mismatch between X and R"
        assert number_of_heads == R.shape[0], f"Number of Heads mismatch between X and R"

        assert number_of_heads == auxfree_bias.shape[0], f"Number of Heads mismatch between X and auxfree_bias"
        assert number_of_experts == auxfree_bias.shape[1], f"Number of Experts mismatch between R and auxfree_bias"

        assert head_size in {16, 32, 64, 128, 256}, f"Head Size not supported"
        assert number_of_experts % 16 == 0, f"Number of experts must be a multiple of 16"

        assert number_of_tokens % N == 0, f"number_of_tokens ({number_of_tokens}) must be divisible by BLOCK_N ({N})"
        assert number_of_experts % M == 0, f"number_of_experts ({number_of_experts}) must be divisible by BLOCK_M ({M})"
        assert number_of_tokens > 0 and batch_size > 0 and number_of_heads > 0 and head_size > 0 and number_of_experts > 0

        assert X.is_contiguous(), f"X must be contiguous"
        assert R.is_contiguous(), f"R must be contiguous"
        assert auxfree_bias.is_contiguous(), f"auxfree_bias must be contiguous"

        assert K > 0, f"K must be at least 1"
        assert K <= M, f"K is too large, will be inefficient/can't handle block size."

        # (batch_size, number_of_tokens, number_of_heads, K); float32
        top_logit = torch.empty((batch_size, number_of_tokens, number_of_heads, K), dtype=torch.float32, device="cuda")
        # (batch_size, number_of_tokens, number_of_heads, K); int64
        top_idx = torch.empty((batch_size, number_of_tokens, number_of_heads, K), dtype=torch.int64,   device="cuda")

        grid = (batch_size, number_of_tokens // N, number_of_heads)
        mh_router_fwd[grid](
            X, R, auxfree_bias, top_logit, top_idx,
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            R.stride(0), R.stride(1), R.stride(2),
            auxfree_bias.stride(0), auxfree_bias.stride(1),
            top_logit.stride(0), top_logit.stride(1), top_logit.stride(2), top_logit.stride(3),
            top_idx.stride(0), top_idx.stride(1), top_idx.stride(2), top_idx.stride(3),
            head_size=head_size,
            number_of_experts=number_of_experts,
            number_of_tokens=number_of_tokens,
            USE_SIGMOID=USE_SIGMOID,
            K=K,
            BLOCK_N=N,
            BLOCK_M=M,
            num_warps=num_of_warps,
            num_stages=num_of_stages,
        )

        # Remove auxfree bias from top_logit
        top_logit = _remove_auxfree_bias(top_logit, top_idx, auxfree_bias)

        ctx.save_for_backward(X, R, top_logit, top_idx)
        ctx.USE_SIGMOID = USE_SIGMOID
        ctx.K = K
        return top_logit, top_idx

    @staticmethod
    def backward(ctx, d_top_logit, d_top_idx):  # `d_top_idx` is a placeholder, it does not have gradient
        """
        X: (batch_size, num_token, num_head, head_size); bfloat16; contiguous
        R: (num_head, head_size, num_expert); float32; contiguous
        top_idx: (batch_size, num_token, num_head, num_expert_active); int64; contiguous
        d_top_logit: (batch_size, num_token, num_head, num_expert_active); float32; contiguous
        """

        # Retrieve from `ctx`
        X, R, top_logit, top_idx = ctx.saved_tensors
        USE_SIGMOID = ctx.USE_SIGMOID
        K = ctx.K

        # Get shapes
        batch_size, number_of_tokens, number_of_heads, head_size = X.shape
        _, _, number_of_experts = R.shape

        # Hyperparameters
        if head_size == 64:
            N, num_of_warps, num_of_stages = 32, 8, 2
        elif head_size == 128:
            N, num_of_warps, num_of_stages = 32, 8, 2
        elif head_size == 256:
            N, num_of_warps, num_of_stages = 16, 8, 3
        else:
            raise Exception("Unexpected head_size")

        # Note: We already know that X, R, and top_idx conform to our specifications, so we only need to check d_top_logit
        assert d_top_logit.ndim == 4, f"d_top_logit must be 4D (B,T,H,K), got {d_top_logit.shape}"
        assert d_top_logit.shape == top_idx.shape, f"Shape mismatch between d_top_logit and top_idx"
        assert d_top_logit.dtype == torch.float32, "d_top_logit must be float32"
        assert d_top_logit.is_contiguous(), f"d_top_logit must be contiguous"

        assert number_of_tokens % N == 0, f"number_of_tokens ({number_of_tokens}) must be divisible by BLOCK_N ({N})"

        # Note: Sigmoid derivative applied outside of kernel for simplicity/speed
        # TODO: `torch.compile` this part
        if USE_SIGMOID:
            d_top_logit = d_top_logit * top_logit * (1 - top_logit)

        # (batch_size, num_token, num_head, head_size); bfloat16; contiguous
        dX = torch.empty(
            size=(batch_size, number_of_tokens, number_of_heads, head_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Note: We need to initialize `dR` into zeros because we atomic_add into it
        # (num_head, head_size, num_expert); float32; contiguous
        dR = torch.zeros(
            size=(number_of_heads, head_size, number_of_experts),
            dtype=torch.float32,
            device="cuda",
        )

        grid = (batch_size, number_of_tokens // N, number_of_heads)
        mh_router_bwd[grid](
            X,
            R,
            top_idx,
            d_top_logit,
            dX,
            dR,
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            R.stride(0), R.stride(1), R.stride(2),
            top_idx.stride(0), top_idx.stride(1), top_idx.stride(2), top_idx.stride(3),
            d_top_logit.stride(0), d_top_logit.stride(1), d_top_logit.stride(2), d_top_logit.stride(3),
            dX.stride(0), dX.stride(1), dX.stride(2), dX.stride(3),
            dR.stride(0), dR.stride(1), dR.stride(2),
            head_size=head_size,
            number_of_experts=number_of_experts,
            number_of_tokens=number_of_tokens,
            K=K,
            BLOCK_N=N,
            num_warps=num_of_warps,
            num_stages=num_of_stages,
        )
        return dX, dR, None, None, None

mh_router = MHRouter.apply
