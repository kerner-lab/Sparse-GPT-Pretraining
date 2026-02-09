import torch


@torch.no_grad()
@torch.compile()
def _batched_bincount(in_tensor, minlength):
    """
    In:  (dim_0, dim_1); int64; contiguous
    Out: (dim_0, minlength); int64; contiguous
    """
    # Define variables
    dim_0, dim_1 = in_tensor.shape
    # (dim_0,); int64; contiguous
    out = minlength * torch.arange(dim_0, dtype=torch.int64, device=in_tensor.device)
    # (dim_0, 1); int64; contiguous
    out = out.view(dim_0, 1)
    # (dim_0, dim_1); int64; contiguous
    out = in_tensor + out
    # (dim_0 * dim_1,); int64; contiguous
    out = out.view(dim_0 * dim_1)
    # (dim_0 * minlength,); int64; contiguous
    out = torch.bincount(out, minlength=dim_0 * minlength)
    # (dim_0, minlength); int64; contiguous
    out = out.view(dim_0, minlength)
    # (dim_0, minlength); int64; contiguous
    return out
