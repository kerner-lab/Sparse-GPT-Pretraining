import torch


def synchronize():
    # Note: torch.distributed.barrier() does not wait for CUDA operations
    #   Therefore, we call torch.cuda.synchronize() explicitly
    torch.cuda.synchronize()
    torch.distributed.barrier()
