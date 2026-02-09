import torch.distributed as dist
from utils.synchronize import synchronize


def breakpoint_on_rank(rank):
    synchronize()
    if dist.get_rank() == rank:
        print("Notice: Remember to move up a level in the call stack with `up`")
        breakpoint()
    synchronize()
