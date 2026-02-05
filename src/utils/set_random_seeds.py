import torch
import random
import numpy as np
import torch.distributed as dist
from config.config_template import ConfigTemplate


def set_random_seeds(config: ConfigTemplate):
    # Note: We set different random seeds on different ranks to ensure IID
    # Note: Do NOT assume each rank has the same random seed; broadcast the initial weights from rank 0 explicitly
    rank = dist.get_rank()
    seed_1 = config.repro_random_seed_value + 3 * rank + 0
    seed_2 = config.repro_random_seed_value + 3 * rank + 1
    seed_3 = config.repro_random_seed_value + 3 * rank + 2
    random.seed(seed_1)
    np.random.seed(seed_2)
    torch.manual_seed(seed_3)
