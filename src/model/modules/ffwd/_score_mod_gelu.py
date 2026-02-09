import torch
import torch.nn.functional as F


def _score_mod_gelu(score, batch, head, q_idx, k_idx):
    return torch.log1p(F.gelu(score))
