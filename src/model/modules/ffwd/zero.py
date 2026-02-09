import torch.nn as nn
from config.config_template import ConfigTemplate


# TODO: Call this `Identity` instead?; `Zero` --> `Identity`
class Zero(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()

    def forward(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # (batch_size, num_token, emb_size); float32; contiguous
        return x
