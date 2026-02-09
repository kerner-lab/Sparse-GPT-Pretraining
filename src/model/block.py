import torch.nn as nn
from config.config_template import ConfigTemplate
from model.modules.self_attention import SelfAttention
from model.modules.ffwd.build_ffwd import build_ffwd
from model.modules.norm.build_norm import build_norm


class Block(nn.Module):
    def __init__(self, config: ConfigTemplate, idx_block):
        super().__init__()
        # Define layers
        self.attn = SelfAttention(config)
        self.ffwd = build_ffwd(config, idx_block)
        self.norm_1 = build_norm(config)
        self.norm_2 = build_norm(config)

    def forward(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # (batch_size, num_token, emb_size); float32; contiguous
        x = x + self.attn(self.norm_1(x))
        # (batch_size, num_token, emb_size); float32; contiguous
        x = x + self.ffwd(self.norm_2(x))
        # (batch_size, num_token, emb_size); float32; contiguous
        return x
