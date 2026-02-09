# Copyright (c) 2025, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn
from config.config_template import ConfigTemplate


# Consider: (a) Enable bias; (b) Implement GLU (https://arxiv.org/abs/2002.05202)
class MLP(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Define attributes
        self.num_block = config.num_block
        self.emb_size = config.emb_size
        self.hid_size = config.ffwd_hid_size
        self.use_bias = False
        # Define layers
        self.fc_in  = nn.Linear(self.emb_size, self.hid_size, bias=self.use_bias)
        self.fc_out = nn.Linear(self.hid_size, self.emb_size, bias=self.use_bias)
        self.activation = nn.GELU()
        # Initialize parameters
        nn.init.normal_(self.fc_in.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))
        if self.use_bias:
            nn.init.zeros_(self.fc_in.bias)
            nn.init.zeros_(self.fc_out.bias)
        # Register parameters for weight decay
        self.params_decay = list()
        self.params_decay.append(self.fc_in.weight)
        self.params_decay.append(self.fc_out.weight)
        # Register parameters for 8-bit optimization
        self.params_8bit = list()
        self.params_8bit.append(self.fc_in.weight)
        self.params_8bit.append(self.fc_out.weight)


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False)
    @torch.compile()
    def _stage_mlp_computation(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # (batch_size, num_token, hid_size); bfloat16; contiguous
        x = self.fc_in(x)
        # (batch_size, num_token, hid_size); bfloat16; contiguous
        x = self.activation(x)
        # (batch_size, num_token, emb_size); bfloat16; contiguous
        x = self.fc_out(x)
        # (batch_size, num_token, emb_size); float32; contiguous
        x = x.to(torch.float32)
        # (batch_size, num_token, emb_size); float32; contiguous
        return x


    def forward(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self._stage_mlp_computation(x)
        # (batch_size, num_token, emb_size); float32; contiguous
        return x
