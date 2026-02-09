# Copyright (c) 2025, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from config.config_template import ConfigTemplate


# Ask: Is the RoPE module bfloat16-safe? (https://github.com/huggingface/transformers/pull/29285)
class SelfAttention(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # Define attributes
        self.emb_size: int = config.emb_size
        self.num_head: int = config.attn_num_head
        self.head_size: int = config.attn_head_size
        self.num_block: int = config.num_block
        self.context_window: int = config.context_window
        # Define layers
        self.fc_1 = nn.Linear(self.emb_size, 3 * (self.num_head * self.head_size), bias=False)
        nn.init.normal_(self.fc_1.weight, mean=0.0, std=0.02)
        self.fc_2 = nn.Linear(self.num_head * self.head_size, self.emb_size, bias=False)
        nn.init.normal_(self.fc_2.weight, mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))
        self.rope = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=self.context_window, base=10000)
        # Register parameters for weight decay
        self.params_decay = list()
        self.params_decay.append(self.fc_1.weight)
        self.params_decay.append(self.fc_2.weight)


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False)
    @torch.compile()
    def _stage_multi_head_handling_part_1(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
             (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
             (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        """
        # Define variables
        batch_size, num_token, emb_size = x.shape
        num_head, head_size = self.num_head, self.head_size
        # (batch_size, num_token, 3 * num_head * head_size); bfloat16; contiguous
        x = self.fc_1(x)
        # (batch_size, num_token, num_head * head_size); bfloat16; non-contiguous
        # (batch_size, num_token, num_head * head_size); bfloat16; non-contiguous
        # (batch_size, num_token, num_head * head_size); bfloat16; non-contiguous
        q, k, v = x.chunk(3, dim=2)
        # (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        q = q.view(batch_size, num_token, num_head, head_size)
        # (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        k = k.view(batch_size, num_token, num_head, head_size)
        # (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        v = v.view(batch_size, num_token, num_head, head_size)
        return q, k, v


    @torch.compile()
    def _stage_multi_head_handling_part_2(self, q, k, v):
        """
        In:  (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
             (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
             (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        Out: (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
             (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
             (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        """
        # (batch_size, num_token, num_head, head_size); bfloat16; contiguous
        q = self.rope(q)
        # (batch_size, num_token, num_head, head_size); bfloat16; contiguous
        k = self.rope(k)

        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        q = q.transpose(1, 2)
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        k = k.transpose(1, 2)
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        v = v.transpose(1, 2)
        return q, k, v


    @torch.compile()
    def _stage_self_attention_computation(self, q, k, v):
        """
        In:  (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
             (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
             (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        Out: (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        """
        # Define variables
        head_size = self.head_size
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=1.0 / math.sqrt(head_size),
        )
        return x


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False)
    @torch.compile()
    def _stage_multi_head_merging(self, x):
        """
        In:  (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # Define variables
        batch_size, num_head, num_token, head_size = x.shape
        # (batch_size, num_token, num_head, head_size); bfloat16; contiguous
        x = x.transpose(1, 2)
        # (batch_size, num_token, num_head * head_size); bfloat16; contiguous
        x = x.view(batch_size, num_token, num_head * head_size)
        # (batch_size, num_token, emb_size); bfloat16; contiguous
        x = self.fc_2(x)
        # (batch_size, num_token, emb_size); float32; contiguous
        x = x.to(torch.float32)
        return x


    def forward(self, x):
        """
        In  shape: (batch_size, num_token, emb_size); float32; contiguous
        Out shape: (batch_size, num_token, emb_size); float32; contiguous
        """
        # ----- #
        # Stage: Multi-head handling
        # ----- #
        # (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        # (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        # (batch_size, num_token, num_head, head_size); bfloat16; non-contiguous
        q, k, v = self._stage_multi_head_handling_part_1(x)
        del x
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        q, k, v = self._stage_multi_head_handling_part_2(q, k, v)
        # ----- #

        # ----- #
        # Stage: Self-Attention
        # ----- #
        # (batch_size, num_head, num_token, head_size); bfloat16; non-contiguous
        x = self._stage_self_attention_computation(q, k, v)
        del q, k, v
        # ----- #

        # ----- #
        # Stage: Multi-head merging
        # ----- #
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self._stage_multi_head_merging(x)
        # ----- #
        return x
