# Copyright (c) 2025, Chenwei Cui, Rockwell Jackson, Kerner Lab
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.attention.flex_attention import flex_attention

from config.config_template import ConfigTemplate

from model.ops.mh_packing.packing import packing
from model.ops.mh_packing.unpacking import unpacking
from model.ops.mh_packing.prepare_packing import prepare_packing
from model.ops.mh_router.mh_router import mh_router
from model.ops.all_to_all import all_to_all

from model.modules.ffwd._get_block_mask import _get_block_mask
from model.modules.ffwd._score_mod_gelu import _score_mod_gelu
from model.modules.ffwd._batched_bincount import _batched_bincount


# TODO: Make MHMoEHPNRT the standard instead (`MHMoEHPNRT` --> `MHMoEHP`);
#   NRT means "no-routing-tokens", which turns out to be better (see Sec 4.4 in https://arxiv.org/pdf/2602.04870v1)
# TODO: Rename all `mh_moe` into `mh_latent_moe`; `mh_moe` is an early project name
#   It conflicts with MH-MoE (https://arxiv.org/pdf/2404.15045)
# Consider: Good news! FlexAttention had implemented the reversal trick. Consider using it.
#   (https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/mods/activation.py)
# Consider: Support (1, batch_size * num_token, num_head, head_size) to reduce transpose?
#   Background A: FlexAttention supports non-contiguous inputs
#   Background B: dim_0 is a dial that optionally temporarily duplicates K and V to trade for more parallelism
# Note: `bath_size` is an intentional name choice to mirror `pool_size`, not a typo for `batch_size`
class MHMoEHPNRT(nn.Module):
    def __init__(self, config: ConfigTemplate, idx_block):
        super().__init__()
        # ----- #
        # Define attributes
        # ----- #
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.config = config
        self.idx_block = idx_block
        self.num_block = config.num_block

        self.emb_size = config.emb_size
        self.num_head = config.ffwd_num_head
        self.head_size = config.ffwd_head_size
        assert self.num_head % self.world_size == 0, "The number of head needs to be a multiple of the number of ranks"
        self.num_head_per_rank = self.num_head // self.world_size

        self.num_expert = config.ffwd_num_expert
        self.num_expert_active = config.ffwd_num_expert_active
        assert self.num_expert_active >= 2
        self.expert_size = config.ffwd_expert_size

        self.flex_attn_block_size = 128
        assert self.expert_size % self.flex_attn_block_size == 0
        # ----- #

        # ----- #
        # Additional outputs
        # ----- #
        # (num_head_per_rank, num_expert); float32; contiguous; detached
        self.expert_load = None
        # ----- #

        # ----- #
        # Define layers
        # ----- #
        self.fc_mh = nn.Linear(self.emb_size, self.num_head * self.head_size, bias=False)
        nn.init.normal_(self.fc_mh.weight, mean=0.0, std=0.02)

        self.fc_mg = nn.Linear(self.num_head * self.head_size, self.emb_size, bias=False)
        nn.init.normal_(self.fc_mg.weight, mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))

        self.router_no_share = nn.Parameter(0.02 * torch.randn(
            size=(self.num_head_per_rank, self.head_size, self.num_expert),
            dtype=torch.float32,
        ))

        self.k_ffwd_no_share = nn.Parameter(0.02 * torch.randn(
            size=(1, self.num_head_per_rank, self.num_expert * self.expert_size, self.head_size),
            dtype=torch.float32,
        ))

        self.v_ffwd_no_share = nn.Parameter(0.02 * torch.randn(
            size=(1, self.num_head_per_rank, self.num_expert * self.expert_size, self.head_size),
            dtype=torch.float32,
        ))
        # ----- #

        # ----- #
        # Register parameters for weight decay
        # ----- #
        self.params_decay = list()
        self.params_decay.append(self.fc_mh.weight)
        self.params_decay.append(self.fc_mg.weight)
        self.params_decay.append(self.router_no_share)
        self.params_decay.append(self.k_ffwd_no_share)
        self.params_decay.append(self.v_ffwd_no_share)
        # ----- #

        # ----- #
        # Register parameters for 8-bit optimization
        # ----- #
        self.params_8bit = list()
        self.params_8bit.append(self.k_ffwd_no_share)
        self.params_8bit.append(self.v_ffwd_no_share)
        # ----- #


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False)
    @torch.compile()
    def _stage_multi_head_handling(self, x, bath_size):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
             int
        Out: (num_head_per_rank, bath_size, head_size); bfloat16; contiguous
        """
        # Define variables
        rank, world_size = self.rank, self.world_size
        batch_size, num_token, emb_size = x.shape
        num_head, head_size = self.num_head, self.head_size
        num_head_per_rank = self.num_head_per_rank

        # (batch_size, num_token, num_head * head_size); bfloat16; contiguous
        x = self.fc_mh(x)
        # (batch_size * num_token, num_head, head_size); bfloat16; contiguous
        x = x.view(batch_size * num_token, num_head, head_size)

        # Reshape for all_to_all: group by destination rank
        # (batch_size * num_token, world_size, num_head_per_rank, head_size); bfloat16; contiguous
        x = x.view(batch_size * num_token, world_size, num_head_per_rank, head_size)
        # TODO: Examine: (prior) you may not need to transpose and contiguous(), because
        #       batch_size * num_token, world_size dim is independent
        #       Counter argument: transpose necessary because `world_size` specifies different GPU
        # (world_size, batch_size * num_token, num_head_per_rank, head_size); bfloat16; contiguous
        x = x.transpose(0, 1).contiguous()
        # (world_size * batch_size * num_token, num_head_per_rank * head_size); bfloat16; contiguous
        x = x.view(world_size * batch_size * num_token, num_head_per_rank * head_size)

        # All-to-all
        input_splits = [batch_size * num_token] * world_size  # TODO: Make it a class attribute (for the same batch_size and num_token)
        x, output_splits = all_to_all(x, input_splits, output_splits=input_splits)

        # (bath_size, num_head_per_rank, head_size); bfloat16; contiguous
        x = x.view(bath_size, num_head_per_rank, head_size)

        # (num_head_per_rank, bath_size, head_size); bfloat16; non-contiguous
        x = x.transpose(0, 1)

        # (num_head_per_rank, bath_size, head_size); bfloat16; contiguous
        x = x.contiguous()

        # (num_head_per_rank, bath_size, head_size); bfloat16; contiguous
        return x


    @torch.no_grad()
    @torch.compile()
    def _stage_get_expert_load(self, expert_bincount, pool_size):
        """
        In:  (num_head_per_rank, num_expert); int64; contiguous; detached
             int
        Out: (num_head_per_rank, num_expert); float32; contiguous; detached
        """
        return expert_bincount.to(torch.float32) / pool_size


    @torch.compile()
    def _stage_token_duplication(self, x, num_expert_active, pool_size):
        """
        In:  (num_head_per_rank, bath_size, head_size); bfloat16; contiguous
             int int
        Out: (num_head_per_rank, pool_size, head_size); bfloat16; contiguous
        """
        # Define variables
        num_head_per_rank, bath_size, head_size = x.shape
        # (num_head_per_rank, bath_size, 1, head_size); bfloat16; contiguous
        x = x.view(num_head_per_rank, bath_size, 1, head_size)
        # (num_head_per_rank, bath_size, num_expert_active, head_size); bfloat16; non-contiguous
        x = x.expand(num_head_per_rank, bath_size, num_expert_active, head_size)
        # (num_head_per_rank, bath_size, num_expert_active, head_size); bfloat16; contiguous
        x = x.contiguous()
        # (num_head_per_rank, pool_size, head_size); bfloat16; contiguous
        x = x.view(num_head_per_rank, pool_size, head_size)
        return x


    @torch.compile()
    def _stage_flex_attention_computation(self, q, k, v, score_mod, block_mask):
        """
        In:  (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
             (1, num_head_per_rank, num_expert * expert_size, head_size); float32; contiguous
             (1, num_head_per_rank, num_expert * expert_size, head_size); float32; contiguous
             score_mod block_mask
        Out: (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
             (1, num_head_per_rank, pool_size + padding_size); float32; contiguous
        """
        # (1, num_head_per_rank, num_expert * expert_size, head_size); bfloat16; contiguous
        k = k.to(torch.bfloat16)
        # (1, num_head_per_rank, num_expert * expert_size, head_size); bfloat16; contiguous
        v = v.to(torch.bfloat16)
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        # (1, num_head_per_rank, pool_size + padding_size); float32; contiguous
        o, lse = flex_attention(
            query=q,
            key=k,
            value=v,
            scale=1.0,
            block_mask=block_mask,
            score_mod=score_mod,
            return_lse=True,
        )
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        # (1, num_head_per_rank, pool_size + padding_size); float32; contiguous
        return o, lse


    @torch.compile()
    def _stage_reversal_trick(self, o, lse, v, expert_assign, num_head_per_rank, head_size, num_expert, expert_size):
        """
        In:  (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
             (1, num_head_per_rank, pool_size + padding_size); float32; contiguous
             (1, num_head_per_rank, num_expert * expert_size, head_size); float32; contiguous
             (num_head_per_rank, pool_size + padding_size); int64; contiguous; detached
             int int int int
        Out: (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        """
        # Stage 1
        # (1, num_head_per_rank, pool_size + padding_size, 1); float32; contiguous
        lse = lse[:, :, :, None]
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        o = o * lse.exp().to(torch.bfloat16)

        # Stage 2
        # (1, num_head_per_rank, num_expert, expert_size, head_size); float32; contiguous
        offsets = v.view(1, num_head_per_rank, num_expert, expert_size, head_size)
        # (1, num_head_per_rank, num_expert, head_size); float32; contiguous
        offsets = offsets.sum(dim=3, keepdim=False)
        # (1, num_head_per_rank, num_expert, head_size); bfloat16; contiguous
        offsets = offsets.to(torch.bfloat16)

        # (1, num_head_per_rank, pool_size + padding_size, 1); int64; contiguous; detached
        expert_assign = expert_assign[None, :, :, None]
        # (1, num_head_per_rank, pool_size + padding_size, head_size); int64; non-contiguous; detached
        expert_assign = expert_assign.expand(-1, -1, -1, head_size)
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        offsets = torch.gather(input=offsets, dim=2, index=expert_assign)
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        o = o - offsets
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        return o


    @torch.compile()
    def _stage_token_aggregation(self, x, router_values, num_head_per_rank, bath_size, num_expert_active, head_size):
        """
        In:  (num_head_per_rank, pool_size, head_size); bfloat16; contiguous
             (num_head_per_rank, bath_size, num_expert_active); float32; contiguous
             int int int int
        Out: (num_head_per_rank, bath_size, head_size); float32; contiguous
        """
        # (num_head_per_rank, bath_size, num_expert_active, head_size); float32; contiguous
        x = x.view(num_head_per_rank, bath_size, num_expert_active, head_size).to(torch.float32)
        # (num_head_per_rank, bath_size, num_expert_active, 1); float32; contiguous
        router_values = router_values[:, :, :, None]
        # (num_head_per_rank, bath_size, num_expert_active, head_size); float32; contiguous
        x = x * router_values
        # (num_head_per_rank, bath_size, head_size); float32; contiguous
        x = x.sum(dim=2, keepdim=False)
        # (num_head_per_rank, bath_size, head_size); float32; contiguous
        return x


    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False)
    @torch.compile()
    def _stage_multi_head_merging(self, x, batch_size, num_token, num_head, head_size, bath_size):
        """
        In:  (num_head_per_rank, bath_size, head_size); float32; contiguous
             int int int int int
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # Define variables
        rank, world_size = self.rank, self.world_size
        num_head_per_rank = self.num_head_per_rank
        # (bath_size, num_head_per_rank, head_size); float32; non-contiguous
        x = x.transpose(0, 1)
        # (bath_size, num_head_per_rank, head_size); bfloat16; contiguous
        x = x.to(torch.bfloat16).contiguous()

        # Reshape for all_to_all
        # (bath_size, num_head_per_rank * head_size); bfloat16; contiguous
        x = x.view(bath_size, num_head_per_rank * head_size)

        # All-to-all
        input_splits = [batch_size * num_token] * world_size
        x, output_splits = all_to_all(x, input_splits, output_splits=input_splits)

        # Reshape after all_to_all
        # (world_size, batch_size * num_token, num_head_per_rank, head_size); bfloat16; contiguous
        x = x.view(world_size, batch_size * num_token, num_head_per_rank, head_size)
        # (batch_size * num_token, world_size, num_head_per_rank, head_size); bfloat16; contiguous
        x = x.transpose(0, 1).contiguous()
        # (batch_size * num_token, num_head, head_size); bfloat16; contiguous
        x = x.view(batch_size * num_token, num_head, head_size)

        # (batch_size, num_token, num_head * head_size); bfloat16; contiguous
        x = x.view(batch_size, num_token, num_head * head_size)
        # (batch_size, num_token, emb_size); bfloat16; contiguous
        x = self.fc_mg(x)
        # (batch_size, num_token, emb_size); float32; contiguous
        x = x.to(torch.float32)
        return x


    def forward(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """


        # ----- #
        # Define variables
        # ----- #
        batch_size, num_token, emb_size = x.shape
        num_head, head_size = self.num_head, self.head_size
        num_expert, num_expert_active, expert_size = self.num_expert, self.num_expert_active, self.expert_size

        rank, world_size = self.rank, self.world_size
        num_head_per_rank = self.num_head_per_rank
        bath_size = batch_size * num_token * world_size
        pool_size = batch_size * num_token * world_size * num_expert_active
        # ----- #


        # ----- #
        # Multi-head handling
        # ----- #
        # (num_head_per_rank, bath_size, head_size); bfloat16; contiguous
        x = self._stage_multi_head_handling(x, bath_size=bath_size)
        # Note: Consider moving all_to_all outside of torch.compile scope
        # ----- #


        # ----- #
        # Routing
        # ----- #
        # TODO: `mh_router` should follow a head-first layout
        # TODO: torch.compile wherever applicable
        # (bath_size, num_head_per_rank, head_size); bfloat16; contiguous
        q_rter = x.transpose(0, 1).contiguous()
        # (1, bath_size, num_head_per_rank, head_size); bfloat16; contiguous
        q_rter = q_rter[None]

        # (num_head_per_rank, head_size, num_expert); float32; contiguous
        router = self.router_no_share
        # (num_head_per_rank, num_expert); float32; contiguous
        auxfree_bias = self.config.runtime["auxfree_bias_all"][self.idx_block]

        # (1, bath_size, num_head_per_rank, num_expert_active); float32; contiguous
        # (1, bath_size, num_head_per_rank, num_expert_active); int64; contiguous; detached
        router_values, expert_assign = mh_router(q_rter, router, auxfree_bias, num_expert_active, False)

        # (num_head_per_rank, bath_size, num_expert_active); float32; contiguous
        # (num_head_per_rank, bath_size, num_expert_active); int64; contiguous; detached
        router_values = router_values[0].transpose(0, 1).contiguous()
        expert_assign = expert_assign[0].transpose(0, 1).contiguous()

        # ----- #
        # Random Expert Assignment for Training Time Estimation
        # ----- #
        # Note: Currently does not reflect the real training cost, therefore not in use at the moment.
        if self.config.runtime.get("enforce_random_routing", False):
            expert_assign = torch.randint(
                low=0,
                high=self.num_expert,
                size=(num_head_per_rank, bath_size, num_expert_active),
                dtype=torch.int64,
                device="cuda",
            )
        # ----- #

        # Release q_rter
        del q_rter
        # (num_head_per_rank, bath_size, num_expert_active); float32; contiguous
        router_values = router_values.softmax(dim=2)
        # ----- #


        # ----- #
        # Calculate `expert_bincount` and update `self.expert_load`
        # ----- #
        # (num_head_per_rank, num_expert); int64; contiguous; detached
        expert_bincount = _batched_bincount(
            in_tensor=expert_assign.view(num_head_per_rank, pool_size),
            minlength=num_expert,
        )
        # (num_head_per_rank, num_expert); float32; contiguous; detached
        self.expert_load = self._stage_get_expert_load(expert_bincount, pool_size)
        # ----- #


        # ----- #
        # Token duplication
        # ----- #
        # (num_head_per_rank, pool_size, head_size); bfloat16; contiguous
        x = self._stage_token_duplication(x, self.num_expert_active, pool_size)
        # ----- #


        # ----- #
        # Packing
        # ----- #
        # (num_head_per_rank * pool_size + num_head_per_rank * padding_size,); int64; contiguous; detached
        # (num_head_per_rank * pool_size + num_head_per_rank * padding_size,); int64; contiguous; detached
        # (); int64; contiguous; detached
        # (num_head_per_rank, num_block_q); int64; contiguous; detached
        # (num_head_per_rank, pool_size + padding_size); int64; contiguous; detached
        mapping, mapping_inv, padding_size, block_level_expert_assign, expert_assign = prepare_packing(
            expert_assign=expert_assign.view(num_head_per_rank, pool_size),
            expert_bincount=expert_bincount,
            block_size=self.flex_attn_block_size,
        )
        # (num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        x = packing(x, padding_size, mapping)
        # ----- #


        # ----- #
        # Get `block_mask`
        # ----- #
        block_mask = _get_block_mask(
            block_level_expert_assign=block_level_expert_assign,
            num_expert=num_expert,
            expert_size=expert_size,
            block_size=self.flex_attn_block_size,
        )
        # ----- #


        # ----- #
        # Flex Attention
        # ----- #
        # Note: `lse` is verified to be float32
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        x = x[None, :, :, :]
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        # (1, num_head_per_rank, pool_size + padding_size); float32; contiguous
        x, lse = self._stage_flex_attention_computation(
            q=x,
            k=self.k_ffwd_no_share,
            v=self.v_ffwd_no_share,
            score_mod=_score_mod_gelu,
            block_mask=block_mask,
        )
        # (1, num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        x = self._stage_reversal_trick(
            o=x, lse=lse, v=self.v_ffwd_no_share, expert_assign=expert_assign,
            num_head_per_rank=num_head_per_rank, head_size=head_size, num_expert=num_expert, expert_size=expert_size,
        )
        # (num_head_per_rank, pool_size + padding_size, head_size); bfloat16; contiguous
        x = x[0]
        # Release `lse`
        del lse
        # ----- #


        # ----- #
        # Unpacking
        # ----- #
        # (num_head_per_rank, pool_size, head_size); bfloat16; contiguous
        x = unpacking(x, padding_size, mapping_inv)
        # ----- #


        # ----- #
        # Token aggregation
        # ----- #
        # (num_head_per_rank, bath_size, head_size); float32; contiguous
        x = self._stage_token_aggregation(
            x=x, router_values=router_values,
            num_head_per_rank=num_head_per_rank, bath_size=bath_size,
            num_expert_active=num_expert_active, head_size=head_size,
        )
        # ----- #


        # ----- #
        # Multi-head merging
        # ----- #
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self._stage_multi_head_merging(
            x=x, batch_size=batch_size, num_token=num_token,
            num_head=num_head, head_size=head_size, bath_size=bath_size,
        )
        # ----- #


        # (batch_size, num_token, emb_size); float32; contiguous
        return x
