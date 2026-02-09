# Copyright (c) 2025, Chenwei Cui, Rockwell Jackson, Benjamin Joseph Herrera, Kerner Lab

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import transformer_engine.pytorch as te
from model.ops.all_to_all import all_to_all
from config.config_template import ConfigTemplate


# Assumption: Rank 0 owns Expert 0~31, Rank 1 owns Expert 32~63, Rank 2 owns Expert 64~95, Rank 3 owns Expert 96~127
class MoEEP(nn.Module):
    def __init__(self, config: ConfigTemplate, idx_block):
        super().__init__()
        # ----- #
        # Define attributes
        # ----- #
        self.config = config
        self.idx_block = idx_block
        self.num_block = config.num_block
        self.emb_size = config.emb_size
        self.expert_size = config.ffwd_expert_size
        self.num_expert = config.ffwd_num_expert
        self.num_expert_active = config.ffwd_num_expert_active
        assert self.num_expert_active >= 2  # Note: Required by post-softmax
        # ----- #


        # ----- #
        # Additional outputs
        # ----- #
        # (num_expert,); float32; contiguous; detached
        self.expert_load = None
        # ----- #


        # ----- #
        # Define Expert Parallel
        # ----- #
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        assert self.num_expert > 0 and self.num_expert % self.world_size == 0, f"Number of experts must be divisible by world size"
        self.num_expert_per_rank = self.num_expert // self.world_size
        self.metadata_a2a_split_size = [1] * self.world_size
        # Ask: It is better to register the following two as pytorch buffers?
        self.weaved_chunk_indices     = torch.arange(0, self.world_size * self.num_expert_per_rank).view(self.world_size, self.num_expert_per_rank).T.contiguous().view(self.world_size * self.num_expert_per_rank).cuda()
        self.weaved_chunk_indices_inv = self.weaved_chunk_indices.argsort()
        # self.weaved_chunk_indices     = torch.arange(0, self.num_expert_per_rank).repeat(self.world_size).cuda()
        # self.weaved_chunk_indices_inv = torch.arange(0, self.world_size).repeat(self.num_expert_per_rank).cuda()
        # ----- #


        # ----- #
        # Define layers
        # ----- #
        self.router = nn.Linear(self.emb_size, self.num_expert, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        self.grouped_fc_1_no_share = te.GroupedLinear(
            num_gemms=self.num_expert_per_rank,
            in_features=self.emb_size,
            out_features=self.expert_size,
            bias=False,
            params_dtype=torch.float32,
        )
        for idx in range(self.num_expert_per_rank):
            current = getattr(self.grouped_fc_1_no_share, f"weight{idx}")
            nn.init.normal_(current, mean=0.0, std=0.02)

        self.grouped_fc_2_no_share = te.GroupedLinear(
            num_gemms=self.num_expert_per_rank,
            in_features=self.expert_size,
            out_features=self.emb_size,
            bias=False,
            params_dtype=torch.float32,
        )
        for idx in range(self.num_expert_per_rank):
            current = getattr(self.grouped_fc_2_no_share, f"weight{idx}")
            nn.init.normal_(current, mean=0.0, std=0.02 / math.sqrt(2.0 * self.num_block))
        # ----- #


        # ----- #
        # Register parameters for weight decay
        # ----- #
        self.params_decay = list()
        # `self.router`
        self.params_decay.append(self.router.weight)
        # `self.grouped_fc_1_no_share`
        for idx in range(self.num_expert_per_rank):
            current = getattr(self.grouped_fc_1_no_share, f"weight{idx}")
            self.params_decay.append(current)
        # `self.grouped_fc_2_no_share`
        for idx in range(self.num_expert_per_rank):
            current = getattr(self.grouped_fc_2_no_share, f"weight{idx}")
            self.params_decay.append(current)
        # ----- #


        # ----- #
        # Register parameters for 8-bit optimization
        # ----- #
        self.params_8bit = list()
        # `self.grouped_fc_1_no_share`
        for idx in range(self.num_expert_per_rank):
            current = getattr(self.grouped_fc_1_no_share, f"weight{idx}")
            self.params_8bit.append(current)
        # `self.grouped_fc_2_no_share`
        for idx in range(self.num_expert_per_rank):
            current = getattr(self.grouped_fc_2_no_share, f"weight{idx}")
            self.params_8bit.append(current)
        # ----- #


    @torch.compile()
    def _stage_routing(self, x, auxfree_bias):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
             (num_expert,); float32; contiguous; detached
        Out: (batch_size, num_token, num_expert_active); float32; contiguous
             (batch_size, num_token, num_expert_active); int64; contiguous; detached
        """
        # Compute `router_values`
        # (batch_size, num_token, num_expert); float32; contiguous
        router_values = self.router(x)

        # Get `expert_assign`
        # (batch_size, num_token, num_expert); float32; contiguous; detached
        topk_input = router_values.detach()
        # (batch_size, num_token, num_expert); float32; contiguous; detached
        topk_input = topk_input + auxfree_bias.view(1, 1, self.num_expert)
        # (batch_size, num_token, num_expert_active); int64; contiguous; detached
        expert_assign = torch.topk(input=topk_input, k=self.num_expert_active, dim=-1, largest=True, sorted=False)
        expert_assign = expert_assign.indices.detach()
        del topk_input

        # Gather elements from `router_values` according to `expert_assign`
        # (batch_size, num_token, num_expert_active); float32; contiguous
        router_values = torch.gather(input=router_values, dim=2, index=expert_assign)

        # (batch_size, num_token, num_expert_active); float32; contiguous
        # (batch_size, num_token, num_expert_active); int64; contiguous; detached
        return router_values, expert_assign


    @torch.no_grad()
    @torch.compile()
    def _stage_get_expert_bincount_and_expert_load(self, expert_assign):
        """
        In:  (batch_size, num_token, num_expert_active); int64; contiguous; detached
        Out: (num_expert,); int64; contiguous; detached
             (num_expert,); float32; contiguous; detached
        """
        # Define variables
        batch_size, num_token, num_expert_active = expert_assign.shape
        # Flatten `expert_assign`
        # (batch_size * num_token * num_expert_active,); int64; contiguous; detached
        expert_assign = expert_assign.view(batch_size * num_token * num_expert_active)
        # (num_expert,); int64; contiguous; detached
        expert_bincount = torch.bincount(expert_assign, minlength=self.num_expert)
        # (num_expert,); float32; contiguous; detached
        expert_load = expert_bincount.to(torch.float32) / (batch_size * num_token * num_expert_active)

        # (num_expert,); int64; contiguous; detached
        # (num_expert,); float32; contiguous; detached
        return expert_bincount, expert_load


    @torch.compile()
    def _stage_token_duplication(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, num_expert_active, emb_size); bfloat16; contiguous
        """
        # Define variables
        batch_size, num_token, emb_size = x.shape
        # (batch_size, num_token, emb_size); bfloat16; contiguous
        x = x.to(torch.bfloat16)
        # (batch_size, num_token, 1, emb_size); bfloat16; contiguous
        x = x.view(batch_size, num_token, 1, emb_size)
        # (batch_size, num_token, num_expert_active, emb_size); bfloat16; non-contiguous
        x = x.expand(batch_size, num_token, self.num_expert_active, emb_size)
        # (batch_size, num_token, num_expert_active, emb_size); bfloat16; contiguous
        x = x.contiguous()
        # (batch_size, num_token, num_expert_active, emb_size); bfloat16; contiguous
        return x


    @torch.compile()
    def _stage_token_aggregation(self, router_values, x):
        """
        In:  (batch_size, num_token, num_expert_active); float32; contiguous
             (batch_size, num_token, num_expert_active, emb_size); bfloat16; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # Define variables
        batch_size, num_token, num_expert_active = router_values.shape
        # (batch_size, num_token, num_expert_active, 1); float32; contiguous
        router_values = router_values.view(batch_size, num_token, num_expert_active, 1)
        # (batch_size, num_token, num_expert_active, emb_size); float32; contiguous
        x = x.to(torch.float32)
        # (batch_size, num_token, num_expert_active, emb_size); float32; contiguous
        x = x * router_values
        # (batch_size, num_token, emb_size); float32; contiguous
        x = x.sum(dim=2, keepdim=False)
        # (batch_size, num_token, emb_size); float32; contiguous
        return x


    def forward(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """

        # ----- #
        # Stage: Routing
        # ----- #
        # (num_expert,); float32; contiguous; detached
        auxfree_bias = self.config.runtime["auxfree_bias_all"][self.idx_block]
        # (batch_size, num_token, num_expert_active); float32; contiguous
        # (batch_size, num_token, num_expert_active); int64; contiguous; detached
        router_values, expert_assign = self._stage_routing(x, auxfree_bias)
        del auxfree_bias

        # ----- #
        # Random Expert Assignment for Training Time Estimation
        # ----- #
        # Note: Currently does not reflect the real training cost, therefore not in use at the moment.
        if self.config.runtime.get("enforce_random_routing", False):
            batch_size, num_token, _ = expert_assign.shape
            expert_assign = torch.randint(
                low=0,
                high=self.num_expert,
                size=(batch_size, num_token, self.num_expert_active),
                dtype=torch.int64,
                device="cuda",
            )
        # ----- #

        # (batch_size, num_token, num_expert_active); float32; contiguous
        router_values = router_values.softmax(dim=2)
        # ----- #


        # ----- #
        # Stage: Calculate `expert_bincount` and update `self.expert_load`
        # ----- #
        # (num_expert,); int64; contiguous; detached
        # (num_expert,); float32; contiguous; detached
        expert_bincount, self.expert_load = self._stage_get_expert_bincount_and_expert_load(expert_assign)
        # ----- #


        # ----- #
        # Stage: Token duplication
        # ----- #
        # (batch_size, num_token, num_expert_active, emb_size); bfloat16; contiguous
        x = self._stage_token_duplication(x)
        # ----- #


        # ----- #
        # Stage: Packing
        # ----- #
        # Define variables
        batch_size, num_token, num_expert_active, emb_size = x.shape
        # (batch_size * num_token * num_expert_active, 1); int32; contiguous; detached
        routing_map = expert_assign.to(torch.int32).view(batch_size * num_token * num_expert_active, 1)
        # (batch_size * num_token * num_expert_active, emb_size); bfloat16; contiguous
        x = x.view(batch_size * num_token * num_expert_active, emb_size)
        # (batch_size * num_token * num_expert_active, emb_size); bfloat16; contiguous
        x, permute_info = te.moe_permute(
            inp=x,
            routing_map=routing_map,
            num_out_tokens=batch_size * num_token * num_expert_active,
            max_token_num=-1,
            map_type="index",
        )
        del routing_map
        # ----- #

        # Prepare metadata
        # Note: This part can happen before token duplication and packing
        #       Therefore, dist.all_to_all_single could execute asynchronously (at the same time)
        # Note: `chunk_sizes_distribute` must follow the assumption (contiguous expert ids on each gpu)
        # (world_size, num_expert_per_rank); int64; contiguous; detached
        chunk_sizes_distribute = expert_bincount.view(self.world_size, self.num_expert_per_rank)
        # Get `chunk_sizes_collect` through all-to-all
        # (world_size, num_expert_per_rank); int64; contiguous; detached
        chunk_sizes_collect = torch.empty_like(chunk_sizes_distribute)
        dist.all_to_all_single(
            output=chunk_sizes_collect,
            input=chunk_sizes_distribute,
            output_split_sizes=self.metadata_a2a_split_size,
            input_split_sizes=self.metadata_a2a_split_size,
        )

        # Do the token all-to-all
        # (world_size,); python list of integers
        input_splits = chunk_sizes_distribute.sum(dim=1).tolist()
        # (world_size,); python list of integers
        output_splits = chunk_sizes_collect.sum(dim=1).tolist()
        # (dyn_pool_size, emb_size); bfloat16; contiguous
        x, _ = all_to_all(input=x, input_splits=input_splits, output_splits=output_splits)

        # Sort by chunk
        # BUG! do not use "with_probs", invert sorted_index explicitly
        # Ask: Is te.moe_sort_chunks_by_index a stable sort
        #     Important issue: if not stable sort, there could be nasty hidden bug where future token goes to the past
        # Important TODO: enforce stable sort by jittering the weaved_chunk_indices_inv and weaved_chunk_indices
        # Ask: Do we need to worry about packing/unpacking in general? (stable sort?)
        split_sizes_chunk_level = chunk_sizes_collect.view(self.world_size * self.num_expert_per_rank)
        x = te.moe_sort_chunks_by_index(
            inp=x,
            split_sizes=split_sizes_chunk_level,
            sorted_index=self.weaved_chunk_indices,
        )


        # ----- #
        # Stage: Expert computation
        # ----- #
        # (num_expert_per_rank,); int; python list on cpu
        group_sizes = chunk_sizes_collect.sum(dim=0).tolist()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False):
            # (dyn_pool_size, expert_size); bfloat16; contiguous
            x = self.grouped_fc_1_no_share(x, group_sizes)
            # (dyn_pool_size, expert_size); bfloat16; contiguous
            x = F.gelu(x)
            # (dyn_pool_size, emb_size); bfloat16; contiguous
            x = self.grouped_fc_2_no_share(x, group_sizes)
        # ----- #

        # inv_split_sizes = chunk_sizes_collect.transpose(0,1).reshape(self.world_size*self.num_expert_per_rank)
        inv_split_sizes = split_sizes_chunk_level[self.weaved_chunk_indices]
        # Inverse sort by chunks
        x = te.moe_sort_chunks_by_index(
            inp=x,
            split_sizes=inv_split_sizes,
            sorted_index=self.weaved_chunk_indices_inv,
        )

        # Do the inverse token all-to-all
        # TODO: Consider better naming for `output_splits` and `input_splits`
        x, _ = all_to_all(input=x, input_splits=output_splits, output_splits=input_splits)

        # ----- #
        # Stage: Unpacking
        # ----- #
        # (batch_size * num_token * num_expert_active, emb_size); bfloat16; contiguous
        x = te.moe_unpermute(
            inp=x,
            row_id_map=permute_info,
            restore_shape=torch.Size([batch_size * num_token * num_expert_active, emb_size]),
            map_type="index",
        )
        # (batch_size, num_token, num_expert_active, emb_size); bfloat16; contiguous
        x = x.view(batch_size, num_token, num_expert_active, emb_size)
        # ----- #


        # ----- #
        # Stage: Token aggregation
        # ----- #
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self._stage_token_aggregation(router_values, x)
        # ----- #


        # (batch_size, num_token, emb_size); float32; contiguous
        return x
