import torch
import torch.nn as nn
from config.config_template import ConfigTemplate
from model.block import Block
from model.modules.norm.build_norm import build_norm


# TODO: Remove `build_norm`; Use torch.nn.RMSNorm instead
#       https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
class Model(nn.Module):
    def __init__(self, config: ConfigTemplate):
        super().__init__()
        # ----- #
        # Define attributes
        # ----- #
        self.config = config
        self.num_block = config.num_block
        self.vocab_size = config.vocab_size
        self.num_class = config.num_class
        assert self.vocab_size == self.num_class
        self.context_window = config.context_window
        self.emb_size = config.emb_size
        self.num_expert = config.ffwd_num_expert
        # ----- #


        # ----- #
        # Define layers
        # ----- #
        # Pre-processing stage
        self.wte = nn.Embedding(self.vocab_size, self.emb_size)
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)

        # Transformation stage
        self.block_all = nn.ModuleList()
        for idx_block in range(self.num_block):
            self.block_all.append(Block(config, idx_block))

        # Post-processing stage
        self.norm_cls = build_norm(config)
        self.fc_cls = nn.Linear(self.emb_size, self.num_class, bias=False)
        nn.init.normal_(self.fc_cls.weight, mean=0.0, std=0.02)
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)
        # ----- #


        # ----- #
        # Register parameters for weight decay
        # ----- #
        self.params_decay = list()
        self.params_decay.append(self.wte.weight)
        self.params_decay.append(self.fc_cls.weight)
        # ----- #


    @torch.compile()
    def _stage_pre_processing(self, x):
        """
        In:  (batch_size, num_token); int64; contiguous
        Out: (batch_size, num_token, emb_size); float32; contiguous
        """
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self.wte(x)
        # (batch_size, num_token, emb_size); float32; contiguous
        return x


    @torch.compile()
    def _stage_post_processing_type_1(self, x, y):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
             (batch_size, num_token); int64; contiguous
        Out: (); float32; contiguous
        """
        # Define variables
        batch_size, num_token, emb_size = x.shape
        num_class = self.num_class
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self.norm_cls(x)
        # (batch_size, num_token, num_class); float32; contiguous
        logits = self.fc_cls(x)
        # (); float32; contiguous
        loss_lm = self.cross_entropy_loss(
            logits.view(batch_size * num_token, num_class),
            y.view(batch_size * num_token)
        )
        # (); float32; contiguous
        return loss_lm


    @torch.compile()
    def _stage_post_processing_type_2(self, x):
        """
        In:  (batch_size, num_token, emb_size); float32; contiguous
        Out: (batch_size, num_token, num_class); float32; contiguous
        """
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self.norm_cls(x)
        # (batch_size, num_token, num_class); float32; contiguous
        logits = self.fc_cls(x)
        # (batch_size, num_token, num_class); float32; contiguous
        return logits


    def forward(self, x, y=None):
        """
        In:  (batch_size, num_token); int64; contiguous
             (batch_size, num_token); int64; contiguous; optional
        Out: ...
        """
        # ----- #
        # Stage: Pre-Processing
        # ----- #
        # (batch_size, num_token, emb_size); float32; contiguous
        x = self._stage_pre_processing(x)
        # ----- #


        # ----- #
        # Stage: Transformation
        # ----- #
        # DEBUG START; Temporary workaround
        if self.config.runtime["auxfree_enabled"]:
            # (num_block, num_expert); float32; contiguous; detached; or
            # (num_block, num_head, num_expert); float32; contiguous; detached; or
            # (num_block, num_head_per_rank, num_expert); float32; contiguous; detached
            expert_load_all = torch.zeros(
                size=self.config.runtime["auxfree_shape"],
                dtype=torch.float32,
                device="cuda",
            )
        # DEBUG END

        # Apply the transformer blocks
        for idx_block, block in enumerate(self.block_all):
            # (batch_size, num_token, emb_size); float32; contiguous
            x = block(x)

            # DEBUG START; Temporary workaround
            # Update `expert_load_all`, block-by-block
            if self.config.runtime["auxfree_enabled"]:
                if hasattr(block.ffwd, "expert_load"):
                    if block.ffwd.expert_load is not None:
                        expert_load_all[idx_block] = block.ffwd.expert_load
            # DEBUG END
        # ----- #


        # ----- #
        # Stage: Post-Processing
        # ----- #
        # Note: `loss` is the total loss; For example, `loss = loss_lm + loss_lb`
        if y is not None:
            # (); float32; contiguous
            loss_lm = self._stage_post_processing_type_1(x, y)
            # (); float32; contiguous
            loss = loss_lm

            # Construct `telemetry`
            # Note: All values in `telemetry` are independent (i.e. detached and cloned)
            telemetry = dict()
            telemetry["loss"] = loss.detach().clone()
            telemetry["loss_lm"] = loss_lm.detach().clone()

            # DEBUG START; Temporary workaround
            if self.config.runtime["auxfree_enabled"]:
                telemetry["expert_load_all"] = expert_load_all.detach().clone()
            else:
                telemetry["expert_load_all"] = None
            # DEBUG END

            # (); float32; contiguous
            # dict
            return loss, telemetry
        else:
            # (batch_size, num_token, num_class); float32; contiguous
            logits = self._stage_post_processing_type_2(x)

            # (batch_size, num_token, num_class); float32; contiguous
            return logits
        # ----- #
