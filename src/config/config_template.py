from pydantic import BaseModel, model_validator


class ConfigTemplate(BaseModel):
    # Operational Settings
    project_directory: str                  # Directory where the outputs and logs will be stored
    project_entity: str | None              # Unique name for the wandb entity
    project_name: str                       # Unique name for the wandb project
    run_name: str                           # Unique name for the wandb training run
    use_diagnostic_mode: bool               # Enable extra checks that may slow down execution
    # Basic Settings
    data_name: str                          # Which data to train on
    data_dir: str                           # Which directory the data is located
    vocab_size: int                         # Vocabulary size of the tokenizer
    num_class: int                          # Number of classes to output
    context_window: int                     # Context window of the model
    num_block: int                          # Number of transformer blocks
    emb_size: int                           # Size of the embedding vectors
    num_gpu: int                            # How many GPUs to use
    accu_steps: int                         # Gradient accumulation steps
    batch_size_fwd: int                     # The actual batch size seen in fwd pass
    batch_size: int                         # The effective batch size (GPU & Accu)
    num_batch_override: int | None          # If set, truncates the training set to the first num_batch_override batches
    # Feedforward Settings
    ffwd_name: str                          # "MLP", "MoE", etc
    ffwd_num_head: int | None               # Number of feedforward heads
    ffwd_head_size: int | None              # Size of each feedforward head
    ffwd_hid_size: int | None               # Number of hidden neurons
    ffwd_num_expert: int | None             # Number of experts (active + passive)
    ffwd_num_expert_active: int | None      # Number of experts (active)
    ffwd_expert_size: int | None            # Number of hidden neurons in each expert
    # Attention Settings
    attn_name: str                          # "SelfAttention"
    attn_num_head: int                      # Number of attention heads
    attn_head_size: int                     # Size of each attention head
    # Normalization Settings
    norm_name: str                          # "LayerNorm" "RMSNorm"
    norm_use_affine: bool                   # Whether to use elementwise affine
    norm_use_bias: bool                     # Whether to use bias terms
    norm_eps: float                         # Small constant for numerical stability
    # LR Schedule Settings
    lrsched_max_lr: float                   # Maximum learning rate
    lrsched_min_lr: float                   # Minimum learning rate
    lrsched_warmup_steps: int               # Number of steps for warmup
    lrsched_decay_steps: int                # Number of steps for decay
    # AdamW Settings
    adamw_beta_1: float                     # The beta 1 parameter for AdamW
    adamw_beta_2: float                     # The beta 2 parameter for AdamW
    adamw_eps: float                        # Small constant for numerical stability
    adamw_weight_decay: float               # Weight decay coefficient
    # Gradient Clipping Settings
    gradclip_enabled: bool                  # Whether to enable gradient clipping
    gradclip_max_norm: float                # Maximum norm for the gradient vector
    gradclip_norm_type: float               # Type of norm for gradient clipping
    # Evaluation Settings
    eval_enable_validation: bool            # Whether to perform training-time validation
    eval_evaluators: list[str]              # List of evaluators
    # Performance Settings
    perf_use_profiler: bool                 # Whether to use profiler
    perf_use_8bit_adamw: bool               # Whether to enable 8-bit AdamW
    # Dataloader Settings
    dataloader_num_worker: int              # Number of worker processes (>= 0)
    dataloader_pin_memory: bool             # Whether to use pinned memory
    # Reproducibility Settings
    repro_use_random_seed: bool             # Whether to set a random seed
    repro_random_seed_value: int            # Random seed value
    # Checkpoint Settings
    ckpt_enabled: bool                      # Whether to enable model checkpointing
    # Runtime Variables
    runtime: dict                           # To store variables at runtime


    @model_validator(mode="after")
    def validate_config(self):
        # Validation 1
        assert self.vocab_size == self.num_class
        # Validation 2:
        # Note: To make the classification layer tensor-core-friendly, we require num_class to be a multiple of 128
        assert self.num_class % 128 == 0
        # Validation 3
        assert self.batch_size == self.num_gpu * self.accu_steps * self.batch_size_fwd
        # Validation 4
        assert self.runtime == {}
        return self
