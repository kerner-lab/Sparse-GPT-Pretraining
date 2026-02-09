# Copyright (c) 2025, Chenwei Cui, Benjamin Joseph Herrera, Rockwell Jackson, Kerner Lab
# SPDX-License-Identifier: MIT

import os
import gc
import uuid
import time
import json
import torch
import wandb
import shutil
import argparse
import numpy as np
import bitsandbytes as bnb
import torch.distributed as dist

from tqdm import tqdm
from datetime import datetime, timedelta

from model.model import Model
from config.get_config import get_config
from data.get_dataloader import get_dataloader
from evaluation.evaluation import evaluation

from utils.synchronize import synchronize
from utils.set_random_seeds import set_random_seeds
from utils.to_cpu_recursive import to_cpu_recursive
from utils.breakpoint_on_rank import breakpoint_on_rank
from utils.report_parameter_count import report_parameter_count

from training.validation import validation
from training.should_validate import should_validate
from training.gradient_accumulation import gradient_accumulation
from training.save_checkpoint import process_iter_folders, save_checkpoint

from optimization.get_param_flags import get_param_flags
from optimization.get_learning_rate import get_learning_rate


# TODO: IMPORTANT: Training resumption logic is too messy and scattered around
#       On this note, the current vault handling is also messy
# TODO: IMPORTANT: Auxfree balancing logic is too messy and scattered around
#       Too much branching! Call for a good abstraction. Look at how people handle batch normalization w/ online stats

# TODO: (module-level engineering) Create the model directly on CUDA
#   Ask: In that case, do we need to addtionally call `.to("cuda")` just to be safe? For example some 3rd party modules initializing on cpu
# TODO: It is important to be conscious about the RAM usage -- if there are 8 ranks, there are 8 processes, 8 times the RAM usage
# TODO: Implement a custom "nn_Linear" class that allows control over out_dtype (see torch.mm, out_dtype)
#       Useful for multi-head merge layers and the final classification layer: higher precision + less casting
# TODO: Implement a `CheckpointHandler` class that
#       (1) Occupies CPU RAM only in the main rank
#       (2) Broadcast to other ranks as needed, e.g. `idx_iter_previous`
#       (3) Handles clean-up
#       (4) Makes sure to save the RNG seed to ensure the same dataloader ordering
# TODO: In diagnostic mode,
#       (1) verify that len(dataloader_train) and len(dataloader_val) are the same, across ranks
#       (2) verify that the hashing values of status_dicts (model and optimizers) are the same, across ranks
#       (3) keep track of the real VRAM usage stage-by-stage*
#           *: For example, before and after forward/backward pass, but not within fwd or bwd
def launch_training(remaining_args):
    # ----- #
    # Prologue
    # ----- #
    # region Stage: Prologue
    # Parse `remaining_args`
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--vault_path", type=str, required=False, default=None)
    parser.add_argument("--skip_training", action="store_true")  # Note: Intented for evaluation
    # Note: Currently does not reflect the real training cost, therefore not in use at the moment.
    parser.add_argument("--estimate_training_time", action="store_true")  # Note: Intended for estimating the training time of a model
    parser.add_argument("--enable_training_time_validation", action="store_true")  # Note: Otherwise, only validate after the last iteration
    args = parser.parse_args(remaining_args)

    # Set up torch.distributed
    # Note: Some useful environment variables: RANK, LOCAL_RANK, WORLD_SIZE
    torch.cuda.set_device(torch.device(int(os.environ["LOCAL_RANK"])))
    # See also: https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(int(os.environ["LOCAL_RANK"])),
        timeout=timedelta(minutes=120),  # Extend the NCCL timeout to 120 minutes instead of 10 minutes
    )
    synchronize()

    # Initialize config
    config = get_config(args.config_file)
    if dist.get_rank() == 0:
        print("\n\n", "Loaded config from:", args.config_file, "\n\n", config, "\n\n")
    # Additional validation
    assert config.num_gpu == dist.get_world_size()
    # Create runtime variables
    # config.runtime["config_file"] = args.config_file

    # Override configs if estimate training time
    if args.estimate_training_time:
        config.num_batch_override = 120
        config.lrsched_warmup_steps = 2
        config.lrsched_decay_steps = 2
        config.eval_enable_validation = False
        config.ckpt_enabled = False
        config.runtime["enforce_random_routing"] = True
        runtime_buff = []

    # Define variables
    rank, world_size = dist.get_rank(), dist.get_world_size()
    vault_path = args.vault_path
    is_new_run = args.vault_path is None
    # Note: Models with distributed weights need per-rank checkpointing
    #   This includes Head Parallel (HP) models and Expert Parallel (EP) models
    has_distributed_weights = config.ffwd_name in {"MHMoEHP", "MHMoEHPNRT", "MoEEP", "LatentMoE"}
    if config.ffwd_name in {"MLP"}:
        auxfree_type = "OFF"
    elif config.ffwd_name in {"MoE", "MoEEP", "LatentMoE"}:
        auxfree_type = "SINGLE_HEAD"
    elif config.ffwd_name in {"MHMoE"}:
        auxfree_type = "MULTI_HEAD_NO_HP"
    elif config.ffwd_name in {"MHMoEHP", "MHMoEHPNRT"}:
        auxfree_type = "MULTI_HEAD_WITH_HP"
    else:
        raise Exception("Unexpected config.ffwd_name")
    # endregion
    # ----- #


    # ----- #
    # Random Seed
    # ----- #
    # region Stage: Random Seed
    # Ask: (DDP) Should we use different seeds on different ranks?
    # Consideration: Different seeds allow for IID, that said, model weights needs to be broadcast at initialization
    # Note: Q: What happens if we are resuming a training run?
    #       A: Random seed is determined in config, so we do not need to save/restore it.
    #          That said, even if the random seed is the same during resumption, it does not yet ensure
    #            identical training because generator state is not saved/restored.
    #          Saving generator state is a more involved approach and is left for future work.
    if config.repro_use_random_seed:
        set_random_seeds(config)
    # endregion
    # ----- #


    # ----- #
    # PyTorch Settings
    # ----- #
    # region Stage: PyTorch Settings
    # Note: We need to be aware of the numerical implication of using TF32 in pytorch and in triton
    # TODO: Find a way to explicitly manage TF32 vs FP32 at an op level
    # CUDA
    torch.backends.cuda.matmul.allow_tf32 = True
    # CUDNN
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    # Dtype
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("high")
    # torch.compile
    if config.use_diagnostic_mode:
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.recompile_limit = 8  # Note: 8 or 1; Includes the initial compile
        torch._dynamo.config.fail_on_recompile_limit_hit = True
    # endregion
    # ----- #


    # ----- #
    # Vault
    # ----- #
    # region Stage: Vault
    # Note: vault_name example: E1-01_251012_134031_8a109b4a-e4d3-4bc9-b3c1-e08724f09573
    if vault_path is None:
        # If `vault_path` is not provided, create a new vault
        if rank == 0:
            # Define `vault_path`
            timestamp  = datetime.now().strftime("%y%m%d_%H%M%S")
            vault_uuid = str(uuid.uuid4())
            vault_name = f"{config.run_name}_{timestamp}_{vault_uuid}"  # TODO: Verify that `vault_name` is a valid name
            vault_path = os.path.join(config.project_directory, vault_name)
            # Create the vault folder
            os.makedirs(vault_path, exist_ok=False)  # Note: We expect it to be unique
            # Create the checkpoints folder
            os.makedirs(os.path.join(vault_path, "checkpoints"), exist_ok=False)
            # Copy the yaml file and store it in the vault
            # Consider: Read the file only once; Read in bytes during `get_config(.)` and later save it in the vault
            shutil.copy2(args.config_file, os.path.join(vault_path, "config.yaml"))
        synchronize()
        # Broadcast `vault_path` from rank 0 to all ranks
        vault_path = [vault_path]
        dist.broadcast_object_list(vault_path, src=0)
        vault_path = vault_path[0]
        if rank == 0:
            print(f"\n\nCreated a new vault: {vault_path}\n\n")
    else:
        # If `vault_path` is provided, use the existing one
        if rank == 0:
            print(f"\n\nUsing an existing vault: {vault_path}\n\n")
    # endregion
    # ----- #


    # ----- #
    # Wandb Initialization
    # ----- #
    # region Stage: Wandb Initialization
    # Note: Wandb is always offline (`WANDB_MODE="offline"`) to prevent any internet-related issues
    #       Explicitly upload to the wandb server with `wandb sync your_run`
    # Note: If resume training, wandb would create an entirely new run in the same vault
    if rank == 0:
        # Initialize wandb
        # wandb.login()  # Debug: "wandb: WARNING Unable to verify login in offline mode."
        run = wandb.init(
            entity=config.project_entity,
            project=config.project_name,
            dir=vault_path,
            name=config.run_name,
            id=None,
        )
    else:
        run = None
    synchronize()
    # endregion
    # ----- #


    # ----- #
    # Data
    # ----- #
    dataloader_train = get_dataloader(config, mode="training")
    dataloader_val   = get_dataloader(config, mode="validation")
    # ----- #


    # ----- #
    # Model
    # ----- #
    # region Stage: Model
    # DEBUG START; TODO: Implement auxfree manager
    config.runtime["expert_load_all"] = None
    config.runtime["auxfree_update_ratio"] = None
    if auxfree_type == "OFF":
        config.runtime["auxfree_enabled"] = False
        config.runtime["expert_load_no_share"] = None
        config.runtime["auxfree_shape"] = None
        config.runtime["auxfree_bias_all"] = None
    elif auxfree_type == "SINGLE_HEAD":
        config.runtime["auxfree_enabled"] = True
        config.runtime["expert_load_no_share"] = False
        # (num_block, num_expert); float32; contiguous; detached
        config.runtime["auxfree_bias_all"] = torch.zeros(
            size=(config.num_block, config.ffwd_num_expert),
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        config.runtime["auxfree_shape"] = config.runtime["auxfree_bias_all"].shape
    elif auxfree_type == "MULTI_HEAD_NO_HP":
        config.runtime["auxfree_enabled"] = True
        config.runtime["expert_load_no_share"] = False
        # (num_block, num_head, num_expert); float32; contiguous; detached
        config.runtime["auxfree_bias_all"] = torch.zeros(
            size=(config.num_block, config.ffwd_num_head, config.ffwd_num_expert),
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        config.runtime["auxfree_shape"] = config.runtime["auxfree_bias_all"].shape
    elif auxfree_type == "MULTI_HEAD_WITH_HP":
        config.runtime["auxfree_enabled"] = True
        config.runtime["expert_load_no_share"] = True
        # Define `ffwd_num_head_per_rank`
        assert config.ffwd_num_head % world_size == 0
        ffwd_num_head_per_rank = config.ffwd_num_head // world_size
        # (num_block, num_head_per_rank, num_expert); float32; contiguous; detached
        config.runtime["auxfree_bias_all"] = torch.zeros(
            size=(config.num_block, ffwd_num_head_per_rank, config.ffwd_num_expert),
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        config.runtime["auxfree_shape"] = config.runtime["auxfree_bias_all"].shape
    else:
        raise Exception("Unexpected auxfree_type")
    # DEBUG END

    # Define `model`, on each rank
    if rank == 0:
        print("\n\n")
    # Workaround: One rank at a time to reduce CPU RAM pressure
    for idx_rank in tqdm(range(world_size), desc="Model Init", disable=rank != 0):
        if rank == idx_rank:
            model = Model(config).to("cuda")
        synchronize()
    if rank == 0:
        print("\n\n")

    # Broadcast parameters and buffers from rank 0 (except _no_share ones)
    # TODO: Learn about `model.buffers()`; E.g. There are buffers even though we are not aware (RoPE)
    for n, p in model.named_parameters():
        if "_no_share" not in n:
            dist.broadcast(p.data, src=0)
    for n, b in model.named_buffers():
        if "_no_share" not in n:
            dist.broadcast(b.data, src=0)

    # Report parameter count and print the model
    if rank == 0:
        print("\n\n")
        report_parameter_count(config, model, verbose=config.use_diagnostic_mode)
        print("\n\n")
        print(model)
        print("\n\n")
    # endregion
    # ----- #


    # ----- #
    # Optimization
    # ----- #
    # region Stage: Optimization
    # Note: Do not duplicate/overlap parameters; This will increase their effective learning rate
    # Note: We apply 8-bit adamw only to select parameters, to ensure numerical stability
    #       This may still be subject to the `min_8bit_size=4096` limit
    # Consider: Find a more concise way to report the parameters that are decay or 8bit
    #       The current approach prints too many lines
    # Get `params_decay`, `params_no_decay`, `params_decay_8bit`, and `params_no_decay_8bit`
    params_decay = list()
    params_no_decay = list()
    params_decay_8bit = list()
    params_no_decay_8bit = list()
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_flags = get_param_flags(model, p)
            if param_flags["decay"]:
                if param_flags["8bit"]:
                    params_decay_8bit.append(p)
                    if config.use_diagnostic_mode and rank == 0:
                        print(f"{name} is in `params_decay_8bit`")
                else:
                    params_decay.append(p)
                    if config.use_diagnostic_mode and rank == 0:
                        print(f"{name} is in `params_decay`")
            else:
                if param_flags["8bit"]:
                    params_no_decay_8bit.append(p)
                    if config.use_diagnostic_mode and rank == 0:
                        print(f"{name} is in `params_no_decay_8bit`")
                else:
                    params_no_decay.append(p)
                    if config.use_diagnostic_mode and rank == 0:
                        print(f"{name} is in `params_no_decay`")
    # Define `optimizer_1` and `optimizer_2`
    optimizer_1 = torch.optim.AdamW(
        params=[
            {"params": params_decay,    "weight_decay": config.adamw_weight_decay},
            {"params": params_no_decay, "weight_decay": 0.0},
        ],
        betas=(config.adamw_beta_1, config.adamw_beta_2),
        eps=config.adamw_eps,
        fused=False,  # Consider enabling `fused`
    )
    optimizer_2 = bnb.optim.AdamW(
        params=[
            {"params": params_decay_8bit,    "weight_decay": config.adamw_weight_decay},
            {"params": params_no_decay_8bit, "weight_decay": 0.0},
        ],
        betas=(config.adamw_beta_1, config.adamw_beta_2),
        eps=config.adamw_eps,
        optim_bits=8,
        min_8bit_size=4096,
    )
    # endregion
    # ----- #


    # ----- #
    # Resume from a previous run
    # ----- #
    # region Stage: Resume from a previous run
    if not is_new_run:
        if rank == 0:
            print("\n\n", "Resuming from a previous run", "\n\n")
        # Synchronize before proceeding
        synchronize()
        # Define `iter_folder_all` and sort it
        iter_folder_all = os.listdir(os.path.join(vault_path, "checkpoints"))
        iter_folder_all = sorted(iter_folder_all, key=lambda x: int(x))  # Smallest idx_iter first
        assert len(iter_folder_all) > 0
        # Workaround: One rank at a time to reduce CPU RAM pressure
        for idx_rank in tqdm(range(world_size), disable=rank != 0):
            if rank == idx_rank:
                print(f"\n\nProcessing rank {rank}\n\n")
                # Define `checkpoint_path`
                checkpoint_path = os.path.join(vault_path, "checkpoints", iter_folder_all[-1])  # Get the last one
                if has_distributed_weights:
                    # Note: Head Parallel, each rank loads from its own `.pt` file
                    checkpoint_path = os.path.join(checkpoint_path, f"rank_{idx_rank}.pt")
                else:
                    # Note: Data Parallel, each rank loads from `rank_0.pt`
                    checkpoint_path = os.path.join(checkpoint_path, "rank_0.pt")
                # Load `checkpoint_dict`, on each rank
                checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
                # Define `idx_iter_previous` and `world_size_previous`, on each rank
                idx_iter_previous = checkpoint_dict["idx_iter"]
                world_size_previous = checkpoint_dict["world_size"]
                # Consider: Explicitly handle the `world_size != world_size_previous` scenario
                if rank == 0:
                    print("\n\n")
                    print(f"world_size_previous is {world_size_previous}")
                    print(f"world_size          is {world_size}")
                    print(f"idx_iter_previous   is {idx_iter_previous}")
                    print(f"wandb run.step      is {run.step}")
                    print("\n\n")
                # Update `config.runtime["auxfree_bias_all"]`, on each rank
                config.runtime["auxfree_bias_all"] = checkpoint_dict["auxfree_bias_all"]
                if config.runtime["auxfree_bias_all"] is not None:
                    config.runtime["auxfree_bias_all"] = config.runtime["auxfree_bias_all"].cuda()
                # Load from `state_dict_model`, on each rank
                # Ask: Does `load_state_dict` support {CPU source, CUDA destination}?
                load_state_dict_result = model.load_state_dict(checkpoint_dict["state_dict_model"], strict=False)
                print("Missing keys:", load_state_dict_result.missing_keys)  # Possible excessive printing
                print("Unexpected keys:", load_state_dict_result.unexpected_keys)
                # Load from `state_dict_optimizer_1` and `state_dict_optimizer_2`, on each rank
                # Ask: How do we make sure strictly 1:1?
                optimizer_1.load_state_dict(checkpoint_dict["state_dict_optimizer_1"])
                optimizer_2.load_state_dict(checkpoint_dict["state_dict_optimizer_2"])
                # Release `checkpoint_dict`
                del checkpoint_dict
                gc.collect()
            synchronize()
    # endregion
    # ----- #


    # ----- #
    # Training Loop
    # ----- #
    # region Stage: Training loop
    if not args.skip_training:
        for idx_iter, (inputs, targets) in enumerate(tqdm(
            iterable=dataloader_train,
            desc="Training model",
            total=len(dataloader_train),
            disable=rank != 0,  # Enable tqdm only for rank 0
        )):
            # Note: If we resume training, skip until AFTER `idx_iter_previous`
            #   (1) We save `idx_iter` after completing that iteration
            #   (2) Thus, we skip up to and including `idx_iter_previous` to resume at the next unseen iteration
            if not is_new_run:
                if idx_iter <= idx_iter_previous:
                    continue


            # Reset wandb_log
            wandb_log = dict()
            wandb_log["step"] = idx_iter


            # Start the timer
            synchronize()
            t1 = time.perf_counter()


            # Set the model to training mode
            model.train()


            # ----- #
            # Set learning rate
            # ----- #
            # DEBUG START
            # Consider: setting update ratio to 0.0 during the last 2000 decay steps (See https://arxiv.org/pdf/2502.16982)
            # Note: When the main lr is 2.0e-4, 0.01 is a good choice, therefore choosing 50 * max_lr
            config.runtime["auxfree_update_ratio"] = 50.0 * config.lrsched_max_lr
            wandb_log["auxfree_update_ratio"] = config.runtime["auxfree_update_ratio"]
            # DEBUG END

            # Get learning rate
            lr = get_learning_rate(
                idx_iter=idx_iter,
                max_lr=config.lrsched_max_lr,
                min_lr=config.lrsched_min_lr,
                warmup_steps=config.lrsched_warmup_steps,
                decay_steps=config.lrsched_decay_steps,
                num_iter=len(dataloader_train),
            )
            # Apply learning rate
            for param_group in optimizer_1.param_groups:
                param_group["lr"] = lr
            for param_group in optimizer_2.param_groups:
                param_group["lr"] = lr
            # Update wandb_log
            wandb_log["lr"] = lr
            # ----- #


            # ----- #
            # Take one gradient step
            # ----- #
            # Reset the optimizers
            optimizer_1.zero_grad(set_to_none=True)
            optimizer_2.zero_grad(set_to_none=True)
            # Perform gradient accumulation
            loss_lm = gradient_accumulation(config, inputs, targets, model)
            # Update wandb_log
            wandb_log["loss_lm"] = loss_lm
            # Apply gradient clipping
            if config.gradclip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=config.gradclip_max_norm,
                    norm_type=config.gradclip_norm_type,
                )
            # Update the parameters
            optimizer_1.step()
            optimizer_2.step()
            # ----- #


            # ----- #
            # Stop the timer
            # ----- #
            synchronize()
            t2 = time.perf_counter()
            # Get iter_time
            iter_time = t2 - t1
            # Update wandb_log
            wandb_log["iter_time"] = iter_time
            # ----- #




            # ----- #
            # Calculate and handle estimating training time if enabled
            # ----- #
            if args.estimate_training_time and (rank == 0):
                # Skip the first 20 iterations as performance warmup
                if idx_iter >= 20:
                    runtime_buff.append(iter_time)
                    # Calculate statistics
                    runtime_arr = np.array(runtime_buff)
                    n = len(runtime_buff)
                    mean_time = np.mean(runtime_arr)
                    if n >= 2:
                        std_time = np.std(runtime_arr, ddof=1)  # Note: This is unbiased estimation
                    else:
                        std_time = float("nan")
                    print(f"Runtime stats: mean={mean_time:.4f}s, std={std_time:.4f}s, n={n}")
            # ----- #




            # ----- #
            # Validation
            # ----- #
            # Note: We perform validation 32 times, including after the first iter and after the last iter.
            # Consider: Instead, perform validation 32 times, including **before** the first iter and after the last iter.
            if config.eval_enable_validation:
                if args.enable_training_time_validation:
                    _should_validate = should_validate(idx_iter=idx_iter, num_iter=len(dataloader_train), num_val=32)
                else:
                    _should_validate = (idx_iter + 1) == len(dataloader_train)
                if _should_validate:
                    # Workaround: Avoid hiding the training progress bar
                    if rank == 0:
                        print("\n\n")
                    # Synchronize before validation
                    synchronize()
                    # Start validation
                    perplexity_val = validation(config, model, dataloader_val)
                    # Update wandb_log
                    wandb_log["perplexity_val"] = perplexity_val
                    # Synchronize before proceeding
                    synchronize()
            # ----- #


            # ----- #
            # Optionally visualize `expert_load_all`
            # ----- #
            # TODO: Encapsulate it into a util function e.g. `visualize_expert_load_all(config, wandb_log, expert_load_all)`
            # Note: We assume `config.runtime["expert_load_all"]` has been synchronized globally in prior steps
            # Note: We move `expert_load_all` to cpu first, to avoid moving data element-by-element
            if rank == 0:
                if config.runtime["expert_load_all"] is not None:
                    if auxfree_type == "OFF":
                        pass
                    elif auxfree_type == "SINGLE_HEAD":
                        expert_load_all_cpu = config.runtime["expert_load_all"].cpu()
                        percentile_all = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00], dtype=torch.float32, device="cpu")
                        for idx_block in range(config.num_block):
                            # (num_expert,); float32; contiguous
                            current = expert_load_all_cpu[idx_block]
                            # (5,); float32; contiguous
                            current = torch.quantile(current, percentile_all, interpolation="linear")
                            wandb_log[f"expert_load/block_{idx_block}/P000"] = current[0].item()
                            wandb_log[f"expert_load/block_{idx_block}/P025"] = current[1].item()
                            wandb_log[f"expert_load/block_{idx_block}/P050"] = current[2].item()
                            wandb_log[f"expert_load/block_{idx_block}/P075"] = current[3].item()
                            wandb_log[f"expert_load/block_{idx_block}/P100"] = current[4].item()
                    elif auxfree_type == "MULTI_HEAD_NO_HP":
                        expert_load_all_cpu = config.runtime["expert_load_all"].cpu()
                        percentile_all = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00], dtype=torch.float32, device="cpu")
                        for idx_block in range(config.num_block):
                            for idx_head in range(config.ffwd_num_head):
                                # (num_expert,); float32; contiguous
                                current = expert_load_all_cpu[idx_block, idx_head]
                                # (5,); float32; contiguous
                                current = torch.quantile(current, percentile_all, interpolation="linear")
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P000"] = current[0].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P025"] = current[1].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P050"] = current[2].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P075"] = current[3].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P100"] = current[4].item()
                    elif auxfree_type == "MULTI_HEAD_WITH_HP":
                        expert_load_all_cpu = config.runtime["expert_load_all"].cpu()
                        percentile_all = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00], dtype=torch.float32, device="cpu")
                        # Define `ffwd_num_head_per_rank`
                        assert config.ffwd_num_head % world_size == 0
                        ffwd_num_head_per_rank = config.ffwd_num_head // world_size
                        for idx_block in range(config.num_block):
                            for idx_head in range(ffwd_num_head_per_rank):
                                # (num_expert,); float32; contiguous
                                current = expert_load_all_cpu[idx_block, idx_head]
                                # (5,); float32; contiguous
                                current = torch.quantile(current, percentile_all, interpolation="linear")
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P000"] = current[0].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P025"] = current[1].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P050"] = current[2].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P075"] = current[3].item()
                                wandb_log[f"expert_load/block_{idx_block}/head_{idx_head}/P100"] = current[4].item()
                    else:
                        raise Exception("Unexpected auxfree_type")
            # ----- #


            # ----- #
            # Visualize `watchdog`
            # ----- #
            for name, module in model.named_modules():
                if hasattr(module, "watchdog"):
                    for key, value in module.watchdog.items():
                        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.AVG)
                        name = name if name else "Model"
                        wandb_log[f"{key}/{name}"] = value.item()
            # ----- #


            # ----- #
            # Submit wandb_log
            # ----- #
            if rank == 0:
                run.log(wandb_log, step=idx_iter)
            # ----- #


            # ----- #
            # Checkpointing
            # ----- #
            # Note: Perform checkpointing at the end of the iteration, after gradient updates and after updating wandb
            # Ask: What may happen if only one of the ranks failed to save?
            # Ask: We delete before we save; How do we make sure the one that remains is not a corrupted checkpoint?
            # Condition 1: If 4000 iterations has completed (e.g. 3999, 7999 ...)
            condition_1 = (idx_iter + 1) % 4000 == 0
            # Condition 2: Is the last iteration
            condition_2 = idx_iter == len(dataloader_train) - 1
            if (condition_1 or condition_2) and config.ckpt_enabled:
                if rank == 0:
                    print("\n\nCheckpointing - Start\n\n")

                # Delete the old iter folders and then create a new one, on rank 0
                if rank == 0:
                    process_iter_folders(vault_path, idx_iter)
                synchronize()

                # Define `idx_rank_all`
                if has_distributed_weights:
                    # Note: If using Head Parallel, we save everything on each rank
                    # Reasoning: The majority of the weights are already distributed, the redundant part is small
                    idx_rank_all = list(range(world_size))
                else:
                    # Note: If using Data Parallel, we save everything on rank 0
                    idx_rank_all = [0]

                # Workaround: One rank at a time to reduce CPU RAM pressure
                # Known Issue: NCCL timeout at 600 seconds mark
                #   In part due to: (a) Processing one rank at a time; (b) The loop does not refresh the NCCL timeout timer
                # TODO: Allow retry, especially at the end of training to ensure successful checkpointing
                flag_no_timeout_yet = True
                for idx_rank in tqdm(idx_rank_all, disable=rank != 0):
                    if (rank == idx_rank) and flag_no_timeout_yet:
                        # Define `checkpoint_dict`, on rank `idx_rank`
                        checkpoint_dict = dict()
                        # Save `idx_iter` and `world_size`, on rank `idx_rank`
                        checkpoint_dict["idx_iter"] = idx_iter
                        checkpoint_dict["world_size"] = world_size
                        # Save `auxfree_bias_all`, on rank `idx_rank`
                        checkpoint_dict["auxfree_bias_all"] = config.runtime["auxfree_bias_all"]
                        # Save `state_dict_model`, on rank `idx_rank`
                        checkpoint_dict["state_dict_model"] = model.state_dict()
                        # Save `state_dict_optimizer_1` and `state_dict_optimizer_2`, on rank `idx_rank`
                        checkpoint_dict["state_dict_optimizer_1"] = optimizer_1.state_dict()
                        checkpoint_dict["state_dict_optimizer_2"] = optimizer_2.state_dict()  # Ask: Does it save in 8-bit?
                        # Move `checkpoint_dict` to cpu, on rank `idx_rank`
                        checkpoint_dict = to_cpu_recursive(checkpoint_dict)
                        # Save into the new iter folder, on rank `idx_rank`
                        try:
                            save_checkpoint(vault_path, checkpoint_dict, idx_iter, rank=rank)
                        except TimeoutError as e:
                            print("\n\n")
                            print(f"Checkpointing timed out on rank {rank}!")
                            print(e)
                            print("\n\n")
                            # Cancel checkpointing on all ranks
                            flag_no_timeout_yet = False
                        # Release `checkpoint_dict`, on rank `idx_rank`
                        del checkpoint_dict
                        gc.collect()
                    synchronize()
                    # Broadcast the flag from rank `idx_rank` to all ranks
                    flag_no_timeout_yet = [flag_no_timeout_yet]
                    dist.broadcast_object_list(flag_no_timeout_yet, src=idx_rank)
                    flag_no_timeout_yet = flag_no_timeout_yet[0]
                if rank == 0:
                    print("\n\nCheckpointing - End\n\n")
            # ----- #


            # ----- #
            # Clean up to end this iteration
            # ----- #
            synchronize()
            # ----- #
    # endregion
    # ----- #


    # ----- #
    # Save Training Time Estimation Results
    # ----- #
    if args.estimate_training_time and (rank == 0):
        # Create folder if not exists
        estimation_folder = "iter_time_estimations"
        os.makedirs(estimation_folder, exist_ok=True)
        # Save results
        estimation_file = os.path.join(estimation_folder, f"{config.run_name}.json")
        estimation_data = {
            "run_name": config.run_name,
            "mean_time": mean_time,
            "std_time": std_time,
            "n_samples": n,
        }
        with open(estimation_file, "w") as f:
            json.dump(estimation_data, f, indent=2)
        print(f"Training time estimation saved to: {estimation_file}")
    # ----- #


    # ----- #
    # Evaluation
    # ----- #
    # region Stage: Evaluation
    # BUG: This part may suffer from internet instability and fail
    """ Information
    Key error:
    ```
    OSError: [Errno 101] Network is unreachable
    HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded
    ```
    This happens during the evaluation phase when calling `AutoTokenizer.from_pretrained()`.
    Common on HPC clusters where compute nodes lack external internet access.
    Solutions:
    1. Pre-download the tokenizer on a login node with internet, then use `local_files_only=True`
    2. Set `HF_HOME` or `TRANSFORMERS_CACHE` environment variable to a pre-cached location
    3. Use a local path instead of "gpt2" model identifier
    4. Run the download/cache step before submitting the job
    """
    # Skip evaluation if estimating training time
    if not args.estimate_training_time:
        # Get `evaluation_results`
        evaluation_results = evaluation(config, model)
        # Present `evaluation_results` and save to file
        if dist.get_rank() == 0:
            print("\n\n\n\nEvaluation Results:")
            for k, v in evaluation_results.items():
                print(k)
                print(v)
                print("\n")
            print("\n\n\n\n")
            # Save evaluation results to vault
            eval_results_path = os.path.join(vault_path, "eval_results.txt")
            with open(eval_results_path, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"Evaluation results saved to: {eval_results_path}\n")
        synchronize()
    # endregion
    # ----- #


    # ----- #
    # Clean up to end this training run
    # ----- #
    # End the wandb run
    if rank == 0:
        run.finish()
    # Ensure all processes finish
    synchronize()
    # Final clean-up
    dist.destroy_process_group()
    # ----- #
