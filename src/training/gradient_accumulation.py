import torch
import torch.distributed as dist


# Note: We accumulate `loss_lm` on-device to avoid ".item()" call in the accumulation loop
#       https://github.com/karpathy/nanoGPT/pull/207#issuecomment-1506348629
def gradient_accumulation(config, inputs, targets, model) -> float:
    # ----- #
    # Define variables
    # ----- #
    accu_steps = config.accu_steps
    # ----- #

    # ----- #
    # Gradient accumulation
    # ----- #
    # Chunk data
    inputs  = torch.chunk(inputs,  accu_steps, dim=0)
    targets = torch.chunk(targets, accu_steps, dim=0)
    for idx_accu in range(accu_steps):
        # Get `inputs_current` and `targets_current`
        inputs_current = inputs[idx_accu].to(device="cuda", non_blocking=True)  # Ask: Should we use non_blocking?
        targets_current = targets[idx_accu].to(device="cuda", non_blocking=True)

        # Forward and backward pass
        loss, telemetry = model(inputs_current, targets_current)
        loss.backward()

        # Accumulate `loss_lm`
        # Note: For better numerical stability, we defer division by `accu_steps` until after accumulation
        if idx_accu == 0:
            loss_lm = telemetry["loss_lm"].clone()
        else:
            loss_lm += telemetry["loss_lm"]

        # Accumulate `expert_load_all`
        if config.runtime["auxfree_enabled"]:
            if idx_accu == 0:
                expert_load_all = telemetry["expert_load_all"].clone()
            else:
                expert_load_all += telemetry["expert_load_all"]
    # ----- #


    # ----- #
    # Average the gradients
    # ----- #
    for n, p in model.named_parameters():
        if p.grad is not None:
            # Average across accumulation steps
            p.grad = p.grad / accu_steps
            if "_no_share" not in n:
                # Average across ranks
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    # ----- #


    # ----- #
    # Average `loss_lm`
    # ----- #
    # Ask: Do we need to call synchronize() before and after `torch.distributed.all_reduce`?
    #      Specifically, `torch.cuda.synchronize` and `torch.distributed.barrier`
    # Average across accumulation steps
    loss_lm = loss_lm / accu_steps
    # Average across ranks
    dist.all_reduce(loss_lm, op=dist.ReduceOp.AVG)
    # Move to cpu
    loss_lm = loss_lm.item()
    # ----- #


    # ----- #
    # Average `expert_load_all`
    # ----- #
    # Ask: Do we need to call `synchronize()` before and after `torch.distributed.all_reduce`?
    if config.runtime["auxfree_enabled"]:
        # Average across accumulation steps
        expert_load_all = expert_load_all / accu_steps
        # Note: MHMoEHP does not average across ranks
        if not config.runtime["expert_load_no_share"]:
            # Average across ranks
            dist.all_reduce(expert_load_all, op=dist.ReduceOp.AVG)
        # Update `config.runtime["expert_load_all"]` on cpu
        # (num_block, num_head, num_expert); float32; contiguous; detached; or
        # (num_block, num_head_per_rank, num_expert); float32; contiguous; detached; or
        # (num_block, num_expert); float32; contiguous; detached
        config.runtime["expert_load_all"] = expert_load_all.cpu()
    # ----- #


    # ----- #
    # Update `config.runtime["auxfree_bias_all"]`
    # ----- #
    # Note: `auxfree_bias_all` is initialized as zeros, independently on each rank
    #     All updates are synchronized across ranks; Therefore, `auxfree_bias_all` ought to be in sync across ranks
    # Note: We subtract `sign(e).mean()` to control the magnitude of the bias, while not changing topk selection
    # See: Appendix C in https://arxiv.org/pdf/2502.16982
    if config.runtime["auxfree_enabled"]:
        # (num_block, num_head, num_expert); float32; contiguous; detached; or
        # (num_block, num_head_per_rank, num_expert); float32; contiguous; detached; or
        # (num_block, num_expert); float32; contiguous; detached
        auxfree_update_ratio = config.runtime["auxfree_update_ratio"]
        # Note: dim=-1 implicitly respects all the cases
        # TODO: Create an auxfree manager or other good strategies/abstraction to manage the auxfree load balancing system
        auxfree_update = torch.sign(expert_load_all.mean(dim=-1, keepdim=True) - expert_load_all)
        auxfree_update = auxfree_update - auxfree_update.mean(dim=-1, keepdim=True)
        auxfree_update = auxfree_update_ratio * auxfree_update
        config.runtime["auxfree_bias_all"] += auxfree_update
    # ----- #


    # float
    return loss_lm
