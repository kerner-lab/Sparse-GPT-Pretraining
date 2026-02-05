import math


# TODO: Support turning off warmup or decay
def get_learning_rate(
        idx_iter: int,  # index of the current iteration
        max_lr: float,  # maximum learning rate
        min_lr: float,  # minimum learning rate
        warmup_steps: int,  # number of iterations for the warmup phase
        decay_steps: int,  # number of iterations for the decay phase at the end
        num_iter: int,  # the total number of iterations for the training run
) -> float:
    # Note: Set max_lr == min_lr for trapezoidal lr schedule
    # Validation
    assert 0 <= idx_iter < num_iter, "bad idx_iter/num_iter"
    assert 0 < min_lr <= max_lr, "bad max_lr/min_lr"
    assert 2 <= warmup_steps, "bad warmup_steps"
    assert 2 <= decay_steps, "bad decay_steps"
    assert (warmup_steps + decay_steps) <= num_iter, "bad warmup_steps/decay_steps"
    # Stage: Linear Warmup
    if 0 <= idx_iter <= (warmup_steps - 1):
        return max_lr * idx_iter / (warmup_steps - 1.0)
    # Stage: Cosine Decay
    if (warmup_steps - 1) <= idx_iter <= (num_iter - decay_steps):
        ratio = (idx_iter - warmup_steps + 1) / (num_iter - decay_steps - warmup_steps + 1)
        cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + (max_lr - min_lr) * cosine_coeff
    # Stage: Linear Decay
    if (num_iter - decay_steps) <= idx_iter <= (num_iter - 1):
        return min_lr * (1.0 + (num_iter - decay_steps - idx_iter) / (decay_steps - 1.0))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    max_lr       = 2.0e-4
    min_lr       = 2.0e-5
    warmup_steps = 5
    decay_steps  = 5
    num_iter     = 20
    idx_iter_all = list(range(num_iter))
    lr_all = [get_learning_rate(
        idx_iter, max_lr, min_lr, warmup_steps, decay_steps, num_iter
    ) for idx_iter in idx_iter_all]
    plt.scatter(idx_iter_all, lr_all, s=5)
    plt.xlabel("Iteration (idx_iter)")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlim(-1, num_iter)
    plt.ylim(-1e-5, 2.1e-4)
    plt.grid(True)
    plt.axvline(num_iter - 1, color='red', linestyle='--')
    plt.show()
