import torch
import torch.distributed as dist


class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_splits, output_splits):
        assert input.is_contiguous()  # Due to `dist.all_to_all_single`

        if output_splits is None:
            world_size = dist.get_world_size()
            input_splits_tensor = torch.tensor(input_splits, device=input.device)
            output_splits_tensor = torch.empty(world_size, dtype=torch.int64, device=input.device)
            dist.all_to_all_single(output_splits_tensor, input_splits_tensor)
            output_splits = output_splits_tensor.cpu().tolist()

        output = torch.empty(sum(output_splits), input.shape[1], device=input.device, dtype=input.dtype)
        dist.all_to_all_single(
            output=output,
            input=input,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
        )

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        return output, output_splits

    @staticmethod
    def backward(ctx, grad_output, _):
        assert grad_output.is_contiguous()  # Due to `dist.all_to_all_single`

        grad_input = torch.empty(sum(ctx.input_splits), grad_output.shape[1],
                                  device=grad_output.device, dtype=grad_output.dtype)
        dist.all_to_all_single(
            grad_input, grad_output,
            output_split_sizes=ctx.input_splits,
            input_split_sizes=ctx.output_splits,
        )
        return grad_input, None, None


def all_to_all(input, input_splits, output_splits=None):
    # input: (num_token, emb_size)
    # input_splits: list
    # output_splits: list; optional
    return AllToAll.apply(input, input_splits, output_splits)
