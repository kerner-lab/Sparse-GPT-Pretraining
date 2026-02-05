import torch
import torch.distributed as dist
from config.config_template import ConfigTemplate
from data.fineweb_edu_10b import FineWebEdu10B


def get_dataloader(config: ConfigTemplate, mode: str):
    # Note: `num_batch_override` is only applied to the training set
    num_batch_override = config.num_batch_override if mode == "training" else None

    # Define `dataset`
    if config.data_name == "FineWebEdu10B":
        dataset = FineWebEdu10B(
            data_dir=config.data_dir,
            mode=mode,
            batch_size=config.batch_size,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            num_batch_override=num_batch_override,
        )
    else:
        raise Exception("Unexpected data_name")

    # Define `dataloader`
    # Ask: When to use pin_memory and non_blocking?
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=config.dataloader_num_worker,
        pin_memory=config.dataloader_pin_memory,
        drop_last=False,
    )
    return dataloader
