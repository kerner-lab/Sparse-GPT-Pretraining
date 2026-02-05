"""
Training:
    ./shards_training/0000000000.h5    (1202240, 2049) uint16
    ./shards_training/0000000001.h5    (1202240, 2049) uint16
    ./shards_training/0000000002.h5    (1202240, 2049) uint16
    ./shards_training/0000000003.h5    (1202240, 2049) uint16
Validation:
    ./shards_validation/0000000000.h5  (  12160, 2049) uint16
    ./shards_validation/0000000001.h5  (  12160, 2049) uint16
    ./shards_validation/0000000002.h5  (  12160, 2049) uint16
    ./shards_validation/0000000003.h5  (  12160, 2049) uint16
Note: The integers are in range [0, 50256]
Note: We increase the vocab_size and num_class to 50304, which is a multiple of 128
"""

import os
import h5py
import torch
from torch.utils.data import Dataset


# Note: A batch is divided into `world_size` amount of local batches
# Note: Setting `num_batch_override` truncates the dataset from the beginning.
#       It does not let you sample from the entire dataset.
#       Example: If you want to validate on 25% of the validation set, when you validate for the second time
#                it would be the same 25% of the validation set, not another 25% uniformly sampled from the entire
#                `12160 * 4` validation set
class FineWebEdu10B(Dataset):
    def __init__(self, data_dir, mode, batch_size, rank, world_size, num_batch_override=None):
        super().__init__()
        # Define attributes
        self.data_dir   = data_dir
        self.mode       = mode
        self.batch_size = batch_size
        assert batch_size in {80, 160, 320}
        self.rank       = rank
        self.world_size = world_size
        assert self.world_size in {1, 2, 4, 8}
        self.context_window = 2048
        # Define attributes that depend on `mode`
        if self.mode == "training":
            self.data_dir = os.path.join(self.data_dir, "shards_training")
            self.num_shard = 4
            self.num_sample_per_shard = 1202240
        elif self.mode == "validation":
            self.data_dir = os.path.join(self.data_dir, "shards_validation")
            self.num_shard = 4
            self.num_sample_per_shard = 12160
        else:
            raise Exception("Unexpected dataset mode")
        # Define `num_batch_per_shard`
        assert self.num_sample_per_shard % self.batch_size == 0
        self.num_batch_per_shard = self.num_sample_per_shard // self.batch_size
        # Define `self.num_batch`
        self.num_batch = self.num_shard * self.num_batch_per_shard
        # Define `self.local_batch_size`
        assert self.batch_size % self.world_size == 0
        self.local_batch_size = self.batch_size // self.world_size
        # Apply `num_batch_override`
        if num_batch_override is not None:
            assert 1 <= num_batch_override <= self.num_batch
            self.num_batch = num_batch_override

    def __len__(self):
        return self.num_batch

    def __getitem__(self, idx_batch):
        # Note: __getitem__ is compatible with randomly ordered idx_batch, as long as no indices are skipped
        # Note: A single sample takes ~50 ms to load (Tested on a tensor of (320, 2049) uint16)
        #       The data loader hides this latency through pre-fetching
        # Get idx_shard
        idx_shard = idx_batch // self.num_batch_per_shard
        # Get idx_batch_in_shard
        idx_batch_in_shard = idx_batch % self.num_batch_per_shard
        # Get row_start and row_end
        row_start = idx_batch_in_shard * self.batch_size + self.rank * self.local_batch_size
        row_end   = row_start + self.local_batch_size
        # Note: h5py reads lazily -- we only load the needed slices into RAM, not the entire shard
        # Note: We reopen .h5 file, just in case
        # See also: https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        with h5py.File(os.path.join(self.data_dir, "{:010d}.h5".format(idx_shard)), "r") as f:
            # (local_batch_size, context_window + 1) uint16
            data = f["data"][row_start:row_end].copy()  # Ask: Do we have to call .copy()?
        # (local_batch_size, context_window + 1) int64
        data = torch.from_numpy(data).to(torch.int64)
        assert data.shape == (self.local_batch_size, self.context_window + 1)
        # (local_batch_size, context_window) int64
        inputs, targets = data[:, :-1], data[:, 1:]
        return inputs, targets
