import os
import gc
import h5py
import numpy as np
from tqdm import tqdm


def read_one_chunk(file_path):
    with open(file_path, "rb") as f:
        # Skip the first 256 * 4 = 1024 bytes (header)
        # See: https://github.com/KellerJordan/modded-nanogpt/blob/master/data/fineweb.py
        f.seek(1024)
        out = np.fromfile(f, dtype=np.uint16)
    return out

"""
INPUT:
    Training:
        ./chunks/finewebedu_train_000001.bin  # (100_000_000,) uint16
        ./chunks/finewebedu_train_000002.bin  # (100_000_000,) uint16
        ...
        ./chunks/finewebedu_train_000098.bin  # (100_000_000,) uint16
        ./chunks/finewebedu_train_000099.bin  # ( 53_989_344,) uint16
    Validation:
        ./chunks/finewebedu_val_000000.bin    # (100_000_000,) uint16
"""

# ----- #
# Stage: Preparation
# ----- #
# Config
CHUNK_DIR       = "./chunks"
SHARD_DIR_TRAIN = "./shards_training"
SHARD_DIR_VAL   = "./shards_validation"
CONTEXT_WINDOW  = 2048
# Note: We only support CONTEXT_WINDOW == 2048 for now
assert CONTEXT_WINDOW == 2048
# Create the directories (strict)
os.makedirs(SHARD_DIR_TRAIN, exist_ok=False)
os.makedirs(SHARD_DIR_VAL, exist_ok=False)
# ----- #


# ----- #
# Stage: Get data_train and data_val
# ----- #
# Assumption: Each chunk has 100_000_000 tokens, except for finewebedu_train_000099.bin, which has 53_989_344 tokens
data_train = []
for idx in tqdm(range(99)):  # From finewebedu_train_000001.bin to finewebedu_train_000099.bin
    current_chunk = read_one_chunk(os.path.join(CHUNK_DIR, "finewebedu_train_{:06d}.bin".format(idx + 1)))
    data_train.append(current_chunk)
# (9853989344,) uint16
data_train = np.concatenate(data_train)
# ( 100000000,) uint16
data_val   = read_one_chunk("./chunks/finewebedu_val_000000.bin")
# Verify assumptions
assert data_train.shape == (9_853_989_344,)
assert data_train.dtype == np.uint16
assert data_val.shape   == (100_000_000,)
assert data_val.dtype == np.uint16
# ----- #


# ----- #
# Stage: Get samples_train and samples_val
# ----- #
"""
samples_train: (4811518, 2049) uint16
samples_val:   (  48828, 2049) uint16
"""
# Get length_train and length_val
length_train = len(data_train)  # 9853989344
length_val   = len(data_val)    #  100000000
# Overlap-by-one selection with drop-last
window_size = CONTEXT_WINDOW + 1
step_size   = CONTEXT_WINDOW
num_step_train = (length_train - window_size) // step_size + 1
num_step_val   = (length_val   - window_size) // step_size + 1
assert num_step_train == 4_811_518
assert num_step_val   == 48_828
# For training set
samples_train = []
for idx in tqdm(range(num_step_train)):
    start = idx * step_size
    samples_train.append(data_train[start:start + window_size])
samples_train = np.stack(samples_train)  # (4811518, 2049) uint16
# Clean-up
del data_train
gc.collect()
# For validation set
samples_val = []
for idx in tqdm(range(num_step_val)):
    start = idx * step_size
    samples_val.append(data_val[start:start + window_size])
samples_val = np.stack(samples_val)  # (48828, 2049) uint16
# Clean-up
del data_val
gc.collect()
# Verify assumptions
assert samples_train.shape == (4811518, 2049,)
assert samples_train.dtype == np.uint16
assert samples_val.shape == (  48828, 2049,)
assert samples_val.dtype == np.uint16
# ----- #

# ----- #
# Stage: Shuffling
# ----- #
np.random.seed(42)
np.random.shuffle(samples_train)  # (4811518, 2049)
np.random.shuffle(samples_val)    # (  48828, 2049)
# ----- #

# ----- #
# Stage: Sharding
# ----- #
"""
For training set,
    Out of 4811518 samples, we keep 4808960 = 4 * 3757 * 320 = 15028 * 320 samples
    This is in alignment with existing experiments (Trained on 15028 * 320 tokens)
For validation set,
    Out of 48828 samples, we keep 48640 = 4 * 320 * 38 samples
OUTPUT:
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
"""
# Token dropping
samples_train = samples_train[:4808960]  # (4808960, 2049) uint16
samples_val   = samples_val[:48640]      # (  48640, 2049) uint16
# Training set
num_shard  = 4
shard_size = 1202240
assert num_shard * shard_size == len(samples_train)
for idx in tqdm(range(num_shard)):
    start = idx * shard_size
    end   = start + shard_size
    shard_path = os.path.join(SHARD_DIR_TRAIN, f"{idx:010d}.h5")
    with h5py.File(shard_path, "w") as f:
        f.create_dataset("data", data=samples_train[start:end], dtype=np.uint16, compression=None)
# Validation set
num_shard  = 4
shard_size = 12160
assert num_shard * shard_size == len(samples_val)
for idx in tqdm(range(num_shard)):
    start = idx * shard_size
    end   = start + shard_size
    shard_path = os.path.join(SHARD_DIR_VAL, f"{idx:010d}.h5")
    with h5py.File(shard_path, "w") as f:
        f.create_dataset("data", data=samples_val[start:end], dtype=np.uint16, compression=None)
# ----- #
