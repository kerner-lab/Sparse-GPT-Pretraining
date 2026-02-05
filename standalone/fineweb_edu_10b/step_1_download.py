# Modified from: https://github.com/KellerJordan/modded-nanogpt/blob/master/data/cached_fineweb10B.py
import os
from huggingface_hub import hf_hub_download
# Config
CHUNK_DIR = "./chunks"
# Create the chunk directory (strict)
os.makedirs(CHUNK_DIR, exist_ok=False)
# Note: Each chunk has 100_000_000 tokens, except for finewebedu_train_000099.bin, which has 53_989_344 tokens
# Download finewebedu_val_000000.bin
hf_hub_download(repo_id="kjj0/finewebedu10B-gpt2", filename="finewebedu_val_000000.bin", repo_type="dataset", local_dir=CHUNK_DIR)
# Download finewebedu_train_000001.bin ~ finewebedu_train_000099.bin
for idx in range(99):
    hf_hub_download(repo_id="kjj0/finewebedu10B-gpt2", filename="finewebedu_train_{:06d}.bin".format(idx + 1), repo_type="dataset", local_dir=CHUNK_DIR)
