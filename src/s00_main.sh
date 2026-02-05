#!/usr/bin/env bash

# TODO: (QoL) In the old resume training script, the config file was hardcoded as vault_path/config.yaml, so should this script

# ----- #
# Usage
# ----- #
# Normal Training:
#     s00_main.sh ./config_files/Example.yaml
# Resume Training:
#     s00_main.sh ./config_files/Example.yaml --vault_name <path_to_vault>
# After-Training Evaluation:
#     s00_main.sh ./config_files/Example.yaml --skip_training --vault_name <path_to_vault>
# Estimate Training Time:
#     s00_main.sh ./config_files/Example.yaml --estimate_training_time
# ----- #

# ----- #
# Basic settings
# ----- #
source "$HOME/.bashrc"
conda activate moe
__CONFIG_FILE="${1}"
__NUM_GPU="$(yq .num_gpu "${__CONFIG_FILE}")"
# ----- #

# ----- #
# Additional settings
# ----- #
# Note: Increase `OMP_NUM_THREADS` if CPU performance starts to become a bottleneck
# Related: https://github.com/pytorch/pytorch/pull/22501#issuecomment-511515722
export OMP_NUM_THREADS="1"
# Do not print every logged value at the end
export WANDB_QUIET="true"
# Offline training
export WANDB_MODE="offline"
# To suppress a {huggingface, lm_eval} warning; See: https://github.com/huggingface/transformers/issues/5486
export TOKENIZERS_PARALLELISM="false"
# No __pycache__
# export PYTHONDONTWRITEBYTECODE="1"  # Note: Disabled; Unless there is a good reason to apply this
# ----- #

# ----- #
# Expose libraries and tools that were installed in the user space
# ----- #
# GCC and G++
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
# CUDA
export CUDA_HOME="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_PATH="${CONDA_PREFIX}/targets/x86_64-linux"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
# CUDNN
export CUDNN_PATH="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
# Transformer Engine 2.5 (see documentation)
export NVTE_CUDA_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include"
# ----- #

# ----- #
# Prepare for run logging
# ----- #
mkdir -p "./_run_outputs"
__RUN_TIMESTAMP="$(date +%y%m%d%H%M%S)"
__RUN_UUID="$(uuidgen)"
__RUN_LOG_FILE="./_run_outputs/_run_output_${__RUN_TIMESTAMP}_${__RUN_UUID}.txt"
# ----- #

# ----- #
# Start
# ----- #
# Note: https://docs.pytorch.org/docs/stable/elastic/run.html#single-node-multi-worker
# Note: https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py
# Note: We write stdout and stderr to a local file
# Note: "${@:2}" forwards extra arguments (other than __CONFIG_FILE) to main.py
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${__NUM_GPU}" \
  ./main.py \
  --action training \
  --config_file "${__CONFIG_FILE}" \
  "${@:2}" \
  2>&1 | tee "${__RUN_LOG_FILE}"
# ----- #
