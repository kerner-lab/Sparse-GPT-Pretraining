```
# Copyright (c) 2025, Chenwei Cui, Kerner Lab
# SPDX-License-Identifier: MIT
```

## **System Requirements**
```
Tested on:
- Rocky Linux 8.10
- Python 3.12.9
- Nvidia Driver Version: 565.57.01 (Supports up to CUDA 12.7)
```

## **Step 1: Install Miniconda and Python 3.12.9**
- Install Miniconda (See [link](https://www.anaconda.com/docs/getting-started/miniconda/install))
- Run `conda create -n moe python=3.12.9`
- Run `conda activate moe`

## **Step 2: Install PyTorch 2.8.0 and Triton 3.4.0**
- We install the cu126 build
- Run the following
  ```
  pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
  ```
- Verify that `triton==3.4.0`

## **Step 3: Install CUDA toolkit 12.6**
- We match the PyTorch build (cu126)
- Run `conda install --override-channels --strict-channel-priority -c nvidia -c defaults cuda=12.6`
- Verify that `cuda.h` is in `$CONDA_PREFIX/targets/x86_64-linux/include`
- Run the following lines (also at runtime)
  ```
  export CUDA_HOME=$CONDA_PREFIX/targets/x86_64-linux
  export CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux
  export LD_LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
  ```

## **Step 4: Install CUDNN 9.12.0**
- `CUDNN 9.12.0` is compatible with `CUDA 12.6`
- Run `conda install --override-channels --strict-channel-priority -c nvidia -c defaults cudnn=9.12.0`
- Verify that `cudnn.h` is in `$CONDA_PREFIX/include`
- Run the following lines (also at runtime)
  ```
  export CUDNN_PATH=$CONDA_PREFIX  
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  ```

## **Step 5: Install GCC and G++**
- This is because our server has an old GCC/G++ version
- Run `conda install -c conda-forge gcc_linux-64=11.2.0 gxx_linux-64=11.2.0`
- Verify that `x86_64-conda-linux-gnu-gcc` is in `$CONDA_PREFIX/bin`
- Verify that `x86_64-conda-linux-gnu-g++` is in `$CONDA_PREFIX/bin`
- Run the following lines (also at runtime)
  ```
  export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
  export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
  ```

## **Step 6: Compile Transformer Engine 2.5**
- Install `ninja` for faster compiling
- Run `pip install ninja==1.13.0`
- The installation requires `CUDA_PATH`, `CUDNN_PATH`, and `CXX` be set, which we did
- Run `pip install --no-build-isolation transformer_engine[pytorch]==2.5`
- Run the following lines (also at runtime)
  ```
  export NVTE_CUDA_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include
  ```
- Verify that the following test script works
  ```
  import torch
  import transformer_engine.pytorch as te
  x = torch.randn(288, 768, device="cuda", dtype=torch.float32)
  a = torch.randn(288, 4).softmax(dim=-1).topk(k=1, dim=-1)[1]
  a = torch.nn.functional.one_hot(a.view(-1), 4)
  print(te.moe_permute(x.cuda(), a.cuda(), 288))
  ```

## **Step 7: Install other dependencies**
- Run the following
  ```
  pip install \
  yq==3.4.3 \
  tqdm==4.67.1 \
  numpy==2.1.2 \
  wandb==0.21.0 \
  PyYAML==6.0.2 \
  pandas==2.3.1 \
  lm-eval==0.4.9 \
  tiktoken==0.9.0 \
  pydantic==2.11.7 \
  transformers==4.55.0 \
  bitsandbytes==0.47.0 \
  torchao==0.12.0 \
  torchtune==0.6.1
  ```
- Run `wandb login` and follow the prompt
- Run `check_bnb_install.py` ([Link](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.47.0/check_bnb_install.py)) 
- Note that `torchao` and `torchtune` are only for RoPE

## **Step 8: (Optional) Compile `tgale96/grouped_gemm`**
- Run `git clone --recursive https://github.com/tgale96/grouped_gemm.git`
- We use `--recursive` because `grouped_gemm` has `third_party/cutlass @ 8783c41`
- Run `cd grouped_gemm`
- Assuming H100 GPU (For A100, set `TORCH_CUDA_ARCH_LIST="8.0"` below)
- Run `export TORCH_CUDA_ARCH_LIST="9.0"`
- Run `export GROUPED_GEMM_CUTLASS=1`
- Run `pip install .`
- Verify that `python benchmark.py` runs properly

## **Step 9: (Optional) Compile `torch_scatter`**
- We compile `torch_scatter` because it was built against a newer GLIBC (≥ 2.32) than the server provides
- Verify that `echo $PATH` contains `$CONDA_PREFIX/bin`
- Run `export CPATH=$CONDA_PREFIX/include`
- Assuming H100 GPU (For A100, set `TORCH_CUDA_ARCH_LIST="8.0"` below)
- Run `export TORCH_CUDA_ARCH_LIST="9.0"`
- Run `pip install torch-scatter`
