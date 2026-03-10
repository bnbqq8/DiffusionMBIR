#!/bin/bash

set -e

# 1. CUDA 环境
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 2. RTX 4090 架构代号
export TORCH_CUDA_ARCH_LIST="8.9"

# 3. 使用 naf 环境 Python
PYTHON_EXEC="/root/miniconda3/envs/naf/bin/python"

# 4. 4 卡训练（与你的 main_script.py 保持一致）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 5. 直接走 main_script.py，训练参数由该文件统一管理
$PYTHON_EXEC main_script.py