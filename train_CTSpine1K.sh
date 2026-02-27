#!/bin/bash

# 设置 CUDA 环境 (保持您原来的设置)
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3090 显卡架构代码是 8.6
export TORCH_CUDA_ARCH_LIST="8.6"

# 给 main.py 执行权限
chmod +x main.py

# --- 核心运行命令 ---
# 注意：
# 1. 确保 /home/jym/python38/bin/python 是正确的解释器路径
# 2. 这里的 batch_size=32 是总数 (4张卡平分，每张卡8个)
# 3. 每一行末尾必须有 \ (最后一行除外)

CUDA_VISIBLE_DEVICES=0,1,2,3 /home/jym/python38 main.py \
  --config=configs/ve/CTSpine1K_256_ncsnpp_continuous.py \
  --mode='train' \
  --workdir=workdir/CTSpine1K_256_result \
  --config.training.batch_size=32 \
  --config.training.n_epochs=20