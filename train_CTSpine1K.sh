#!/bin/bash

# 1. 设置最新的 CUDA 12.4 环境 (适配你刚换的机器)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 2. RTX 4090 的显卡架构代号是 8.9 (极其重要，8.6 是 3090 的，写错会报错)
export TORCH_CUDA_ARCH_LIST="8.9"

# 给 main.py 执行权限
chmod +x main.py

# 3. 指定你刚刚创建好的 Conda naf 环境的 Python 绝对路径
PYTHON_EXEC="/root/miniconda3/envs/naf/bin/python"

# --- 核心运行命令 ---
# 注意：
# 如果你现在这台 4090 机器是单卡，请保持 CUDA_VISIBLE_DEVICES=0
# 如果是多卡（比如4张4090），请改回 0,1,2,3
# 单卡情况下，batch_size=32 绝对会爆显存(OOM)，建议先从 8 开始试！

CUDA_VISIBLE_DEVICES=0 $PYTHON_EXEC main.py \
  --config=configs/ve/CTSpine1K_256_ncsnpp_continuous.py \
  --mode='train' \
  --workdir=workdir/CTSpine1K_256_result \
  --config.training.batch_size=8 \
  --config.training.n_epochs=20