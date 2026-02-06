#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6"
chmod +x main.py

CUDA_VISIBLE_DEVICES=1 /home/jym/python38 main.py \
  --config=configs/ve/CTSpine1K_256_ncsnpp_continuous.py \
  --mode='train' \
  --workdir=workdir/CTSpine1K_256_result
