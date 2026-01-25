#!/bin/bash

TORCH_CUDA_ARCH_LIST="8.9" python main.py \
  --config=configs/ve/IXI_256_ncsnpp_continuous.py \
  --mode='train' \
  --workdir=workdir/HCP256