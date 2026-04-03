#!/bin/bash
# 1. 设置代理（利用你配置好的环境变量）
source ~/.bashrc
cd /root/epfs/DiffusionMBIR

# 1. 自动检测 GPU 型号
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "检测到 GPU: $GPU_NAME"

# 2. 根据型号设置编译架构
if [[ $GPU_NAME == *"A100"* ]]; then
    echo "设置 TORCH_CUDA_ARCH_LIST 为 8.0 (Ampere)"
    export TORCH_CUDA_ARCH_LIST="8.0"
elif [[ $GPU_NAME == *"4090"* ]]; then
    echo "设置 TORCH_CUDA_ARCH_LIST 为 8.9 (Ada Lovelace)"
    export TORCH_CUDA_ARCH_LIST="8.9"
elif [[ $GPU_NAME == *"3090"* ]]; then
    echo "设置 TORCH_CUDA_ARCH_LIST 为 8.6 (Ampere)"
    export TORCH_CUDA_ARCH_LIST="8.6"
else
    # 如果是其他型号或无法识别，建议设置一个常用列表以增强兼容性
    echo "未匹配到特定优化型号，使用通用架构列表"
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9+PTX"
fi
# 4. 根据型号设置batch size
if [[ $GPU_NAME == *"A100"* ]]; then
    BATCH_SIZE=24
elif [[ $GPU_NAME == *"4090"* ]]; then
    BATCH_SIZE=12
elif [[ $GPU_NAME == *"3090"* ]]; then
    BATCH_SIZE=12
else
    BATCH_SIZE=8  # 默认较小的 batch size 以适应不同 GPU
fi

LOG_FILE="tmp/training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p tmp
# 5. run train (default N=2000, ccdf=0.5, dsg=False)
# --dps-weight 0.2 \
# 5.1 run_tpdm_mrzsr_IXI_conv_noDecouple.py T2(default N=2000, ccdf=0.5, dsg=False)
# python run_tpdm_mrzsr_IXI_conv_noDecouple.py \
# --problem-name 'SyN_noT1_hcp' \
# --patient-id '249947' \
# --zsr-factor 4 \
# --batch-size $BATCH_SIZE \
# --save-measurement \
# --contrast T2 \
# 2>&1 | tee -a "$LOG_FILE"

# commented in 2026年3月10日
# python run_tpdm_mrzsr_IXI_conv_noDecouple.py \
# --problem-name 'SyN_noT2_hcp_A100' \
# --patient-id '249947' \
# --zsr-factor 4 \
# --batch-size $BATCH_SIZE \
# --save-measurement \
# 2>&1 | tee -a "$LOG_FILE"

# commented in 2026年3月11日
# 6. run two-contrast with HCP dataset (DPS basline)
# python run_tpdm_mrzsr_IXI_conv_2contrast.py \
# --problem-name 'two_contrast_pc_ccdfHat' \
# --patient-id '249947' \
# 2>&1 | tee -a "$LOG_FILE"

# 7. run two-contrast with HCP dataset (DSG ablation studies)
# 8. run two-contrast with HCP dataset (DSG ablation studies with lamb2)
python inverse_problem_solver_IXI_3d_total_20260324.py \
2>&1 | tee -a "$LOG_FILE"
