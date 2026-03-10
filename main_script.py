import os

# ================= 配置区域 =================
# 1. 替换为全新的 naf 环境 Python 解释器路径
PYTHON_EXE = "/root/miniconda3/envs/naf/bin/python"

# 【关键调整：4 卡 4090 配置】
# 该项目使用 DataParallel，batch_size 是“总 batch”。
# 尝试总 batch=16（4 卡下约每卡 4）。
BATCH_SIZE = 16

# 2. 核心修改：架构代码改为 8.9，并设置使用 4 张卡
base_cmd = (
    "TORCH_CUDA_ARCH_LIST=8.9 "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "CUDA_VISIBLE_DEVICES=0,1,2,3 "
    f"{PYTHON_EXE} main.py "
    "--config=configs/ve/CTSpine1K_256_ncsnpp_continuous.py "
    "--mode='train' "
    "--workdir=workdir/CTSpine1K_256_result_4090 "
    "--config.training.epochs=20 "
)
# ===========================================

print(f"\n>>> 正在启动 4x4090 训练策略 (总 Batch Size = {BATCH_SIZE}) ...")
print(">>> 显存策略: 此项目 batch_size 是总批量，会自动分摊到可见 GPU。")

# 运行命令
exit_code = os.system(f"{base_cmd} --config.training.batch_size={BATCH_SIZE}")

if exit_code != 0:
    print(f"\n>>> ⚠️  任务失败。建议检查是否爆显存 (OOM)，可用 watch -n 1 nvidia-smi 监控。如果爆显存，请把 BATCH_SIZE 进一步调小。")
else:
    print("\n>>> ✅ 训练成功完成！")