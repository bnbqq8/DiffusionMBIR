import os

# ================= 配置区域 =================
# 1. 替换为全新的 naf 环境 Python 解释器路径
PYTHON_EXE = "/root/miniconda3/envs/naf/bin/python"

# 【关键调整：单卡 4090 配置】
# 因为现在是单张 4090 (24G显存)，不再是 4 张卡分担。
# 之前每张卡跑 4 张图 (4x4=16)。现在我们先尝试 Batch Size = 4 (或者 8)，防止单卡 OOM。
BATCH_SIZE = 4

# 2. 核心修改：架构代码改为 8.9，并设置仅使用 0 号卡
base_cmd = (
    "TORCH_CUDA_ARCH_LIST=8.9 "
    "CUDA_VISIBLE_DEVICES=0 "
    f"{PYTHON_EXE} main.py "
    "--config=configs/ve/CTSpine1K_256_ncsnpp_continuous.py "
    "--mode='train' "
    "--workdir=workdir/CTSpine1K_256_result_4090 "
    "--config.training.epochs=20 "
)
# ===========================================

print(f"\n>>> 正在启动 4090 专属训练策略 (单卡 Batch Size = {BATCH_SIZE}) ...")
print(f">>> 显存策略: 这里的 {BATCH_SIZE} 是指单张 4090 独立运行的批量大小。")

# 运行命令
exit_code = os.system(f"{base_cmd} --config.training.batch_size={BATCH_SIZE}")

if exit_code != 0:
    print(f"\n>>> ⚠️  任务失败。建议检查是否爆显存 (OOM)，可用 watch -n 1 nvidia-smi 监控。如果爆显存，请把 BATCH_SIZE 进一步调小。")
else:
    print("\n>>> ✅ 训练成功完成！")