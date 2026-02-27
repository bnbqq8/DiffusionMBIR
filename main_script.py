import os

# ================= 配置区域 =================
PYTHON_EXE = "/home/jym/.conda/envs/dif-net/bin/python"

# 【关键调整】
# 刚才 32 (8x4) 导致 0号卡显存溢出 (OOM)。
# 因为 DataParallel 模式下 0号卡 负担极重。
# 我们降级到 16 (平均每张卡跑 4 张图)，确保 0号卡 能活下来。
BATCH_SIZE = 16

base_cmd = (
    "TORCH_CUDA_ARCH_LIST=8.6 "
    "CUDA_VISIBLE_DEVICES=0,1,2,3 "
    f"{PYTHON_EXE} main.py "
    "--config=configs/ve/CTSpine1K_256_ncsnpp_continuous.py "
    "--mode='train' "
    "--workdir=workdir/CTSpine1K_256_result_3090 "
    "--config.training.epochs=20 "
)
# ===========================================

print(f"\n>>> 正在启动保底策略 (Total Batch Size = {BATCH_SIZE}) ...")
print(">>> 显存策略: 这里的 16 是指 4张卡总共跑16张 (每张卡4张)。")

# 运行命令
exit_code = os.system(f"{base_cmd} --config.training.batch_size={BATCH_SIZE}")

if exit_code != 0:
    print(f"\n>>> ⚠️  任务失败。建议检查是否还有其他程序占用显存 (nvidia-smi)。")
else:
    print("\n>>> ✅ 训练成功完成！")