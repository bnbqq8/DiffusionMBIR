import os

for seqs in ["T2", "T1"]:
    for orientation in ["SAG", "COR"]:
        os.system(
            f"TORCH_CUDA_ARCH_LIST=8.9 python main.py --config=configs/ve/IXI_256_ncsnpp_continuous.py --mode='train'  --workdir=workdir/IXI256 --config.data.orientation={orientation} --config.data.seq={seqs} --config.training.batch_size=64"
        )
# TORCH_CUDA_ARCH_LIST=8.9 python main.py --config=configs/ve/IXI_256_ncsnpp_continuous.py --mode='train'  --workdir=workdir/IXI256 --config.data.orientation=SAG --config.data.seq=T2
