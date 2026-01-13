import shutil
from pathlib import Path

import lmdb
import numpy as np
import torch
from monai.data import CacheDataset, Dataset, GridPatchDataset, PatchIter, ShuffleBuffer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ResizeWithPadOrCrop,
    ScaleIntensityRangePercentilesd,
    SqueezeDim,
)
from tqdm import tqdm

source_dir = Path("/root/aicp-data/IXI_downsampledx4_iacl")
target_dir = Path("/root/aicp-data/IXI_dataset_slices")
target_dir.mkdir(parents=True, exist_ok=True)
seqs = ["T1", "T2"]
orientations = ["AX", "SAG", "COR"]
# selected_dim = {"AX": (0, 1), "SAG": (1, 2), "COR": (0, 2)}
selected_dim = {"AX": 2, "SAG": 0, "COR": 1}
transforms = Compose(
    [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.05, upper=99.9, b_min=0.0, b_max=1.0, clip=False
        ),
    ]
)

# patch_ds = GridPatchDataset(
#     data=volume_ds,
#     patch_iter=patch_func,
#     transform=patch_transform,
#     with_coordinates=False,
# )

# for seq in seqs:
#     vol_path_list = list(source_dir.glob(f"*/{seq}_gt.nii.gz"))
#     mask_path_list = list(source_dir.glob(f"*/mask_gt.nii.gz"))
#     path_dict = [
#         {"image": str(vp), "mask": str(mp)}
#         for vp, mp in zip(vol_path_list, mask_path_list)
#     ]
#     vol_dataset = Dataset(data=path_dict, transform=transforms)
#     for orientation in orientations:
#         dim = selected_dim[orientation]
#         reduction_dim = tuple(d for d in (0,1,2) if d not in dim)
#         # lmdb
#         save_path = target_dir / f"{seq}_{orientation}.lmdb"
#         env = lmdb.open(save_path, map_size=1099511627776)
#         with env.begin(write=True) as txn:
#             for vol_data in tqdm(vol_dataset, desc=f"Processing {seq} {orientation}"):
#                 img = vol_data["image"].squeeze(dim=0)  # (x,y,z)
#                 mask = vol_data["mask"]
#                 filepath = Path(img.meta["filename_or_obj"])
#                 patient_id = filepath.parent.name
#                 mask = mask.sum(dim = reduction_dim).to(torch.bool)
#                 for slice_idx, foreground in enumerate(mask):
#                     if not foreground:
#                         continue
#                     slice_img = img.select(dim=dim, index=slice_idx)  # 2D slice
#                     byte_data = slice_img.numpy().tobytes()
#                     key = f"{patient_id}_{slice_idx:03d}".encode('ascii')
#                     txn.put(key, byte_data)
#         env.close()
# slice_filename = target_dir / orientation / seq / f"{patient_id}_{slice_idx:03d}.pt"
# slice_filename.parent.mkdir(parents=True, exist_ok=True)
# torch.save(slice_img, slice_filename)
# gemini version
for seq in seqs:
    # 1. 安全地匹配路径 (防止 zip 错位)
    vols = {p.parent.name: p for p in source_dir.glob(f"*/{seq}_gt.nii.gz")}
    masks = {p.parent.name: p for p in source_dir.glob("*/mask_gt.nii.gz")}
    common_ids = set(vols.keys()) & set(masks.keys())

    path_dict = [
        {"image": str(vols[pid]), "mask": str(masks[pid])} for pid in sorted(common_ids)
    ]
    vol_dataset = Dataset(data=path_dict, transform=transforms)

    for orientation in orientations:
        save_path = target_dir / f"{seq}_{orientation}.lmdb"
        if save_path.exists():
            print(f"检测到旧数据库 {save_path}，正在删除...")
            shutil.rmtree(save_path)  # LMDB 通常是文件夹形式，包含 data.mdb 和 lock.mdb

        env = lmdb.open(str(save_path), map_size=1099511627776)

        # 将事务放入循环内或分批 commit 以防内存溢出
        with env.begin(write=True) as txn:
            for vol_data in tqdm(vol_dataset, desc=f"Processing {seq} {orientation}"):
                img = vol_data["image"].squeeze(dim=0)
                mask = vol_data["mask"].squeeze(dim=0)

                # 获取元数据
                filepath = Path(img.meta["filename_or_obj"])
                patient_id = filepath.parent.name

                slice_axis = selected_dim[orientation]  # 必须是 int, 如 0, 1, 2

                # 确定哪些层包含前景
                # 假设 mask 与 img 维度一致
                reduction_dims = tuple(d for d in range(3) if d != slice_axis)
                has_foreground = mask.sum(dim=reduction_dims) > 0

                for slice_idx, is_fg in enumerate(has_foreground):
                    if not is_fg:
                        continue

                    # 切片并确保内存连续
                    slice_img = img.select(dim=slice_axis, index=slice_idx)
                    slice_np = slice_img.numpy()

                    # 存储数据
                    key = f"{patient_id}_{slice_idx:03d}".encode("ascii")
                    txn.put(key, slice_np.tobytes())

                    # 【关键】存储 shape 信息，否则读取时无法 reshape
                    shape_key = f"{patient_id}_{slice_idx:03d}_shape".encode("ascii")
                    txn.put(
                        shape_key, np.array(slice_np.shape, dtype=np.int32).tobytes()
                    )

        env.close()
