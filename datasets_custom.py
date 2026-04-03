from pathlib import Path

import numpy as np
from matplotlib.pylab import f
from torch.utils.data import DataLoader, Dataset

from physics.zsr import ZAxisSuperResolutionConv


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def get_IXI_sample(
    dir_name="/root/aicp-data/IXI_downsampledx2_iacl/IXI002-Guys-0828",
    seq="T2",
    factor=4,
    dual_field=False,
    mask=False,
    crop=True,
):
    print("Loading data from:", dir_name)
    import json
    import os.path as osp

    from monai.data import CacheDataset, GridPatchDataset, PatchIter, ShuffleBuffer
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        Identity,
        LoadImage,
        ResizeWithPadOrCrop,
        ScaleIntensityRangePercentiles,
        ToTensor,
    )

    # TODO make sure the transforms is consistent with training procedure
    transforms = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            ScaleIntensityRangePercentiles(
                lower=0.05, upper=99.9, b_min=0.0, b_max=1.0, clip=False
            ),  # TODO check consistency with DPM training  √01/08/2026
            ResizeWithPadOrCrop(spatial_size=(256, 256, 256)) if crop else Identity(),
        ]
    )
    toTensor = ToTensor(track_meta=False)
    # degradation op
    forward_op_pri = ZAxisSuperResolutionConv(
        factor=factor, dir_name=dir_name, pri_direction="sag"
    )
    forward_op_aux = ZAxisSuperResolutionConv(
        factor=factor, dir_name=dir_name, pri_direction="ax"
    )

    # if not Path(dir_name).exists():
    label_path = Path(dir_name) / f"{seq}_gt.nii.gz"
    # measurement_path = Path(dir_name) / f"{seq}.nii.gz"
    # label, measurement = transforms(str(label_path)), transforms(str(measurement_path))
    label = transforms(str(label_path)).permute(
        (1, 0, 2, 3)
    )  # sagittal (c,x,y,z)→(x,c,y,z)
    measurement = forward_op_pri.A(label)
    affine, meta = label.affine, label.meta
    if mask:
        mask_path = label_path.parent / "mask_gt.nii.gz"
        mask_transform = Compose(
            [
                LoadImage(),
                EnsureChannelFirst(),
                (
                    ResizeWithPadOrCrop(spatial_size=(256, 256, 256))
                    if crop
                    else Identity()
                ),
            ]
        )
        _mask = mask_transform(str(mask_path)).permute((1, 0, 2, 3))
    if dual_field:
        aux_seq = "T2" if seq != "T2" else "T1"
        label2_path = Path(dir_name) / f"{aux_seq}_gt.nii.gz"
        label2 = transforms(str(label2_path))
        _label2 = label2.permute((3, 0, 1, 2))  # axial (c,x,y,z)→(z,c,x,y)
        measurement2 = forward_op_aux.A(_label2)
        # measurement2_path = Path(dir_name) / f"{aux_seq}.nii.gz"
        # measurement2 = transforms(str(measurement2_path))

        # label = label.permute((1, 0, 2, 3))
        # measurement = measurement.permute((1, 0, 2, 3))
        # measurement2 = measurement2.permute((1, 0, 2, 3))

        # if forward_op is not None:

        #     measurement = forward_op.A(label)
        #     measurement2 = forward_op.A2(label)
        print(
            f"label shape: {label.shape}, measurement1 shape: {measurement.shape}, measurement2 shape: {measurement2.shape}"
        )
        label2 = label2.permute(
            (1, 0, 2, 3)
        )  # make it sagittal (x,c,y,z), for evaluation
        if mask:
            return (
                toTensor(_mask).contiguous().to(bool).cuda(),
                toTensor(label).contiguous().cuda(),
                toTensor(label2).contiguous().cuda(),
                toTensor(measurement).contiguous().cuda(),
                toTensor(measurement2).contiguous().cuda(),
                affine,
                meta,
            )
        else:
            return (
                toTensor(label).contiguous().cuda(),
                toTensor(label2).contiguous().cuda(),
                toTensor(measurement).contiguous().cuda(),
                toTensor(measurement2).contiguous().cuda(),
                affine,
                meta,
            )
    else:
        # since primary model direction is sagittal, we need to permute the (C,X,Y,Z) data to (X,C,Y,Z)
        # label = label.permute((1, 0, 2, 3))

        print(f"label shape: {label.shape}, measurement shape: {measurement.shape}")
        return toTensor(label).cuda(), toTensor(measurement).cuda(), affine, meta
        # TODO add mask return


if __name__ == "__main__":
    # standardize HCP dataset
    dir = Path("/root/epfs/IXI_dataset/hcp_T1w_T2w")
    name_map = {
        "Head.nii.gz": "mask_gt.nii.gz",
        "T2w_acpc_dc_restore.nii.gz": "T2_gt.nii.gz",
        "T1w_acpc_dc_restore.nii.gz": "T1_gt.nii.gz",
    }
    for file_path in dir.glob("*/T1w/*.nii.gz"):
        original_name = file_path.name
        if original_name in name_map:
            # 计算目标路径：移动到 T1w 的上一级目录，并更名
            # file_path.parent 是 T1w 目录，file_path.parent.parent 是 958976 这一层
            target_dir = file_path.parent.parent
            new_file_path = target_dir / name_map[original_name]

            try:
                # 执行移动并重命名
                file_path.rename(new_file_path)
                print(f"成功: {original_name} -> {new_file_path.relative_to(dir)}")
            except Exception as e:
                print(f"跳过 {file_path}: {e}")
    # delete T1w directory if it's empty
    for t1w_dir in dir.glob("*/T1w"):
        try:
            t1w_dir.rmdir()
            print(f"删除空目录: {t1w_dir.relative_to(dir)}")
        except Exception as e:
            print(f"无法删除 {t1w_dir}: {e}")
