from pathlib import Path

import torch
import torch.nn as nn
from degrade.degrade import fwhm_needed, fwhm_units_to_voxel_space, select_kernel
from monai.data.utils import affine_to_spacing
from monai.transforms import LoadImage


class ZAxisSuperResolutionConv(nn.Module):
    def __init__(
        self,
        factor: int,
        dir_name: Path | str,
        pri_direction="sag",
    ):
        super().__init__()
        kernel_path = Path(dir_name) / f"T2_blur_kernel.pt"
        # check if kernel file exists, if not, create it
        if not kernel_path.exists():
            self.make_kernel(factor, dir_name)
        self.kernel = torch.load(kernel_path, weights_only=False).to(torch.float32)
        self.kernel = self.kernel.squeeze(0)  # shape (1,1,1,kernel_size)
        self.factor = factor
        if pri_direction == "sag":
            self.A_conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(1, self.kernel.shape[-1]),
                stride=(1, self.factor),
                padding=(0, self.kernel.shape[-1] // 2),
                padding_mode="zeros",
                bias=False,
            )
            self.A_T_conv = torch.nn.ConvTranspose2d(
                1,
                1,
                kernel_size=(1, self.kernel.shape[-1]),
                stride=(1, self.factor),
                padding=(0, self.kernel.shape[-1] // 2),
                output_padding=(0, 0),
                bias=False,
            )
        elif pri_direction == "ax":  # axial
            self.A_conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(self.kernel.shape[-1], 1),
                stride=(self.factor, 1),
                padding=(self.kernel.shape[-1] // 2, 0),
                padding_mode="zeros",
                bias=False,
            )
            self.A_T_conv = torch.nn.ConvTranspose2d(
                1,
                1,
                kernel_size=(self.kernel.shape[-1], 1),
                stride=(self.factor, 1),
                padding=(self.kernel.shape[-1] // 2, 0),
                output_padding=(0, 0),
                bias=False,
            )
        else:
            raise ValueError(
                "Invalid primary direction. Should be either 'sag' or 'ax'."
            )

        # initialize weight
        with torch.no_grad():
            self.A_conv.weight.data = self.kernel.clone()
            self.A_T_conv.weight.data = self.kernel.clone()
        self.A_conv.requires_grad_(False)
        self.A_T_conv.requires_grad_(False)

    def expand_1d_kernel_to_3d(self, kernel_1d, axis):
        """
        将一维卷积核扩展为三维卷积核，只在指定 axis 上卷积。
        kernel_1d: shape [window_size]
        axis: 1, 2, or 3 (对应conv3d的depth, height, width)
        返回: shape [1, 1, D, H, W] 的3D kernel
        """
        assert kernel_1d.ndim == 1
        window_size = kernel_1d.shape[0]
        # conv3d kernel shape: [out_channels, in_channels, D, H, W]
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = window_size  # axis=1->D, 2->H, 3->W
        kernel_3d = torch.zeros(shape, dtype=kernel_1d.dtype, device=kernel_1d.device)
        # 填充到对应维度
        idx = [0, 0, 0, 0, 0]
        for i in range(window_size):
            idx[axis + 2] = i
            kernel_3d[tuple(idx)] = kernel_1d[i]
        return kernel_3d

    def make_kernel(self, factor, dir_name):
        _loader = LoadImage(ensure_channel_first=True)
        x = _loader(str(Path(dir_name) / "T2_gt.nii.gz"))
        axis = 2  # T2 axial
        spacing_from_affine = affine_to_spacing(x.affine)
        hr_res = spacing_from_affine[axis]
        lr_res = hr_res * factor
        fwhm = fwhm_units_to_voxel_space(fwhm_needed(hr_res, lr_res), hr_res)
        window_size = int(2 * fwhm.round() + 1)
        blur_kernel = select_kernel(window_size, "rf-pulse-slr", float(fwhm))
        blur_kernel /= blur_kernel.sum()
        blur_kernel = torch.from_numpy(blur_kernel).float()  # [window_size]
        blur_kernel = self.expand_1d_kernel_to_3d(blur_kernel, axis)
        torch.save(blur_kernel, Path(dir_name) / f"T2_blur_kernel.pt")

    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.A_conv(x)

    def AT(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        # Use PyTorch's native output_size parameter to elegantly resolve stride ambiguity
        return self.A_T_conv(x, output_size=output_size)
