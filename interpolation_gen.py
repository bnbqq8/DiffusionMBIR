# generate interpolated results given the measurements
import os

import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from generative.networks.nets import DiffusionModelUNet
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Lambda,
    LoadImage,
    LoadImaged,
    SaveImage,
    ScaleIntensityRangePercentilesd,
)
from torch._C import device
from tqdm import tqdm

import controllable_generation_TV
import datasets
from guided_diffusion.unet import create_my_model
from losses import get_optimizer
from models import ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage

# for radon
from physics.ct import CT
from sampling import (
    AncestralSamplingPredictor,
    LangevinCorrector,
    ReverseDiffusionPredictor,
)
from sde_lib import DDPM, VESDE, VPSDE
from utils import (
    batchfy,
    clear,
    img_wise_min_max,
    patient_wise_min_max,
    restore_checkpoint,
)

parser = argparse.ArgumentParser(description="3D reconstruction")
parser.add_argument("--M_iter", type=int, default=1, help="outer iterations")
parser.add_argument("--K_iter", type=int, default=1, help="inner iterations")
parser.add_argument("--lamb", type=float, default=0.04, help="lambda")
parser.add_argument("--rho", type=float, default=10, help="rho")
args = parser.parse_args()

loader = LoadImage(ensure_channel_first=True)
###############################################
# Configurations
###############################################
factor = 2
problem = "MRI_through_plane_SR_ADMM_TV_total"
config_name = "IXI_256_ncsnpp_continuous"
sde = "vpsde"
num_scales = 50
ckpt_num = 185
N = num_scales
M_iter = args.M_iter
K_iter = args.K_iter
niter = M_iter
n_inner = K_iter
lamb = args.lamb
rho = args.rho

vol_name = "IXI002-Guys-0828"
seq = "T2"
root = Path(
    # f"/root/aicp-data/IXI_downsampledx{int(factor)}_iacl/{vol_name}/{seq}.nii.gz"
    f"../IXI_dataset/IXI_downsampledx{int(factor)}_iacl/{vol_name}/{seq}.nii.gz"
)
gpu = 0
# Device setting
device_str = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
print(f"Device set to {device_str}.")
device = torch.device(device_str)

# Parameters for the inverse problem
det_spacing = 1.0
size = 256
det_count = int((size * (2 * torch.ones(1)).sqrt()).ceil())

freq = 1

if sde.lower() == "vesde":
    from configs.ve import AAPM_256_ncsnpp_continuous as configs

    ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales,
    )
    sde.N = N
    sampling_eps = 1e-5
elif sde.lower() == "ddpm":
    config_name += "_ddpm"
    from configs.ddpm import IXI_256_ncsnpp_continuous as configs

    config = configs.get_config()
    config.model.num_scales = N
    sde = DDPM(N=num_scales, beta_start=1e-4, beta_end=2e-2)

elif sde.lower() == "vpsde":
    config_name += "_vpsde"
    from configs.ddpm import IXI_256_ncsnpp_continuous as configs

    config = configs.get_config()
    sde = VPSDE(beta_min=0.1, beta_max=20, N=1000)
    sde.N = N

# predictor = ReverseDiffusionPredictor
predictor = AncestralSamplingPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

batch_size = 12
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
# -------------score_model----------------
# score_model = mutils.create_model(config)
# ckpt_dir = "/home/czfy/diffusion-posterior-sampling/model64"
ckpt_dir = "/home/czfy/IXI_diffusion/model"
score_model = create_my_model(f"{ckpt_dir}/{seq}_latest.pth")
score_model = score_model.to(device)
score_model.eval()
# -------------score_model----------------

# -------------EMA----------------
# optimizer = get_optimizer(config, score_model.parameters())
# ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
# state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

# state = restore_checkpoint(
#     ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True
# )
# ema.copy_to(score_model.parameters())
# -------------EMA----------------
# Specify save directory for saving generated samples
config_name += "_gridSearch"
save_root = Path(
    f"./results/{config_name}/{problem}/M_iter{M_iter}/K_iter{K_iter}/rho{rho}/lambda{lamb}/AncestralSamplingPredictor/N{N}_noFinalConsistency"
)
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ["input", "recon", "label", "BP"]
for t in irl_types:
    if t == "recon":
        save_root_f = save_root / t / "progress"
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

# -------------read all data-------------
transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.1, upper=99.9, b_min=0.0, b_max=1.0, clip=False
        ),
        Lambda(func=lambda x: x["image"]),
    ]
)

all_img = transforms({"image": str(root).replace(".nii.gz", "_gt.nii.gz")})
degraded_img = transforms({"image": root})
if seq == "T2":
    all_img = all_img.permute(3, 0, 1, 2)
    degraded_img = degraded_img.permute(3, 0, 1, 2)
else:
    all_img = all_img.permute(1, 0, 2, 3)
    degraded_img = degraded_img.permute(1, 0, 2, 3)
# fname_list = os.listdir(root)
# fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
# print(fname_list)
# all_img = []

# print("Loading data")
# for fname in tqdm(fname_list):
#     just_name = fname.split(".")[0]
#     img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
#     h, w = img.shape
#     img = img.view(1, 1, h, w)
#     all_img.append(img)
#     plt.imsave(
#         os.path.join(save_root, "label", f"{just_name}.png"), clear(img), cmap="gray"
#     )
# all_img = torch.cat(all_img, dim=0)


print(f"Data loaded shape : {all_img.shape}")
img = all_img.to(config.device)
degraded_img = degraded_img.to(config.device)
# -------------read all data-------------
pc_radon, AT = controllable_generation_TV.get_pc_SR_ADMM_TV_vol(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr=snr,
    n_steps=n_steps,
    probability_flow=probability_flow,
    continuous=False,
    denoise=True,
    save_progress=True,
    save_root=save_root,
    final_consistency=False,
    img_shape=img.shape,
    lamb_1=lamb,
    rho=rho,
    factor=factor,
    niter=niter,
    n_inner=n_inner,
)
# Sparse by masking
# sinogram = radon.A(img)
sinogram = degraded_img
# A_dagger
save_root = Path("results")
bp = AT(sinogram)
# saver = SaveImage()
# saver(
#     MetaTensor(bp.permute(1, 2, 3, 0), affine=all_img.affine, meta=all_img.meta),
#     filename=save_root / f"{seq}_interpolated.nii.gz",
# )


def AT_nearest(y: torch.Tensor, img_shape):
    # y: [Nsamp, C, H, W] or general [...], factor from outer scope
    # target_shape = (y.shape[0] * factor, *y.shape[1:])  # tuple of ints, safe
    # [Nsamp, C, H, W] → [1,C,Nsamp,H,W]
    _y = y.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    x_hr = F.interpolate(
        _y,
        size=(img_shape[0], _y.shape[-2], _y.shape[-1]),
        mode="nearest",
    )
    return x_hr.squeeze(0).permute(1, 0, 2, 3).contiguous()


bp = AT_nearest(sinogram, img_shape=img.shape)
saver = SaveImage()
saver(
    MetaTensor(bp.permute(1, 2, 3, 0), affine=all_img.affine, meta=all_img.meta),
    filename=save_root / f"{seq}_interpolated_nearest.nii.gz",
)
