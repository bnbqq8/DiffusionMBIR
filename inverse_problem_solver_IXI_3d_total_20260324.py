import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import SaveImage
from torch._C import device
from tqdm import tqdm

import controllable_generation_TV
import datasets
from datasets_custom import get_IXI_sample
from losses import get_optimizer
from models import ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage

# for radon
from physics.zsr import ZAxisSuperResolutionConv
from sampling import LangevinCorrector, ReverseDiffusionPredictor
from sde_lib import VESDE
from utils import (
    batchfy,
    clear,
    img_wise_min_max,
    patient_wise_min_max,
    restore_checkpoint,
)

###############################################
# Configurations
###############################################

config_name = "AAPM_256_ncsnpp_continuous"
sde = "VESDE"
num_scales = 2000
contrast = "T2"
N = num_scales

# Parameters for the inverse problem
factor = 4
lamb = 0.04
rho = 10
freq = 1

# patient directory
patient_id = "IXI075-Guys-0754"
if "IXI" in patient_id:
    dir_name = f"../IXI_dataset/IXI_downsampledx{factor}_iacl_SyN/{patient_id}"
else:
    dir_name = f"../IXI_dataset/hcp_T1w_T2w/{patient_id}"

# correct ckpt path for hcp
if "IXI" not in patient_id:
    dataset_name = "hcp"
else:
    dataset_name = "IXI"
# correct downsampling directory for different contrast
pri_direction = "ax" if contrast == "T1" else "sag"

ckpt_filename = (
    f"../weights/{dataset_name}/{contrast}_{pri_direction}/checkpoints/latest.pth"
)


problem = "factor{factor}_ZSR"
###############################################
# Configurations
###############################################
if sde.lower() == "vesde":
    from configs.ve import BMR_ZSR_256 as configs

    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=config.model.num_scales,
    )
    sde.N = N
    sampling_eps = 1e-5
predictor = ReverseDiffusionPredictor
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
score_model = mutils.create_model(config)  ## model

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)

state = restore_checkpoint(
    ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True
)
ema.copy_to(score_model.parameters())

# Specify save directory for saving generated samples
save_root = Path(f"./results/{problem}_rho{rho}_lambda{lamb}_freq{freq}/{patient_id}")
save_root.mkdir(parents=True, exist_ok=True)

irl_types = ["input", "recon", "label", "BP", "sinogram"]
for t in irl_types:
    if t == "recon":
        save_root_f = save_root / t / "progress"
        save_root_f.mkdir(exist_ok=True, parents=True)
    else:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

# read all data
print("Loading all data")
forward_op = ZAxisSuperResolutionConv(
    factor=factor, dir_name=dir_name, pri_direction=pri_direction
)
mask, label1, label2, measure1, measure2, affine, meta = get_IXI_sample(
    # dir_name=f"/home/czfy/IXI_dataset/IXI_downsampledx{args.zsr_factor}_iacl/IXI002-Guys-0828"
    dir_name=dir_name,
    dual_field=True,
    factor=factor,
    mask=True,
)
if contrast == "T2":
    label, measure = label1, measure1
elif contrast == "T1":
    label, measure = label2, measure2
else:
    raise ValueError("Invalid contrast type. Should be either T1 or T2.")

print(f"GT shape : {label.shape}, measure shape: {measure.shape}")

predicted_sinogram = []
label_sinogram = []
img_cache = None

img = label.to(config.device)
pc_radon = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr=snr,
    n_steps=n_steps,
    probability_flow=probability_flow,
    continuous=config.training.continuous,
    denoise=True,
    radon=forward_op.to(config.device),
    save_progress=True,
    save_root=save_root,
    final_consistency=True,
    img_shape=img.shape,
    lamb_1=lamb,
    rho=rho,
)
# Sparse by masking
degradation = forward_op.A(img)

# A_dagger
bp = forward_op.AT(degradation, output_size=img.shape)

# Recon Image
x = pc_radon(score_model, scaler(img), measurement=degradation)
img_cahce = x[-1].unsqueeze(0)

count = 0
for i, recon_img in enumerate(x):
    plt.imsave(save_root / "BP" / f"{count}.png", clear(bp[i]), cmap="gray")
    plt.imsave(save_root / "label" / f"{count}.png", clear(img[i]), cmap="gray")
    plt.imsave(save_root / "recon" / f"{count}.png", clear(recon_img), cmap="gray")

    count += 1
# save the whole volume
saver = SaveImage()
# sag: [x,1,y,z] —> [1,x,y,z]
recon_nii = x.permute((1, 0, 2, 3)).cpu()
saver(
    MetaTensor(recon_nii, affine=affine, meta=meta),
    filename=save_root / f"{'_'.join(save_root.parts[-6:])}_{contrast}.nii.gz",
)

# evaluate result
# sag: [x,1,y,z] —> [1,x,y,z]
mask = mask.permute(1, 0, 2, 3).contiguous()
label1 = label.permute(1, 0, 2, 3).contiguous()
recon1 = x.permute(1, 0, 2, 3).contiguous()

tutils.print_and_save_eval_result_with_mask(
    recon1, label1, mask, save_root, file_path="result_mask.txt"
)

# # Recon and Save Sinogram
# label_sinogram.append(forward_op.A_all(img))
# predicted_sinogram.append(forward_op.A_all(x))

# original_sinogram = torch.cat(label_sinogram, dim=0).detach().cpu().numpy()
# recon_sinogram = torch.cat(predicted_sinogram, dim=0).detach().cpu().numpy()

# np.save(str(save_root / "sinogram" / f"original_{count}.npy"), original_sinogram)
# np.save(str(save_root / "sinogram" / f"recon_{count}.npy"), recon_sinogram)
