import functools
import time
from logging import fatal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing._private.utils import measure
from tqdm import tqdm

from models import utils as mutils
from physics.ct import *
from sampling import (
    NoneCorrector,
    NonePredictor,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)
from utils import (
    batchfy,
    clear,
    clear_color,
    fft2,
    fft2_m,
    ifft2,
    ifft2_m,
    show_samples,
    show_samples_gray,
)


class lambda_schedule:
    def __init__(self, total=2000):
        self.total = total

    def get_current_lambda(self, i):
        pass


class lambda_schedule_linear(lambda_schedule):
    def __init__(self, start_lamb=1.0, end_lamb=0.0):
        super().__init__()
        self.start_lamb = start_lamb
        self.end_lamb = end_lamb

    def get_current_lambda(self, i):
        return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
    def __init__(self, lamb=1.0):
        super().__init__()
        self.lamb = lamb

    def get_current_lambda(self, i):
        return self.lamb


def _Dz_nonperiodic(x):
    # x: shape (Z, C, H, W) or (Z, ...)
    y = torch.zeros_like(x)
    # forward difference for i=0..n-2
    y[:-1] = x[1:] - x[:-1]
    # last slice difference = 0 (or you can set x[-1] - x[-2] if prefer)
    y[-1].zero_()
    return y


def _DzT_nonperiodic(v):
    # v: same shape as _Dz output
    out = torch.zeros_like(v)
    # out[0] = -v[0]
    # out[1..n-2] = v[0..n-3] - v[1..n-2]
    # out[n-1] = v[n-2]
    out[:-1] += -v[:-1]  # out[0..n-2] += -v[0..n-2]
    out[1:] += v[:-1]  # out[1..n-1] += v[0..n-2]
    return out


def _Dz(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]
    return y - x


def _DzT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]

    tempt = -(y - x)
    difft = tempt[:-1]
    y[1:] = difft
    y[0] = x[-1] - x[0]

    return y


def _Dx(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    return y - x


def _DxT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    tempt = -(y - x)
    difft = tempt[:, :, :-1, :]
    y[:, :, 1:, :] = difft
    y[:, :, 0, :] = x[:, :, -1, :] - x[:, :, 0, :]
    return y


def _Dy(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    return y - x


def _DyT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    tempt = -(y - x)
    difft = tempt[:, :, :, :-1]
    y[:, :, :, 1:] = difft
    y[:, :, :, 0] = x[:, :, :, -1] - x[:, :, :, 0]
    return y


def get_pc_radon_ADMM_TV(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    radon=None,
    save_progress=False,
    save_root=None,
    final_consistency=False,
    img_cache=None,
    img_shape=None,
    lamb_1=5,
    rho=10,
):
    """Sparse application of measurement consistency"""
    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    if img_cache != None:
        img_shape[0] += 1
    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        if img_cache != None:
            x = torch.cat([img_cache, x], dim=0)
            idx = list(range(len(x), 0, -1))
            x = x[idx]

        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        if img_cache != None:
            x = x[idx]
            x = x[1:]
            del_z[-1] = 0
            udel_z[-1] = 0
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                x, x_mean = corrector_radon_update_fn(
                    model, data, x, t, measurement=measurement
                )
                if save_progress:
                    if (i % 50) == 0:
                        print(f"iter: {i}/{sde.N}")
                        plt.imsave(
                            save_root / "recon" / f"progress{i}.png",
                            clear(x_mean[0:1]),
                            cmap="gray",
                        )
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(
                    x, x_mean, measurement, lamb=1.0, norm_const=norm_const
                )

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_ADMM_TV_vol(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    radon=None,
    save_progress=False,
    save_root=None,
    final_consistency=False,
    img_shape=None,
    lamb_1=5,
    rho=10,
):
    """Sparse application of measurement consistency"""
    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean

        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 12)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_batch_sing, _ = corrector_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f"iter: {i}/{sde.N}")
                        plt.imsave(
                            save_root / "recon" / "progress" / f"progress{i}.png",
                            clear(x_mean[0:1]),
                            cmap="gray",
                        )
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_SR_ADMM_TV_vol(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    save_progress=False,
    save_root=None,
    final_consistency=False,
    img_shape=None,
    lamb_1=5,
    rho=10,
    factor=2,
    niter=1,
    n_inner=1,
):
    """Sparse application of measurement consistency"""
    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        # return radon.A(x)
        # return F.interpolate(
        #     x, scale_factor=factor, mode="nearest", align_corners=False
        # )
        indices = torch.arange(0, x.size(0), factor, device=x.device, dtype=torch.long)
        return x.index_select(0, indices)

    def _AT(y: torch.Tensor):
        # y: [Nsamp, C, H, W] or general [...], factor from outer scope
        # target_shape = (y.shape[0] * factor, *y.shape[1:])  # tuple of ints, safe
        # [Nsamp, C, H, W] → [1,C,Nsamp,H,W]
        _y = y.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        x_hr = F.interpolate(
            _y,
            size=(img_shape[0], _y.shape[-2], _y.shape[-1]),
            mode="trilinear",
            align_corners=True,
        )
        return x_hr.squeeze(0).permute(1, 0, 2, 3).contiguous()
        # target_shape = img_shape
        # x_hr = torch.zeros(target_shape, device=y.device, dtype=y.dtype)
        # indices = torch.arange(
        #     0, target_shape[0], factor, device=y.device, dtype=torch.long
        # )
        # x_hr.index_copy_(0, indices, y)
        # return x_hr

        # return x_hr

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        eps_k = 1e-10
        indices = torch.arange(0, x.size(0), factor, device=x.device, dtype=torch.long)
        Ax = _A(x)  # size == measurement.shape
        res = measurement - Ax
        # 更新仅针对被采样的 slices
        x_s = x.index_select(0, indices)
        x_s = x_s + lamb * (res / (norm_const.index_select(0, indices) + eps_k))
        x.index_copy_(0, indices, x_s)

        # x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def A_cg(x):
        # return _AT(_A(x)) + rho * _DzT(_Dz(x))
        return _AT(_A(x)) + rho * _DzT_nonperiodic(_Dz_nonperiodic(x))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20, n_inner=1):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            # b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            b_cg = ATy + rho * (_DzT_nonperiodic(del_z) - _DzT_nonperiodic(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=n_inner)

            # del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            del_z = shrink(_Dz_nonperiodic(x) + udel_z, lamb_1 / rho)
            # udel_z = _Dz(x) - del_z + udel_z
            udel_z = _Dz_nonperiodic(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn(niter, n_inner):
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=niter, n_inner=n_inner)
                return x, x_mean

        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn(niter, n_inner)

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 6)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_batch_sing, _ = corrector_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)
                if i == 48:
                    pass
                if save_progress:
                    if (i % 1) == 0:
                        print(f"iter: {i}/{sde.N}")
                        plt.imsave(
                            save_root / "recon" / "progress" / f"progress{i}.png",
                            clear(x_mean[20:21]),
                            cmap="gray",
                        )
                        plt.imsave(
                            save_root
                            / "recon"
                            / "progress"
                            / f"progress{i}_Adjacency.png",
                            clear(x_mean[21:22]),
                            cmap="gray",
                        )
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon, _AT


def get_pc_SR_ADMM_TV_MIND_vol(
    sde1,
    sde2,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    save_progress=False,
    save_root=None,
    final_consistency=False,
    img_shape=None,
    lamb_1=5,
    rho=10,
    factor=2,
    niter=1,
    n_inner=1,
    # Deprecated: mu用于跨模态L2耦合（不再使用）；MIND一致性请使用 mind_mu
    mu: float = 0.0,
    # axes of through-plane (downsample) for each modality in canonical (D,C,H,W)
    axis1: int = 0,  # T2W axial through-plane -> D
    axis2: int = 2,  # T1W sagittal through-plane -> H (after permuting x2 to canonical)
    # MIND consistency params
    mind_mu: float = 0.1,
    mind_lr: float = 1e-3,
    mind_steps: int = 1,
):
    """Sparse application of measurement consistency"""
    # Define predictor & corrector
    predictor_update_fn1 = functools.partial(
        shared_predictor_update_fn,
        sde=sde1,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn1 = functools.partial(
        shared_corrector_update_fn,
        sde=sde1,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )
    predictor_update_fn2 = functools.partial(
        shared_predictor_update_fn,
        sde=sde2,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn2 = functools.partial(
        shared_corrector_update_fn,
        sde=sde2,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    # TV auxiliaries for each modality
    del_z1 = torch.zeros(img_shape)
    udel_z1 = torch.zeros(img_shape)
    del_z2 = torch.zeros(img_shape)
    udel_z2 = torch.zeros(img_shape)
    eps_cg = 1e-10

    def _A_axis(x: torch.Tensor, axis: int):
        # Subsample along the given axis by 'factor'
        length = x.size(axis)
        indices = torch.arange(0, length, factor, device=x.device, dtype=torch.long)
        return torch.index_select(x, dim=axis, index=indices)

    def _AT_axis(y: torch.Tensor, axis: int):
        # Trilinear upsample along given axis back to img_shape
        # y, x are shaped (Z,C,H,W). We reshape to (1,C,D,H,W) for interpolate
        _y = y.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,C,D,H,W_sub]
        target_D, target_H, target_W = img_shape[0], img_shape[2], img_shape[3]
        if axis == 0:
            size = (target_D, _y.shape[-2], _y.shape[-1])
        elif axis == 2:
            size = (_y.shape[-3], target_H, _y.shape[-1])
        elif axis == 3:
            size = (_y.shape[-3], _y.shape[-2], target_W)
        else:
            raise ValueError(f"Unsupported axis {axis} for _AT_axis")
        x_hr = F.interpolate(_y, size=size, mode="trilinear", align_corners=True)
        return x_hr.squeeze(0).permute(1, 0, 2, 3).contiguous()

    # Modality-specific measurement operators (in canonical (D,C,H,W) space)
    def _A1(x):
        return _A_axis(x, axis1)

    def _AT1(y):
        return _AT_axis(y, axis1)

    def _A2(x):
        return _A_axis(x, axis2)

    def _AT2(y):
        return _AT_axis(y, axis2)

    # Backward-compat convenience for callers expecting a single _AT
    def _AT(y: torch.Tensor):
        # Default to modality-1 axis
        return _AT1(y)

    # (removed legacy kaczmarz; final consistency uses axis-specific version)

    def _D_axis_nonperiodic(x: torch.Tensor, axis: int):
        y = torch.zeros_like(x)
        sl_cur = [slice(None)] * x.dim()
        sl_nxt = [slice(None)] * x.dim()
        sl_cur[axis] = slice(0, x.size(axis) - 1)
        sl_nxt[axis] = slice(1, None)
        y[tuple(sl_cur)] = x[tuple(sl_nxt)] - x[tuple(sl_cur)]
        # last slice along axis is zero
        last_narrow = y.narrow(axis, x.size(axis) - 1, 1)
        last_narrow.zero_()
        return y

    def _DT_axis_nonperiodic(v: torch.Tensor, axis: int):
        out = torch.zeros_like(v)
        sl_0n = [slice(None)] * v.dim()
        sl_1n = [slice(None)] * v.dim()
        sl_0n[axis] = slice(0, v.size(axis) - 1)
        sl_1n[axis] = slice(1, None)
        out[tuple(sl_0n)] += -v[tuple(sl_0n)]
        out[tuple(sl_1n)] += v[tuple(sl_0n)]
        return out

    def A_cg1(x):
        # (A^T A + rho D^T D) x for modality 1
        return _AT1(_A1(x)) + rho * _DT_axis_nonperiodic(
            _D_axis_nonperiodic(x, axis1), axis1
        )

    def A_cg2(x):
        # (A^T A + rho D^T D) x for modality 2
        return _AT2(_A2(x)) + rho * _DT_axis_nonperiodic(
            _D_axis_nonperiodic(x, axis2), axis2
        )

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps_cg:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine_mod1(
        x, ATy, warp_from_other=None, x_other=None, niter=20, n_inner=1
    ):
        nonlocal del_z1, udel_z1
        if del_z1.device != x.device:
            del_z1 = del_z1.to(x.device)
            udel_z1 = udel_z1.to(x.device)
        for _ in range(niter):
            b_cg = ATy + rho * (
                _DT_axis_nonperiodic(del_z1, axis1)
                - _DT_axis_nonperiodic(udel_z1, axis1)
            )
            x = CG(A_cg1, b_cg, x, n_inner=n_inner)

            del_z1 = shrink(_D_axis_nonperiodic(x, axis1) + udel_z1, lamb_1 / rho)
            udel_z1 = _D_axis_nonperiodic(x, axis1) - del_z1 + udel_z1
        x_mean = x
        return x, x_mean

    def CS_routine_mod2(
        x, ATy, warp_from_other=None, x_other=None, niter=20, n_inner=1
    ):
        nonlocal del_z2, udel_z2
        if del_z2.device != x.device:
            del_z2 = del_z2.to(x.device)
            udel_z2 = udel_z2.to(x.device)
        for _ in range(niter):
            b_cg = ATy + rho * (
                _DT_axis_nonperiodic(del_z2, axis2)
                - _DT_axis_nonperiodic(udel_z2, axis2)
            )
            x = CG(A_cg2, b_cg, x, n_inner=n_inner)

            del_z2 = shrink(_D_axis_nonperiodic(x, axis2) + udel_z2, lamb_1 / rho)
            udel_z2 = _D_axis_nonperiodic(x, axis2) - del_z2 + udel_z2
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    # --- MIND descriptor helpers ---
    def _gaussian1d(kernel_size: int, sigma: float, device, dtype):
        coords = (
            torch.arange(kernel_size, device=device, dtype=dtype)
            - (kernel_size - 1) / 2.0
        )
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        return g

    def _gaussian_blur_3d(x: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5):
        # x: (D,C,H,W) -> use depthwise conv3d with N=1
        x_in = x.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,D,H,W]
        C = x_in.shape[1]
        g1d = _gaussian1d(kernel_size, sigma, x.device, x.dtype)
        kZ = g1d.view(1, 1, kernel_size, 1, 1).repeat(C, 1, 1, 1, 1)
        kY = g1d.view(1, 1, 1, kernel_size, 1).repeat(C, 1, 1, 1, 1)
        kX = g1d.view(1, 1, 1, 1, kernel_size).repeat(C, 1, 1, 1, 1)
        pad = (kernel_size // 2, kernel_size // 2)
        x_sm = F.conv3d(x_in, kZ, padding=(pad[0], 0, 0), groups=C)
        x_sm = F.conv3d(x_sm, kY, padding=(0, pad[0], 0), groups=C)
        x_sm = F.conv3d(x_sm, kX, padding=(0, 0, pad[0]), groups=C)
        return x_sm.squeeze(0).permute(1, 0, 2, 3).contiguous()  # (D,C,H,W)

    def _shift3d(x: torch.Tensor, dz: int, dy: int, dx: int):
        # x: (D,C,H,W) shift with replicate padding
        D, C, H, W = x.shape
        pad_z = (max(0, dz), max(0, -dz))
        pad_y = (max(0, dy), max(0, -dy))
        pad_x = (max(0, dx), max(0, -dx))
        x_in = x.permute(1, 0, 2, 3).unsqueeze(0)  # [1,C,D,H,W]
        x_pad = F.pad(
            x_in,
            (pad_x[0], pad_x[1], pad_y[0], pad_y[1], pad_z[0], pad_z[1]),
            mode="replicate",
        )
        z0 = pad_z[0] + (-dz if dz < 0 else 0)
        y0 = pad_y[0] + (-dy if dy < 0 else 0)
        x0 = pad_x[0] + (-dx if dx < 0 else 0)
        x_shift = x_pad[:, :, z0 : z0 + D, y0 : y0 + H, x0 : x0 + W]
        return x_shift.squeeze(0).permute(1, 0, 2, 3).contiguous()

    def _mind3d(x: torch.Tensor, sigma: float = 1.0):
        # x: (D,C,H,W)
        eps_m = 1e-8
        xg = _gaussian_blur_3d(x, sigma=sigma, kernel_size=5)
        offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        dists = []
        for dz, dy, dx in offsets:
            xs = _shift3d(xg, dz, dy, dx)
            d = (xg - xs) ** 2  # (D,C,H,W)
            dists.append(d)
        Dst = torch.stack(dists, dim=0)  # [6,D,C,H,W]
        V = Dst.mean(dim=0, keepdim=True)  # [1,D,C,H,W]
        m = Dst.min(dim=0, keepdim=True).values
        M = torch.exp(-(Dst - m) / (V + eps_m))  # [6,D,C,H,W]
        return M

    def get_ADMM_TV_fn(niter, n_inner):
        def ADMM_TV_fn(x1, x2, y1, y2):
            with torch.no_grad():
                # Mod1: x1,y1 already in canonical (D,C,H,W) with through-plane on D
                ATy1 = _AT1(y1)  # y1 shape: (D//f,C,H,W)
                # Mod2: permute x2,y2 to canonical (D,C,H,W) with through-plane on H
                y2p = y2.permute(3, 1, 0, 2)  # (D//f,C,H,W)
                ATy2 = _AT2(y2p)
                x2p = x2.permute(3, 1, 0, 2)
                x1, x_mean1 = CS_routine_mod1(
                    x1,
                    ATy1,
                    niter=niter,
                    n_inner=n_inner,
                )
                x2, x_mean2 = CS_routine_mod2(
                    x2p,
                    ATy2,
                    niter=niter,
                    n_inner=n_inner,
                )
                # After TV subproblem, add a separate MIND consistency sub-step
                if mind_mu > 0 and mind_steps > 0:
                    for _ in range(mind_steps):
                        with torch.enable_grad():
                            x1_req = x1.detach().requires_grad_(True)
                            x2_req = x2.detach().requires_grad_(
                                True
                            )  # x2 is canonical here
                            M1 = _mind3d(x1_req, sigma=1.0)
                            M2 = _mind3d(x2_req, sigma=1.0)
                            loss_mind = mind_mu * F.mse_loss(M1, M2)
                            g1, g2 = torch.autograd.grad(
                                loss_mind, [x1_req, x2_req], retain_graph=False
                            )
                            x1 = (x1_req - mind_lr * g1).detach()
                            x2 = (x2_req - mind_lr * g2).detach()
                # Permute x2 back to original layout (H,C,W,D)
                x2 = x2.permute(2, 1, 3, 0)
                x_mean2 = x2
                return x1, x2, x_mean1, x_mean2

        return ADMM_TV_fn

    predictor_denoise_update_fn1 = get_update_fn(predictor_update_fn1)
    corrector_denoise_update_fn1 = get_update_fn(corrector_update_fn1)
    predictor_denoise_update_fn2 = get_update_fn(predictor_update_fn2)
    corrector_denoise_update_fn2 = get_update_fn(corrector_update_fn2)
    # mc_update_fn will be constructed after x1/x2 are available (needs warp grids)

    def pc_radon(model, data, y1=None, y2=None):
        with torch.no_grad():
            x1 = sde1.prior_sampling(data.shape).to(data.device)
            x2 = sde2.prior_sampling(data.shape).to(data.device)
            ones1 = torch.ones_like(x1).to(data.device)
            ones2 = torch.ones_like(x2).to(data.device)
            norm_const1 = _AT1(_A1(ones1))
            ones2p = ones2.permute(3, 1, 0, 2)
            norm_const2p = _AT2(_A2(ones2p))

            mc_update_fn = get_ADMM_TV_fn(niter, n_inner)
            timesteps = torch.linspace(sde1.T, eps, sde1.N)
            for i in tqdm(range(sde1.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch1 = batchfy(x1, 6)
                x_batch2 = batchfy(x2, 6)
                # 2. Run PC step for each batch
                x_agg1 = list()
                x_agg2 = list()

                for idx, x_batch_sing in enumerate(x_batch1):
                    x_batch_sing, _ = predictor_denoise_update_fn1(
                        model, data, x_batch_sing, t
                    )
                    x_batch_sing, _ = corrector_denoise_update_fn1(
                        model, data, x_batch_sing, t
                    )
                    x_agg1.append(x_batch_sing)
                for idx, x_batch_sing in enumerate(x_batch2):
                    x_batch_sing, _ = predictor_denoise_update_fn2(
                        model, data, x_batch_sing, t
                    )
                    x_batch_sing, _ = corrector_denoise_update_fn2(
                        model, data, x_batch_sing, t
                    )
                    x_agg2.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x1 = torch.cat(x_agg1, dim=0)
                x2 = torch.cat(x_agg2, dim=0)

                # 4. Run ADMM TV
                x1, x2, x_mean1, x_mean2 = mc_update_fn(x1, x2, y1, y2)
                # x, x_mean = mc_update_fn(x, measurement=measurement,mesurement_orth=mesurement_orth)
                if i == 48:
                    pass
                if save_progress:
                    if (i % 1) == 0:
                        print(f"iter: {i}/{sde1.N}")
                        # Save a couple of slices from each modality for quick QA
                        try:
                            plt.imsave(
                                save_root
                                / "recon"
                                / "progress"
                                / f"m1_progress{i}.png",
                                clear(
                                    x_mean1[
                                        x_mean1.shape[0] // 2 : x_mean1.shape[0] // 2
                                        + 1
                                    ]
                                ),
                                cmap="gray",
                            )
                            plt.imsave(
                                save_root
                                / "recon"
                                / "progress"
                                / f"m2_progress{i}.png",
                                clear(
                                    x_mean2[
                                        x_mean2.shape[0] // 2 : x_mean2.shape[0] // 2
                                        + 1
                                    ]
                                ),
                                cmap="gray",
                            )
                        except Exception:
                            pass
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                # axis-specific Kaczmarz with proper A/AT
                def kaczmarz_axis(x, measurement, lamb, norm_const, A_fn, AT_fn):
                    x = x + lamb * AT_fn(measurement - A_fn(x)) / (norm_const + 1e-10)
                    return x, x

                x1, x_mean1 = kaczmarz_axis(
                    x1, y1, lamb=1.0, norm_const=norm_const1, A_fn=_A1, AT_fn=_AT1
                )
                y2p = y2.permute(3, 1, 0, 2)
                x2p = x2.permute(3, 1, 0, 2)
                x2p, x_mean2p = kaczmarz_axis(
                    x2p, y2p, lamb=1.0, norm_const=norm_const2p, A_fn=_A2, AT_fn=_AT2
                )
                x2 = x2p.permute(2, 1, 3, 0)
                x_mean2 = x2

            # Return both reconstructed volumes (denoised mean by default)
            out1 = x_mean1 if denoise else x1
            out2 = x_mean2 if denoise else x2
            return inverse_scaler(out1), inverse_scaler(out2)

    return pc_radon, _AT


def get_pc_radon_ADMM_TV_all_vol(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    radon=None,
    save_progress=False,
    save_root=None,
    final_consistency=False,
    img_shape=None,
    lamb_1=5,
    rho=10,
):
    """Sparse application of measurement consistency"""
    # Define predictor & corrector
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (
                (_DxT(del_x) - _DxT(udel_x))
                + (_DyT(del_y) - _DyT(udel_y))
                + (_DzT(del_z) - _DzT(udel_z))
            )
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_x = shrink(_Dx(x) + udel_x, lamb_1 / rho)
            del_y = shrink(_Dy(x) + udel_y, lamb_1 / rho)
            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_x = _Dx(x) - del_x + udel_x
            udel_y = _Dy(x) - del_y + udel_y
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean

        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 12)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_batch_sing, _ = corrector_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f"iter: {i}/{sde.N}")
                        plt.imsave(
                            save_root / "recon" / "progress" / f"progress{i}.png",
                            clear(x_mean[0:1]),
                            cmap="gray",
                        )
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_ADMM_TV(
    eps=1e-5,
    radon=None,
    save_progress=False,
    save_root=None,
    img_shape=None,
    lamb_1=5,
    rho=10,
    outer_iter=30,
    inner_iter=20,
):

    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    def CG(A_fn, b_cg, x, n_inner=20):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=30):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
        for i in tqdm(range(niter)):
            b_cg = ATy + rho * (
                (_DxT(del_x) - _DxT(udel_x))
                + (_DyT(del_y) - _DyT(udel_y))
                + (_DzT(del_z) - _DzT(udel_z))
            )
            x = CG(A_cg, b_cg, x, n_inner=inner_iter)
            if save_progress:
                plt.imsave(
                    save_root / "recon" / "progress" / f"progress{i}.png",
                    clear(x[0:1]),
                    cmap="gray",
                )

            del_x = shrink(_Dx(x) + udel_x, lamb_1 / rho)
            del_y = shrink(_Dy(x) + udel_y, lamb_1 / rho)
            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_x = _Dx(x) - del_x + udel_x
            udel_y = _Dy(x) - del_y + udel_y
            udel_z = _Dz(x) - del_z + udel_z
        return x

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=outer_iter)
                return x, x_mean

        return ADMM_TV_fn

    mc_update_fn = get_ADMM_TV_fn()

    def ADMM_TV(data, measurement=None):
        with torch.no_grad():
            x = torch.zeros(data.shape).to(data.device)
            x = mc_update_fn(x, measurement=measurement)
            return x

    return ADMM_TV


def get_ADMM_TV_isotropic(
    eps=1e-5,
    radon=None,
    save_progress=False,
    save_root=None,
    img_shape=None,
    lamb_1=5,
    rho=10,
    outer_iter=30,
    inner_iter=20,
):
    """
    (get_ADMM_TV): implements anisotropic TV-ADMM
    In contrast, this function implements isotropic TV, which regularizes with |TV|_{1,2}
    """
    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def A_cg(x):
        return _AT(_A(x)) + rho * (_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x)))

    def CG(A_fn, b_cg, x, n_inner=20):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=30):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
        for i in tqdm(range(niter)):
            b_cg = ATy + rho * (
                (_DxT(del_x) - _DxT(udel_x))
                + (_DyT(del_y) - _DyT(udel_y))
                + (_DzT(del_z) - _DzT(udel_z))
            )
            x = CG(A_cg, b_cg, x, n_inner=inner_iter)
            if save_progress:
                plt.imsave(
                    save_root / "recon" / "progress" / f"progress{i}.png",
                    clear(x[0:1]),
                    cmap="gray",
                )

            # Each of shape [448, 1, 256, 256]
            _Dxx = _Dx(x)
            _Dyx = _Dy(x)
            _Dzx = _Dz(x)
            # shape [448, 3, 256, 256]. dim=1 gradient dimension
            _Dxa = torch.cat((_Dxx, _Dyx, _Dzx), dim=1)
            udel_a = torch.cat((udel_x, udel_y, udel_z), dim=1)

            # prox
            del_a = prox_l21(_Dxa + udel_a, lamb_1 / rho, dim=1)

            # split
            del_x, del_y, del_z = torch.split(del_a, 1, dim=1)

            # del_x = prox_l21(_Dxx + udel_x, lamb_1 / rho, -2)
            # del_y = prox_l21(_Dyx + udel_y, lamb_1 / rho, -1)
            # del_z = prox_l21(_Dzx + udel_z, lamb_1 / rho, 0)

            udel_x = _Dxx - del_x + udel_x
            udel_y = _Dyx - del_y + udel_y
            udel_z = _Dzx - del_z + udel_z
        return x

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x = CS_routine(x, ATy, niter=outer_iter)
                return x

        return ADMM_TV_fn

    mc_update_fn = get_ADMM_TV_fn()

    def ADMM_TV(data, measurement=None):
        with torch.no_grad():
            x = torch.zeros(data.shape).to(data.device)
            x = mc_update_fn(x, measurement=measurement)
            return x

    return ADMM_TV


def prox_l21(src, lamb, dim):
    """
    src.shape = [448(z), 1, 256(x), 256(y)]
    """
    weight_src = torch.linalg.norm(src, dim=dim, keepdim=True)
    weight_src_shrink = shrink(weight_src, lamb)

    weight = weight_src_shrink / weight_src
    return src * weight


def shrink(weight_src, lamb):
    return torch.sign(weight_src) * torch.max(
        torch.abs(weight_src) - lamb, torch.zeros_like(weight_src)
    )


def get_pc_radon_ADMM_TV_mri(
    sde,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    mask=None,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-5,
    save_progress=False,
    save_root=None,
    img_shape=None,
    lamb_1=5,
    rho=10,
):
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    del_z = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return fft2(x) * mask

    def _AT(kspace):
        return torch.real(ifft2(kspace))

    def _Dz(x):  # Batch direction
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]
        return y - x

    def _DzT(x):  # Batch direction
        y = torch.zeros_like(x)
        y[:-1] = x[1:]
        y[-1] = x[0]

        tempt = -(y - x)
        difft = tempt[:-1]
        y[1:] = difft
        y[0] = x[-1] - x[0]

        return y

    def A_cg(x):
        return _AT(_A(x)) + rho * _DzT(_Dz(x))

    def shrink(src, lamb):
        return torch.sign(src) * torch.max(torch.abs(src) - lamb, torch.zeros_like(src))

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

            x += a * p
            r -= a * Ap

            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_z, udel_z
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            udel_z = del_z.to(x.device)
        for i in range(niter):
            b_cg = ATy + rho * (_DzT(del_z) - _DzT(udel_z))
            x = CG(A_cg, b_cg, x, n_inner=1)

            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho)
            udel_z = _Dz(x) - del_z + udel_z
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean

        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, 20)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_batch_sing, _ = corrector_denoise_update_fn(
                        model, data, x_batch_sing, t
                    )
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f"iter: {i}/{sde.N}")
                        plt.imsave(
                            save_root / "recon" / "progress" / f"progress{i}.png",
                            clear(x_mean[0:1]),
                            cmap="gray",
                        )

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon
