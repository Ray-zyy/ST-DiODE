from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as cal_ssim


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def psnr(pred: np.ndarray, true: np.ndarray, data_range: float) -> float:
    mse_value = np.mean((pred - true) ** 2)
    if mse_value <= 1e-12:
        return float('inf')
    return float(20 * np.log10(data_range) - 10 * np.log10(mse_value))


def _denormalize(data: np.ndarray, mean, std) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return data * std + mean


def metric(
    pred: np.ndarray,
    true: np.ndarray,
    mean,
    std,
    return_ssim_psnr: bool = False,
    clip_range: tuple[float, float] | None = (0.0, 1.0),
):
    pred = _denormalize(pred, mean, std)
    true = _denormalize(true, mean, std)

    mse_value = mse(pred, true)
    mae_value = mae(pred, true)

    if not return_ssim_psnr:
        return mse_value, mae_value

    if clip_range is not None:
        low, high = clip_range
        pred = np.clip(pred, low, high)
        true = np.clip(true, low, high)
        data_range = float(high - low)
    else:
        data_range = float(max(pred.max(), true.max()) - min(pred.min(), true.min()))
        data_range = max(data_range, 1e-6)

    ssim_total = 0.0
    psnr_total = 0.0
    num_frames = pred.shape[0] * pred.shape[1]

    for b in range(pred.shape[0]):
        for t in range(pred.shape[1]):
            pred_frame = np.transpose(pred[b, t], (1, 2, 0))
            true_frame = np.transpose(true[b, t], (1, 2, 0))

            if pred_frame.shape[-1] == 1:
                pred_frame = pred_frame[..., 0]
                true_frame = true_frame[..., 0]
                ssim_value = cal_ssim(pred_frame, true_frame, data_range=data_range)
            else:
                ssim_value = cal_ssim(pred_frame, true_frame, channel_axis=-1, data_range=data_range)

            ssim_total += float(ssim_value)
            psnr_total += psnr(pred[b, t], true[b, t], data_range=data_range)

    return mse_value, mae_value, ssim_total / num_frames, psnr_total / num_frames
