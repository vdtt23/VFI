import numpy as np
import torch
from skimage.metrics import structural_similarity


@torch.no_grad()
def batch_psnr(pred, target, eps=1e-10):
    """
    pred/target: float tensors [B, 3, H, W], values in [0, 1].
    Returns mean PSNR in dB.
    """
    pred = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(eps))
    return psnr.mean().item()


@torch.no_grad()
def batch_ssim(pred, target):
    """
    SSIM is computed with scikit-image per image.
    This runs on CPU and is intended for evaluation, not training.
    """
    pred_np = pred.detach().float().clamp(0, 1).cpu().numpy()
    target_np = target.detach().float().clamp(0, 1).cpu().numpy()

    values = []
    for p, t in zip(pred_np, target_np):
        p = np.transpose(p, (1, 2, 0))
        t = np.transpose(t, (1, 2, 0))
        values.append(
            structural_similarity(
                t, p, channel_axis=-1, data_range=1.0
            )
        )
    return float(np.mean(values))
