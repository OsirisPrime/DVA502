"""Image metrics. Same definitions as the research scripts, so numbers are comparable."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(-10 * torch.log10(((pred - gt) ** 2).mean()))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean SSIM (11x11 Gaussian window, sigma 1.5) of two (1, 3, H, W) images in [0, 1]. Differentiable."""
    g = torch.exp(-((torch.arange(11, device=x.device, dtype=x.dtype) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()

    def f(z):
        return F.conv2d(z, win, padding=5, groups=3)

    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()
