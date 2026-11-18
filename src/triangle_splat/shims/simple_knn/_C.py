"""distCUDA2 without the CUDA build: mean squared distance to the 3 nearest neighbours of every point.

Triangle Splatting only uses it once, to size the initial triangles, so a CPU k-d tree is fast enough
(about a second for 100k points).
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    xyz = points.detach().cpu().numpy().astype(np.float64)
    dist, _ = cKDTree(xyz).query(xyz, k=4)          # column 0 is the point itself
    return torch.from_numpy((dist[:, 1:] ** 2).mean(1)).float().to(points.device)
