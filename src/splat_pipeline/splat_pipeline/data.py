"""Training views: camera matrices from the COLMAP model plus images loaded on demand or cached in RAM."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .colmap_io import Model

log = logging.getLogger(__name__)


@dataclass
class View:
    name: str
    path: Path
    K: torch.Tensor        # (3, 3) intrinsics at the training resolution
    viewmat: torch.Tensor  # (4, 4) world -> camera
    width: int
    height: int


def build_views(model: Model, image_dir: Path, downscale: int) -> list[View]:
    views = []
    for im in model.images:
        cam = model.cameras[im.camera_id]
        if not cam.is_undistorted:
            raise ValueError(f"Camera model {cam.model} has lens distortion; rerun sfm with camera_model=OPENCV, "
                             "which undistorts the images, or PINHOLE")
        fx, fy, cx, cy = cam.intrinsics()
        W, H = cam.width // downscale, cam.height // downscale
        sx, sy = W / cam.width, H / cam.height
        K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=torch.float32)
        path = image_dir / im.name
        if not path.exists():
            raise FileNotFoundError(f"Image {path} listed in the COLMAP model is missing")
        views.append(View(im.name, path, K, torch.tensor(im.w2c(), dtype=torch.float32), W, H))
    return views


def split_views(views: list[View], test_every: int) -> tuple[list[View], list[View]]:
    """Every ``test_every``-th view (in name order) is held out; same split as the research scripts."""
    if test_every <= 0:
        return views, []
    return ([v for i, v in enumerate(views) if i % test_every != 0], views[::test_every])


def load_image(view: View) -> torch.Tensor:
    """(H, W, 3) uint8 RGB at the view's training resolution."""
    import cv2

    img = cv2.imread(str(view.path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not read {view.path}")
    if img.shape[1] != view.width or img.shape[0] != view.height:
        img = cv2.resize(img, (view.width, view.height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1]))


class ViewDataset(Dataset):
    """Images are decoded lazily (DataLoader workers) or kept in RAM after the first pass (``cache_in_ram``)."""

    def __init__(self, views: list[View], cache_in_ram: bool):
        self.views = views
        self.cache: list[torch.Tensor] | None = None
        if cache_in_ram:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(8) as ex:
                self.cache = list(ex.map(load_image, views))

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, i: int) -> dict:
        v = self.views[i]
        img = self.cache[i] if self.cache is not None else load_image(v)
        return {"index": i, "image": img, "K": v.K, "viewmat": v.viewmat}


def choose_cache(views: list[View], mode: str, limit_gb: float) -> bool:
    gb = sum(v.width * v.height * 3 for v in views) / 1e9
    in_ram = mode == "ram" or (mode == "auto" and gb <= limit_gb)
    log.info("%d images, %.1f GB decoded -> %s", len(views), gb, "cached in RAM" if in_ram else "read from disk")
    return in_ram


def infinite_loader(dataset: ViewDataset, num_samples: int, num_workers: int, seed: int):
    """Uniformly random views (with replacement) for ``num_samples`` steps."""
    from torch.utils.data import DataLoader, RandomSampler

    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=max(num_samples, 1), generator=g)
    workers = 0 if dataset.cache is not None else num_workers
    return DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
