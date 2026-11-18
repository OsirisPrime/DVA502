"""Typed run configuration.

Defaults live here; a YAML file (``--config``) overrides them, and command-line flags override the file.
The resolved configuration is written to every run folder (``config.yaml``) so a run can be repeated exactly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class FramesConfig:
    fps: float = 2.0                 # frames kept per second of video (sharpest frame in each window)
    max_side: int = 1920             # resize the longest side to at most this (0 = keep)
    jpeg_quality: int = 95


@dataclass
class SfmConfig:
    mapper: Literal["colmap", "glomap"] = "colmap"   # incremental (robust) or global (fast on many images)
    matcher: Literal["sequential", "exhaustive", "vocab_tree"] = "sequential"
    vocab_tree_path: str = ""        # required for matcher=vocab_tree (large unordered photo sets)
    camera_model: Literal["PINHOLE", "OPENCV"] = "PINHOLE"  # OPENCV estimates lens distortion, then undistorts
    single_camera: bool = True       # one camera (same intrinsics) for all images
    max_features: int = 8192
    colmap: str = "colmap"           # executable; "colmap" = $COLMAP_BIN, else the conda env's, else PATH.
    #                                  All steps use the same one (the database format is versioned)
    use_gpu: bool = False            # COLMAP SIFT on GPU (CUDA builds only)


@dataclass
class TrainConfig:
    steps: int = 30_000
    downscale: int = 1               # image downscale factor (1 = full resolution)
    test_every: int = 8              # every N-th image (sorted by name) is held out; 0 = train on all
    sh_degree: int = 3
    sh_degree_interval: int = 1000   # raise the SH degree by one every N steps
    ssim_lambda: float = 0.2
    # densification
    strategy: Literal["default", "mcmc"] = "default"  # default = clone/split/prune; mcmc = fixed budget
    cap_max: int = 3_000_000         # mcmc only: number of Gaussians
    densify_until: float = -1.0      # fraction of steps after which densification stops (-1 = 0.5 for default,
    #                                  25/30 for mcmc, as in gsplat)
    absgrad: bool = False            # default only: AbsGS gradient (finer detail, more Gaussians)
    # learning rates (means is scaled by the scene size and decays 100x over the run)
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    # memory / data
    packed: bool = False             # packed rasterization: less memory for very large scenes
    cache: Literal["auto", "ram", "disk"] = "auto"  # auto = RAM if the images fit in cache_limit_gb
    cache_limit_gb: float = 8.0
    num_workers: int = 4             # image decoding workers when reading from disk
    # bookkeeping
    eval_every: int = 0              # held-out PSNR during training every N steps (0 = only at the end)
    checkpoint_every: int = 5000     # 0 = only at the end
    keep_checkpoints: int = 2
    save_renders: bool = True
    seed: int = 0
    device: str = "cuda"


@dataclass
class Config:
    frames: FramesConfig = field(default_factory=FramesConfig)
    sfm: SfmConfig = field(default_factory=SfmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, object] | None = None) -> "Config":
        cfg = cls()
        if path:
            data = yaml.safe_load(Path(path).read_text()) or {}
            for section, values in data.items():
                for k, v in (values or {}).items():
                    cfg.set(f"{section}.{k}", v)
        for k, v in (overrides or {}).items():
            cfg.set(k, v)
        return cfg

    def set(self, dotted: str, value: object) -> None:
        section, _, key = dotted.partition(".")
        obj = getattr(self, section, None)
        if obj is None or not dataclasses.is_dataclass(obj) or key not in {f.name for f in dataclasses.fields(obj)}:
            raise KeyError(f"Unknown config key '{dotted}'")
        current = getattr(obj, key)
        if isinstance(value, str) and not isinstance(current, str):  # from the command line
            value = yaml.safe_load(value)
        if isinstance(current, bool) and not isinstance(value, bool):
            raise TypeError(f"{dotted} must be true/false, got {value!r}")
        if isinstance(current, float) and isinstance(value, int):
            value = float(value)
        if type(current) is not type(value) and not (isinstance(current, (int, float)) and
                                                     isinstance(value, (int, float)) and not isinstance(value, bool)):
            raise TypeError(f"{dotted} must be {type(current).__name__}, got {value!r}")
        setattr(obj, key, value)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
