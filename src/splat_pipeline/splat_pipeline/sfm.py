"""Stage 2: structure-from-motion with COLMAP, using COLMAP's incremental mapper or GLOMAP's global mapper.

Writes into the scene folder (see the README for the full disk contract):

    database.db                 features + matches (recreated on every run)
    sparse/<k>/ | sparse_glomap/<k>/   models from the COLMAP | GLOMAP mapper (folder cleared on every run)
    undistorted_<mapper>/       only for camera_model=OPENCV: undistorted images + PINHOLE model
    scene.json                  which model + image folder training should use, plus statistics
    sparse_model.txt            same model path, for the research scripts (PINHOLE only)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .colmap_io import read_model
from .config import SfmConfig

log = logging.getLogger(__name__)


def _run(cmd: list[str], log_file: Path) -> None:
    log.info(">> %s", " ".join(cmd))
    with open(log_file, "a") as fh:
        fh.write(f"\n>> {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        tail = log_file.read_text().splitlines()[-15:]
        raise RuntimeError(f"{cmd[0]} {cmd[1]} failed (exit {r.returncode}). Last lines of {log_file}:\n"
                           + "\n".join(tail))


def _help(exe: str, sub: str) -> str:
    try:
        r = subprocess.run([exe, sub, "-h"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout + r.stderr if r.returncode == 0 else ""


def resolve_colmap(name: str) -> str:
    """'colmap' (the default) means: $COLMAP_BIN, else the COLMAP of the conda env this Python belongs to (also
    from a venv built on it), else colmap on PATH. Any other value is used as given (~ is expanded)."""
    if name != "colmap":
        return os.path.expanduser(name)
    if os.environ.get("COLMAP_BIN"):
        return os.environ["COLMAP_BIN"]
    env_colmap = Path(sys.base_prefix) / "bin" / "colmap"
    return str(env_colmap) if env_colmap.exists() else "colmap"


def colmap_version(exe: str) -> str:
    out = _help(exe, "help")
    return out.splitlines()[0].strip() if out else "unknown"


def mapper_command(cfg: SfmConfig) -> list[str]:
    if shutil.which(cfg.colmap) is None and not Path(cfg.colmap).exists():
        raise FileNotFoundError(f"COLMAP executable not found: {cfg.colmap}")
    if cfg.mapper == "colmap":
        return [cfg.colmap, "mapper"]
    # GLOMAP is part of COLMAP >= 3.13 (`colmap global_mapper`); older setups ship a separate `glomap` binary
    if "--database_path" in _help(cfg.colmap, "global_mapper"):
        return [cfg.colmap, "global_mapper"]
    if shutil.which("glomap"):
        return ["glomap", "mapper"]
    raise FileNotFoundError(
        f"GLOMAP not available: '{cfg.colmap}' has no global_mapper (needs COLMAP >= 3.13) and no `glomap` on "
        "PATH. Install COLMAP 4 (see README) or set sfm.colmap to its path")


def _gpu_flags(exe: str, stage: str, use_gpu: bool) -> list[str]:
    """The GPU option was renamed in COLMAP 3.13 (SiftExtraction.use_gpu -> FeatureExtraction.use_gpu)."""
    sub = "feature_extractor" if stage == "extract" else "exhaustive_matcher"
    help_text = _help(exe, sub)
    for name in (("FeatureExtraction", "SiftExtraction") if stage == "extract" else
                 ("FeatureMatching", "SiftMatching")):
        if f"--{name}.use_gpu" in help_text:
            return [f"--{name}.use_gpu", "1" if use_gpu else "0"]
    return []


def run_sfm(scene: str | Path, cfg: SfmConfig) -> dict:
    """Run the SfM stage on ``<scene>/images`` and return the statistics written to ``scene.json``."""
    scene = Path(scene)
    cfg = dataclasses.replace(cfg, colmap=resolve_colmap(cfg.colmap),
                              vocab_tree_path=os.path.expanduser(cfg.vocab_tree_path))
    images = scene / "images"
    n_input = len([f for f in images.iterdir() if f.is_file()]) if images.is_dir() else 0
    if n_input < 3:
        raise FileNotFoundError(f"Need at least 3 images in {images}")
    mapper = mapper_command(cfg)
    db = scene / "database.db"
    sparse = scene / ("sparse_glomap" if cfg.mapper == "glomap" else "sparse")
    log_file = scene / f"sfm_{cfg.mapper}.log"
    log_file.write_text(f"{colmap_version(cfg.colmap)}\n")
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()
    db.unlink(missing_ok=True)

    t0 = time.time()
    _run([cfg.colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(images),
          "--ImageReader.single_camera", str(int(cfg.single_camera)), "--ImageReader.camera_model", cfg.camera_model,
          "--SiftExtraction.max_num_features", str(cfg.max_features)]
         + _gpu_flags(cfg.colmap, "extract", cfg.use_gpu), log_file)
    match = [cfg.colmap, f"{cfg.matcher}_matcher", "--database_path", str(db)]
    if cfg.matcher == "vocab_tree":
        if not cfg.vocab_tree_path:
            raise ValueError("matcher=vocab_tree needs sfm.vocab_tree_path (download from colmap.github.io)")
        match += ["--VocabTreeMatching.vocab_tree_path", cfg.vocab_tree_path]
    _run(match + _gpu_flags(cfg.colmap, "match", cfg.use_gpu), log_file)
    t1 = time.time()
    _run(mapper + ["--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse)], log_file)
    t2 = time.time()

    models = [m for m in sparse.iterdir() if (m / "images.bin").exists()]
    if not models:
        raise RuntimeError(f"{cfg.mapper} produced no model: need more overlap, texture or sharper images "
                           f"(log: {log_file})")
    best = max(models, key=lambda m: len(read_model(m).images))
    model_dir, image_dir = best, images
    if cfg.camera_model != "PINHOLE":
        und = scene / f"undistorted_{cfg.mapper}"
        shutil.rmtree(und, ignore_errors=True)
        _run([cfg.colmap, "image_undistorter", "--image_path", str(images), "--input_path", str(best),
              "--output_path", str(und), "--output_type", "COLMAP"], log_file)
        model_dir, image_dir = und / "sparse", und / "images"

    m = read_model(model_dir)
    stats = {
        "model": str(model_dir.relative_to(scene)),
        "images": str(image_dir.relative_to(scene)),
        "mapper": cfg.mapper,
        "colmap": colmap_version(cfg.colmap),
        "registered_images": len(m.images),
        "input_images": n_input,
        "points": int(len(m.xyz)),
        "num_models": len(models),
        "seconds": {"features_matching": round(t1 - t0, 1), "mapping": round(t2 - t1, 1)},
        "config": {k: getattr(cfg, k) for k in ("mapper", "matcher", "camera_model", "single_camera", "max_features")},
    }
    (scene / "scene.json").write_text(json.dumps(stats, indent=2))
    if cfg.camera_model == "PINHOLE":  # research scripts read text models via sparse_model.txt
        _run([cfg.colmap, "model_converter", "--input_path", str(best), "--output_path", str(best),
              "--output_type", "TXT"], log_file)
        (scene / "sparse_model.txt").write_text(str(best.relative_to(scene)))
    log.info("%s: registered %d/%d images, %d points, %d model(s); features+matching %.0f s, mapping %.0f s",
             cfg.mapper, len(m.images), n_input, len(m.xyz), len(models), t1 - t0, t2 - t1)
    if len(m.images) < 0.8 * n_input:
        log.warning("Only %d of %d images registered; consider exhaustive matching or more overlap",
                    len(m.images), n_input)
    return stats


def resolve_scene(scene: str | Path) -> tuple[Path, Path]:
    """(model folder, image folder) to train on: from scene.json, else sparse_model.txt, else sparse/0."""
    scene = Path(scene)
    if (scene / "scene.json").exists():
        info = json.loads((scene / "scene.json").read_text())
        return scene / info["model"], scene / info["images"]
    if (scene / "sparse_model.txt").exists():
        m = Path((scene / "sparse_model.txt").read_text().strip())
        for cand in (m, scene / m, scene / "sparse" / m.name):
            if (cand / "images.bin").exists() or (cand / "images.txt").exists():
                return cand, scene / "images"
    return scene / "sparse" / "0", scene / "images"
