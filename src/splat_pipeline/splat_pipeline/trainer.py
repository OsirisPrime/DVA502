"""Stage 3: gsplat optimisation with checkpoint/resume, periodic evaluation and structured logs.

Parameterisation (also the .ply layout): means, log-scales, quaternions (w, x, y, z), logit-opacities,
SH band 0 ``sh0`` (N, 1, 3) and higher bands ``shN`` (N, K-1, 3). Activations (exp / sigmoid) are applied only
when rasterizing.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .colmap_io import read_model
from .config import Config, TrainConfig
from .data import ViewDataset, build_views, choose_cache, infinite_loader, load_image, split_views
from .metrics import psnr, ssim
from .ply import save_ply
from .sfm import resolve_scene

log = logging.getLogger(__name__)
C0 = 0.28209479177387814  # SH band-0 constant
PARAM_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


# ---------------------------------------------------------------------------- model setup
def init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, init_opacity: float) -> torch.nn.ParameterDict:
    from scipy.spatial import KDTree

    n = len(xyz)
    if n < 4:
        raise ValueError("Need at least 4 SfM points to initialise")
    dist, _ = KDTree(xyz).query(xyz, k=4)
    scale = np.sqrt((dist[:, 1:] ** 2).mean(1)).clip(1e-7)  # mean distance to the 3 nearest points
    sh0 = (torch.from_numpy(rgb.astype(np.float32) / 255.0) - 0.5) / C0
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz.astype(np.float32))),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy(scale).float())[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), init_opacity))),
        "sh0": torch.nn.Parameter(sh0[:, None, :]),
        "shN": torch.nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3)),
    })


def make_strategy(cfg: TrainConfig):
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    if cfg.strategy == "mcmc":
        until = cfg.densify_until if cfg.densify_until > 0 else 25 / 30
        return MCMCStrategy(cap_max=cfg.cap_max, refine_stop_iter=int(until * cfg.steps), verbose=False)
    until = cfg.densify_until if cfg.densify_until > 0 else 0.5
    kw = {"grow_grad2d": 0.0008} if cfg.absgrad else {}
    return DefaultStrategy(refine_stop_iter=int(until * cfg.steps), absgrad=cfg.absgrad, verbose=False, **kw)


def make_optimizers(params, cfg: TrainConfig, scene_scale: float):
    lrs = {"means": cfg.lr_means * scene_scale, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    opts: dict[str, torch.optim.Optimizer] = {
        k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], 0.01 ** (1 / cfg.steps))
    return opts, sched


def render(params, K, viewmat, width, height, sh_degree, cfg: TrainConfig):
    from gsplat import rasterization

    img, _, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=torch.cat([params["sh0"], params["shN"]], 1),
        viewmats=viewmat[None], Ks=K[None], width=width, height=height, sh_degree=sh_degree,
        packed=cfg.packed, absgrad=cfg.absgrad and cfg.strategy == "default")
    return img[0], info


# ---------------------------------------------------------------------------- run bookkeeping
def repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists():
            return p
    return Path.cwd()


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(repo_root()), "status", "--porcelain"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _manifest(scene: Path, cfg: Config, n_train: int, n_test: int) -> dict:
    import gsplat

    return {
        "scene": str(scene.resolve()), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "git": _git_commit(),
        "python": platform.python_version(), "torch": torch.__version__, "gsplat": gsplat.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_views": n_train, "test_views": n_test,
    }


def save_checkpoint(run: Path, step: int, params, opts, sched, state, keep: int) -> Path:
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    path = ckpt_dir / f"ckpt_{step:07d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "step": step,
        "params": {k: params[k].detach().cpu() for k in PARAM_KEYS},
        "optimizers": {k: o.state_dict() for k, o in opts.items()},
        "scheduler": sched.state_dict(),
        "strategy_state": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state.items()},
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate()},
    }, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint
    for old in sorted(ckpt_dir.glob("ckpt_*.pt"))[:-keep]:
        old.unlink()
    return path


def latest_checkpoint(run: Path) -> Path | None:
    ckpts = sorted((run / "checkpoints").glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(params, views, cfg: TrainConfig, out_dir: Path | None = None) -> dict:
    import cv2

    dev = cfg.device
    per_view = {}
    for v in views:
        gt = load_image(v).to(dev).float() / 255.0
        pred, _ = render(params, v.K.to(dev), v.viewmat.to(dev), v.width, v.height, cfg.sh_degree, cfg)
        pred = pred.clamp(0, 1)
        per_view[v.name] = {"psnr": psnr(pred, gt),
                            "ssim": float(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            pair = (torch.cat([pred, gt], 1).cpu().numpy()[:, :, ::-1] * 255).round().astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{Path(v.name).stem}.jpg"), pair)  # render | ground truth
    if not per_view:
        return {"views": 0}
    return {"views": len(per_view), "psnr": float(np.mean([m["psnr"] for m in per_view.values()])),
            "ssim": float(np.mean([m["ssim"] for m in per_view.values()])), "per_view": per_view}


# ---------------------------------------------------------------------------- training
def train(scene: str | Path, cfg: Config, run: Path | None = None, resume: bool = False) -> Path:
    """Train on a scene prepared by the sfm stage. Returns the run folder.

    With ``resume=True`` the run folder's config.yaml and latest checkpoint are used and training continues.
    """
    from tqdm import tqdm

    scene = Path(scene)
    tc = cfg.train
    dev = tc.device
    if run is None:
        run = repo_root() / "outputs" / "splat_pipeline" / f"{scene.resolve().name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    _add_file_log(run / "train.log")
    if not resume:
        cfg.save(run / "config.yaml")
    log.info("run folder: %s", run)

    random.seed(tc.seed)
    np.random.seed(tc.seed)
    torch.manual_seed(tc.seed)

    # ---- data ----
    model_dir, image_dir = resolve_scene(scene)
    model = read_model(model_dir)
    views = build_views(model, image_dir, tc.downscale)
    train_views, test_views = split_views(views, tc.test_every)
    log.info("model %s: %d posed images (%d train / %d test), %d points", model_dir, len(views), len(train_views),
             len(test_views), len(model.xyz))
    dataset = ViewDataset(train_views, choose_cache(train_views, tc.cache, tc.cache_limit_gb))
    centers = model.camera_centers
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)

    # ---- model / optimiser / strategy (fresh or from checkpoint) ----
    strategy = make_strategy(tc)
    ckpt = latest_checkpoint(run) if resume else None
    if resume and ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {run / 'checkpoints'} to resume from")
    if ckpt is not None:
        log.info("resuming from %s", ckpt)
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(data["params"][k]) for k in PARAM_KEYS}).to(dev)
    else:
        params = init_params(model.xyz, model.rgb, tc.sh_degree, 0.5 if tc.strategy == "mcmc" else 0.1).to(dev)
    opts, sched = make_optimizers(params, tc, scene_scale)
    strategy.check_sanity(params, opts)
    if tc.strategy == "mcmc":
        state = strategy.initialize_state()
    else:
        state = strategy.initialize_state(scene_scale=scene_scale)
    start = 0
    if ckpt is not None:
        for k, o in opts.items():
            o.load_state_dict(data["optimizers"][k])
        sched.load_state_dict(data["scheduler"])
        state = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in data["strategy_state"].items()}
        torch.set_rng_state(data["rng"]["torch"])
        torch.cuda.set_rng_state_all(data["rng"]["cuda"])
        np.random.set_state(data["rng"]["numpy"])
        random.setstate(data["rng"]["python"])
        start = data["step"]
        del data
    else:
        (run / "run.json").write_text(json.dumps(_manifest(scene, cfg, len(train_views), len(test_views)), indent=2))

    # ---- loop ----
    loader = infinite_loader(dataset, tc.steps - start, tc.num_workers, tc.seed + start)
    metrics_log = open(run / "metrics.jsonl", "a")
    pbar = tqdm(total=tc.steps, initial=start, desc="train", dynamic_ncols=True)
    step, ema, t0 = start, None, time.time()
    try:
        for batch in loader:
            gt = batch["image"].to(dev, non_blocking=True).float() / 255.0
            K, viewmat = batch["K"].to(dev), batch["viewmat"].to(dev)
            sh_deg = min(step // tc.sh_degree_interval, tc.sh_degree)
            pred, info = render(params, K, viewmat, gt.shape[1], gt.shape[0], sh_deg, tc)
            strategy.step_pre_backward(params, opts, state, step, info)
            l1 = (pred - gt).abs().mean()
            s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            loss = (1 - tc.ssim_lambda) * l1 + tc.ssim_lambda * (1 - s)
            if tc.strategy == "mcmc":  # regularisers from the 3DGS-MCMC paper
                loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() \
                    + 0.01 * torch.exp(params["scales"]).mean()
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            if tc.strategy == "mcmc":
                strategy.step_post_backward(params, opts, state, step, info, lr=sched.get_last_lr()[0])
            else:
                strategy.step_post_backward(params, opts, state, step, info, packed=tc.packed)
            step += 1

            if step % 100 == 0 and not math.isfinite(loss.item()):
                raise FloatingPointError(f"loss is {loss.item()} at step {step}; last checkpoint is still valid")
            if step % 50 == 0:
                ema = loss.item() if ema is None else 0.6 * ema + 0.4 * loss.item()
                pbar.set_postfix(loss=f"{ema:.4f}", gaussians=f"{len(params['means']):,}")
            pbar.update(1)
            if step % 1000 == 0:
                rec = {"step": step, "loss": ema, "gaussians": len(params["means"]),
                       "elapsed_s": round(time.time() - t0, 1),
                       "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
                if tc.eval_every and step % tc.eval_every == 0 and test_views:
                    m = evaluate(params, test_views, tc)
                    rec.update(psnr=m["psnr"], ssim=m["ssim"])
                metrics_log.write(json.dumps(rec) + "\n")
                metrics_log.flush()
                tqdm.write(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
            if tc.checkpoint_every and step % tc.checkpoint_every == 0 and step < tc.steps:
                save_checkpoint(run, step, params, opts, sched, state, tc.keep_checkpoints)
    except KeyboardInterrupt:
        pbar.close()
        path = save_checkpoint(run, step, params, opts, sched, state, tc.keep_checkpoints)
        log.warning("Interrupted at step %d; saved %s. Continue with: train --resume %s", step, path, run)
        raise SystemExit(130)
    finally:
        metrics_log.close()
    pbar.close()

    # ---- final export + evaluation ----
    save_checkpoint(run, step, params, opts, sched, state, tc.keep_checkpoints)
    save_ply(run / "point_cloud.ply", {k: params[k].detach().cpu().numpy() for k in PARAM_KEYS})
    result = evaluate(params, test_views, tc, run / "renders" if tc.save_renders else None)
    result.update(steps=step, gaussians=len(params["means"]), train_minutes=round((time.time() - t0) / 60, 1))
    (run / "metrics.json").write_text(json.dumps(result, indent=2))
    if result["views"]:
        log.info("test views %d | PSNR %.2f dB | SSIM %.4f | %d Gaussians", result["views"], result["psnr"],
                 result["ssim"], result["gaussians"])
    log.info("saved %s", run / "point_cloud.ply")
    return run


def export_ply(run: Path, out: Path | None = None) -> Path:
    ckpt = latest_checkpoint(run)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {run}")
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    out = out or run / f"point_cloud_{data['step']:07d}.ply"
    save_ply(out, {k: v.numpy() for k, v in data["params"].items()})
    log.info("exported %s (step %d, %d Gaussians)", out, data["step"], len(data["params"]["means"]))
    return out


def evaluate_run(run: Path, scene: Path | None = None) -> dict:
    """Re-evaluate the latest checkpoint of a run on its held-out views."""
    cfg = Config.load(run / "config.yaml")
    scene = scene or Path(json.loads((run / "run.json").read_text())["scene"])
    ckpt = latest_checkpoint(run)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint in {run}")
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    params = {k: v.to(cfg.train.device) for k, v in data["params"].items()}
    model_dir, image_dir = resolve_scene(scene)
    _, test_views = split_views(build_views(read_model(model_dir), image_dir, cfg.train.downscale),
                                cfg.train.test_every)
    result = evaluate(params, test_views, cfg.train, run / "renders")
    result["step"] = data["step"]
    (run / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info("step %d | test views %d | PSNR %.2f dB | SSIM %.4f", data["step"], result["views"],
             result.get("psnr", float("nan")), result.get("ssim", float("nan")))
    return result


def _add_file_log(path: Path) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(path.resolve()) for h in root.handlers):
        h = logging.FileHandler(path)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
