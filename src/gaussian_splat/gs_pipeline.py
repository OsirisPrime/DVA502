#!/usr/bin/env python3
"""
Gaussian Splatting pipeline (research version, one self-contained file; also runs on Colab)
  video / photos -> COLMAP or GLOMAP (camera poses + sparse points) -> gsplat training -> .ply + eval

Requirements: gsplat (AMD: build it with rocm/build_gsplat_rocm.sh, then `source ~/gsplat-rocm/venv/bin/activate`;
NVIDIA/Colab: `pip install gsplat`), opencv-python, scipy, tqdm, and COLMAP (conda-forge colmap 4.2.1, see docs).
GLOMAP needs COLMAP >= 3.13 (`colmap global_mapper`); we use conda-forge COLMAP 4.2.1 in the lerobot env.

Usage (from the repo root):
  # 1a) from a video: keep the sharpest frame in every 1/fps-second window
  python src/gaussian_splat/gs_pipeline.py frames --video local_data/videos/scene.mp4 --out local_data/scenes/scene --fps 2
  # 1b) from photos: put them in local_data/scenes/scene/images/

  # 2) structure-from-motion (--sequential for video frames; --mapper glomap for the faster global mapper)
  python src/gaussian_splat/gs_pipeline.py sfm --data local_data/scenes/scene --sequential
  python src/gaussian_splat/gs_pipeline.py sfm --data local_data/scenes/scene --sequential --mapper glomap

  # 3) train (every 8th image held out for test PSNR/SSIM)
  python src/gaussian_splat/gs_pipeline.py train --data local_data/scenes/scene --steps 30000 --downscale 1

Output (default <repo>/outputs/gsplat/<scene>_<timestamp>/, or --out):
        point_cloud.ply  (open in https://superspl.at/editor or any 3DGS viewer)
        renders/         (test views: render | ground truth)
        metrics.txt, args.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def repo_root():
    """The git repository this script lives in (outputs/ goes there); the script's folder when run elsewhere,
    e.g. on Colab."""
    here = Path(__file__).resolve().parent
    for p in here.parents:
        if (p / ".git").exists():
            return p
    return here


# ----------------------------------------------------------------------------
# 1. Frame extraction from video
# ----------------------------------------------------------------------------
def cmd_frames(a):
    """Extract the sharpest frame at regular intervals from a video.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command‑line arguments for the ``frames`` sub‑command.
        ``a.video`` – path to the input video.
        ``a.out`` – directory where extracted JPEGs will be stored.
        ``a.fps`` – desired number of frames per second to keep.
        ``a.max_side`` – optional maximum resolution for the longest side; frames
        larger than this will be down‑scaled before sharpness evaluation.

    The function opens the video using OpenCV, iterates over all frames, and
    keeps the most "sharp" frame within each window defined by ``fps``.
    Sharpness is measured via the variance of the Laplacian; a higher value
    indicates less motion blur.  Selected frames are written as JPEGs named
    ``00000.jpg`` … ``NNNNN.jpg``.
    """
    import cv2

    out = Path(a.out) / "images"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {a.video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = max(1, round(vfps / a.fps))

    best, best_score, i, n = None, -1.0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if a.max_side and max(frame.shape[:2]) > a.max_side:
            s = a.max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()  # sharpness: higher = less motion blur
        if score > best_score:
            best, best_score = frame, score
        i += 1
        if i % window == 0:
            cv2.imwrite(str(out / f"{n:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
            best, best_score = None, -1.0
    if best is not None:
        cv2.imwrite(str(out / f"{n:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    print(f"Saved {n} frames to {out}")


# ----------------------------------------------------------------------------
# 2. Structure from Motion: COLMAP features + matches, then COLMAP (incremental) or GLOMAP (global) mapper
# ----------------------------------------------------------------------------
def run(cmd):
    """Print a command (prefixed with ``>>``) and run it, raising if it fails."""
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _has_subcommand(exe, sub):
    try:
        r = subprocess.run([exe, sub, "-h"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and "--database_path" in (r.stdout + r.stderr)


def glomap_command(colmap):
    """GLOMAP is built into COLMAP >= 3.13 as `colmap global_mapper`; older setups have a separate `glomap`."""
    if _has_subcommand(colmap, "global_mapper"):
        return [colmap, "global_mapper"]
    if shutil.which("glomap"):
        return ["glomap", "mapper"]
    raise SystemExit(
        f"GLOMAP not found: '{colmap}' has no global_mapper and there is no `glomap` on PATH.\n"
        "Install COLMAP 4 into the lerobot env (see docs/gaussian_splatting.md, setup) or pass --colmap.")


def default_colmap():
    """$COLMAP_BIN, else the COLMAP of the conda env this Python belongs to (also from a venv built on it, e.g.
    ~/gsplat-rocm), else `colmap` on PATH (/usr/bin/colmap is the older 3.9 without GLOMAP)."""
    if os.environ.get("COLMAP_BIN"):
        return os.environ["COLMAP_BIN"]
    env_colmap = Path(sys.base_prefix) / "bin" / "colmap"
    return str(env_colmap) if env_colmap.exists() else "colmap"


def resolve_model(d):
    """COLMAP model folder of a scene: the one named in sparse_model.txt, else sparse/0."""
    d = Path(d)
    f = d / "sparse_model.txt"
    if f.exists():
        m = Path(f.read_text().strip())
        for cand in (m, d / m, d / "sparse" / m.name):  # absolute, relative to the scene, old layout
            if (cand / "images.txt").exists() or (cand / "images.bin").exists():
                return cand
    return d / "sparse" / "0"


def cmd_sfm(a):
    """Recover camera poses and sparse points with COLMAP (+ optionally GLOMAP).

    Steps: feature extraction (one PINHOLE camera for all images) -> matching (sequential or exhaustive)
    -> mapper -> keep the largest model, convert it to TXT and record it in <data>/sparse_model.txt.
    The database is recreated on every run. Each mapper writes to its own folder (sparse/ for COLMAP,
    sparse_glomap/ for GLOMAP), so both can be kept and compared; sparse_model.txt points at the latest run.
    All steps use the same COLMAP binary (--colmap), because the database format changes between versions.
    """
    d = Path(a.data)
    img_dir, db = d / "images", d / "database.db"
    sparse = d / ("sparse_glomap" if a.mapper == "glomap" else "sparse")
    mapper = glomap_command(a.colmap) if a.mapper == "glomap" else [a.colmap, "mapper"]
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    if db.exists():
        db.unlink()

    # PINHOLE = no lens distortion model -> directly usable by the rasterizer.
    # For strong wide-angle distortion, use OPENCV here and run `colmap image_undistorter` after.
    t0 = time.time()
    run([a.colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(img_dir),
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "PINHOLE",
         "--SiftExtraction.max_num_features", str(a.max_features)])
    matcher = "sequential_matcher" if a.sequential else "exhaustive_matcher"
    run([a.colmap, matcher, "--database_path", str(db)])
    t1 = time.time()
    run(mapper + ["--database_path", str(db), "--image_path", str(img_dir), "--output_path", str(sparse)])
    t2 = time.time()

    # the mapper can split the scene into several models; keep the biggest one
    models = [m for m in sparse.iterdir() if (m / "images.bin").exists()]
    if not models:
        raise SystemExit("Reconstruction failed. Need more overlap / texture / sharper images.")
    best = max(models, key=lambda m: (m / "images.bin").stat().st_size)
    run([a.colmap, "model_converter", "--input_path", str(best), "--output_path", str(best),
         "--output_type", "TXT"])
    (d / "sparse_model.txt").write_text(str(best.relative_to(d)))
    n_img = sum(1 for l in open(best / "images.txt") if l.strip() and not l.startswith("#")) // 2
    n_pts = sum(1 for l in open(best / "points3D.txt") if l.strip() and not l.startswith("#"))
    n_all = len([f for f in img_dir.iterdir() if f.is_file()])
    print(f"\n{a.mapper}: registered {n_img}/{n_all} images, {n_pts:,} points | features+matching "
          f"{t1 - t0:.0f} s, mapper {t2 - t1:.0f} s | using {best} ({len(models)} model(s) found)")


# ----------------------------------------------------------------------------
# COLMAP text-model reader
# ----------------------------------------------------------------------------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def load_colmap(model_dir):
    """Parse COLMAP text‑model files and return images, points and colors.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing COLMAP ``cameras.txt``, ``images.txt`` and
        ``points3D.txt``.

    Returns
    -------
    tuple
        ``(images, xyz, rgb)`` where ``images`` is a list of dictionaries with
        ``name``, ``cam`` and ``w2c`` keys; ``xyz`` is an ``(N,3)`` float32
        array of 3‑D points and ``rgb`` is an ``(N,3)`` float32 array in the
        ``[0,1]`` range.
    """
    model_dir = Path(model_dir)
    cams = {}
    for line in open(model_dir / "cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cid, model, w, h, prm = int(p[0]), p[1], int(p[2]), int(p[3]), list(map(float, p[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = prm[:4]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = prm[0]
            cx, cy = prm[1:3]
        else:  # OPENCV etc: use focal/center, ignore distortion (undistort first for best results)
            fx, fy, cx, cy = prm[:4]
        cams[cid] = (w, h, fx, fy, cx, cy)

    lines = [l for l in open(model_dir / "images.txt") if not l.startswith("#")]
    images = []
    for i in range(0, len(lines) - 1, 2):  # 2 lines per image (2nd = 2D points, may be empty)
        p = lines[i].split()
        if len(p) < 10:
            continue
        q = np.array(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(q)
        w2c[:3, 3] = t
        images.append(dict(name=p[9], cam=cams[int(p[8])], w2c=w2c))
    images.sort(key=lambda x: x["name"])

    xyz, rgb = [], []
    for line in open(model_dir / "points3D.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append(list(map(float, p[1:4])))
        rgb.append(list(map(int, p[4:7])))
    return images, np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


# ----------------------------------------------------------------------------
# 3. Training with gsplat
# ----------------------------------------------------------------------------
C0 = 0.28209479177387814  # SH band-0 constant


def ssim(x, y):
    """Compute Structural Similarity (SSIM) between two images.

    Parameters
    ----------
    x, y : torch.Tensor
        Images of shape ``(1, 3, H, W)`` with values in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        The mean SSIM over all spatial locations.
    """
    import torch
    import torch.nn.functional as F
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    win = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda z: F.conv2d(z, win, padding=5, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return m.mean()


def save_ply(path, params):
    """Write a point cloud in the standard 3DGS ``.ply`` format.

    Parameters
    ----------
    path : str | Path
        Destination file.
    params : torch.nn.ParameterDict
        Dictionary containing the Gaussian parameters exported by the
        training loop.  Keys expected are ``means``, ``scales``, ``quats``,
        ``opacities``, ``sh0`` and ``shN``.

    The function converts the tensors to NumPy, reshapes SH coefficients into
    channel‑major layout, builds the binary PLY header with the appropriate
    properties, and writes the vertex data as little‑endian floats.
    """
    t = lambda k: params[k].detach().cpu().numpy()
    means, scales, quats, opac = t("means"), t("scales"), t("quats"), t("opacities")
    sh0 = t("sh0").reshape(len(means), -1)                      # (N,3)
    shN = t("shN").transpose(0, 2, 1).reshape(len(means), -1)   # channel-major, as in 3DGS
    attrs = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(shN.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, opac[:, None], scales, quats],
                          axis=1).astype(np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(data)}\n"
              + "".join(f"property float {a}\n" for a in attrs) + "end_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(data.tobytes())


def cmd_train(a):
    """Train a Gaussian Splatting model and evaluate it.

    Parameters
    ----------
    a : argparse.Namespace
        Arguments for the ``train`` sub‑command.  See the ``__main__`` block for the
        full list of accepted options.

    The function performs the following steps:

    1. Load the COLMAP reconstruction using :func:`load_colmap`.
    2. Build a list of image views with pre‑computed camera intrinsics and
       pose matrices.  Images can be optionally down‑scaled.
    3. Initialise Gaussian parameters from the sparse point cloud.
    4. Run a training loop for ``a.steps`` iterations, gradually increasing
       the spherical‑harmonics degree and using a custom strategy from
       :mod:`gsplat.strategy`.
    5. After training, render held‑out views, compute PSNR and SSIM, and
       write the results, a point‑cloud PLY, and per‑view renders.
    """
    import cv2
    import torch
    from scipy.spatial import cKDTree
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from tqdm import tqdm

    dev = "cuda"
    d = Path(a.data)
    model_dir = resolve_model(d)
    # default: <repo>/outputs/gsplat/<scene>_<timestamp>/ (one folder per run, never overwritten)
    out = Path(a.out) if a.out else (repo_root() / "outputs" / "gsplat"
                                     / f"{d.resolve().name}_{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "renders").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "cmd"}, indent=2))
    print("output:", out)

    # ---- data ----
    images, xyz, rgb = load_colmap(model_dir)
    print(f"{len(images)} posed images, {len(xyz)} sparse points")
    views = []
    for im in images:
        img = cv2.imread(str(d / "images" / im["name"]))[:, :, ::-1]
        W0, H0, fx, fy, cx, cy = im["cam"]
        if a.downscale > 1:
            img = cv2.resize(img, (W0 // a.downscale, H0 // a.downscale), interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
        sx, sy = W / W0, H / H0
        K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=torch.float32)
        views.append(dict(name=im["name"], img=torch.from_numpy(img.copy()),  # uint8 on CPU
                          K=K, viewmat=torch.tensor(im["w2c"], dtype=torch.float32), W=W, H=H))
    test = views[::a.test_every]
    train = [v for i, v in enumerate(views) if i % a.test_every != 0]

    # scene scale from camera spread (used for LR + densification thresholds)
    centers = np.stack([np.linalg.inv(im["w2c"])[:3, 3] for im in images])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)

    # ---- init Gaussians from SfM points ----
    N = len(xyz)
    dist, _ = cKDTree(xyz).query(xyz, k=4)
    init_scale = np.sqrt((dist[:, 1:] ** 2).mean(1)).clip(1e-7)
    K_sh = (a.sh_degree + 1) ** 2
    sh0 = ((torch.from_numpy(rgb) - 0.5) / C0)[:, None, :]
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(xyz)),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy(init_scale)).float()[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]]).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1))),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(torch.zeros(N, K_sh - 1, 3)),
    }).to(dev)

    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], 0.01 ** (1 / a.steps))

    strategy = DefaultStrategy(verbose=False)  # clone/split/prune + opacity reset
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    def render(v, sh_deg):
        """Render a single view with a specified SH degree.

        Parameters
        ----------
        v : dict
            View dictionary containing image tensor, camera intrinsics and
            pose.
        sh_deg : int
            Spherical‑harmonics degree to use.
        """
        colors = torch.cat([params["sh0"], params["shN"]], 1)
        img, alpha, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors,
            viewmats=v["viewmat"][None].to(dev), Ks=v["K"][None].to(dev),
            width=v["W"], height=v["H"], sh_degree=sh_deg, packed=False)
        return img[0].clamp(0, 1), info

    # ---- optimisation loop ----
    pbar = tqdm(range(a.steps), desc="train", dynamic_ncols=True)
    for step in pbar:
        v = train[np.random.randint(len(train))]
        gt = v["img"].to(dev).float() / 255.0
        sh_deg = min(step // 1000, a.sh_degree)  # grow SH degree gradually

        pred, info = render(v, sh_deg)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (pred - gt).abs().mean()
        s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = 0.8 * l1 + 0.2 * (1 - s)
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", gaussians=f"{len(params['means']):,}")
        if step % 1000 == 0:  # permanent log line every 1000 steps
            tqdm.write(f"step {step:6d}  loss {loss.item():.4f}  #gaussians {len(params['means']):,}")

    # ---- evaluation on held-out views ----
    psnrs, ssims = [], []
    with torch.no_grad():
        for v in tqdm(test, desc="eval", dynamic_ncols=True):
            gt = v["img"].to(dev).float() / 255.0
            pred, _ = render(v, a.sh_degree)
            mse = ((pred - gt) ** 2).mean()
            psnrs.append((-10 * torch.log10(mse)).item())
            ssims.append(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
            pair = torch.cat([pred, gt], 1).cpu().numpy()
            cv2.imwrite(str(out / "renders" / v["name"]), (pair[:, :, ::-1] * 255).astype(np.uint8))

    msg = (f"test views: {len(test)}\nPSNR: {np.mean(psnrs):.2f} dB\nSSIM: {np.mean(ssims):.4f}\n"
           f"gaussians: {len(params['means'])}\n")
    print(msg)
    (out / "metrics.txt").write_text(msg)
    save_ply(out / "point_cloud.ply", params)
    print(f"Saved {out / 'point_cloud.ply'}")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    f = sp.add_parser("frames")
    f.add_argument("--video", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--fps", type=float, default=2.0, help="frames kept per second of video")
    f.add_argument("--max_side", type=int, default=1920, help="resize longest side (0 = keep)")

    s = sp.add_parser("sfm")
    s.add_argument("--data", required=True)
    s.add_argument("--sequential", action="store_true", help="sequential matching (video frames, faster)")
    s.add_argument("--mapper", choices=["colmap", "glomap"], default="colmap",
                   help="colmap = incremental (robust, slower); glomap = global (much faster on many images)")
    s.add_argument("--colmap", default=default_colmap(),
                   help="COLMAP executable (default: $COLMAP_BIN, else the conda env's colmap, else colmap on "
                        "PATH). GLOMAP needs COLMAP >= 3.13")
    s.add_argument("--max_features", type=int, default=8192, help="SIFT features per image")

    t = sp.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default=None,
                   help="output folder (default: outputs/gsplat/<scene>_<timestamp> in the repo)")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--downscale", type=int, default=1)
    t.add_argument("--sh_degree", type=int, default=3)
    t.add_argument("--test_every", type=int, default=8)

    a = ap.parse_args()
    {"frames": cmd_frames, "sfm": cmd_sfm, "train": cmd_train}[a.cmd](a)
