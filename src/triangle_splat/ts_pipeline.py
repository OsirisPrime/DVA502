#!/usr/bin/env python3
"""
Triangle Splatting trainer (Held et al., "Triangle Splatting for Real-Time Radiance Field Rendering", 3DV 2026)
  COLMAP scene (from gs_pipeline.py frames + sfm)  ->  triangle soup  ->  model + mesh + eval

Uses the official code (https://github.com/trianglesplatting/triangle-splatting) and its GPU rasterizer,
built for the AMD GPU by build_triangle_splat_rocm.sh into ~/triangle-splat-rocm. The training loop below is
the official train.py loop with the same hyperparameters; what differs is the interface (same options,
progress bar and output layout as the Gaussian splatting scripts) and that LPIPS/TensorBoard are left out.

Setup (once):
  bash src/triangle_splat/build_triangle_splat_rocm.sh

Usage:
  source ~/triangle-splat-rocm/venv/bin/activate
  python src/triangle_splat/ts_pipeline.py train --data local_data/scenes/img_1453 --steps 30000 --downscale 1

Output (default <repo>/outputs/triangle_splat/<scene>_<timestamp>/, or --out):
  renders/                        test views: render | ground truth
  metrics.txt, args.json
  triangles.ply                   coloured triangle mesh (MeshLab, Blender, game engines)
  point_cloud/iteration_<N>/      model in the official format (works with the official render.py -m <out>)

Licence note: the official code is derived from Inria's 3D Gaussian Splatting code, which is free for
non-commercial research and evaluation only (LICENSE_GS.md in the source tree).
"""
import argparse
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("TRIANGLE_SPLAT_SRC", Path.home() / "triangle-splat-rocm" / "src"))
C0 = 0.28209479177387814  # SH band-0 constant


def repo_root():
    for p in HERE.parents:
        if (p / ".git").exists():
            return p
    return HERE.parents[1]


def import_official():
    """Import the official Triangle Splatting modules (plus the simple_knn replacement in shims/)."""
    if not (SRC / "triangle_renderer").is_dir():
        raise SystemExit(f"Triangle Splatting source not found in {SRC}.\n"
                         f"Run: bash {HERE / 'build_triangle_splat_rocm.sh'}")
    sys.path[:0] = [str(HERE / "shims"), str(SRC)]
    try:
        import diff_triangle_rasterization  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"Triangle rasterizer not available ({e}).\n"
                         f"Activate the venv (source ~/triangle-splat-rocm/venv/bin/activate) or build it with "
                         f"bash {HERE / 'build_triangle_splat_rocm.sh'}")
    import arguments
    import scene
    import scene.dataset_readers as readers
    import triangle_renderer
    import utils.loss_utils as losses
    return arguments, scene, readers, triangle_renderer.render, losses


def scene_root(d, out):
    """The official loader reads <root>/images and <root>/sparse/0. Our sfm step may have picked another
    model (sparse_model.txt, e.g. sparse/1 or sparse_glomap/0), so then build a folder of symlinks with the
    expected layout."""
    model = d / "sparse" / "0"
    if (d / "sparse_model.txt").exists():
        m = Path((d / "sparse_model.txt").read_text().strip())
        for cand in (m, d / m, d / "sparse" / m.name):  # absolute, relative to the scene, old layout
            if (cand / "images.txt").exists() or (cand / "images.bin").exists():
                model = cand
                break
    if not (model / "images.txt").exists() and not (model / "images.bin").exists():
        raise SystemExit(f"No COLMAP model in {model}. Run src/gaussian_splat/gs_pipeline.py sfm first.")
    if model.resolve() == (d / "sparse" / "0").resolve():
        return d
    stage = out / "_scene"
    (stage / "sparse").mkdir(parents=True, exist_ok=True)
    for link, target in ((stage / "images", d / "images"), (stage / "sparse" / "0", model)):
        if not link.exists():
            link.symlink_to(target.resolve(), target_is_directory=True)
    return stage


def save_mesh_ply(path, triangles):
    """Triangle soup as a binary PLY mesh: 3 vertices per triangle, colour = SH band 0, alpha = opacity."""
    pts = triangles.get_triangles_points.detach().cpu().numpy().astype("<f4")          # (N,3,3)
    rgb = (triangles._features_dc.detach()[:, 0, :] * C0 + 0.5).clamp(0, 1)            # (N,3)
    rgba = torch.cat([rgb, triangles.get_opacity.detach().reshape(-1, 1)], 1)
    rgba = (rgba * 255).round().to(torch.uint8).cpu().numpy()
    n = len(pts)
    verts = np.empty(n * 3, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")])
    flat = pts.reshape(-1, 3)
    verts["x"], verts["y"], verts["z"] = flat[:, 0], flat[:, 1], flat[:, 2]
    col = np.repeat(rgba, 3, axis=0)
    verts["red"], verts["green"], verts["blue"], verts["alpha"] = col[:, 0], col[:, 1], col[:, 2], col[:, 3]
    faces = np.empty(n, dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    faces["n"] = 3
    faces["idx"] = np.arange(n * 3, dtype="<i4").reshape(n, 3)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n * 3}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
              f"element face {n}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode())
        fh.write(verts.tobytes())
        fh.write(faces.tobytes())


def cmd_train(a):
    import cv2

    arguments, scene_mod, readers, render, losses = import_official()

    d = Path(a.data)
    # default: <repo>/outputs/triangle_splat/<scene>_<timestamp>/ (one folder per run, never overwritten)
    out = Path(a.out) if a.out else (repo_root() / "outputs" / "triangle_splat"
                                     / f"{d.resolve().name}_{time.strftime('%Y%m%d-%H%M%S')}")
    (out / "renders").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "cmd"}, indent=2))
    print("output:", out)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    # ---- official hyperparameters, with our options on top ----
    parser = argparse.ArgumentParser()
    lp = arguments.ModelParams(parser)
    op = arguments.OptimizationParams(parser)
    pp = arguments.PipelineParams(parser)
    ns = parser.parse_args([])
    ns.source_path, ns.model_path = str(scene_root(d, out)), str(out)
    ns.resolution, ns.sh_degree, ns.eval, ns.data_device = a.downscale, a.sh_degree, True, a.data_device
    ns.iterations, ns.max_shapes = a.steps, a.max_triangles
    # the official schedule is defined for 30k steps; keep its proportions for other lengths
    ns.position_lr_max_steps = a.steps
    ns.densify_until_iter = round(25_000 * a.steps / 30_000)
    ns.iteration_mesh = round(5_000 * a.steps / 30_000)
    dataset, opt, pipe = lp.extract(ns), op.extract(ns), pp.extract(ns)
    (out / "cfg_args").write_text(str(argparse.Namespace(**vars(ns))))  # read by the official render.py

    # ---- data + initial triangles from the SfM points ----
    readers.sceneLoadTypeCallbacks["Colmap"] = partial(readers.readColmapSceneInfo, llffhold=a.test_every)
    triangles = scene_mod.TriangleModel(dataset.sh_degree)
    scene = scene_mod.Scene(dataset, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points,
                            opt.set_sigma, a.no_dome)
    triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma,
                             opt.lr_triangles_points_init)
    background = torch.zeros(3, device="cuda")
    n_views = len(scene.getTrainCameras())
    print(f"{n_views} training views, {len(scene.getTestCameras())} test views, "
          f"{len(triangles.get_opacity):,} initial triangles")
    loss_fn = losses.l2_loss if (triangles.large and a.outdoor) else losses.l1_loss

    def dead_triangles(new_round):
        low_opacity = (triangles.get_opacity <= opt.opacity_dead).squeeze()
        if n_views >= 250 and new_round:
            return low_opacity
        return torch.logical_or((triangles.importance_score < opt.importance_threshold).squeeze(), low_opacity)

    # ---- optimisation loop (official train.py) ----
    viewpoint_stack = []
    opacity_now, new_round, removed_them = True, False, False
    ema_loss = 0.0
    pbar = tqdm(range(1, a.steps + 1), desc="train", dynamic_ncols=True)
    for iteration in pbar:
        triangles.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()  # grow SH degree gradually

        if not viewpoint_stack:  # every training view once per round
            first_round = iteration == 1
            viewpoint_stack = scene.getTrainCameras().copy()
            if not first_round:
                if not new_round and removed_them:
                    new_round, removed_them = True, False
                else:
                    new_round = False
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

        pkg = render(cam, triangles, pipe, background)
        image = pkg["render"]

        # per-triangle statistics used for pruning / densification
        triangle_area = pkg["density_factor"].detach()
        image_size = pkg["scaling"].detach()
        importance = pkg["max_blending"].detach()
        if new_round:
            triangles.triangle_area[triangle_area > 1] += 1
        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance > triangles.importance_score
        triangles.importance_score[mask] = importance[mask]

        gt = cam.original_image.cuda()
        pixel_loss = loss_fn(image, gt)
        loss = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - losses.ssim(image, gt))
        loss = loss + torch.abs(triangles.get_opacity).mean() * opt.lambda_opacity
        if iteration > opt.iteration_mesh:  # normal consistency + depth distortion regularisers
            normal_error = 1 - (pkg["rend_normal"] * pkg["surf_normal"]).sum(dim=0)
            loss = loss + opt.lambda_normals * normal_error.mean() + opt.lambda_dist * pkg["rend_dist"].mean()
        if iteration < opt.densify_until_iter:
            loss = loss + opt.lambda_size / losses.equilateral_regularizer(triangles.get_triangles_points).mean()
        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0 or iteration == 1:
                ema_loss = loss.item() if iteration == 1 else 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 50 == 0:
                pbar.set_postfix(loss=f"{ema_loss:.4f}", triangles=f"{len(triangles.get_opacity):,}")
            if iteration % 1000 == 0 or iteration == 1:  # permanent log line every 1000 steps
                tqdm.write(f"step {iteration:6d}  loss {ema_loss:.4f}  #triangles {len(triangles.get_opacity):,}")

            interval = iteration % opt.densification_interval == 0
            if interval and opt.densify_from_iter < iteration < opt.densify_until_iter:
                # replace dead triangles and add new ones by sampling from the alive ones
                dead = dead_triangles(new_round)
                if iteration > 1000 and not new_round:
                    dead = torch.logical_or(dead, (triangles.triangle_area < 2).squeeze())
                    if not a.outdoor:
                        dead = torch.logical_or(dead, (triangles.image_size > 1400).squeeze())
                if opt.proba_distr == 0:
                    odd_group = True
                elif opt.proba_distr == 1:
                    odd_group = False
                else:  # alternate between sampling by opacity and by sharpness (sigma)
                    odd_group, opacity_now = opacity_now, not opacity_now
                removed_them, new_round = True, False
                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=odd_group, dead_mask=dead)
            elif interval and iteration > opt.densify_until_iter:
                # after densification: only remove dead triangles
                dead = dead_triangles(new_round)
                if not new_round:
                    dead = torch.logical_or(dead, (triangles.triangle_area < 2).squeeze())
                triangles.remove_final_points(dead)
                removed_them, new_round = True, False

            if iteration < a.steps:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none=True)

    # ---- evaluation on held-out views (same PSNR/SSIM definitions as gs_pipeline.py) ----
    psnrs, ssims = [], []
    test = sorted(scene.getTestCameras(), key=lambda c: c.image_name)
    with torch.no_grad():
        for cam in tqdm(test, desc="eval", dynamic_ncols=True):
            pred = render(cam, triangles, pipe, background)["render"].clamp(0, 1)
            gt = cam.original_image.cuda().clamp(0, 1)
            psnrs.append((-10 * torch.log10(((pred - gt) ** 2).mean())).item())
            ssims.append(losses.ssim(pred, gt).item())
            pair = torch.cat([pred, gt], 2).permute(1, 2, 0).cpu().numpy()  # render | ground truth
            cv2.imwrite(str(out / "renders" / f"{cam.image_name}.jpg"), (pair[:, :, ::-1] * 255).astype(np.uint8))

    n = len(triangles.get_opacity)
    msg = f"test views: {len(test)}\n"
    if test:
        msg += f"PSNR: {np.mean(psnrs):.2f} dB\nSSIM: {np.mean(ssims):.4f}\n"
    msg += f"triangles: {n}\n"
    print(msg)
    (out / "metrics.txt").write_text(msg)
    scene.save(a.steps)
    save_mesh_ply(out / "triangles.ply", triangles)
    print(f"Saved {out / 'triangles.ply'} and {out / 'point_cloud' / f'iteration_{a.steps}'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--data", required=True, help="scene folder with images/ and sparse/ (from src/gaussian_splat/gs_pipeline.py sfm)")
    t.add_argument("--out", default=None,
                   help="output folder (default: outputs/triangle_splat/<scene>_<timestamp> in the repo)")
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--downscale", type=int, default=1, choices=[1, 2, 4, 8])
    t.add_argument("--sh_degree", type=int, default=3)
    t.add_argument("--test_every", type=int, default=8, help="every N-th image is held out for PSNR/SSIM")
    t.add_argument("--max_triangles", type=int, default=4_000_000)
    t.add_argument("--outdoor", action="store_true", help="unbounded outdoor scene (official --outdoor)")
    t.add_argument("--no_dome", action="store_true", help="do not add the sky dome of far triangles")
    t.add_argument("--data_device", default="cpu", choices=["cpu", "cuda"],
                   help="where training images are kept; cuda is a bit faster but needs ~25 MB GPU memory "
                        "per full-resolution image")
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"train": cmd_train}[a.cmd](a)
