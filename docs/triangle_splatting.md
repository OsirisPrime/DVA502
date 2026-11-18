# Triangle splatting from a phone video (AMD GPU, local)

How to train a [Triangle Splatting](https://trianglesplatting.github.io/) model (Held et al., 3DV 2026) on our
machine (WSL2, RX 9070 XT / gfx1201, ROCm PyTorch). It uses the same COLMAP scenes as Gaussian splatting (see
[gaussian_splatting.md](gaussian_splatting.md) for filming, frame extraction and COLMAP), but represents the scene
with semi-transparent 3D triangles instead of Gaussians. The result can also be exported as a triangle mesh.

```
video ──frames──▶ images/ ──sfm (COLMAP)──▶ camera poses + sparse points ──train (triangles)──▶ model + triangles.ply
```

The training uses the [official code](https://github.com/trianglesplatting/triangle-splatting) and its GPU
rasterizer, compiled for our GPU.

> **Status:** a full 30k-step run at full resolution on `img_1453` reached **PSNR 28.45 dB, SSIM 0.925** with
> 3.25M triangles. Gaussian splatting on the same scene gives 28.80 dB / 0.935 with 1.42M Gaussians.
> A quick test (3k steps, quarter resolution) gives 25.3 dB in about 1 minute.

> **Licence:** the official code builds on Inria's 3D Gaussian Splatting code, which is **free for
> non-commercial research and evaluation only** (`LICENSE_GS.md` in the source). Check this with ABB before using it
> beyond the thesis research.

## Files

Everything is in `src/triangle_splat/`:

| File | Purpose |
|---|---|
| `build_triangle_splat_rocm.sh` | One-time build into an isolated venv; ends with a correctness check. |
| `ts_pipeline.py` | Training, evaluation and export. Same options and output layout as the Gaussian scripts. |
| `verify_triangle_rasterizer.py` | Checks the compiled rasterizer's image and gradients against a pure-PyTorch reference. |
| `shims/simple_knn/` | Small CPU replacement for the official CUDA `simple_knn` package (used once, to size the initial triangles). |

## 1. One-time setup

Prerequisites: the `lerobot` conda env with ROCm PyTorch and `~/rocm-torch.txt` (see the [README](../README.md#setup)),
and COLMAP (see [gaussian_splatting.md](gaussian_splatting.md), setup). The build also reuses the ROCm compiler from the gsplat build
(`~/gsplat-rocm`) if it exists; otherwise it downloads the ROCm 7.13 developer SDK (~9 GB) into its own venv.

```bash
bash src/triangle_splat/build_triangle_splat_rocm.sh
```

- Everything goes into `~/triangle-splat-rocm/` (venv, source, build log). The `lerobot` env is not changed.
  Undo with `rm -rf ~/triangle-splat-rocm`.
- Takes a few minutes. It must end with **`RASTERIZER OK`**.

## 2. Prepare a scene

Use the Gaussian splatting pipeline for steps 2–4 (film, extract frames, COLMAP), see
[gaussian_splatting.md](gaussian_splatting.md). A scene folder needs `images/` and `sparse/0/` (or
`sparse_model.txt` pointing at another model, e.g. `sparse_glomap/0` from `sfm --mapper glomap`).

## 3. Train

```bash
source ~/triangle-splat-rocm/venv/bin/activate
cd ~/projects/DVA502
python src/triangle_splat/ts_pipeline.py train --data local_data/scenes/img_1453 --steps 30000 --downscale 1
```

The progress bar shows loss and triangle count, with a permanent log line every 1000 steps.

| Option | Default | Meaning |
|---|---|---|
| `--steps` | 30000 | Official schedule. Densification runs until 25k (scaled for other lengths). |
| `--downscale` | 1 | 1, 2, 4 or 8. 1 = full resolution. Use 4 for quick tests. |
| `--sh_degree` | 3 | View-dependent colour detail. |
| `--test_every` | 8 | Every 8th frame is held out for PSNR/SSIM and not trained on. |
| `--max_triangles` | 4,000,000 | Upper limit on the number of triangles. |
| `--outdoor` | off | Official setting for unbounded outdoor scenes. |
| `--no_dome` | off | Don't add the ring of far-away "sky" triangles around the scene. |
| `--data_device` | cpu | Where training images are kept. `cuda` is slightly faster but uses more GPU memory. |
| `--seed` | 0 | Random seed. |
| `--out` | `outputs/triangle_splat/<scene>_<timestamp>/` | Output folder. Each run gets a new folder. |

Speed on the RX 9070 XT: about 58 ms/step at full resolution at the start (slows as the triangle count grows).
A quick test is `--steps 3000 --downscale 4` (~1 min).

The hyperparameters and training loop follow the official `train.py`. Differences: LPIPS and TensorBoard are left
out, and the evaluation uses the same PSNR/SSIM as `gs_pipeline.py` so the numbers are comparable.

## 4. Results

Each run folder contains:

- `metrics.txt`: PSNR / SSIM on the held-out frames, plus the triangle count.
- `renders/`: each held-out frame as *render | ground truth*. Look at these first.
- `triangles.ply`: the triangles as a coloured mesh with per-triangle opacity in the alpha channel. Opens in MeshLab
  or Blender. Colours are the base colour only, so it looks flatter than the renders (which include view-dependent
  colour) and it is a "triangle soup" (unconnected triangles), not a clean mesh.
- `point_cloud/iteration_<N>/`: the model in the official format, plus `cfg_args` and `cameras.json`. Works with the
  official tools, e.g. `python ~/triangle-splat-rocm/src/render.py -m <run folder>`.
- `args.json`: exact settings of the run.

The model files are large (hundreds of MB at full resolution) and are git-ignored (`outputs/**/*.pt`,
`outputs/**/*.ply`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Triangle rasterizer not available` | Activate the venv (`source ~/triangle-splat-rocm/venv/bin/activate`) or run the build script. |
| `RASTERIZER WRONG` from the check | Do not train with that build. See the build log in `~/triangle-splat-rocm/build.log`. |
| `No COLMAP model in ...` | Run `src/gaussian_splat/gs_pipeline.py sfm` on the scene first. |
| Out of GPU memory | Use `--downscale 2` or lower `--max_triangles`. |
| Build fails | Check that torch is still `2.11.0+rocm7.13.0`; the ROCm SDK version must match it. |

## How the AMD build works

The official rasterizer is CUDA code. PyTorch converts it to AMD's HIP automatically (hipify) when compiling, but
a few CUDA-only things need small fixes, which the build script applies to the pinned source (commit `685ff27`):

1. Removes two CUDA-only headers (`device_launch_parameters.h`, `cooperative_groups/reduce.h`) that have no HIP
   counterpart and aren't used.
2. Writes kernel launches as `<<<...>>>` instead of `<< <...>> >` (nvcc accepts the spaces, the AMD compiler does not).
3. Replaces CUDA's `__trap()` with `__builtin_trap()` (only reached on an internal error).
4. Replaces the nvcc flag `--use_fast_math` with `-ffp-contract=fast`.

Unlike gsplat, no changes to the rasterizer's maths or GPU reductions were needed: the code uses only per-thread
work and atomic adds, which behave the same on AMD.

### Verification

`verify_triangle_rasterizer.py` runs automatically after every build. It compares the compiled rasterizer with an
independent pure-PyTorch re-implementation of the official forward pass:

| Check | Result |
|---|---|
| Rendered image vs reference | 98.0 dB PSNR |
| Gradients (colour, opacity, sigma, vertices) vs reference | cosine 1.00000, norm ratio 1.0000 |
| Finite differences (colour, opacity, sigma) | worst error 1% |

Note on vertex gradients: the official backward pass is **not the exact derivative** of its forward pass. It leaves
out the pixel-scale factor (W/2, H/2) of the NDC → pixel conversion and treats the incircle normalisation as a
constant. That is how the method is defined and trained (the Adam optimizer is insensitive to the overall scale),
so the check compares against a reference with the same two choices rather than against finite differences.
Against the exact derivative, the official vertex gradient points in nearly the same direction (cosine 0.96) but is
about 64× smaller. This is a property of the official method, not of our port.
