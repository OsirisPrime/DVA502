# Gaussian splatting from a phone video (AMD GPU, local)

How to go from a video of a real scene to a trained 3D Gaussian splatting (3DGS) model on our machine
(WSL2, RX 9070 XT / gfx1201, ROCm PyTorch). Training uses [gsplat](https://github.com/nerfstudio-project/gsplat)
built for our GPU; see [How the AMD build works](#how-the-amd-build-works).

```
video ──frames──▶ images/ ──sfm (COLMAP or GLOMAP)──▶ camera poses + sparse points ──train (gsplat)──▶ point_cloud.ply
```

Reference result: `img_1453` (stool + controller, 132 s iPhone video, 265 frames) trained at full
1080×1920 for 30k steps in 33 min: **PSNR 28.8 dB, SSIM 0.935** on 34 held-out frames, 1.42M Gaussians.

The code is in `src/gaussian_splat/`: `gs_pipeline.py` (frames, sfm, train), `gsplat.ipynb` (Colab) and `rocm/`
(the AMD build). This guide uses that research version. For large datasets, checkpoint/resume and team use there is
a packaged version with the same stages in `src/splat_pipeline/` (see [splat_pipeline.md](splat_pipeline.md)).

## 1. One-time setup

1. **LeRobot conda env** with the ROCm PyTorch and `~/rocm-torch.txt`, as in the [README](../README.md#setup).
2. **COLMAP 4.2.1** (computes camera poses on the CPU; includes GLOMAP), installed into the `lerobot` env. The pins keep
   LeRobot working: COLMAP 4.2.1 needs ffmpeg ≥ 8.1, torchcodec supports ffmpeg 4–9, and numpy must stay at the
   pip version. (`sudo apt install colmap` gives 3.9, which works with `--mapper colmap` only.)
   ```bash
   conda install -n lerobot -c conda-forge "colmap=4.2.1=cpu*" "ffmpeg=8.1" "numpy=2.2.6" "libopenimageio=3.1"
   ```
   The scripts use this COLMAP automatically, also from the gsplat venv. `--colmap PATH` or `$COLMAP_BIN`
   overrides it.
3. **gsplat for the AMD GPU** (about 10 min; installs into its own venv in `~/gsplat-rocm`, the `lerobot` env is not touched):
   ```bash
   bash src/gaussian_splat/rocm/build_gsplat_rocm.sh
   ```
   It ends with a gradient check that must print `GRADIENTS OK`. Undo everything with `rm -rf ~/gsplat-rocm`.

Activate the environment before every step below. The venv sees all `lerobot` packages and adds gsplat. Activating
`lerobot` first puts COLMAP 4.2.1 on the PATH for the manual `colmap` commands below; `/usr/bin/colmap`, if
installed, is the older 3.9.
```bash
conda activate lerobot
source ~/gsplat-rocm/venv/bin/activate
cd ~/projects/DVA502
```

## 2. Film the scene

The result depends more on the video than on any training setting. COLMAP needs texture it can track across frames.

- **Texture:** textured surroundings (wood, rugs, books, printed paper). A plain object on a plain table fails:
  our first try (white mug, grey desk) only got camera poses for 15% of the frames.
- **Light:** bright and even. Dim light means blur and noise.
- **Motion:** walk slowly around the scene at 40–60 cm or more, in 2–3 loops at different heights. Keep the
  surroundings in view. No fast pans.
- **Static scene:** no people, pets, moving shadows or screens with changing content in the frame.
- **iPhone settings:** stay on the 1× lens (no automatic lens switching; turn off *Macro Control*), use AE/AF lock
  (long-press the screen), and preferably turn off *HDR Video*. 4K/30 fps is fine; `.MOV` (HEVC or H.264) needs no conversion.
- **Length:** 1–2 minutes is plenty.

Put the video in `local_data/videos/` (git-ignored).

## 3. Extract frames

Keeps the sharpest frame in every 1/`fps`-second window and resizes so the longest side is 1920 px.

```bash
python src/gaussian_splat/gs_pipeline.py frames --video local_data/videos/IMG_1453.MOV --out local_data/scenes/img_1453 --fps 2
```

Aim for **150–300 frames** (`--fps 2` for a ~2 min video). More frames make COLMAP much slower without improving the result.

## 4. Camera poses (COLMAP or GLOMAP)

```bash
python src/gaussian_splat/gs_pipeline.py sfm --data local_data/scenes/img_1453
```

- Takes tens of minutes on the CPU for ~250 frames.
- Uses exhaustive matching (every frame against every other), which is robust. Add `--sequential` only for very
  long videos (>400 frames). It is faster but split our first video into disconnected pieces.

**Mapper choice** (`--mapper`). Feature extraction and matching are always done by COLMAP; only the step that
computes the poses from the matches differs:

| | `--mapper colmap` (default) | `--mapper glomap` |
|---|---|---|
| Method | incremental: adds one image at a time | global: solves all cameras at once |
| Speed | slow on many images (grows quickly with the number of frames) | 10–100× faster mapping |
| Robustness | most robust, the reference | slightly less robust on difficult scenes |
| Output folder | `sparse/` | `sparse_glomap/` (so both can be kept) |

```bash
python src/gaussian_splat/gs_pipeline.py sfm --data local_data/scenes/img_1453                              # COLMAP mapper
python src/gaussian_splat/gs_pipeline.py sfm --data local_data/scenes/img_1453 --sequential --mapper glomap # GLOMAP mapper
```

**Which to use:** start with GLOMAP for videos (much faster, and as good on our scenes). Use the COLMAP mapper if
GLOMAP registers clearly fewer than ~80% of the frames or splits the scene, or as the reference. If frames are
missing with either one, drop `--sequential` so every frame is matched against every other.

On `img_1453` with sequential matching, GLOMAP registered all 265 frames (about 63,800 points) and its mapping
step took about 1.5 min. Feature extraction and matching took 3–5 min. Sequential matching worked here together
with GLOMAP; if frames are missing, drop `--sequential` (exhaustive matching).

Both `train` scripts (Gaussian and triangle) read the model from `sparse_model.txt`, so they use whichever mapper
ran last. Run `sfm` again with the other `--mapper` to switch. All steps use the same COLMAP
binary, because the database format changed in COLMAP 3.12.

**Check the result before training:**
```bash
S=local_data/scenes/img_1453
colmap model_analyzer --path $S/$(cat $S/sparse_model.txt)
```

| | good (img_1453) | bad (first try) |
|---|---|---|
| Registered images | 265 / 265 | 41 / 282 |
| Points | 68,300 | 1,802 |
| Mean reprojection error | 0.84 px | 0.87 px |

- `sfm` prints `(N model(s) found)` at the end. More than one means COLMAP split the video and only the largest
  piece is used.
- If under ~80% of frames are registered, **re-film**. Training cannot fix bad poses.

## 5. Train

```bash
python src/gaussian_splat/gs_pipeline.py train --data local_data/scenes/img_1453 --steps 30000 --downscale 1
```

| Option | Default | Meaning |
|---|---|---|
| `--steps` | 30000 | Standard 3DGS schedule. Densification runs until 15k, then refinement. |
| `--downscale` | 1 | 1 = full resolution (1080×1920 frames). Use 2 if memory runs out or for quick tests. |
| `--sh_degree` | 3 | View-dependent colour detail. |
| `--test_every` | 8 | Every 8th frame is held out for PSNR/SSIM and not trained on. |
| `--out` | `outputs/gsplat/<scene>_<timestamp>/` | Output folder. Each run gets a new folder. |

Speed on the RX 9070 XT: ~50 ms/step at the start and ~70 ms/step once the scene has >1M Gaussians.
A quick test is `--steps 5000 --downscale 2`.

## 6. Results

Each run folder in `outputs/gsplat/` contains:

- `metrics.txt`: PSNR / SSIM on the held-out frames. Above ~25 dB is good for a real scene.
- `renders/`: each held-out frame as *render | ground truth*. Look at these first.
- `point_cloud.ply`: the 3D scene in the standard 3DGS format, a few hundred MB.
- `args.json`: exact settings of the run.

**View in 3D:** open https://superspl.at/editor and drag `point_cloud.ply` into it. The file is loaded in the
browser and only uploaded if you press Publish. From WSL, `explorer.exe outputs/gsplat` opens the folder in
Windows. The scene may appear tilted (COLMAP has no notion of "up"); rotate it in the editor. It looks best from
viewpoints near where the camera went.

## Alternatives

- **Google Colab (NVIDIA GPU, free tier):** `src/gaussian_splat/gsplat.ipynb`. Upload the frames (`images/`), the
  three `.txt` files of the model named in `sparse_model.txt` (e.g. `sparse/0/` or `sparse_glomap/0/`) and
  `gs_pipeline.py` to Google Drive, open the notebook in VS Code with the *Colab* extension,
  select a GPU runtime and run the cells.
- **Production version** (`src/splat_pipeline/`): same stages as a Python package, with YAML configs,
  checkpoint/resume, images streamed from disk for large datasets, the MCMC densification strategy and tests.
  See [splat_pipeline.md](splat_pipeline.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| COLMAP registers few frames / several models | Re-film with more texture, light and overlap. Use exhaustive matching (no `--sequential`). |
| `CUDA out of memory` while training | `--downscale 2`. |
| `GLOMAP not found` | Install COLMAP 4 into the lerobot env (step 1), or pass `--colmap PATH`. |
| Build script fails | Full log in `~/gsplat-rocm/build.log`. Check that torch is still `2.11.0+rocm7.13.0`: the SDK version in the script must match it. |
| `GRADIENTS WRONG` from the check | Do not train with that build. See the next section. |
| Washed-out colours | Video was HDR. Film with *HDR Video* off. |

## How the AMD build works

`src/gaussian_splat/rocm/build_gsplat_rocm.sh` builds [AMD's ROCm port of gsplat](https://github.com/ROCm/gsplat) at a pinned
commit and applies `gsplat_rocm_gfx1201.patch` (same folder). AMD's port targets Instinct datacenter GPUs, which run
threads in groups (*wavefronts*) of 64. Our RDNA4 card defaults to 32, so the port does not work out of the box.
The patch:

1. **Build settings:** compiles for gfx1201 in 64-wide mode (`-mwavefrontsize64`). It tells rocprim the wave size
   (`ROCPRIM_NAVI=0`) and uses its shuffle fallback, because RDNA4 lacks the DPP broadcast instructions
   (`ROCPRIM_DISABLE_DPP=1`). It also uses the ROCm 7.13 developer SDK that matches our PyTorch.
2. **PyTorch 2.11 API:** the GPU caching allocator kept its CUDA name in the ROCm headers.
3. **Sorting:** hipcub's device-wide radix sort picks its kernel at runtime from the GPU's default wave size (32),
   which does not exist in a wave64 build. It is replaced by PyTorch's stable sort, which gives the same order.
4. **Gradient reductions (important):** the backward kernels summed gradients with rocprim warp reductions that
   were given no scratch memory and only return the result in lane 0. On our GPU this silently lost 30–90% of
   every gradient while the rendered images were still correct. They are replaced by shuffle-based reductions that
   need no scratch memory and give every lane the result, like the original CUDA code.

The fix was checked in two ways. Forward renders match an independent pure-PyTorch renderer to 100 dB PSNR.
Gradients match that renderer (cosine similarity 1.0000) and finite differences.
`src/gaussian_splat/rocm/verify_gsplat_rocm.py` repeats the finite-difference check and runs automatically after every build.
