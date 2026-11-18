# Gaussian splatting, production version (`src/splat_pipeline`)

Video or photos → COLMAP/GLOMAP → [gsplat](https://github.com/nerfstudio-project/gsplat) → standard 3DGS `.ply`, as an
installable Python package. It is meant for long runs, large datasets and use by several people.

```
video ──frames──▶ images/ ──sfm (COLMAP or GLOMAP)──▶ scene.json + model ──train (gsplat)──▶ run folder
                                                                            (checkpoints, .ply, metrics, renders)
```

The maths, defaults, train/test split and PSNR/SSIM are the same as in the research script
`src/gaussian_splat/gs_pipeline.py` ([gaussian_splatting.md](gaussian_splatting.md)), so results can be compared.
Filming advice and the AMD gsplat build are described there and not repeated here.

## When to use which

| | research `gs_pipeline.py` | production `splat_pipeline` |
|---|---|---|
| Use for | quick experiments, Colab, changing the method | long runs, many images, reproducible comparisons, handing over |
| Form | one file | package with modules and tests |
| Settings | command-line flags | typed defaults + YAML files + `--set key=value`; resolved config saved per run |
| Images | all in RAM | in RAM if they fit (`cache_limit_gb`), else streamed from disk by worker processes |
| Long runs | no checkpoints | checkpoints every N steps, `--resume`, Ctrl-C saves a checkpoint |
| Densification | gsplat default | default, or MCMC (fixed number of Gaussians → predictable GPU memory) |
| COLMAP models | text | binary or text; distorted cameras (OPENCV) are undistorted |
| Logs | console | `train.log`, `metrics.jsonl`, `run.json` (git commit, versions, GPU) |

Triangle splatting is not included: its official code is licensed for non-commercial research only.

## 1. One-time setup

1. The `lerobot` env with COLMAP 4.2.1, and gsplat built for the AMD GPU: steps 1–3 of
   [gaussian_splatting.md](gaussian_splatting.md#1-one-time-setup).
2. Install the package into the gsplat venv. `--no-deps` because everything it needs (numpy, scipy, OpenCV, PyYAML,
   tqdm) is already there, and pip must never touch the ROCm torch:
   ```bash
   conda activate lerobot
   source ~/gsplat-rocm/venv/bin/activate
   cd ~/projects/DVA502
   pip install --no-deps -e src/splat_pipeline -c ~/rocm-torch.txt
   ```
   This adds the `splat-pipeline` command (`python -m splat_pipeline` does the same). `-e` means code changes
   take effect without reinstalling. On an NVIDIA machine: `pip install torch gsplat`, then
   `pip install -e src/splat_pipeline`.

Check: `splat-pipeline --help`.

## 2. Run

Activate as above (`conda activate lerobot`, `source ~/gsplat-rocm/venv/bin/activate`) and work from the repo root.

```bash
S=local_data/scenes/my_scene
splat-pipeline frames --scene $S --video local_data/videos/my_scene.MOV --fps 2   # 1. frames
splat-pipeline sfm    --scene $S --mapper glomap                                  # 2. camera poses
splat-pipeline train  --scene $S --steps 30000                                    # 3. train

# or all three in one go
splat-pipeline run --scene $S --video local_data/videos/my_scene.MOV --mapper glomap
```

A scene prepared with the research script (`gs_pipeline.py sfm`) can be trained directly with `train`.

| Command | What it does |
|---|---|
| `frames --scene S --video V` | keeps the sharpest frame in every 1/fps-second window → `S/images/` |
| `sfm --scene S` | COLMAP features + matching, then the COLMAP or GLOMAP mapper → model + `S/scene.json` |
| `train --scene S` | trains into a new run folder `outputs/splat_pipeline/<scene>_<timestamp>/` (or `--out`) |
| `train --scene S --resume RUN` | continues RUN from its latest checkpoint with its saved settings |
| `export RUN` | writes a `.ply` from the latest checkpoint, e.g. to look at an unfinished run |
| `eval RUN` | re-evaluates the latest checkpoint on the held-out frames |
| `run --scene S [--video V]` | frames (if a video is given) + sfm + train |

**Stopping and resuming:** press Ctrl-C at any time. A checkpoint is saved and the exact resume command is printed:
```bash
splat-pipeline train --scene $S --resume outputs/splat_pipeline/my_scene_<timestamp>
```

### Camera poses: COLMAP or GLOMAP

`sfm` always uses COLMAP for the first two steps (finding features in every frame and matching them between
frames). Only the last step, which computes the camera poses from the matches, can be done by either COLMAP or
GLOMAP. Choose with `--mapper`:

```bash
splat-pipeline sfm --scene $S --mapper colmap    # COLMAP's incremental mapper (default)
splat-pipeline sfm --scene $S --mapper glomap    # GLOMAP's global mapper
```

| | `--mapper colmap` (default) | `--mapper glomap` |
|---|---|---|
| How it works | incremental: starts from two frames and adds one frame at a time | global: solves all cameras at once |
| Speed of the mapping step | slow, grows quickly with the number of frames | much faster (1–2 min for ~300 frames) |
| Robustness | the reference; most robust on difficult footage | slightly less robust when frames overlap little |
| Output folder | `sparse/` | `sparse_glomap/` |

**Which to use:**
- Start with **GLOMAP** for videos. It is fast, and on our scenes it registered as many frames as COLMAP.
- Use **COLMAP** if GLOMAP registers clearly fewer than ~80% of the frames or splits the scene into several large
  models, or as the reference to compare against.
- For either one, if frames are missing, switch the matching from `sequential` (neighbouring frames only, default)
  to `exhaustive` (every frame against every other), which is slower but finds more connections:
  `splat-pipeline sfm --scene $S --mapper glomap --matcher exhaustive`.

GLOMAP needs COLMAP 3.13 or newer, which has it built in as `colmap global_mapper`. Our `lerobot` env has
COLMAP 4.2.1 (setup step 1); `/usr/bin/colmap` from apt (3.9) only supports `--mapper colmap`.

**Check the result before training:** `sfm` ends with a line like
```
glomap: registered 276/292 images, 29847 points, 2 model(s); features+matching 111 s, mapping 54 s
```
The same numbers are in `$S/scene.json`.

| Check | Good | What to do otherwise |
|---|---|---|
| Registered images | above ~80% of the frames | try `--matcher exhaustive`, the other mapper, or re-film (slower, more texture, more light) |
| Models | 1, or one large model plus a few tiny ones | several large models means the video was split; only the largest is used. Re-film with more overlap |

Missing frames are often blurry ones (fast camera motion), and dropping them is fine: training only uses
registered frames. Example, `img_1477` with GLOMAP: 276/292 registered. The 16 missing frames were all blurry
(sharpness well below the median), and the second model held 5 of them, so it could be ignored.

**Switching between the two:** each mapper keeps its own output folder, but `train` uses the model of the *latest*
`sfm` run (recorded in `scene.json`). To train on the other one, run `sfm` again with that `--mapper`. Every `sfm`
run starts from scratch: it recreates `database.db` and repeats feature extraction and matching. Each run's
complete COLMAP output is in `$S/sfm_<mapper>.log`.

## 3. Settings

All settings, with defaults and comments, are in
[`config.py`](../src/splat_pipeline/splat_pipeline/config.py), in three sections: `frames`, `sfm` and `train`. Later
sources override earlier ones:

```
defaults (config.py)  →  --config my.yaml  →  --set section.key=value  →  --steps / --mapper / ... flags
```

```bash
splat-pipeline train --scene $S --config src/splat_pipeline/configs/large_scene.yaml --set train.cap_max=2000000
```

Unknown keys and wrong types are rejected (`--set train.stepz=1` and `--set train.steps=many` both fail). Each run saves
its resolved settings as `config.yaml`; `--config <run>/config.yaml` repeats a run.

The most useful settings:

| Key | Default | Meaning |
|---|---|---|
| `sfm.mapper` | `colmap` | `colmap` (incremental, most robust) or `glomap` (global, much faster on many images); see [Camera poses](#camera-poses-colmap-or-glomap) |
| `sfm.matcher` | `sequential` | `sequential` (video), `exhaustive` (<~500 photos, most robust), `vocab_tree` (large photo sets, needs `vocab_tree_path`) |
| `sfm.camera_model` | `PINHOLE` | `OPENCV` estimates lens distortion and undistorts the images (wide-angle cameras) |
| `sfm.colmap` | `colmap` | COLMAP executable; `colmap` = `$COLMAP_BIN`, else the conda env's COLMAP, else PATH |
| `train.steps` | 30000 | |
| `train.downscale` | 1 | image downscale factor |
| `train.test_every` | 8 | every N-th frame held out for PSNR/SSIM (0 = train on all) |
| `train.strategy` | `default` | `default` (grow/prune) or `mcmc` (fixed budget of `cap_max` Gaussians) |
| `train.densify_until` | -1 | fraction of the run with densification (-1 = 0.5 for default, 25/30 for mcmc) |
| `train.packed` | false | packed rasterization: less GPU memory for large scenes |
| `train.cache` | `auto` | `ram`, `disk`, or `auto` (RAM if the decoded images fit in `cache_limit_gb` = 8) |
| `train.eval_every` | 0 | PSNR/SSIM on the held-out frames every N steps (0 = only at the end) |
| `train.checkpoint_every` | 5000 | |

Example files: [`configs/default.yaml`](../src/splat_pipeline/configs/default.yaml) and
[`configs/large_scene.yaml`](../src/splat_pipeline/configs/large_scene.yaml) (1000+ images or limited GPU memory:
GLOMAP, vocab-tree matching, MCMC, packed, images from disk).

Unlike the research script, densification stops after a *fraction* of the run. At 30k steps both stop at 15k; in a
short test run (e.g. 1000 steps) this version densifies only briefly.

## 4. Results

**Scene folder** (written by `frames` and `sfm`):

| Path | Content |
|---|---|
| `images/` | input frames |
| `database.db` | COLMAP features + matches, recreated on every `sfm` run |
| `sparse/<k>/`, `sparse_glomap/<k>/` | models from the COLMAP / GLOMAP mapper (that folder is cleared when its mapper reruns) |
| `undistorted_<mapper>/` | only with `camera_model: OPENCV`: undistorted images + PINHOLE model |
| `scene.json` | the model and image folder to train on, registered images, points, timings |
| `sparse_model.txt` | the same model, for the research scripts (PINHOLE only) |
| `sfm_<mapper>.log` | full COLMAP output |

`train` uses `scene.json`, else `sparse_model.txt`, else `sparse/0`.

**Run folder** (`outputs/splat_pipeline/<scene>_<timestamp>/`):

| Path | Content |
|---|---|
| `metrics.json` | PSNR/SSIM on the held-out frames, mean and per frame; number of Gaussians; training time |
| `renders/` | each held-out frame as *render \| ground truth*. Look at these first. |
| `point_cloud.ply` | the model in the standard 3DGS format: open in https://superspl.at/editor |
| `metrics.jsonl` | one line per 1000 steps: loss, Gaussians, peak GPU memory (+ PSNR/SSIM with `eval_every`) |
| `config.yaml`, `run.json` | exact settings; git commit, Python/torch/gsplat versions, GPU |
| `train.log` | full log |
| `checkpoints/ckpt_<step>.pt` | everything needed to resume; the newest 2 are kept, each about 3× the `.ply` size |

`.ply` and `.pt` files are git-ignored (GitHub's 100 MB limit).

## 5. Large datasets

- **SfM is usually the bottleneck.** Use `sequential` matching for video and `vocab_tree` for large unordered photo
  sets (exhaustive matching grows with the square of the number of images), plus `mapper: glomap`.
- **GPU memory:** `strategy: mcmc` with `cap_max` fixes the number of Gaussians; `packed: true` saves more;
  `downscale: 2` cuts memory by about 4×.
- **RAM:** with `cache: disk` (or automatically above `cache_limit_gb`), `num_workers` processes decode images while
  the GPU trains.

## Conventions (for maintainers)

- **Stages only communicate through the files above**, so any stage can be rerun, replaced or run on another
  machine.
- **Cameras:** the rasterizer has no lens distortion, so training accepts only PINHOLE/SIMPLE_PINHOLE. Other models
  give an error instead of being silently treated as pinhole.
- **Poses:** COLMAP stores world-to-camera rotation (quaternion w, x, y, z) and translation; used directly as gsplat
  `viewmats`.
- **Parameters:** log-scales, logit-opacities, SH band 0 in `sh0`; activations only when rasterizing. In the `.ply`,
  higher SH bands are channel-major (`f_rest_0..14` = red). Do not reorder; viewers depend on it.
- **Imports:** torch, gsplat and OpenCV are imported inside the commands, so `--help` is instant and `sfm` runs
  without a GPU.
- **Dependencies:** torch and gsplat are not listed in `pyproject.toml`, so installing the package can never replace
  the ROCm build.

Code layout (`src/splat_pipeline/splat_pipeline/`): `cli.py` (commands), `config.py` (settings), `frames.py`,
`sfm.py`, `colmap_io.py` (COLMAP readers), `data.py` (cameras + image loading), `trainer.py` (training,
checkpoints, evaluation), `metrics.py`, `ply.py`.

**Tests** (CPU only: COLMAP readers, `.ply` layout, config handling):
```bash
cd src/splat_pipeline && python -m pytest
```

## Verified

On `img_1453` (265 frames) on the RX 9070 XT:
- `frames`, and `sfm` with both mappers (GLOMAP: 265/265 registered);
- `train`, interrupt with Ctrl-C, `--resume`, `eval`, `export`;
- MCMC, packed rasterization and images streamed from disk.

PSNR matches the research script (24.36 vs 24.30 dB after 1k steps at quarter resolution).

Not yet verified:
- a dataset with 1000+ images;
- `camera_model: OPENCV` (undistortion);
- the gradients of MCMC and packed mode. They run on our AMD build, but the gradient check
  (`src/gaussian_splat/rocm/verify_gsplat_rocm.py`) only covers the default mode.
