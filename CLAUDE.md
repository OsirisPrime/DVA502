# Project: DVA502 master thesis (ABB): synthetic data fidelity for VLA policies on ABB GoFa

- LeRobot is a git submodule in third_party/lerobot, pinned to a tested commit.
  Never edit files there; update it only deliberately with a separate commit. Work around upstream issues in
  scripts/ or src/ instead.
- Policies: VLA models (pi0.5, GR00T N1.7). Evaluated in LIBERO simulation; GoFa model/env not built yet.

## Layout
- src/: gaussian_splat/ (research 3DGS pipeline + rocm/ gsplat build), triangle_splat/ (triangle splatting,
  non-commercial licence), splat_pipeline/ (production 3DGS package with tests). Guides in docs/.
- scripts/: exact commands of every experiment (one script per run type, measured results in its header).
  Run LeRobot commands with an explicit --output_dir=outputs/... from the repo root.
- outputs/: results. Committed, except large files (*.ply, *.pt, *.safetensors are git-ignored; GitHub limit
  100 MB). Never commit checkpoints.
- local_data/ (git-ignored): videos, scenes, local model copies, env backups.

## Local AMD machine (RX 9070 XT, ROCm)
- Separate venvs on top of the lerobot conda env's Python (--system-site-packages): ~/gsplat-rocm (gsplat),
  ~/triangle-splat-rocm, ~/groot-rocm (GR00T extras, scripts/setup_groot_venv.sh). Install new or risky
  dependencies into a venv like these first, not into lerobot.
- In a venv, start LeRobot as `python -m lerobot.scripts.lerobot_train` / `lerobot_eval`: the `lerobot-*`
  launchers use the conda Python and bypass the venv.
- Use --policy.compile_model=false (torch.compile max-autotune hangs for >13 min on this GPU).
- pi0.5 training fits only expert-only (VLM frozen) at batch <= 8 (14 GB); full fine-tunes need a bigger GPU.
- COLMAP 4.2.1 (with GLOMAP) is in the lerobot env, installed with pins; see docs/gaussian_splatting.md.
- Checks after environment changes: src/gaussian_splat/rocm/verify_gsplat_rocm.py,
  src/triangle_splat/verify_triangle_rasterizer.py, pytest in src/splat_pipeline, and the LIBERO smoke evals
  scripts/eval_pi05_libero_smoke.sh and scripts/eval_groot_libero_smoke.sh.
