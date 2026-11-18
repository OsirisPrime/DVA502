# splat_pipeline

Gaussian splatting, production version: video/photos → COLMAP/GLOMAP → gsplat → standard 3DGS `.ply`, with YAML
configs, checkpoint/resume, large-dataset support and tests.

**Full guide: [docs/splat_pipeline.md](../../docs/splat_pipeline.md)**

Quick start (AMD machine, from the repo root, after the setup in the guide):

```bash
conda activate lerobot && source ~/gsplat-rocm/venv/bin/activate
pip install --no-deps -e src/splat_pipeline -c ~/rocm-torch.txt     # once

S=local_data/scenes/my_scene
splat-pipeline run --scene $S --video local_data/videos/my_scene.MOV --mapper glomap
```

Tests: `cd src/splat_pipeline && python -m pytest`.
