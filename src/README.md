# src

| Folder | What | Guide |
|---|---|---|
| `gaussian_splat/` | Research version of the Gaussian splatting pipeline: `gs_pipeline.py` (frames → COLMAP/GLOMAP → gsplat), the Colab notebook, and `rocm/` (gsplat build for the RX 9070 XT) | [docs/gaussian_splatting.md](../docs/gaussian_splatting.md) |
| `triangle_splat/` | Triangle splatting on the same scenes (official code, built for our AMD GPU). Non-commercial licence. | [docs/triangle_splatting.md](../docs/triangle_splatting.md) |
| `splat_pipeline/` | Production version of the Gaussian pipeline: package, YAML configs, checkpoint/resume, large datasets, tests | [docs/splat_pipeline.md](../docs/splat_pipeline.md) |

Data (videos, scenes) goes in `local_data/` (git-ignored) and training results in `outputs/` (the large `.ply`/`.pt` files are git-ignored).
