#!/usr/bin/env bash
# Smoke test: evaluate NVIDIA's published GR00T N1.7 LIBERO policy (nvidia/gr00t17-lerobot-libero_object-640,
# 100% on LIBERO Object in third_party/lerobot/docs/source/groot.mdx) on a few episodes on our GPU.
# Same task as eval_pi05_libero_smoke.sh for comparison. Needs the venv from scripts/setup_groot_venv.sh.
#
# Usage:  bash scripts/eval_groot_libero_smoke.sh [suite] [task_ids] [episodes] [model]
#   defaults: libero_object, [0], 3, nvidia/gr00t17-lerobot-libero_object-640
# Output: outputs/eval/groot_libero_smoke_<timestamp>/
#
# rename_map: the simulator calls the wrist camera "image2", NVIDIA's checkpoint expects "wrist_image" (the eval
# command in groot.mdx leaves this out and fails with "Checkpoint feature missing from the simulator").
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUITE="${1:-libero_object}"
TASK_IDS="${2:-[0]}"
EPISODES="${3:-3}"
MODEL="${4:-nvidia/gr00t17-lerobot-libero_object-640}"

eval "$(conda shell.bash hook)"
conda activate lerobot   # env vars: TORCH_BLAS_PREFER_HIPBLASLT=0, MUJOCO_GL=egl, GALLIUM_DRIVER=d3d12
# shellcheck disable=SC1091
source "$HOME/groot-rocm/venv/bin/activate"

# NVIDIA's checkpoint stores image shapes channels-last (256, 256, 3); LeRobot's pre-eval check reads shape[0] as
# the channel count and rejects it. GR00T resizes images itself, so the shape is only metadata: make a local copy
# (weights symlinked from the HF cache, nothing duplicated) whose config.json lists the images as (3, 256, 256),
# and whose base_model_path is the Hub name instead of a path on NVIDIA's training cluster.
LOCAL_MODEL=$(python - "$MODEL" <<'EOF'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
src = Path(snapshot_download(sys.argv[1]))
dst = Path("local_data/models") / (sys.argv[1].replace("/", "--") + "--chw")
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    link = dst / f.name
    if f.name != "config.json" and not link.exists():
        link.symlink_to(f.resolve())
cfg = json.loads((src / "config.json").read_text())
for feat in cfg.get("input_features", {}).values():
    s = feat.get("shape")
    if feat.get("type") == "VISUAL" and len(s) == 3 and s[2] in (1, 3) and s[0] not in (1, 3):
        feat["shape"] = [s[2], s[0], s[1]]
# base_model_path is a path on NVIDIA's training cluster (/cache/huggingface/models--nvidia--GR00T-N1.7-3B/...)
bmp = cfg.get("base_model_path", "")
if bmp.startswith("/") and not Path(bmp).exists():
    cfg["base_model_path"] = "nvidia/GR00T-N1.7-3B"
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
print(dst)
EOF
)
echo "policy: $LOCAL_MODEL"

# Run via the venv's python: the `lerobot-eval` launcher belongs to the conda env and would bypass this venv.
python -m lerobot.scripts.lerobot_eval \
  --policy.path="$LOCAL_MODEL" \
  --policy.embodiment_tag=libero_sim \
  --rename_map='{"observation.images.image2": "observation.images.wrist_image"}' \
  --policy.device=cuda \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --env.max_parallel_tasks=1 \
  --output_dir="outputs/eval/groot_libero_smoke_$(date +%Y%m%d-%H%M%S)"
