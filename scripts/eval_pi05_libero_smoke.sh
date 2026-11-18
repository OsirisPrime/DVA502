#!/usr/bin/env bash
# Smoke test: evaluate the published pi0.5 LIBERO policy (lerobot/pi05_libero_finetuned, 97.5% average in the
# LeRobot docs) on a few LIBERO episodes on our GPU. Checks model download, bf16 inference on ROCm and MuJoCo
# simulation before we train anything ourselves.
#
# Usage (from anywhere):  bash scripts/eval_pi05_libero_smoke.sh [suite] [task_ids] [episodes]
#   defaults: libero_object, task [0], 3 episodes
# Output: outputs/eval/pi05_libero_smoke_<timestamp>/ (eval_info.json, videos/)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUITE="${1:-libero_object}"
TASK_IDS="${2:-[0]}"
EPISODES="${3:-3}"

eval "$(conda shell.bash hook)"
conda activate lerobot   # sets TORCH_BLAS_PREFER_HIPBLASLT=0, MUJOCO_GL=egl, GALLIUM_DRIVER=d3d12

# n_action_steps=10 matches OpenPI and the LeRobot reproduction (docs/source/libero.mdx).
# compile_model=false: the published config uses torch.compile max-autotune, which on our GPU spent >13 min
# benchmarking kernels before the first step (many candidates need more shared memory than RDNA4's 64 KB).
lerobot-eval \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --policy.compile_model=false \
  --policy.device=cuda \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.task_ids="$TASK_IDS" \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --env.max_parallel_tasks=1 \
  --output_dir="outputs/eval/pi05_libero_smoke_$(date +%Y%m%d-%H%M%S)"
