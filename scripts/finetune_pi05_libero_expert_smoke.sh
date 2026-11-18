#!/usr/bin/env bash
# Smoke test: can pi0.5 be fine-tuned on our RX 9070 XT (17 GB)? Expert-only fine-tune (vision-language model
# frozen, only the action expert + projections train) of lerobot/pi05_libero_base on lerobot/libero, following the
# "VLM frozen" command in third_party/lerobot/docs/source/pi05.mdx, with a small batch and few steps.
# Measures memory/speed; does not save a checkpoint.
#
# Measured 2026-11-18 (RX 9070 XT, 17 GB):
#   batch 4:  1.43 s/step (2.8 samples/s), loss 2.10 -> 0.97 in 100 steps
#   batch 8:  3.3 s/step  (2.4 samples/s), peak 14.0 GB
#   batch 16: fails at step 1 with "hipErrorInvalidValue" (out of memory)
#
# Usage:  bash scripts/finetune_pi05_libero_expert_smoke.sh [batch_size] [steps]     (defaults: 4, 100)
# Output: outputs/train/pi05_libero_expert_smoke_bs<B>_<timestamp>/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BATCH="${1:-4}"
STEPS="${2:-100}"

eval "$(conda shell.bash hook)"
conda activate lerobot   # sets TORCH_BLAS_PREFER_HIPBLASLT=0, MUJOCO_GL=egl, GALLIUM_DRIVER=d3d12

# compile_model=false: torch.compile max-autotune spends very long benchmarking kernels on our GPU (see
# eval_pi05_libero_smoke.sh). PYTORCH_HIP_ALLOC_CONF reduces memory fragmentation.
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id=lerobot/libero \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero_base \
  --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
  --policy.n_action_steps=10 \
  --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.compile_model=false \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --save_checkpoint=false \
  --batch_size="$BATCH" \
  --num_workers=4 \
  --steps="$STEPS" \
  --log_freq=10 \
  --seed=1000 \
  --job_name="pi05_libero_expert_smoke_bs$BATCH" \
  --output_dir="outputs/train/pi05_libero_expert_smoke_bs${BATCH}_$(date +%Y%m%d-%H%M%S)"
