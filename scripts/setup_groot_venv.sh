#!/usr/bin/env bash
# Isolated venv for GR00T N1.7 (LeRobot policy.type=groot) on our AMD machine.
#
# - Built on top of the lerobot conda env's Python (--system-site-packages): it sees lerobot, the ROCm torch and
#   everything else, and installs only GR00T's extra packages (peft, diffusers, timm, dm-tree, decord) into itself.
#   The lerobot env is not changed. Undo: rm -rf ~/groot-rocm
# - Every package already in the lerobot env is pinned to its current version, so pip cannot replace torch,
#   transformers, numpy, ... (version ranges from third_party/lerobot/pyproject.toml, extra "groot").
# - flash-attn is not installed: it is optional in LeRobot's GR00T N1.7 (falls back to PyTorch SDPA attention).
#
# Usage: bash scripts/setup_groot_venv.sh     then: source ~/groot-rocm/venv/bin/activate
set -euo pipefail

ROOT="${GROOT_DIR:-$HOME/groot-rocm}"
BASE_PY="$HOME/miniforge3/envs/lerobot/bin/python"

mkdir -p "$ROOT"
[ -d "$ROOT/venv" ] || "$BASE_PY" -m venv --system-site-packages "$ROOT/venv"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

# constraints = everything currently installed in lerobot (except editable installs) + the ROCm torch pins
"$BASE_PY" -m pip freeze --exclude-editable | grep -vE "^-e |@ file://" > "$ROOT/constraints.txt"
cat "$HOME/rocm-torch.txt" >> "$ROOT/constraints.txt"

pip install -c "$ROOT/constraints.txt" \
  "peft>=0.18.0,<1.0.0" "timm>=1.0.0,<1.1.0" "dm-tree>=0.1.8,<1.0.0" "decord>=0.6.0,<1.0.0"
# diffusers 0.38-0.39 (LeRobot's range) declare huggingface-hub<2.0, but LeRobot itself needs huggingface-hub>=1.31
# and the env has 2.4.0, so pip cannot resolve it. GR00T only uses diffusers' model building blocks (attention,
# feed-forward, embeddings), so install it without its dependency check and verify the import below.
pip install --no-deps -c "$ROOT/constraints.txt" "diffusers==0.39.0"

python - <<'EOF'
import sys, torch, transformers, peft, diffusers, timm, tree, decord, lerobot
assert sys.prefix.endswith("venv"), sys.prefix
print("venv OK | torch", torch.__version__, "| GPU", torch.cuda.is_available(), "| transformers",
      transformers.__version__, "| peft", peft.__version__, "| diffusers", diffusers.__version__, "| timm",
      timm.__version__, "| decord", decord.__version__)
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: F401
print("lerobot GR00T policy importable")
EOF
