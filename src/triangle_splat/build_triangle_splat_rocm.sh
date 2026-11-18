#!/usr/bin/env bash
# Build Triangle Splatting (Held et al., 3DV 2026) for the RX 9070 XT (gfx1201) in an ISOLATED venv.
#
# - Official code: https://github.com/trianglesplatting/triangle-splatting (pinned commit below).
# - The venv sits on top of the `lerobot` conda env's Python (--system-site-packages), so it uses the existing
#   ROCm PyTorch. Nothing is installed into the lerobot env. Everything lives in $TRIANGLE_SPLAT_DIR
#   (default ~/triangle-splat-rocm); delete that folder to undo it all.
# - The ROCm compiler + headers are reused from the gsplat build (~/gsplat-rocm) when present, otherwise
#   rocm-sdk-devel (~9 GB) is installed into this venv.
# - Only the triangle rasterizer is compiled. The `simple_knn` CUDA package is replaced by shims/simple_knn.
# - Ends with a correctness check (image + gradients against a pure-PyTorch reference) and fails if it does not pass.
#
# Usage:  bash src/triangle_splat/build_triangle_splat_rocm.sh
# Then:   source ~/triangle-splat-rocm/venv/bin/activate
set -euo pipefail

ROOT="${TRIANGLE_SPLAT_DIR:-$HOME/triangle-splat-rocm}"
BASE_PY="${BASE_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"  # Python with the ROCm torch
CONSTRAINTS="${ROCM_TORCH_CONSTRAINTS:-$HOME/rocm-torch.txt}"       # pins torch/rocm so pip never replaces them
TS_REPO="https://github.com/trianglesplatting/triangle-splatting.git"
TS_COMMIT="685ff270a8273afc3a2b094ae3148dd0b98e31e9"
ROCM_SDK_VERSION="7.13.0"                                           # must match torch's ROCm (torch 2.11.0+rocm7.13.0)
ROCM_INDEX="https://repo.amd.com/rocm/whl/gfx120X-all/"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Checking prerequisites"
[ -x "$BASE_PY" ] || { echo "Base Python not found: $BASE_PY (set BASE_PYTHON)"; exit 1; }
[ -f "$CONSTRAINTS" ] || { echo "Constraints file not found: $CONSTRAINTS (see README, AMD setup)"; exit 1; }
"$BASE_PY" - <<EOF
import torch
assert torch.version.hip, "base Python's torch is not a ROCm build"
assert torch.cuda.is_available(), "no GPU visible to torch"
assert "$ROCM_SDK_VERSION" in torch.__version__, f"torch {torch.__version__} does not match ROCm SDK $ROCM_SDK_VERSION"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
EOF

say "1/6 venv at $ROOT/venv (sees the lerobot packages, installs only into itself)"
mkdir -p "$ROOT"
[ -d "$ROOT/venv" ] || "$BASE_PY" -m venv --system-site-packages "$ROOT/venv"
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"
python -c "import torch, sys; assert sys.prefix.endswith('venv'); print('venv python OK, torch', torch.__version__)"

say "2/6 Python dependencies (plyfile) into the venv"
pip install -q -c "$CONSTRAINTS" plyfile
python -c "import plyfile, tqdm, PIL, cv2, scipy, matplotlib; print('dependencies OK')"

say "3/6 ROCm $ROCM_SDK_VERSION developer SDK (hipcc, HIP headers, hipcub)"
SDK="${ROCM_SDK_DEVEL:-}"
if [ -z "$SDK" ]; then
    for cand in "$HOME"/gsplat-rocm/venv/lib/python3.*/site-packages/_rocm_sdk_devel; do
        [ -x "$cand/bin/hipcc" ] && SDK="$cand" && echo "reusing the SDK from the gsplat build"
    done
fi
if [ -z "$SDK" ]; then
    pip install -q --no-deps "rocm-sdk-devel==$ROCM_SDK_VERSION" --index-url "$ROCM_INDEX"
    SP_BASE=$("$BASE_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    SP_VENV=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    for d in "$SP_BASE"/_rocm_sdk_core "$SP_BASE"/_rocm_sdk_libraries_*; do
        ln -sfn "$d" "$SP_VENV/$(basename "$d")"
    done
    SDK=$(python -m rocm_sdk path --root | tail -1)
fi
[ -e "$SDK/include/hip/hip_runtime.h" ] || { echo "ROCm SDK headers not found under $SDK"; exit 1; }
echo "ROCm SDK root: $SDK"

say "4/6 Triangle Splatting source (@ ${TS_COMMIT:0:7})"
[ -d "$ROOT/src/.git" ] || git clone -q "$TS_REPO" "$ROOT/src"
if [ "$(git -C "$ROOT/src" rev-parse HEAD)" != "$TS_COMMIT" ]; then
    git -C "$ROOT/src" fetch -q origin
    git -C "$ROOT/src" checkout -q "$TS_COMMIT"
fi
git -C "$ROOT/src" submodule update -q --init --recursive submodules/diff-triangle-rasterization
RAST="$ROOT/src/submodules/diff-triangle-rasterization"
# nvcc-only flag: clang (hipcc) does not know it. -ffp-contract=fast keeps the fused multiply-add part of
# it without -ffast-math's "no NaN/Inf" assumptions.
sed -i 's/"--use_fast_math"/"-ffp-contract=fast"/' "$RAST/setup.py"
# CUDA-only headers with no HIP counterpart (torch's hipify converts the rest: cuda.h, cub, cooperative_groups.h)
sed -i -E '/#include "device_launch_parameters.h"|#include <cooperative_groups\/reduce.h>/d' \
    "$RAST"/cuda_rasterizer/*.cu "$RAST"/cuda_rasterizer/*.h
# Kernel launches are written "<< < ... >> >" (nvcc accepts the spaces, clang does not)
sed -i -E 's/<< </<<</g; s/>> >/>>>/g' "$RAST"/cuda_rasterizer/*.cu
# __trap() is a CUDA intrinsic (only reached on an internal consistency error)
sed -i 's/__trap();/__builtin_trap();/' "$RAST/cuda_rasterizer/auxiliary.h"
rm -rf "$RAST/hip_rasterizer" "$RAST"/*.hip   # stale hipify output from an earlier build

say "5/6 Compiling the triangle rasterizer (gfx1201)"
cd "$RAST"
rm -rf build ./*.egg-info
export ROCM_HOME="$SDK" ROCM_PATH="$SDK" HIP_PATH="$SDK" PATH="$SDK/bin:$PATH" MAX_JOBS="${MAX_JOBS:-12}"
if ! pip install --no-build-isolation --no-deps -c "$CONSTRAINTS" -v . > "$ROOT/build.log" 2>&1; then
    grep -nE "fatal error|error:" "$ROOT/build.log" | head -20
    echo "Build failed, full log: $ROOT/build.log"; exit 1
fi
echo "built: $(pip show diff_triangle_rasterization | grep -E '^Version')"

say "6/6 Correctness check (image + gradients)"
TRIANGLE_SPLAT_SRC="$ROOT/src" python "$HERE/verify_triangle_rasterizer.py" 2>&1 | grep -vE "MIOpen|TORCH_BLAS|frozen importlib"

say "Done. Use it with:  source $ROOT/venv/bin/activate"
