#!/usr/bin/env bash
# Build gsplat (AMD's ROCm port, pinned + patched) for the RX 9070 XT (gfx1201) in an ISOLATED venv.
#
# - The venv is built on top of the `lerobot` conda env's Python (--system-site-packages), so it uses the
#   existing ROCm PyTorch. Nothing is installed into the lerobot env itself.
# - Everything lives in $GSPLAT_ROCM_DIR (default ~/gsplat-rocm). Delete that folder to undo it all (~10 GB).
# - Patches (gsplat_rocm_gfx1201.patch, next to this script) make the wave64 port work on RDNA4 and fix wrong gradients;
#   see docs/gaussian_splatting.md. The build ends with a gradient check and fails if it does not pass.
#
# Usage:   bash src/gaussian_splat/rocm/build_gsplat_rocm.sh           (takes ~10 min, ~6 of them compiling)
# Then:    source ~/gsplat-rocm/venv/bin/activate
set -euo pipefail

ROOT="${GSPLAT_ROCM_DIR:-$HOME/gsplat-rocm}"
BASE_PY="${BASE_PYTHON:-$HOME/miniforge3/envs/lerobot/bin/python}"  # Python with the ROCm torch
CONSTRAINTS="${ROCM_TORCH_CONSTRAINTS:-$HOME/rocm-torch.txt}"       # pins torch/rocm so pip never replaces them
GSPLAT_REPO="https://github.com/ROCm/gsplat.git"
GSPLAT_COMMIT="b01acd43e3c7fa942f95fda0974e9125e4de7395"            # ROCm/gsplat main, gsplat 1.5.3
ROCM_SDK_VERSION="7.13.0"                                           # must match torch's ROCm (torch 2.11.0+rocm7.13.0)
ROCM_INDEX="https://repo.amd.com/rocm/whl/gfx120X-all/"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/gsplat_rocm_gfx1201.patch"

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

say "2/6 ROCm $ROCM_SDK_VERSION developer SDK (headers, hipcc, rocprim/hipcub) + gsplat's Python deps, into the venv"
pip install -q --no-deps "rocm-sdk-devel==$ROCM_SDK_VERSION" --index-url "$ROCM_INDEX"
pip install -q -c "$CONSTRAINTS" jaxtyping rich ninja
# The devel package symlinks into _rocm_sdk_core / _rocm_sdk_libraries_* next to itself; those live in the base env.
SP_BASE=$("$BASE_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
SP_VENV=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
for d in "$SP_BASE"/_rocm_sdk_core "$SP_BASE"/_rocm_sdk_libraries_*; do
    ln -sfn "$d" "$SP_VENV/$(basename "$d")"
done
SDK=$(python -m rocm_sdk path --root | tail -1)
[ -e "$SDK/include/hip/hip_runtime.h" ] || { echo "ROCm SDK headers not found under $SDK"; exit 1; }
echo "ROCm SDK root: $SDK"

say "3/6 gsplat source (ROCm/gsplat @ ${GSPLAT_COMMIT:0:7})"
if [ ! -d "$ROOT/src/.git" ]; then
    git clone -q "$GSPLAT_REPO" "$ROOT/src"
fi
git -C "$ROOT/src" fetch -q origin "$GSPLAT_COMMIT" 2>/dev/null || true
if git -C "$ROOT/src" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patch already applied"
else
    [ -z "$(git -C "$ROOT/src" status --porcelain --untracked-files=no)" ] || {
        echo "$ROOT/src has local changes that are not this patch; move it away and rerun."; exit 1; }
    git -C "$ROOT/src" checkout -q "$GSPLAT_COMMIT"
    git -C "$ROOT/src" submodule update -q --init --recursive
    git -C "$ROOT/src" apply "$PATCH"
    echo "patch applied"
fi

say "4/6 glm headers outside the source tree (torch's hipify must not rewrite them)"
rm -rf "$ROOT/glm_include" && mkdir -p "$ROOT/glm_include"
cp -r "$ROOT/src/gsplat/cuda/csrc/third_party/glm/glm" "$ROOT/glm_include/"

say "5/6 Compiling (gfx1201, wave64) - about 6 minutes"
cd "$ROOT/src"
rm -rf build gsplat/hip
export ROCM_HOME="$SDK" ROCM_PATH="$SDK" HIP_PATH="$SDK" PATH="$SDK/bin:$PATH"
export GSPLAT_GLM_INCLUDE="$ROOT/glm_include" MAX_JOBS="${MAX_JOBS:-12}"
if ! pip install --no-build-isolation --no-deps -c "$CONSTRAINTS" -v . > "$ROOT/build.log" 2>&1; then
    grep -nE "fatal error|error:" "$ROOT/build.log" | head -20
    echo "Build failed, full log: $ROOT/build.log"; exit 1
fi
echo "built: $(pip show amd_gsplat | grep -E '^Version')"

say "6/6 Gradient check"
cd "$HERE"
python "$HERE/verify_gsplat_rocm.py" 2>&1 | grep -vE "MIOpen|TORCH_BLAS|frozen importlib"

say "Done. Use it with:  source $ROOT/venv/bin/activate"
