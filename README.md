# DVA502
## Setup

```bash
git clone --recursive <repo-url> && cd DVA502
conda create -y -n lerobot python=3.12 && conda activate lerobot
conda install -y ffmpeg -c conda-forge
```

PyTorch:
- NVIDIA: nothing extra, installed with LeRobot below.
- AMD (ROCm, gfx120X): `pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ torch torchvision`,
  then `pip freeze | grep -E '^(torch|torchvision|triton|rocm)' > ~/rocm-torch.txt`
  and add `-c ~/rocm-torch.txt` to every pip install below.

```bash
pip install -e "./third_party/lerobot[training,libero,pi]"
hf auth login   # accept the PaliGemma license on huggingface.co first
```

Check: `python -c "import torch, lerobot; print(torch.cuda.is_available())"`
