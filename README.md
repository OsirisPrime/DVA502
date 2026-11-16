# Synthetic Training Data for Physical AI

Master's thesis (DVA502), Mälardalen University, in collaboration with ABB Robotics.


**Period:** January – June 2027\
**Authors:** Kim Svedberg, Zebastian Thorsén\
**Supervisors:** Rickard Holm (ABB Robotics),

## Overview

Vision-language-action (VLA) models can learn robot manipulation skills from demonstrations, but collecting real-world demonstrations is slow and expensive. Synthetic data from simulation is a scalable alternative, but how realistic does it need to be?

This thesis investigates how the fidelity of synthetic training data affects the performance of VLA policies, and how well policies trained on synthetic data transfer to a real **ABB GoFa** collaborative robot.

## Setup

```bash
git clone --recursive https://github.com/OsirisPrime/DVA502 && cd DVA502
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
