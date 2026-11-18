"""Standard 3DGS .ply (INRIA layout), readable by SuperSplat and other 3DGS viewers.

Values are stored as the raw parameters: log-scales, logit-opacities, SH coefficients with band 0 in
``f_dc_*`` and the rest channel-major in ``f_rest_*`` (all R coefficients, then G, then B). Do not reorder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_ply(path: str | Path, params: dict[str, np.ndarray]) -> None:
    """params: means (N,3), scales (N,3) log, quats (N,4) wxyz, opacities (N,) logit, sh0 (N,1,3), shN (N,K-1,3)."""
    means = params["means"]
    n = len(means)
    sh0 = params["sh0"].reshape(n, 3)
    shN = params["shN"].transpose(0, 2, 1).reshape(n, -1)  # (N,K-1,3) -> channel-major
    attrs = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(shN.shape[1])] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, params["opacities"].reshape(n, 1),
                           params["scales"], params["quats"]], axis=1).astype("<f4")
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              + "".join(f"property float {a}\n" for a in attrs) + "end_header\n")
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(header.encode())
        fh.write(data.tobytes())
    tmp.replace(path)


def load_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Inverse of :func:`save_ply` (only for files with that exact layout)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode().splitlines()
    n = int(next(l for l in header if l.startswith("element vertex")).split()[-1])
    names = [l.split()[-1] for l in header if l.startswith("property float")]
    data = np.frombuffer(raw[end:], dtype="<f4").reshape(n, len(names))
    col = {name: i for i, name in enumerate(names)}
    n_rest = sum(1 for a in names if a.startswith("f_rest_"))

    def take(prefix, k):
        return data[:, [col[f"{prefix}{i}"] for i in range(k)]]

    return {
        "means": data[:, [col["x"], col["y"], col["z"]]].copy(),
        "sh0": take("f_dc_", 3).reshape(n, 1, 3).copy(),
        "shN": take("f_rest_", n_rest).reshape(n, 3, n_rest // 3).transpose(0, 2, 1).copy(),
        "opacities": data[:, col["opacity"]].copy(),
        "scales": take("scale_", 3).copy(),
        "quats": take("rot_", 4).copy(),
    }
