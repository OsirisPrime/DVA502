"""Read COLMAP sparse models (binary or text) into plain numpy structures.

Conventions: COLMAP stores the world-to-camera pose of each image as a quaternion (w, x, y, z) and a
translation t, so ``x_cam = R(q) @ x_world + t``. :func:`Image.w2c` returns that as a 4x4 matrix, which is
what gsplat calls a ``viewmat``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# model_id -> (name, number of parameters); from colmap/src/colmap/sensor/models.h
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
UNDISTORTED_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy). Only valid without lens distortion, see :attr:`is_undistorted`."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
                          "FOV"):
            return float(p[0]), float(p[0]), float(p[1]), float(p[2])
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])

    @property
    def is_undistorted(self) -> bool:
        return self.model in UNDISTORTED_MODELS


@dataclass
class Image:
    id: int
    name: str
    camera_id: int
    qvec: np.ndarray  # (w, x, y, z), world -> camera
    tvec: np.ndarray

    def w2c(self) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = qvec2rotmat(self.qvec)
        m[:3, 3] = self.tvec
        return m


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: list[Image]  # sorted by name
    xyz: np.ndarray      # (N, 3) float32
    rgb: np.ndarray      # (N, 3) uint8

    @property
    def camera_centers(self) -> np.ndarray:
        return np.stack([np.linalg.inv(im.w2c())[:3, 3] for im in self.images])


def qvec2rotmat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_model(path: str | Path) -> Model:
    """Read a COLMAP model folder; binary files are preferred when both formats are present."""
    path = Path(path)
    if (path / "images.bin").exists():
        cams, imgs, (xyz, rgb) = (_read_cameras_bin(path / "cameras.bin"), _read_images_bin(path / "images.bin"),
                                  _read_points_bin(path / "points3D.bin"))
    elif (path / "images.txt").exists():
        cams, imgs, (xyz, rgb) = (_read_cameras_txt(path / "cameras.txt"), _read_images_txt(path / "images.txt"),
                                  _read_points_txt(path / "points3D.txt"))
    else:
        raise FileNotFoundError(f"No COLMAP model (images.bin / images.txt) in {path}")
    imgs.sort(key=lambda im: im.name)
    return Model(cams, imgs, xyz, rgb)


# ---------------------------------------------------------------------------- binary
class _Reader:
    def __init__(self, path: Path):
        self.buf = path.read_bytes()
        self.pos = 0

    def read(self, fmt: str):
        vals = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return vals

    def cstring(self) -> str:
        end = self.buf.index(b"\x00", self.pos)
        s = self.buf[self.pos:end].decode("utf-8")
        self.pos = end + 1
        return s


def _read_cameras_bin(path: Path) -> dict[int, Camera]:
    r, cams = _Reader(path), {}
    (n,) = r.read("Q")
    for _ in range(n):
        cid, model_id, w, h = r.read("iiQQ")
        name, n_params = CAMERA_MODELS[model_id]
        cams[cid] = Camera(cid, name, w, h, np.array(r.read("d" * n_params)))
    return cams


def _read_images_bin(path: Path) -> list[Image]:
    r, imgs = _Reader(path), []
    (n,) = r.read("Q")
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = r.read("idddddddi")
        name = r.cstring()
        (n_pts,) = r.read("Q")
        r.pos += n_pts * 24  # (double x, double y, int64 point3D_id) per 2D point, not needed
        imgs.append(Image(iid, name, cid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz])))
    return imgs


def _read_points_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    r = _Reader(path)
    (n,) = r.read("Q")
    xyz, rgb = np.empty((n, 3), np.float32), np.empty((n, 3), np.uint8)
    for i in range(n):
        _, x, y, z, cr, cg, cb, _err, track_len = r.read("QdddBBBdQ")
        xyz[i], rgb[i] = (x, y, z), (cr, cg, cb)
        r.pos += track_len * 8  # (int32 image_id, int32 point2D_idx) per observation
    return xyz, rgb


# ---------------------------------------------------------------------------- text
def _data_lines(path: Path):
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                yield line


def _read_cameras_txt(path: Path) -> dict[int, Camera]:
    cams = {}
    for line in _data_lines(path):
        p = line.split()
        cams[int(p[0])] = Camera(int(p[0]), p[1], int(p[2]), int(p[3]), np.array(list(map(float, p[4:]))))
    return cams


def _read_images_txt(path: Path) -> list[Image]:
    imgs = []
    with open(path) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):  # 2 lines per image; the 2nd (2D points) may be empty
        p = lines[i].split()
        if len(p) < 10:
            continue
        imgs.append(Image(int(p[0]), " ".join(p[9:]), int(p[8]), np.array(list(map(float, p[1:5]))),
                          np.array(list(map(float, p[5:8])))))
    return imgs


def _read_points_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    for line in _data_lines(path):
        p = line.split()
        xyz.append(p[1:4])
        rgb.append(p[4:7])
    return np.array(xyz, np.float32).reshape(-1, 3), np.array(rgb, np.uint8).reshape(-1, 3)
