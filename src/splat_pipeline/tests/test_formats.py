"""CPU-only tests of the file formats and config handling (no GPU, gsplat or COLMAP needed)."""
import struct

import numpy as np
import pytest

from splat_pipeline.colmap_io import qvec2rotmat, read_model
from splat_pipeline.config import Config
from splat_pipeline.ply import load_ply, save_ply


def _write_model(d, binary):
    """Two cameras-worth of a tiny COLMAP model in binary or text format."""
    q = np.array([0.9238795, 0.0, 0.3826834, 0.0])  # 45 deg about y
    t = np.array([0.1, -0.2, 3.0])
    pts = np.array([[0, 0, 0], [1, 2, 3], [-1, 0.5, 2]], np.float64)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    d.mkdir()
    if binary:
        (d / "cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 1, 1, 640, 480, 500.0, 501.0, 320.0, 240.0))
        (d / "images.bin").write_bytes(
            struct.pack("<Q", 2)
            + struct.pack("<idddddddi", 7, *q, *t, 1) + b"b image.jpg\x00" + struct.pack("<Q", 1)
            + struct.pack("<ddq", 1.0, 2.0, -1)
            + struct.pack("<idddddddi", 3, 1, 0, 0, 0, 0, 0, 0, 1) + b"a.jpg\x00" + struct.pack("<Q", 0))
        body = struct.pack("<Q", len(pts))
        for i, (p, c) in enumerate(zip(pts, rgb)):
            body += struct.pack("<QdddBBBdQ", i + 1, *p, *c, 0.5, 2) + struct.pack("<iiii", 7, 0, 3, 0)
        (d / "points3D.bin").write_bytes(body)
    else:
        (d / "cameras.txt").write_text("# c\n1 PINHOLE 640 480 500 501 320 240\n")
        (d / "images.txt").write_text(
            "# i\n7 " + " ".join(map(str, q)) + " " + " ".join(map(str, t)) + " 1 b image.jpg\n1.0 2.0 -1\n"
            "3 1 0 0 0 0 0 0 1 a.jpg\n\n")
        (d / "points3D.txt").write_text("".join(
            f"{i + 1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0.5 7 0 3 0\n" for i, (p, c) in enumerate(zip(pts, rgb))))
    return q, t, pts, rgb


@pytest.mark.parametrize("binary", [True, False])
def test_read_model(tmp_path, binary):
    q, t, pts, rgb = _write_model(tmp_path / "m", binary)
    m = read_model(tmp_path / "m")
    assert [im.name for im in m.images] == ["a.jpg", "b image.jpg"]  # sorted, names with spaces kept
    assert m.cameras[1].model == "PINHOLE" and m.cameras[1].intrinsics() == (500.0, 501.0, 320.0, 240.0)
    w2c = m.images[1].w2c()
    np.testing.assert_allclose(w2c[:3, :3], qvec2rotmat(q), atol=1e-6)
    np.testing.assert_allclose(w2c[:3, 3], t)
    np.testing.assert_allclose(m.xyz, pts, atol=1e-6)
    np.testing.assert_array_equal(m.rgb, rgb)


def test_rotation_is_orthonormal():
    R = qvec2rotmat([0.3, -0.5, 0.2, 0.7])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    params = {"means": rng.normal(size=(n, 3)), "scales": rng.normal(size=(n, 3)), "quats": rng.normal(size=(n, 4)),
              "opacities": rng.normal(size=n), "sh0": rng.normal(size=(n, 1, 3)), "shN": rng.normal(size=(n, 15, 3))}
    params = {k: v.astype(np.float32) for k, v in params.items()}
    save_ply(tmp_path / "x.ply", params)
    header = (tmp_path / "x.ply").read_bytes().split(b"end_header")[0].decode()
    assert "property float f_rest_44" in header and "property float rot_3" in header
    back = load_ply(tmp_path / "x.ply")
    for k in params:
        np.testing.assert_array_equal(back[k], params[k])
    # channel-major: f_rest_0..14 are the red coefficients
    names = [l.split()[-1] for l in header.splitlines() if l.startswith("property")]
    raw = np.frombuffer((tmp_path / "x.ply").read_bytes().split(b"end_header\n", 1)[1], "<f4").reshape(n, -1)
    np.testing.assert_array_equal(raw[:, names.index("f_rest_14")], params["shN"][:, 14, 0])
    np.testing.assert_array_equal(raw[:, names.index("f_rest_15")], params["shN"][:, 0, 1])


def test_config_layers(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("sfm:\n  mapper: glomap\ntrain:\n  steps: 7000\n  lr_sh0: 1\n")
    cfg = Config.load(f, {"train.steps": "100", "train.packed": "true"})
    assert cfg.sfm.mapper == "glomap" and cfg.train.steps == 100 and cfg.train.packed is True
    assert cfg.train.lr_sh0 == 1.0 and isinstance(cfg.train.lr_sh0, float)
    cfg.save(tmp_path / "out.yaml")
    assert Config.load(tmp_path / "out.yaml").to_dict() == cfg.to_dict()
    with pytest.raises(KeyError):
        Config.load(None, {"train.stepz": "1"})
    with pytest.raises(TypeError):
        Config.load(None, {"train.steps": "many"})
