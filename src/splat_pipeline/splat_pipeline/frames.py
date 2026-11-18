"""Stage 1: extract sharp frames from a video into ``<scene>/images/``."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import FramesConfig

log = logging.getLogger(__name__)


def extract_frames(video: str | Path, scene: str | Path, cfg: FramesConfig) -> int:
    """Keep the sharpest frame (highest variance of the Laplacian, i.e. least motion blur) in every
    ``1 / cfg.fps`` second window. Frames are written as ``00000.jpg``, ``00001.jpg``, ...

    iPhone .MOV (HEVC) works if the OpenCV/FFmpeg build can decode it; rotation metadata is applied by OpenCV.
    Returns the number of frames written.
    """
    import cv2

    out = Path(scene) / "images"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; use a new scene folder")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = max(1, round(vfps / cfg.fps))
    params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]

    best, best_score, i, n = None, -1.0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cfg.max_side and max(frame.shape[:2]) > cfg.max_side:
            s = cfg.max_side / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if score > best_score:
            best, best_score = frame, score
        i += 1
        if i % window == 0:
            cv2.imwrite(str(out / f"{n:05d}.jpg"), best, params)
            n += 1
            best, best_score = None, -1.0
    if best is not None:
        cv2.imwrite(str(out / f"{n:05d}.jpg"), best, params)
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"No frames decoded from {video} (unsupported codec?)")
    log.info("Saved %d frames (%.1f fps video, 1 per %d frames) to %s", n, vfps, window, out)
    return n
