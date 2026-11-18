"""Command line: ``python -m splat_pipeline <command>`` (or ``splat-pipeline <command>`` when installed).

Commands map 1:1 to stages that only talk through files in the scene / run folders:
  frames  video -> <scene>/images
  sfm     <scene>/images -> camera poses + sparse points (COLMAP or GLOMAP mapper)
  train   scene -> run folder (checkpoints, point_cloud.ply, metrics, renders); --resume continues a run
  eval    re-evaluate the latest checkpoint of a run
  export  write a .ply from the latest checkpoint of a run (e.g. of an unfinished run)
  run     frames (if --video) + sfm + train in one go

Settings: defaults in config.py < --config file.yaml < --set section.key=value < dedicated flags.
Heavy libraries (torch, gsplat, cv2) are imported inside the commands so --help and sfm start instantly.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _config(a):
    from .config import Config

    overrides = dict(kv.split("=", 1) for kv in (a.set or []))
    for flag, key in (("mapper", "sfm.mapper"), ("matcher", "sfm.matcher"), ("colmap", "sfm.colmap"),
                      ("steps", "train.steps"), ("downscale", "train.downscale"), ("strategy", "train.strategy"),
                      ("fps", "frames.fps")):
        if getattr(a, flag, None) is not None:
            overrides[key] = getattr(a, flag)
    return Config.load(a.config, overrides)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="splat_pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, *, scene=True):
        if scene:
            p.add_argument("--scene", required=True, help="scene folder (contains images/)")
        p.add_argument("--config", help="YAML file with frames/sfm/train sections (see configs/)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. --set train.packed=true")
        return p

    p = common(sp.add_parser("frames", help="extract sharp frames from a video"))
    p.add_argument("--video", required=True)
    p.add_argument("--fps", type=float)

    def sfm_flags(p):
        p.add_argument("--mapper", choices=["colmap", "glomap"])
        p.add_argument("--matcher", choices=["sequential", "exhaustive", "vocab_tree"])
        p.add_argument("--colmap", help="COLMAP executable (GLOMAP needs COLMAP >= 3.13)")

    sfm_flags(common(sp.add_parser("sfm", help="camera poses + sparse points")))

    def train_flags(p):
        p.add_argument("--steps", type=int)
        p.add_argument("--downscale", type=int)
        p.add_argument("--strategy", choices=["default", "mcmc"])
        p.add_argument("--out", help="run folder (default outputs/splat_pipeline/<scene>_<timestamp>)")

    train_flags(common(sp.add_parser("train", help="train (or --resume) a model")))
    sp.choices["train"].add_argument("--resume", metavar="RUN", help="continue the run in this folder")

    p = sp.add_parser("eval", help="evaluate the latest checkpoint of a run")
    p.add_argument("run")
    p.add_argument("--scene", help="default: the scene recorded in run.json")

    p = sp.add_parser("export", help="write a .ply from the latest checkpoint of a run")
    p.add_argument("run")
    p.add_argument("--out")

    p = common(sp.add_parser("run", help="frames (with --video) + sfm + train"))
    p.add_argument("--video")
    p.add_argument("--fps", type=float)
    sfm_flags(p)
    train_flags(p)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from tqdm.contrib.logging import logging_redirect_tqdm

    try:
        with logging_redirect_tqdm():  # log lines are printed above the progress bars instead of through them
            return _dispatch(a)
    except (FileNotFoundError, FileExistsError, ValueError, KeyError, TypeError, RuntimeError) as e:
        logging.getLogger("splat_pipeline").error("%s", e)
        if a.verbose:
            raise
        return 1


def _dispatch(a) -> int:
    if a.cmd == "eval":
        from .trainer import evaluate_run

        evaluate_run(Path(a.run), Path(a.scene) if a.scene else None)
        return 0
    if a.cmd == "export":
        from .trainer import export_ply

        export_ply(Path(a.run), Path(a.out) if a.out else None)
        return 0
    if a.cmd == "train" and a.resume:
        from .config import Config
        from .trainer import train

        run = Path(a.resume)
        cfg = Config.load(run / "config.yaml", dict(kv.split("=", 1) for kv in (a.set or [])))
        train(a.scene, cfg, run, resume=True)
        return 0

    cfg = _config(a)
    if a.cmd in ("frames", "run") and getattr(a, "video", None):
        from .frames import extract_frames

        extract_frames(a.video, a.scene, cfg.frames)
    if a.cmd in ("sfm", "run"):
        from .sfm import run_sfm

        run_sfm(a.scene, cfg.sfm)
    if a.cmd in ("train", "run"):
        from .trainer import train

        train(a.scene, cfg, Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
