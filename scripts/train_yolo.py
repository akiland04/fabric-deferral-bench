"""Train a YOLO detector from a named profile in config.yaml (T20b smoke run; T27 full run).

This is the only training code that imports Ultralytics (AGPL-3.0). The deferral harness
reads prediction files and never imports it. Settings come from `train.<profile>` in
config.yaml; Ultralytics also saves them to <run>/args.yaml, and this script adds
<run>/run_info.json with timing, library versions and the machine used.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.yolo import count_train_images  # noqa: E402


def data_yaml_for(cfg: dict, view: str) -> Path:
    """'smoke' -> <layout>/smoke/data.yaml, 'full' -> <layout>/data.yaml."""
    root = Path(cfg["data_root"]).expanduser() / cfg["derived_dir"] / "zju_yolo"
    views = {"smoke": root / "smoke" / "data.yaml", "full": root / "data.yaml"}
    if view not in views:
        raise SystemExit(f"unknown data view '{view}': use one of {sorted(views)}")
    return views[view]


def read_results(run_dir: Path) -> list[dict[str, float]]:
    """Ultralytics' per-epoch log (results.csv) as a list of {column: value} rows."""
    with open(run_dir / "results.csv", newline="") as f:
        return [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO from a config.yaml profile.")
    parser.add_argument("--profile", default="smoke", help="key under train: in config.yaml (default: smoke)")
    parser.add_argument("--name", help="run folder name (default: the profile name)")
    args = parser.parse_args()

    cfg = load_config()
    p = cfg["train"][args.profile]
    data_yaml = data_yaml_for(cfg, p["data"])
    if not data_yaml.exists():
        raise SystemExit(f"{data_yaml} not found: run scripts/convert_zju_yolo.py"
                         + (" and scripts/make_zju_smoke.py" if p["data"] == "smoke" else ""))
    n_images = count_train_images(data_yaml)

    import torch  # imported here so --help and config errors don't wait for PyTorch to load
    import ultralytics
    from ultralytics import YOLO

    if p["device"] == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("device is 'mps' but PyTorch cannot see the Apple GPU")

    project = Path(cfg["repo_root"]) / cfg["outputs_dir"] / "train"
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    model = YOLO(p["model"])
    model.train(
        data=str(data_yaml),
        epochs=p["epochs"],
        imgsz=p["imgsz"],
        batch=p["batch"],
        workers=p["workers"],
        device=p["device"],
        seed=cfg["seed"],
        deterministic=True,
        project=str(project),
        name=args.name or args.profile,
        exist_ok=False,
    )
    seconds = time.perf_counter() - t0

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training ended but {best} is missing")

    rows = read_results(run_dir)
    train_seconds = rows[-1]["time"]  # cumulative seconds of the training loop, incl. per-epoch validation
    info = {
        "profile": args.profile,
        "data_yaml": str(data_yaml),
        "train_images": n_images,
        "epochs": p["epochs"],
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_seconds": round(seconds, 1),
        "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / len(rows), 1),
        "seconds_per_image_epoch": round(train_seconds / (len(rows) * n_images), 4),
        "device": p["device"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "machine": platform.platform(),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print(f"run folder: {run_dir}")
    print(f"best weights: {best}")
    print(f"{info['wall_seconds']} s wall clock; {info['seconds_per_epoch']} s per epoch "
          f"({info['seconds_per_image_epoch']} s per image) on {n_images} training images")
    for loss in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
        print(f"{loss:15s} epoch 1: {rows[0][loss]:.3f}   epoch {len(rows)}: {rows[-1][loss]:.3f}")


if __name__ == "__main__":
    main()