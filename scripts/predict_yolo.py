"""Run a trained YOLO model over split manifests and write per-image predictions (T20c).

The second and last file that imports Ultralytics (AGPL-3.0). For each split it writes
runs/predict/<name>/predictions_<split>.jsonl in the fdb.predictions format (one line per
image, including images with no boxes), plus meta.json with the weights' SHA-256, the
inference settings and the library versions, so every number can be traced to its model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.layout import read_manifests  # noqa: E402
from fdb.predictions import Detection, FramePrediction, check_complete, read_predictions, write_predictions  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict_frames(model, image_paths: list[Path], split: str, p: dict) -> Iterator[FramePrediction]:
    """One FramePrediction per image, in input order, including images with no boxes."""
    results = model.predict(source=[str(x) for x in image_paths], conf=p["conf"], iou=p["iou"],
                            imgsz=p["imgsz"], max_det=p["max_det"], device=p["device"],
                            stream=True, verbose=False)
    for path, r in zip(image_paths, results, strict=True):
        if Path(r.path).stem != path.stem:
            raise RuntimeError(f"result for {r.path} arrived where {path} was expected")
        height, width = r.orig_shape
        boxes = tuple(
            Detection(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(conf, 6), int(cls))
            for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                                   r.boxes.cls.tolist())
        )
        yield FramePrediction(path.stem, split, width, height, boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write per-image YOLO predictions for split manifests.")
    parser.add_argument("--weights", required=True, help="trained weights, e.g. runs/train/smoke/weights/best.pt")
    parser.add_argument("--subset", choices=("smoke", "full"), default="smoke",
                        help="smoke: splits/zju/smoke_<split>.txt; full: splits/zju/<split>.txt")
    parser.add_argument("--splits", nargs="+", default=["cal", "test"], help="splits to predict (default: cal test)")
    parser.add_argument("--name", help="output folder under runs/predict (default: the subset name)")
    args = parser.parse_args()

    cfg = load_config()
    p = cfg["predict"]
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    root = Path(cfg["data_root"]).expanduser() / cfg["derived_dir"] / "zju_yolo"
    prefix = "smoke_" if args.subset == "smoke" else ""
    manifests = read_manifests(Path(cfg["repo_root"]) / "splits" / "zju",
                               tuple(prefix + s for s in args.splits))
    out = Path(cfg["repo_root"]) / cfg["outputs_dir"] / "predict" / (args.name or args.subset)
    if out.exists():
        raise SystemExit(f"{out} already exists: pass --name or delete it (results are never overwritten)")

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out.mkdir(parents=True)
    started, t0 = datetime.now(timezone.utc), time.perf_counter()
    meta_splits = {}
    for split in args.splits:
        ids = manifests[prefix + split]
        paths = [root / "images" / split / f"{i}.jpg" for i in ids]
        missing = [x for x in paths if not x.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} {split} images missing from the layout, e.g. {missing[0]}")
        file = out / f"predictions_{split}.jsonl"
        write_predictions(file, predict_frames(model, paths, split, p))

        frames = read_predictions(file)   # read back what was written, then check it
        check_complete(frames, ids)
        scores = sorted(max((b.conf for b in f.boxes), default=0.0) for f in frames)
        with_boxes = sum(1 for f in frames if f.boxes)
        print(f"{split:5s} {len(frames):6d} frames  {with_boxes:6d} with boxes  "
              f"s(x) min {scores[0]:.4f}  median {statistics.median(scores):.4f}  max {scores[-1]:.4f}  "
              f"({len(set(scores))} distinct values)")
        meta_splits[split] = {"manifest": f"splits/zju/{prefix}{split}.txt", "images": len(frames),
                              "file": file.name}

    meta = {
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "subset": args.subset,
        "splits": meta_splits,
        "settings": p,
        "started_utc": started.isoformat(timespec="seconds"),
        "seconds": round(time.perf_counter() - t0, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "machine": platform.platform(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"written to {out}")


if __name__ == "__main__":
    main()