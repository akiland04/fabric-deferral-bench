"""Per-image predictions: the contract between a detector adapter and the deferral harness (T20c).

JSON Lines, one object per image, including images with no boxes:
  {"image_id":"000101","split":"cal","width":512,"height":512,
   "boxes":[{"x1":..,"y1":..,"x2":..,"y2":..,"conf":..,"cls":0}, ...]}
Boxes are in absolute pixels. Ground truth is deliberately absent; the harness joins it by
image_id. This module must never import a detector library (Ultralytics is AGPL-3.0).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0


@dataclass(frozen=True)
class FramePrediction:
    image_id: str
    split: str
    width: int
    height: int
    boxes: tuple[Detection, ...] = ()


def write_predictions(path: Path, frames: Iterable[FramePrediction]) -> int:
    """Write one JSON line per frame; returns the number of frames written."""
    n = 0
    with open(path, "w") as f:
        for frame in frames:
            f.write(json.dumps(asdict(frame), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_predictions(path: Path) -> list[FramePrediction]:
    """Read a predictions file. A record without a "boxes" key is an error, not 'no boxes'."""
    frames = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if "boxes" not in d:
                raise ValueError(f"{path}:{line_no}: record for {d.get('image_id')} has no 'boxes' key")
            boxes = tuple(Detection(**b) for b in d["boxes"])
            frames.append(FramePrediction(d["image_id"], d["split"], d["width"], d["height"], boxes))
    return frames


def check_complete(frames: list[FramePrediction], expected_ids: Iterable[str]) -> None:
    """Fail unless there is exactly one prediction per expected image ID."""
    counts = Counter(f.image_id for f in frames)
    expected = set(expected_ids)
    duplicated = sorted(i for i, c in counts.items() if c > 1)
    missing = sorted(expected - counts.keys())
    unexpected = sorted(counts.keys() - expected)
    if duplicated or missing or unexpected:
        raise ValueError(f"predictions do not match the manifest: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected, {len(duplicated)} duplicated "
                         f"(e.g. {(missing or unexpected or duplicated)[:3]})")