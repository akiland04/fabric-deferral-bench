"""The common label format every dataset adapter writes: YOLO text labels, one class."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFECT_CLASS = 0  # every defect type maps to the single class 'defect' (Class Mapping sheet)
CLASS_NAMES = {DEFECT_CLASS: "defect"}


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in absolute pixels."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def to_yolo_line(box: Box, width: int, height: int, cls: int = DEFECT_CLASS) -> str | None:
    """'cls cx cy w h' with coordinates divided by image size; None if the box has no area."""
    x0, x1 = max(0.0, box.xmin), min(float(width), box.xmax)
    y0, y1 = max(0.0, box.ymin), min(float(height), box.ymax)
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    w, h = (x1 - x0) / width, (y1 - y0) / height
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def write_data_yaml(root: Path, train: str, val: str, names: dict[int, str] = CLASS_NAMES) -> Path:
    """Write <root>/data.yaml, the only data a YOLO training run is allowed to see.

    train and val are paths relative to root (a folder of images, or a .txt list of images).
    Calibration and test data are deliberately absent, so training cannot read them.
    """
    root = root.resolve()
    for rel in (train, val):
        if not (root / rel).exists():
            raise FileNotFoundError(f"data.yaml would point at a missing path: {root / rel}")
    doc = {
        "path": str(root),          # absolute: Ultralytics resolves a relative path elsewhere
        "train": train,
        "val": val,
        "nc": len(names),
        "names": dict(names),
    }
    out = root / "data.yaml"
    out.write_text(yaml.safe_dump(doc, sort_keys=False))
    return out

def from_yolo_line(line: str, width: int, height: int) -> tuple[int, Box]:
    """Inverse of to_yolo_line: (class, box in absolute pixels) decoded from one label line."""
    cls, cx, cy, w, h = line.split()
    cx, w = float(cx) * width, float(w) * width
    cy, h = float(cy) * height, float(h) * height
    return int(cls), Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)