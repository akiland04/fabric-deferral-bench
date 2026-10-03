"""The common label format every dataset adapter writes: YOLO text labels, one class."""
from __future__ import annotations

from dataclasses import dataclass

DEFECT_CLASS = 0  # every defect type maps to the single class 'defect' (Class Mapping sheet)


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