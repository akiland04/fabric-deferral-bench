"""Frame scores: one number per image from its detections (T21a; measure M1 in papers.xlsx sheet 6).

Higher score = more likely defective; the deferral rule (D1) auto-passes a frame when s(x) < t.
Scorers read the predictions contract (fdb.predictions) and never import a detector library.
"""
from __future__ import annotations

from typing import Callable, Iterable

from .predictions import FramePrediction

Scorer = Callable[[FramePrediction], float]


def max_conf_score(frame: FramePrediction) -> float:
    """s(x) = the highest box confidence on the frame, or 0.0 if it has no boxes (M1, D1, D7)."""
    best = 0.0
    for box in frame.boxes:
        conf = box.conf
        if not 0.0 <= conf <= 1.0:  # also rejects NaN, which fails every comparison
            raise ValueError(f"{frame.image_id}: confidence {conf!r} is outside [0, 1]")
        best = max(best, conf)
    return best


def score_frames(frames: Iterable[FramePrediction], scorer: Scorer = max_conf_score) -> dict[str, float]:
    """{image_id: score} for every frame, using any scorer with the same shape as max_conf_score."""
    scores: dict[str, float] = {}
    for frame in frames:
        if frame.image_id in scores:
            raise ValueError(f"image {frame.image_id} appears twice")
        scores[frame.image_id] = scorer(frame)
    return scores