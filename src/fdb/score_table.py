"""Per-frame score table: the model's score joined with the ground truth (T21b).

One CSV row per frame: image_id, split, score, defective (1 = truly defective, 0 = not).
This is the only place where ground truth meets the model's output; coverage, risk and
the threshold (T22, T31) read these tables and nothing else.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

COLUMNS = ("image_id", "split", "score", "defective")


@dataclass(frozen=True)
class ScoredFrame:
    image_id: str
    split: str
    score: float
    defective: bool


def join_truth(scores: Mapping[str, float], truth: Mapping[str, bool], split: str) -> list[ScoredFrame]:
    """Strict join on image_id: every scored image needs a label and every labelled image a score."""
    no_label = sorted(scores.keys() - truth.keys())
    no_score = sorted(truth.keys() - scores.keys())
    if no_label or no_score:
        raise ValueError(f"join mismatch: {len(no_score)} labelled images have no score, "
                         f"{len(no_label)} scored images have no label (e.g. {(no_score or no_label)[:3]})")
    return [ScoredFrame(i, split, float(scores[i]), bool(truth[i])) for i in sorted(scores)]


def write_score_table(path: Path, rows: Iterable[ScoredFrame]) -> None:
    """Write the table sorted by image_id, 6-decimal scores, LF line endings (identical on re-run)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in sorted(rows, key=lambda r: r.image_id):
            w.writerow([r.image_id, r.split, f"{r.score:.6f}", int(r.defective)])


def read_score_table(path: Path) -> list[ScoredFrame]:
    """Read a score table, refusing a wrong header, a score outside [0, 1] or a label other than 0/1."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = tuple(next(reader, ()))
        if header != COLUMNS:
            raise ValueError(f"{path}: header {header} is not {COLUMNS}")
        rows = []
        for line_no, (image_id, split, score, defective) in enumerate(reader, start=2):
            s = float(score)
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"{path}:{line_no}: score {score} is outside [0, 1]")
            if defective not in ("0", "1"):
                raise ValueError(f"{path}:{line_no}: defective must be 0 or 1, got {defective!r}")
            rows.append(ScoredFrame(image_id, split, s, defective == "1"))
    return rows