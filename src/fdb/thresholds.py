"""Frozen thresholds file (T31c; D3, D7).

thresholds/<name>.json binds every t(eps) to the exact weights (SHA-256), score, decision rule
and inference settings it was chosen with, plus a fingerprint of the image IDs it was chosen on.
Written once: re-writing identical content is a no-op, different content is refused. No
timestamp and no machine paths, so the same selection gives the same bytes on any machine.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .selection import Selection

SCHEMA_VERSION = 1
SCORE = "max_conf"                      # M1: s(x) = highest box confidence, 0 if no boxes
RULE = "auto-pass if s(x) < t"          # D1: a score equal to t goes to a human


def id_fingerprint(image_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted IDs, one per line: the same set gives the same hash in any order."""
    ids = sorted(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image IDs")
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def build_record(selections: list[Selection], meta: dict, split: str, image_ids: list[str]) -> dict:
    """Everything needed to re-apply the thresholds and to audit how they were chosen."""
    if not selections:
        raise ValueError("no selections to freeze")
    c = selections[0].counts
    if len(image_ids) != c.n:
        raise ValueError(f"{len(image_ids)} image IDs for a curve over {c.n} frames")
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"weights_sha256": meta["weights_sha256"], "subset": meta["subset"]},
        "score": SCORE,
        "rule": RULE,
        "inference": meta["settings"],
        "selection": {
            "decision": "D3",
            "split": split,
            "frames": c.n,
            "defective": c.defective,
            "defect_rate": c.defective / c.n,
            "image_ids_sha256": id_fingerprint(image_ids),
        },
        "thresholds": [
            {"eps": s.eps, "t": s.t, "feasible": s.feasible,
             "coverage": s.coverage, "selective_risk": s.selective_risk, "escape_rate": s.escape_rate,
             "accepted": s.counts.accepted, "missed": s.counts.missed}
            for s in selections
        ],
    }


def read_thresholds(path: Path) -> dict:
    record = json.loads(Path(path).read_text())
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {record.get('schema_version')}, expected {SCHEMA_VERSION}")
    return record


def write_thresholds(path: Path, record: dict) -> bool:
    """Freeze the record at path. True if written, False if an identical file was already there."""
    path = Path(path)
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() == text:
            return False
        raise FileExistsError(f"{path} already holds different thresholds; frozen thresholds are never "
                              "overwritten. Delete it deliberately, and say why in the commit, to re-freeze.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(text)
    if read_thresholds(path) != record:
        raise RuntimeError(f"{path} does not read back as written")
    return True