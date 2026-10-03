"""ZJU-Leaper metadata: official split, fabric pattern and group for every image."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import dataset_path

N_PATTERNS = 19
N_GROUPS = 5


@dataclass(frozen=True)
class ZjuImage:
    image_id: str        # "000001" -> Images/000001.jpg
    pattern_id: int      # 1..19
    group_id: int        # 1..5
    defective: bool
    official_split: str  # "train" or "test"


def _entries(path: Path):
    """Yield (image_id, defective, split) from a {normal|defect: {train|test: [ids]}} file."""
    data = json.loads(path.read_text())
    for label, splits in data.items():
        for split, ids in splits.items():
            for image_id in ids:
                yield image_id, label == "defect", split


def load_index(cfg: dict) -> list[ZjuImage]:
    """Every image in the release, sorted by ID, built from ImageSets/Patterns and Groups."""
    image_sets = dataset_path(cfg, "zju_leaper", "official_splits").parent
    group_of = {}
    for g in range(1, N_GROUPS + 1):
        for image_id, _, _ in _entries(image_sets / "Groups" / f"group{g}.json"):
            group_of[image_id] = g
    index = [
        ZjuImage(image_id, p, group_of[image_id], defective, split)
        for p in range(1, N_PATTERNS + 1)
        for image_id, defective, split in _entries(image_sets / "Patterns" / f"pattern{p}.json")
    ]
    return sorted(index, key=lambda r: r.image_id)
