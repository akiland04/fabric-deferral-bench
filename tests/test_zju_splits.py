"""ZJU-Leaper split files: disjoint, complete, and the official test split untouched (T12e)."""
from pathlib import Path

import pytest

SPLIT_DIR = Path(__file__).resolve().parents[1] / "splits" / "zju"
NAMES = ("train", "val", "cal", "test")
TOTAL_IMAGES = 94833


@pytest.fixture(scope="module")
def splits():
    if not SPLIT_DIR.exists():
        pytest.skip("split files missing: run scripts/make_zju_splits.py first")
    return {n: (SPLIT_DIR / f"{n}.txt").read_text().split() for n in NAMES}


def test_no_image_in_two_splits(splits):
    seen = {}
    for name, ids in splits.items():
        assert len(ids) == len(set(ids)), f"duplicate IDs inside {name}"
        for image_id in ids:
            assert image_id not in seen, f"{image_id} is in both {seen[image_id]} and {name}"
            seen[image_id] = name


def test_splits_cover_every_image(splits):
    assert sum(len(ids) for ids in splits.values()) == TOTAL_IMAGES


def test_official_test_split_untouched(splits):
    try:
        from fdb.config import load_config
        from fdb.zju import load_index
        index = load_index(load_config())
    except (RuntimeError, FileNotFoundError):
        pytest.skip("dataset not available on this machine")
    official_test = sorted(r.image_id for r in index if r.official_split == "test")
    assert splits["test"] == official_test
