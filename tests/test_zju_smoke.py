"""Committed ZJU-Leaper smoke subset (T20a): configured sizes, each piece inside its parent split."""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO / "splits" / "zju"
NAMES = ("train", "val", "cal", "test")


@pytest.fixture(scope="module")
def lists():
    if not (SPLIT_DIR / "smoke_train.txt").exists():
        pytest.skip("smoke files missing: run scripts/make_zju_smoke.py first")
    return {n: ((SPLIT_DIR / f"{n}.txt").read_text().split(),
                (SPLIT_DIR / f"smoke_{n}.txt").read_text().split()) for n in NAMES}


def test_each_smoke_piece_is_inside_its_parent_split(lists):
    for name, (parent, smoke) in lists.items():
        assert set(smoke) <= set(parent), f"smoke_{name} has IDs outside {name}"


def test_smoke_sizes_match_config(lists):
    sizes = yaml.safe_load((REPO / "config.yaml").read_text())["smoke"]["zju_leaper"]
    for name, (_, smoke) in lists.items():
        assert len(smoke) == len(set(smoke)) == sizes[name]