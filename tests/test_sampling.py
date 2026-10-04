"""Stratified sampling (T20a). Runs without the dataset."""
from collections import Counter

import pytest

from fdb.sampling import stratified_sample

# 1,000 fake IDs: 25% defective ("d"), 75% normal ("n"), like ZJU-Leaper
IDS = [f"{i:06d}" for i in range(1000)]
KIND = {i: ("d" if int(i) % 4 == 0 else "n") for i in IDS}


def test_exact_size_and_subset():
    picked = stratified_sample(IDS, KIND.get, 300, seed=42)
    assert len(picked) == len(set(picked)) == 300
    assert set(picked) <= set(IDS)


def test_keeps_the_parent_mix():
    picked = stratified_sample(IDS, KIND.get, 300, seed=42)
    assert Counter(KIND[i] for i in picked) == {"d": 75, "n": 225}


def test_same_seed_same_sample_different_seed_different_sample():
    assert stratified_sample(IDS, KIND.get, 100, seed=42) == stratified_sample(IDS, KIND.get, 100, seed=42)
    assert stratified_sample(IDS, KIND.get, 100, seed=42) != stratified_sample(IDS, KIND.get, 100, seed=7)


def test_input_order_does_not_matter():
    assert stratified_sample(IDS, KIND.get, 100, seed=42) == stratified_sample(IDS[::-1], KIND.get, 100, seed=42)


def test_rounding_still_gives_exact_size():
    kind = {i: ("a", "b", "c")[int(i) % 3] for i in IDS}  # 334 / 333 / 333: quotas are not whole numbers
    assert len(stratified_sample(IDS, kind.get, 100, seed=42)) == 100


def test_refuses_more_than_available():
    with pytest.raises(ValueError):
        stratified_sample(IDS[:10], KIND.get, 11, seed=42)