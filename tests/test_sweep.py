"""The epsilon sweep (T31b, D3): one t per budget, chosen the same way from one curve."""
import random

import pytest

from fdb.curve import risk_coverage_curve
from fdb.score_table import ScoredFrame
from fdb.selection import select_threshold, sweep

GRID = [0.005, 0.01, 0.02, 0.05]


def random_curve(seed: int = 42, n: int = 2000, rate: float = 0.05):
    rng = random.Random(seed)
    frames = []
    for i in range(n):
        defective = rng.random() < rate
        score = round(min(1.0, rng.random() * 0.6 + (0.4 if defective else 0.0)), 3)  # defects score higher
        frames.append(ScoredFrame(f"{i:06d}", "cal", score, defective))
    return risk_coverage_curve(frames)


def test_one_selection_per_budget_in_ascending_order():
    results = sweep(random_curve(), [0.05, 0.005, 0.02, 0.01])
    assert [s.eps for s in results] == GRID


def test_each_result_is_the_single_budget_selection():
    points = random_curve()
    assert sweep(points, GRID) == [select_threshold(points, eps) for eps in GRID]


@pytest.mark.parametrize("seed", range(5))
def test_looser_budget_never_lowers_coverage_or_t(seed):
    results = sweep(random_curve(seed), GRID + [0.1, 0.3])
    for tighter, looser in zip(results, results[1:]):
        assert looser.counts.accepted >= tighter.counts.accepted
        assert looser.t >= tighter.t


def test_rejects_an_empty_grid():
    with pytest.raises(ValueError):
        sweep(random_curve(), [])


def test_rejects_duplicate_budgets():
    with pytest.raises(ValueError):
        sweep(random_curve(), [0.01, 0.02, 0.01])


def test_rejects_a_percent_typo():
    with pytest.raises(ValueError):
        sweep(random_curve(), [0.005, 0.01, 2, 0.05])    # 2 instead of 0.02