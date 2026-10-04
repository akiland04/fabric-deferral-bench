"""Threshold selection t(eps) (T31a, D3). Runs without the dataset or a detector."""
import math
import random

import pytest

from fdb.curve import risk_coverage_curve
from fdb.metrics import counts_at
from fdb.score_table import ScoredFrame
from fdb.selection import DEFER_ALL, select_threshold


def frames(*pairs: tuple[float, bool]) -> list[ScoredFrame]:
    return [ScoredFrame(f"{i:06d}", "cal", s, d) for i, (s, d) in enumerate(pairs)]


# Sorted by score: normal, DEFECTIVE, 8 normal, DEFECTIVE.
# Risk after accepting k frames: 0, 1/2, 1/3, ..., 1/10, then 2/11.
DIP = frames((0.1, False), (0.2, True), *((0.3 + i / 100, False) for i in range(8)), (0.99, True))


def test_takes_the_highest_coverage_not_the_first_breach():
    # risk breaks 10% at k = 2 and only comes back to 10% at k = 10;
    # stopping at the first breach would return k = 1
    sel = select_threshold(risk_coverage_curve(DIP), 0.10)
    assert sel.feasible
    assert sel.counts.accepted == 10
    assert sel.t == 0.99                   # everything strictly below the last frame passes


def test_budget_is_inclusive():
    sel = select_threshold(risk_coverage_curve(DIP), 0.10)
    assert sel.selective_risk == 0.10      # exactly at eps is allowed
    tighter = select_threshold(risk_coverage_curve(DIP), 0.099)
    assert tighter.counts.accepted == 1
    assert tighter.t == 0.2


def test_whole_set_when_the_budget_allows_it():
    fs = frames(*((i / 100, i % 4 == 0) for i in range(100)))     # defect rate 25%
    sel = select_threshold(risk_coverage_curve(fs), 0.30)
    assert sel.coverage == 1.0
    assert sel.t > max(f.score for f in fs)


def test_no_feasible_threshold_defers_everything():
    fs = frames((0.1, True), (0.2, False), (0.3, False))         # best possible risk is 1/3
    sel = select_threshold(risk_coverage_curve(fs), 0.05)
    assert not sel.feasible
    assert sel.t == DEFER_ALL
    assert sel.coverage == 0.0
    assert sel.selective_risk is None      # undefined, never 0
    assert sel.counts == counts_at(fs, sel.t)


def test_never_splits_tied_scores():
    fs = frames((0.1, False), (0.2, True), (0.2, False), (0.3, False))
    sel = select_threshold(risk_coverage_curve(fs), 0.20)
    assert sel.counts.accepted == 1        # the tied pair moves together, so both are deferred
    assert sel.counts == counts_at(fs, sel.t)


def test_agrees_with_counts_at_on_random_data():
    rng = random.Random(42)
    fs = frames(*((round(rng.random(), 2), rng.random() < 0.25) for _ in range(500)))  # many ties
    points = risk_coverage_curve(fs)
    for eps in (0.005, 0.01, 0.02, 0.05, 0.2, 0.5):
        sel = select_threshold(points, eps)
        assert sel.counts == counts_at(fs, sel.t)
        if sel.feasible:
            assert sel.selective_risk <= eps


@pytest.mark.parametrize("eps", [0, 1, -0.01, 1.5, math.nan])
def test_rejects_an_impossible_budget(eps):
    with pytest.raises(ValueError):
        select_threshold(risk_coverage_curve(DIP), eps)


def test_rejects_an_empty_curve():
    with pytest.raises(ValueError):
        select_threshold([], 0.01)