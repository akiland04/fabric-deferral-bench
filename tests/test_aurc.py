"""AURC, optimal AURC and E-AURC (T22c). Runs without the dataset or a detector."""
import random

import pytest

from fdb.curve import aurc, excess_aurc, optimal_aurc, risk_coverage_curve
from fdb.score_table import ScoredFrame


def frames(*pairs: tuple[float, bool]) -> list[ScoredFrame]:
    return [ScoredFrame(f"{i:06d}", "cal", s, d) for i, (s, d) in enumerate(pairs)]


def labels(n: int, defective: int) -> list[bool]:
    return [True] * defective + [False] * (n - defective)


def test_all_scores_tied_gives_the_defect_rate():
    fs = frames(*((0.5, d) for d in labels(100, 25)))
    assert aurc(risk_coverage_curve(fs)) == pytest.approx(0.25)


def test_perfect_ranking_reaches_the_optimum():
    # every normal frame scores below every defective one; defective scores are distinct
    fs = frames(*((0.1, False) for _ in range(75)), *((0.5 + j / 100, True) for j in range(25)))
    points = risk_coverage_curve(fs)
    assert aurc(points) == pytest.approx(optimal_aurc(100, 25))
    assert excess_aurc(points) == pytest.approx(0.0)


def test_without_ties_equals_the_mean_of_risk_after_each_frame():
    rng = random.Random(7)
    fs = frames(*((rng.random(), rng.random() < 0.3) for _ in range(200)))   # distinct scores
    ranked = sorted(fs, key=lambda f: f.score)
    missed = 0
    risks = []
    for k, f in enumerate(ranked, start=1):
        missed += f.defective
        risks.append(missed / k)
    assert aurc(risk_coverage_curve(fs)) == pytest.approx(sum(risks) / len(risks))


def test_random_scores_land_near_the_defect_rate():
    rng = random.Random(8)
    fs = frames(*((rng.random(), rng.random() < 0.25) for _ in range(20000)))
    prevalence = sum(f.defective for f in fs) / len(fs)
    assert aurc(risk_coverage_curve(fs)) == pytest.approx(prevalence, abs=0.01)


def test_optimum_matches_known_values():
    assert optimal_aurc(100, 0) == 0.0             # nothing defective: no risk at all
    assert optimal_aurc(4, 2) == pytest.approx((1 / 3 + 2 / 4) / 4)
    assert optimal_aurc(100, 25) == pytest.approx(0.0355, abs=0.0001)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        optimal_aurc(10, 11)
    with pytest.raises(ValueError):
        aurc([])


def test_tied_scores_count_at_the_end_of_their_group():
    # the same perfect order, but all defective frames share one score: no threshold can
    # split them, so the curve only reaches their group's end and the area is larger
    fs = frames(*((0.1, False) for _ in range(75)), *((0.9, True) for _ in range(25)))
    assert aurc(risk_coverage_curve(fs)) == pytest.approx(0.25 * 0.25)
    assert aurc(risk_coverage_curve(fs)) > optimal_aurc(100, 25)