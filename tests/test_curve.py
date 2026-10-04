"""Risk-coverage curve (T22b). Runs without the dataset or a detector."""
import csv
import math
import random

import pytest

from fdb.curve import risk_coverage_curve, write_curve
from fdb.metrics import Counts, counts_at
from fdb.score_table import ScoredFrame


def frames(*pairs: tuple[float, bool]) -> list[ScoredFrame]:
    return [ScoredFrame(f"{i:06d}", "cal", s, d) for i, (s, d) in enumerate(pairs)]


# 5 frames, two of them tied at 0.2; 2 defective
FIVE = frames((0.9, False), (0.2, True), (0.6, True), (0.1, False), (0.2, False))


def test_hand_worked_curve_with_a_tie():
    points = risk_coverage_curve(FIVE)
    assert [p.t for p in points[:-1]] == [0.1, 0.2, 0.6, 0.9]
    assert points[-1].t > 0.9
    assert [p.counts for p in points] == [
        Counts(5, 0, 0, 2),   # t = 0.1: nothing below
        Counts(5, 1, 0, 2),   # t = 0.2: the 0.1 frame
        Counts(5, 3, 1, 2),   # t = 0.6: plus both 0.2 frames together (one defective)
        Counts(5, 4, 2, 2),   # t = 0.9: plus the defective 0.6 frame
        Counts(5, 5, 2, 2),   # above 0.9: everything
    ]


def random_frames(seed: int, n: int = 300) -> list[ScoredFrame]:
    rng = random.Random(seed)
    # scores rounded to 2 decimals so that many frames tie, like real detector output can
    return frames(*((round(rng.random(), 2), rng.random() < 0.25) for _ in range(n)))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_point_matches_the_single_threshold_reference(seed):
    fs = random_frames(seed)
    for p in risk_coverage_curve(fs):
        assert p.counts == counts_at(fs, p.t)


def test_curve_runs_from_nothing_to_everything():
    fs = random_frames(4)
    points = risk_coverage_curve(fs)
    prevalence = sum(f.defective for f in fs) / len(fs)
    assert points[0].coverage == 0.0 and points[0].selective_risk is None
    assert points[-1].coverage == 1.0
    assert points[-1].selective_risk == pytest.approx(prevalence)


def test_coverage_and_threshold_never_decrease():
    points = risk_coverage_curve(random_frames(5))
    assert [p.coverage for p in points] == sorted(p.coverage for p in points)
    assert [p.t for p in points] == sorted(p.t for p in points)


def test_empty_table_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        risk_coverage_curve([])


def test_csv_keeps_t_exact_and_undefined_as_empty(tmp_path):
    points = risk_coverage_curve(FIVE)
    write_curve(tmp_path / "c.csv", points)
    rows = list(csv.DictReader(open(tmp_path / "c.csv")))
    assert rows[0]["selective_risk"] == ""                   # undefined at zero coverage
    assert float(rows[-1]["t"]) == points[-1].t              # the 'above everything' t survives exactly
    assert all(f.score < float(rows[-1]["t"]) for f in FIVE)