"""Coverage, selective risk and escape rate (T22a). Runs without the dataset or a detector."""
import pytest

from fdb.metrics import Counts, counts_at, coverage, escape_rate, selective_risk
from fdb.score_table import ScoredFrame


def frames(*pairs: tuple[float, bool]) -> list[ScoredFrame]:
    return [ScoredFrame(f"{i:06d}", "cal", s, d) for i, (s, d) in enumerate(pairs)]


# The worked example: 10 frames, 3 defective, threshold 0.5
EXAMPLE = frames((0.9, True), (0.7, False), (0.6, True), (0.4, True),
                 (0.3, False), (0.25, False), (0.2, False), (0.1, False), (0.05, False), (0.0, False))


def test_worked_example():
    c = counts_at(EXAMPLE, 0.5)
    assert c == Counts(n=10, accepted=7, missed=1, defective=3)
    assert coverage(c) == pytest.approx(0.7)
    assert selective_risk(c) == pytest.approx(1 / 7)
    assert escape_rate(c) == pytest.approx(1 / 3)


def test_threshold_zero_auto_passes_nothing():
    c = counts_at(EXAMPLE, 0.0)
    assert (coverage(c), selective_risk(c), escape_rate(c)) == (0.0, None, 0.0)


def test_threshold_above_every_score_auto_passes_everything():
    c = counts_at(EXAMPLE, 1.01)
    assert coverage(c) == 1.0
    assert selective_risk(c) == pytest.approx(0.3)   # = the defect rate of the set
    assert escape_rate(c) == 1.0


def test_score_equal_to_threshold_goes_to_a_human():
    c = counts_at(frames((0.5, True)), 0.5)
    assert c.accepted == 0 and c.missed == 0


def test_coverage_never_falls_as_the_threshold_rises():
    values = [coverage(counts_at(EXAMPLE, t)) for t in (0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.01)]
    assert values == sorted(values)


def test_undefined_ratios_are_none_not_zero():
    no_defects = counts_at(frames((0.1, False), (0.2, False)), 0.5)
    assert escape_rate(no_defects) is None
    assert coverage(counts_at([], 0.5)) is None