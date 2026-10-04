"""Risk-coverage plot (T22d). Checks the reference curve and that a real PNG is written;
whether the picture looks right is checked by eye, not by pixels."""
import pytest

from fdb.curve import optimal_aurc, risk_coverage_curve
from fdb.plots import optimal_curve, plot_risk_coverage
from fdb.score_table import ScoredFrame

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def curve(split: str):
    pairs = [(0.1, False), (0.2, True), (0.3, False), (0.3, True), (0.9, True)]
    return risk_coverage_curve(ScoredFrame(f"{i:06d}", split, s, d) for i, (s, d) in enumerate(pairs))


def test_optimal_curve_is_zero_until_the_normal_frames_run_out():
    xs, ys = optimal_curve(4, 1)
    assert xs == [0.25, 0.5, 0.75, 1.0]
    assert ys == [0.0, 0.0, 0.0, 0.25]


def test_optimal_curve_area_is_the_optimal_aurc():
    # each step is 1/n wide, so the mean height is the area the T22c formula gives
    _, ys = optimal_curve(100, 25)
    assert sum(ys) / 100 == pytest.approx(optimal_aurc(100, 25))


def test_optimal_curve_rejects_impossible_counts():
    with pytest.raises(ValueError):
        optimal_curve(10, 11)
    with pytest.raises(ValueError):
        optimal_curve(0, 0)


def test_writes_a_png_and_returns_its_path(tmp_path):
    out = plot_risk_coverage(tmp_path / "rc.png", {"cal": curve("cal"), "test": curve("test")},
                             reference="test", budgets=[0.05, 0.01], title="t", note="placeholder")
    assert out == tmp_path / "rc.png"
    assert out.read_bytes().startswith(PNG_SIGNATURE)


def test_same_curves_give_identical_bytes(tmp_path):
    curves = {"cal": curve("cal")}
    a = plot_risk_coverage(tmp_path / "a.png", curves, reference="cal").read_bytes()
    b = plot_risk_coverage(tmp_path / "b.png", curves, reference="cal").read_bytes()
    assert a == b


def test_rejects_no_curves_or_an_unknown_reference(tmp_path):
    with pytest.raises(ValueError):
        plot_risk_coverage(tmp_path / "x.png", {}, reference="test")
    with pytest.raises(ValueError):
        plot_risk_coverage(tmp_path / "x.png", {"cal": curve("cal")}, reference="test")
    assert not (tmp_path / "x.png").exists()