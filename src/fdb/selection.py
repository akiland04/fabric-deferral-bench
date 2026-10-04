"""Operating-point selection (T31a; D3 in papers.xlsx sheet 6).

t(eps) = the threshold giving the highest coverage for which selective risk <= eps. The rule
is pure: it reads a risk-coverage curve and knows nothing about splits. Which data it may see
(source calibration only, for the deployed threshold) is enforced by the caller and by the
leakage test, so the same rule can also give D4's target-oracle threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .curve import CurvePoint
from .metrics import Counts, coverage, escape_rate, selective_risk

# s(x) is never below 0, so "auto-pass when s(x) < 0" auto-passes nothing on any dataset.
DEFER_ALL = 0.0


@dataclass(frozen=True)
class Selection:
    eps: float
    t: float
    counts: Counts      # the calibration counts at t
    feasible: bool      # False: no threshold met eps, so everything goes to a human (t = DEFER_ALL)

    @property
    def coverage(self) -> float | None:
        return coverage(self.counts)

    @property
    def selective_risk(self) -> float | None:
        return selective_risk(self.counts)

    @property
    def escape_rate(self) -> float | None:
        return escape_rate(self.counts)


def select_threshold(points: list[CurvePoint], eps: float) -> Selection:
    """The highest-coverage point of the curve whose selective risk is at most eps (D3)."""
    if not 0 < eps < 1:
        raise ValueError(f"eps must be strictly between 0 and 1, got {eps}")
    if not points:
        raise ValueError("cannot select from an empty curve")

    best = None
    for p in points:            # coverage rises along the curve; no early exit, risk is not monotonic
        risk = p.selective_risk
        if risk is not None and risk <= eps:
            best = p

    if best is None:
        c = points[-1].counts
        return Selection(eps, DEFER_ALL, Counts(c.n, 0, 0, c.defective), feasible=False)
    return Selection(eps, best.t, best.counts, feasible=True)

def sweep(points: list[CurvePoint], budgets: Iterable[float]) -> list[Selection]:
    """t(eps) for every budget in the grid (D3), in ascending eps; the curve is built once by the caller."""
    grid = sorted(budgets)
    if not grid:
        raise ValueError("the risk-budget grid is empty")
    if len(set(grid)) != len(grid):
        raise ValueError(f"the risk-budget grid has duplicates: {grid}")
    return [select_threshold(points, eps) for eps in grid]
