"""Risk-coverage curve: the deferral metrics at every distinct threshold (T22b; M8 in papers.xlsx sheet 6).

Auto-passing changes only when t passes a score, so a set with k distinct scores has exactly
k + 1 outcomes. Point i uses t = the i-th distinct score (everything strictly below it is
auto-passed); the last point uses the smallest float above the highest score (everything is
auto-passed). Computed with one sort and running totals; fdb.metrics.counts_at is the
reference each point must agree with.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Iterable

from .metrics import Counts, coverage, escape_rate, selective_risk
from .score_table import ScoredFrame

CURVE_COLUMNS = ("t", "n", "accepted", "missed", "defective", "coverage", "selective_risk", "escape_rate")


@dataclass(frozen=True)
class CurvePoint:
    t: float
    counts: Counts

    @property
    def coverage(self) -> float | None:
        return coverage(self.counts)

    @property
    def selective_risk(self) -> float | None:
        return selective_risk(self.counts)

    @property
    def escape_rate(self) -> float | None:
        return escape_rate(self.counts)


def risk_coverage_curve(frames: Iterable[ScoredFrame]) -> list[CurvePoint]:
    """Every distinct operating point, from 'auto-pass nothing' to 'auto-pass everything'."""
    rows = sorted(frames, key=lambda f: f.score)
    if not rows:
        raise ValueError("cannot build a curve from an empty score table")
    n = len(rows)
    defective = sum(f.defective for f in rows)

    points = []
    accepted = missed = 0
    for score, group in groupby(rows, key=lambda f: f.score):
        points.append(CurvePoint(score, Counts(n, accepted, missed, defective)))  # t = score: all below pass
        group = list(group)                     # frames sharing this score move together
        accepted += len(group)
        missed += sum(f.defective for f in group)
    points.append(CurvePoint(math.nextafter(rows[-1].score, math.inf), Counts(n, accepted, missed, defective)))
    return points


def write_curve(path: Path, points: Iterable[CurvePoint]) -> None:
    """One row per point. t is written exactly (repr); undefined ratios are left empty, never 0."""
    def fmt(x: float | None) -> str:
        return "" if x is None else f"{x:.6f}"

    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CURVE_COLUMNS)
        for p in points:
            c = p.counts
            w.writerow([repr(p.t), c.n, c.accepted, c.missed, c.defective,
                        fmt(p.coverage), fmt(p.selective_risk), fmt(p.escape_rate)])

def aurc(points: list[CurvePoint]) -> float:
    """Area under the risk-coverage curve (M8): step-wise and tie-aware; lower is better.

    Each step adds one group of tied frames: width = the coverage it adds, height = the
    selective risk once it is added. Without ties this equals the usual definition, the mean
    of the selective risk after accepting the 1, 2, ..., N lowest-scoring frames.
    """
    if len(points) < 2:
        raise ValueError("a curve needs at least two points")
    n = points[0].counts.n
    area = 0.0
    for prev, cur in zip(points, points[1:]):
        width = (cur.counts.accepted - prev.counts.accepted) / n
        area += width * cur.counts.missed / cur.counts.accepted
    return area


def optimal_aurc(n: int, defective: int) -> float:
    """The lowest AURC any score can reach on a set with n frames, `defective` of them defective.

    A perfect score auto-passes every normal frame first (risk 0) and only then the defective
    ones, so the j-th defective frame accepted brings the risk to j / (n - defective + j).
    """
    if not 0 <= defective <= n or n == 0:
        raise ValueError(f"need 0 <= defective <= n and n > 0, got n={n}, defective={defective}")
    normal = n - defective
    return sum(j / (normal + j) for j in range(1, defective + 1)) / n


def excess_aurc(points: list[CurvePoint]) -> float:
    """E-AURC = AURC - optimal AURC: distance from a perfect score, comparable across defect rates."""
    c = points[-1].counts
    return aurc(points) - optimal_aurc(c.n, c.defective)