"""Deferral metrics at a threshold (T22a; D1, D2 and measures M2-M4 in papers.xlsx sheet 6).

Rule (D1): a frame is auto-passed when s(x) < t; a score exactly equal to t goes to a human.
  N = frames, A = auto-passed frames, M = auto-passed frames that are truly defective (misses),
  D = truly defective frames.
  coverage = |A| / N    selective risk = M / |A| (primary)    escape rate = M / D (secondary)
A ratio whose denominator is zero is undefined and returned as None, never as 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .score_table import ScoredFrame


@dataclass(frozen=True)
class Counts:
    n: int          # N: frames evaluated
    accepted: int   # |A|: frames auto-passed (s < t)
    missed: int     # M: auto-passed frames that are truly defective
    defective: int  # D: truly defective frames


def counts_at(frames: Iterable[ScoredFrame], t: float) -> Counts:
    """Count N, |A|, M and D for threshold t in one pass, so all metrics share one definition."""
    n = accepted = missed = defective = 0
    for f in frames:
        n += 1
        defective += f.defective
        if f.score < t:
            accepted += 1
            missed += f.defective
    return Counts(n, accepted, missed, defective)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def coverage(c: Counts) -> float | None:
    """|A| / N: the share of frames no human has to look at (M2)."""
    return _ratio(c.accepted, c.n)


def selective_risk(c: Counts) -> float | None:
    """M / |A|: the share of auto-passed frames that were truly defective (M3, primary)."""
    return _ratio(c.missed, c.accepted)


def escape_rate(c: Counts) -> float | None:
    """M / D: the share of all defective frames that were auto-passed (M4, secondary)."""
    return _ratio(c.missed, c.defective)