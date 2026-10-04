"""Print coverage, selective risk and escape rate for a score table at a few thresholds (T22a).

Default thresholds: 0, the 25th/50th/75th percentiles of the table's scores, and 1.0, so
the output always spans 'auto-pass nothing' to 'auto-pass almost everything'.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.metrics import counts_at, coverage, escape_rate, selective_risk  # noqa: E402
from fdb.score_table import read_score_table  # noqa: E402


def pct(x: float | None) -> str:
    return "undefined" if x is None else f"{x:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deferral metrics for a score table at chosen thresholds.")
    parser.add_argument("--scores", required=True, help="a score table, e.g. runs/predict/smoke/scores_cal.csv")
    parser.add_argument("--t", type=float, nargs="+", help="thresholds (default: 0, score quartiles, 1.0)")
    args = parser.parse_args()

    frames = read_score_table(Path(args.scores))
    if args.t:
        thresholds = args.t
    else:
        q1, q2, q3 = statistics.quantiles([f.score for f in frames], n=4)
        thresholds = [0.0, q1, q2, q3, 1.0]

    print(f"{'t':>9}  {'N':>6}  {'|A|':>6}  {'M':>5}  {'D':>5}  {'coverage':>9}  {'sel. risk':>9}  {'escape':>9}")
    for t in thresholds:
        c = counts_at(frames, t)
        print(f"{t:9.6f}  {c.n:6d}  {c.accepted:6d}  {c.missed:5d}  {c.defective:5d}  "
              f"{pct(coverage(c)):>9}  {pct(selective_risk(c)):>9}  {pct(escape_rate(c)):>9}")


if __name__ == "__main__":
    main()