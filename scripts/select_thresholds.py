"""Select the operating point t(eps) for every risk budget on calibration scores (T31b; D3).

Reads ONLY <predictions>/scores_cal.csv: the threshold must never see test or target data. Prints
one row per budget in config.yaml risk_budgets, with the calibration defect rate beside them.
Saving the thresholds is T31c; evaluating them on other data comes later.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.curve import risk_coverage_curve  # noqa: E402
from fdb.score_table import read_score_table  # noqa: E402
from fdb.selection import sweep  # noqa: E402

CAL_TABLE = "scores_cal.csv"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="t(eps) for every risk budget, on calibration scores only.")
    parser.add_argument("--predictions", required=True, help="a predictions folder, e.g. runs/predict/smoke")
    args = parser.parse_args()

    table = Path(args.predictions) / CAL_TABLE
    if not table.exists():
        raise SystemExit(f"no {table}: run scripts/score_frames.py first")
    frames = read_score_table(table)
    splits = sorted({f.split for f in frames})
    if splits != ["cal"]:
        raise SystemExit(f"{table} holds splits {splits}; thresholds are selected on 'cal' only")

    points = risk_coverage_curve(frames)
    c = points[-1].counts
    print(f"calibration: {c.n} frames, {c.defective} defective (defect rate {c.defective / c.n:.2%})")
    print(f"{'eps':>6}  {'t':>22}  {'coverage':>8}  {'risk':>7}  {'escape':>7}  {'passed':>6}  {'missed':>6}  feasible")
    for s in sweep(points, load_config()["risk_budgets"]):
        print(f"{s.eps:>6.1%}  {s.t!r:>22}  {pct(s.coverage):>8}  {pct(s.selective_risk):>7}  "
              f"{pct(s.escape_rate):>7}  {s.counts.accepted:>6}  {s.counts.missed:>6}  {'yes' if s.feasible else 'NO'}")


if __name__ == "__main__":
    main()