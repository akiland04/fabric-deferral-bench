"""Write the risk-coverage curve for every score table in a predictions folder (T22b).

Reads <predictions>/scores_<split>.csv and writes <predictions>/curve_<split>.csv: one row per
distinct operating point. The curve is a pure function of the score table, so re-running
rewrites identical files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.curve import aurc, excess_aurc, optimal_aurc, risk_coverage_curve, write_curve  # noqa: E402
from fdb.score_table import read_score_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk-coverage curves for a predictions folder.")
    parser.add_argument("--predictions", required=True, help="a predictions folder, e.g. runs/predict/smoke")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    tables = sorted(pred_dir.glob("scores_*.csv"))
    if not tables:
        raise SystemExit(f"no scores_*.csv in {pred_dir}: run scripts/score_frames.py first")

    for table in tables:
        split = table.stem.removeprefix("scores_")
        points = risk_coverage_curve(read_score_table(table))
        out = pred_dir / f"curve_{split}.csv"
        write_curve(out, points)

        first = next(p for p in points if p.counts.accepted > 0)
        last = points[-1]
        c = last.counts
        print(f"{split:5s} {len(points):6d} points  "
              f"first non-empty: coverage {first.coverage:.1%}, risk {first.selective_risk:.1%}  "
              f"all auto-passed: risk {last.selective_risk:.1%} (= defect rate)  -> {out.name}")
        print(f"      AURC {aurc(points):.4f}   optimal {optimal_aurc(c.n, c.defective):.4f}   "
              f"E-AURC {excess_aurc(points):.4f}   (a useless score scores about {c.defective / c.n:.4f})")


if __name__ == "__main__":
    main()