"""Select and freeze the operating point t(eps) for every risk budget (T31b, T31c; D3).

Reads ONLY <predictions>/scores_cal.csv: the threshold must never see test or target data. Prints
one row per budget in config.yaml risk_budgets, with the calibration defect rate beside them,
then freezes them to thresholds/<name>.json bound to the weights in <predictions>/meta.json.
An identical existing file is left alone; a different one is refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.curve import risk_coverage_curve  # noqa: E402
from fdb.score_table import read_score_table  # noqa: E402
from fdb.selection import sweep  # noqa: E402
from fdb.thresholds import build_record, write_thresholds  # noqa: E402

CAL_SPLIT = "cal"
CAL_TABLE = f"scores_{CAL_SPLIT}.csv"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.2%}"


def main() -> None:
    parser = argparse.ArgumentParser(description="t(eps) for every risk budget, on calibration scores only.")
    parser.add_argument("--predictions", required=True, help="a predictions folder, e.g. runs/predict/smoke")
    parser.add_argument("--name", help="output file thresholds/<name>.json (default: the predictions folder name)")
    args = parser.parse_args()
    cfg = load_config()

    pred_dir = Path(args.predictions)
    table = pred_dir / CAL_TABLE
    if not table.exists():
        raise SystemExit(f"no {table}: run scripts/score_frames.py first")
    frames = read_score_table(table)
    splits = sorted({f.split for f in frames})
    if splits != [CAL_SPLIT]:
        raise SystemExit(f"{table} holds splits {splits}; thresholds are selected on '{CAL_SPLIT}' only")

    meta = json.loads((pred_dir / "meta.json").read_text())
    expected = meta["splits"][CAL_SPLIT]["images"]
    if len(frames) != expected:
        raise SystemExit(f"{table} has {len(frames)} frames but the predictions had {expected}: re-run score_frames.py")

    points = risk_coverage_curve(frames)
    c = points[-1].counts
    print(f"calibration: {c.n} frames, {c.defective} defective (defect rate {c.defective / c.n:.2%})")
    print(f"{'eps':>6}  {'t':>22}  {'coverage':>8}  {'risk':>7}  {'escape':>7}  {'passed':>6}  {'missed':>6}  feasible")
    selections = sweep(points, cfg["risk_budgets"])
    for s in selections:
        print(f"{s.eps:>6.1%}  {s.t!r:>22}  {pct(s.coverage):>8}  {pct(s.selective_risk):>7}  "
              f"{pct(s.escape_rate):>7}  {s.counts.accepted:>6}  {s.counts.missed:>6}  {'yes' if s.feasible else 'NO'}")

    record = build_record(selections, meta, CAL_SPLIT, [f.image_id for f in frames])
    out = Path(cfg["repo_root"]) / "thresholds" / f"{args.name or pred_dir.name}.json"
    written = write_thresholds(out, record)
    print(f"{'frozen' if written else 'unchanged (identical file already there)'}: {out}  "
          f"weights {meta['weights_sha256'][:12]}")


if __name__ == "__main__":
    main()