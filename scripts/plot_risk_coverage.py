"""Plot the risk-coverage curves of a predictions folder as one PNG (T22d).

Reads <predictions>/scores_<split>.csv, rebuilds each curve with fdb.curve (the same code that
writes curve_<split>.csv, so the picture and the numbers cannot disagree) and writes
<predictions>/risk_coverage.png. The risk budgets come from config.yaml; a smoke run is
labelled as a placeholder on the chart itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.curve import risk_coverage_curve  # noqa: E402
from fdb.plots import plot_risk_coverage  # noqa: E402
from fdb.score_table import read_score_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot risk-coverage curves for a predictions folder.")
    parser.add_argument("--predictions", required=True, help="a predictions folder, e.g. runs/predict/smoke")
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    tables = sorted(pred_dir.glob("scores_*.csv"))
    if not tables:
        raise SystemExit(f"no scores_*.csv in {pred_dir}: run scripts/score_frames.py first")
    curves = {t.stem.removeprefix("scores_"): risk_coverage_curve(read_score_table(t)) for t in tables}
    reference = "test" if "test" in curves else next(iter(curves))

    meta = json.loads((pred_dir / "meta.json").read_text())
    note = f"weights {meta['weights_sha256'][:12]}"
    if meta.get("subset") == "smoke":
        note = "Smoke model (placeholder): shows the mechanism, not a result. " + note

    out = plot_risk_coverage(pred_dir / "risk_coverage.png", curves, reference=reference,
                             budgets=load_config()["risk_budgets"],
                             title=f"Risk-coverage, max-confidence score s(x): {pred_dir.name}",
                             note=note)
    print(f"{', '.join(curves)} -> {out} (references from {reference})")


if __name__ == "__main__":
    main()