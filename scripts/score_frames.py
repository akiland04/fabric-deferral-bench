"""Write the per-frame score tables for a predictions folder (T21b).

For each split in <predictions>/meta.json: score every frame with s(x) (fdb.scoring), join the
ZJU-Leaper ground truth for exactly the images in that split's manifest, and write
<predictions>/scores_<split>.csv. The tables are a pure function of the predictions and the
labels, so re-running rewrites identical files.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.predictions import read_predictions  # noqa: E402
from fdb.score_table import ScoredFrame, join_truth, read_score_table, write_score_table  # noqa: E402
from fdb.scoring import max_conf_score, score_frames  # noqa: E402
from fdb.zju import load_index  # noqa: E402


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def same_table(a: list[ScoredFrame], b: list[ScoredFrame]) -> bool:
    """Equal up to the 6 decimals the CSV stores."""
    def key(r: ScoredFrame) -> tuple:
        return r.image_id, r.split, round(r.score, 6), r.defective
    return [key(r) for r in a] == [key(r) for r in b]


def main() -> None:
    parser = argparse.ArgumentParser(description="Join frame scores with ground truth into score tables.")
    parser.add_argument("--predictions", required=True, help="a predictions folder, e.g. runs/predict/smoke")
    args = parser.parse_args()

    cfg = load_config()
    pred_dir = Path(args.predictions).resolve()
    meta = json.loads((pred_dir / "meta.json").read_text())
    defective = {r.image_id: r.defective for r in load_index(cfg)}

    for split, info in meta["splits"].items():
        ids = (Path(cfg["repo_root"]) / info["manifest"]).read_text().split()
        truth = {i: defective[i] for i in ids}
        scores = score_frames(read_predictions(pred_dir / info["file"]), scorer=max_conf_score)
        rows = join_truth(scores, truth, split)

        out = pred_dir / f"scores_{split}.csv"
        write_score_table(out, rows)
        if not same_table(read_score_table(out), rows):
            raise SystemExit(f"{out} does not read back as written")

        d = [r.score for r in rows if r.defective]
        n = [r.score for r in rows if not r.defective]
        print(f"{split:5s} {len(rows):6d} frames  {len(d):5d} defective  "
              f"mean s(x): defective {mean_or_nan(d):.4f}  normal {mean_or_nan(n):.4f}  -> {out.name}")


if __name__ == "__main__":
    main()