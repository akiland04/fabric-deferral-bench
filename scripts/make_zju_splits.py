"""Carve validation and calibration splits from ZJU-Leaper's official train split (T12a-c).

The official test split is kept untouched. Inside the official train split, each
(fabric pattern, defective) group of images is shuffled with the configured seed and
divided into train / val / cal, so every split keeps the same pattern and defect mix.
Writes one sorted image-ID list per split, plus a count summary, to splits/zju/.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.zju import load_index  # noqa: E402

SPLITS = ("train", "val", "cal", "test")


def carve(index, seed: int, val_frac: float, cal_frac: float) -> dict[str, list[str]]:
    strata = defaultdict(list)
    for r in index:
        if r.official_split == "train":
            strata[(r.pattern_id, r.defective)].append(r.image_id)
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_val, n_cal = round(len(ids) * val_frac), round(len(ids) * cal_frac)
        out["val"] += ids[:n_val]
        out["cal"] += ids[n_val:n_val + n_cal]
        out["train"] += ids[n_val + n_cal:]
    out["test"] = [r.image_id for r in index if r.official_split == "test"]
    return {s: sorted(ids) for s, ids in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the ZJU-Leaper split files.")
    parser.add_argument("--out", help="output folder (default: splits/zju in the repo)")
    args = parser.parse_args()

    cfg = load_config()
    fracs = cfg["splits"]["zju_leaper"]
    index = load_index(cfg)
    splits = carve(index, cfg["seed"], fracs["val_frac"], fracs["cal_frac"])

    out = Path(args.out) if args.out else Path(cfg["repo_root"]) / "splits" / "zju"
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / f"{name}.txt").write_text("\n".join(ids) + "\n")

    meta = {r.image_id: r for r in index}
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "pattern_id", "group_id", "defective", "images"])
        for name, ids in splits.items():
            counts = Counter((meta[i].pattern_id, meta[i].group_id, meta[i].defective) for i in ids)
            for (p, g, d), n in sorted(counts.items()):
                w.writerow([name, p, g, int(d), n])

    for name, ids in splits.items():
        d = sum(meta[i].defective for i in ids)
        print(f"{name:5s} {len(ids):6d} images  ({d} defective, {len(ids) - d} normal)")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
