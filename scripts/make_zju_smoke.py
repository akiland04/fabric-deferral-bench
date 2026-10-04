"""Carve the ZJU-Leaper smoke subset and build its YOLO view (T20a).

1. Spec: from each split's manifest, draw a seeded sample stratified by fabric group and
   defective/normal, and write splits/zju/smoke_<split>.txt (committed, like the main splits).
2. View: write <layout>/smoke/<split>.txt listing absolute image paths into the layout built by
   convert_zju_yolo.py, plus <layout>/smoke/data.yaml (train and val only). No image is copied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import load_config  # noqa: E402
from fdb.layout import read_manifests, reset_dir  # noqa: E402
from fdb.sampling import stratified_sample  # noqa: E402
from fdb.yolo import write_data_yaml  # noqa: E402
from fdb.zju import load_index  # noqa: E402

SPLITS = ("train", "val", "cal", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the ZJU-Leaper smoke subset and its YOLO view.")
    parser.add_argument("--root", help="layout folder (default: <data_root>/<derived_dir>/zju_yolo)")
    args = parser.parse_args()

    cfg = load_config()
    sizes = cfg["smoke"]["zju_leaper"]
    split_dir = Path(cfg["repo_root"]) / "splits" / "zju"
    root = (Path(args.root) if args.root else
            Path(cfg["data_root"]).expanduser() / cfg["derived_dir"] / "zju_yolo").resolve()
    if not (root / "images").is_dir():
        raise SystemExit(f"no layout at {root}: run scripts/convert_zju_yolo.py first")

    manifests = read_manifests(split_dir, SPLITS)
    meta = {r.image_id: r for r in load_index(cfg)}

    def stratum(image_id: str) -> tuple[int, bool]:
        return meta[image_id].group_id, meta[image_id].defective

    smoke = {s: stratified_sample(manifests[s], stratum, sizes[s], cfg["seed"]) for s in SPLITS}
    for split, ids in smoke.items():
        if not set(ids) <= set(manifests[split]):
            raise SystemExit(f"smoke_{split} contains IDs outside the {split} manifest")
        (split_dir / f"smoke_{split}.txt").write_text("\n".join(ids) + "\n")

    view = root / "smoke"
    reset_dir(view)
    for split, ids in smoke.items():
        paths = [root / "images" / split / f"{image_id}.jpg" for image_id in ids]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} smoke images missing from the layout, e.g. {missing[0]}: "
                             "re-run scripts/convert_zju_yolo.py")
        (view / f"{split}.txt").write_text("".join(f"{p}\n" for p in paths))
    data_yaml = write_data_yaml(view, train="train.txt", val="val.txt")

    for split, ids in smoke.items():
        d = sum(meta[i].defective for i in ids)
        print(f"{split:5s} {len(ids):4d} images  ({d} defective, {len(ids) - d} normal)")
    print(f"spec: {split_dir}/smoke_*.txt")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()