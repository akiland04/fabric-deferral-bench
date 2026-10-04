"""Build the YOLO dataset for ZJU-Leaper from the split manifests (T19a, T19b).

For every image ID in splits/zju/<split>.txt it writes
    <data_root>/<derived_dir>/zju_yolo/labels/<split>/<id>.txt   (YOLO labels, class 0 'defect')
    <data_root>/<derived_dir>/zju_yolo/images/<split>/<id>.jpg   (hard link to the raw image)
The layout is deleted and rebuilt on every run, then checked against the manifests.
The raw dataset is only read, never written.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import dataset_path, load_config  # noqa: E402
from fdb.layout import check_layout, link_or_copy, read_manifests, reset_dir  # noqa: E402
from fdb.zju import load_index, read_boxes  # noqa: E402
from fdb.yolo import to_yolo_line, write_data_yaml  # noqa: E402

SPLITS = ("train", "val", "cal", "test")


def yolo_lines(xml_path: Path, img_path: Path, expected_defective: bool) -> tuple[list[str], int]:
    """YOLO label lines for one image, and how many boxes were dropped as zero-area."""
    defective, boxes = read_boxes(xml_path)
    if defective != expected_defective:
        raise ValueError(f"{xml_path.stem}: XML and ImageSets disagree on 'defective'")
    if defective and not boxes:
        raise ValueError(f"{xml_path.stem}: marked defective but has no boxes")
    with Image.open(img_path) as im:
        width, height = im.size
    lines = [line for b in boxes if (line := to_yolo_line(b, width, height)) is not None]
    return lines, len(boxes) - len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ZJU-Leaper YOLO dataset from the split manifests.")
    parser.add_argument("--out", help="output folder (default: <data_root>/<derived_dir>/zju_yolo)")
    parser.add_argument("--splits", help="manifest folder (default: splits/zju in the repo)")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = dataset_path(cfg, "zju_leaper")
    xml_dir = dataset_path(cfg, "zju_leaper", "annotations")
    img_dir = dataset_path(cfg, "zju_leaper", "images")
    split_dir = Path(args.splits) if args.splits else Path(cfg["repo_root"]) / "splits" / "zju"
    out = Path(args.out) if args.out else (
        Path(cfg["data_root"]).expanduser() / cfg["derived_dir"] / "zju_yolo")

    if out.resolve() == raw_dir.resolve() or raw_dir.resolve() in out.resolve().parents:
        raise SystemExit(f"refusing to write inside the raw dataset: {out}")

    manifests = read_manifests(split_dir, SPLITS)
    meta = {r.image_id: r for r in load_index(cfg)}
    listed = {i for ids in manifests.values() for i in ids}
    if listed != set(meta):
        raise SystemExit(f"manifests list {len(listed)} images but the dataset has {len(meta)}: "
                         "re-run scripts/make_zju_splits.py")

    reset_dir(out)
    for split, ids in manifests.items():
        img_out, lbl_out = out / "images" / split, out / "labels" / split
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        n_empty = n_boxes = n_skipped = 0
        how = Counter()
        for image_id in ids:
            src = img_dir / f"{image_id}.jpg"
            lines, skipped = yolo_lines(xml_dir / f"{image_id}.xml", src, meta[image_id].defective)
            (lbl_out / f"{image_id}.txt").write_text("".join(f"{line}\n" for line in lines))
            how[link_or_copy(src, img_out / src.name)] += 1
            n_empty += not lines
            n_boxes += len(lines)
            n_skipped += skipped
        print(f"{split:5s}  images {len(ids):6d}  empty {n_empty:6d}  boxes {n_boxes:6d}  "
              f"skipped {n_skipped}  (linked {how['link']}, copied {how['copy']})")

    problems = check_layout(out, manifests)
    if problems:
        raise SystemExit("layout does not match the manifests:\n  " + "\n  ".join(problems))
    data_yaml = write_data_yaml(out, train="images/train", val="images/val")
    print(f"layout matches the manifests: {out}")
    print(f"data.yaml written: {data_yaml}")


if __name__ == "__main__":
    main()