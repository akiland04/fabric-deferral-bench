"""Convert ZJU-Leaper XML annotations to YOLO label files (T19a).

Writes one <image_id>.txt per image to <data_root>/<derived_dir>/zju_yolo/labels/.
Every box becomes class 0 ('defect'); normal images get an empty file, which YOLO
treats as a background image. The raw dataset is only read, never written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import dataset_path, load_config  # noqa: E402
from fdb.yolo import to_yolo_line  # noqa: E402
from fdb.zju import load_index, read_boxes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Write YOLO labels for every ZJU-Leaper image.")
    parser.add_argument("--out", help="output folder (default: <data_root>/<derived_dir>/zju_yolo/labels)")
    args = parser.parse_args()

    cfg = load_config()
    xml_dir = dataset_path(cfg, "zju_leaper", "annotations")
    img_dir = dataset_path(cfg, "zju_leaper", "images")
    out = Path(args.out) if args.out else (
        Path(cfg["data_root"]).expanduser() / cfg["derived_dir"] / "zju_yolo" / "labels")
    out.mkdir(parents=True, exist_ok=True)

    n_empty = n_boxes = n_skipped = 0
    index = load_index(cfg)
    for r in index:
        defective, boxes = read_boxes(xml_dir / f"{r.image_id}.xml")
        if defective != r.defective:
            raise ValueError(f"{r.image_id}: XML and ImageSets disagree on 'defective'")
        if defective and not boxes:
            raise ValueError(f"{r.image_id}: marked defective but has no boxes")

        with Image.open(img_dir / f"{r.image_id}.jpg") as im:
            width, height = im.size

        lines = [line for b in boxes if (line := to_yolo_line(b, width, height)) is not None]
        (out / f"{r.image_id}.txt").write_text("".join(f"{line}\n" for line in lines))

        n_empty += not lines
        n_boxes += len(lines)
        n_skipped += len(boxes) - len(lines)

    n_files = len(list(out.glob("*.txt")))
    print(f"images {len(index)}  label files {n_files}  empty {n_empty}  "
          f"boxes {n_boxes}  skipped {n_skipped}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()