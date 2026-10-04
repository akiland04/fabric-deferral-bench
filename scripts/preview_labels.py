"""Draw a seeded, stratified sample of converted ZJU-Leaper labels for a visual check (T19d).

Boxes are decoded from the YOLO label files the trainer reads (not from the XML) and drawn
over the image, with the ZJU defect mask tinted red as an independent reference.
Writes one contact-sheet PNG and a CSV of what was sampled and why.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import dataset_path, load_config  # noqa: E402
from fdb.yolo import Box, from_yolo_line  # noqa: E402
from fdb.zju import load_index  # noqa: E402

SPLITS = ("train", "val", "cal", "test")
REFERENCE_ID = "002678"   # the worked example from T19a
EDGE = 1e-6               # normalised: a box side this close to 0 or 1 touches the border
TINY = 0.01               # normalised: a box whose shorter side is under 1% of the image
COLS, CAPTION_H = 5, 22


def load_labels(root: Path) -> dict[str, tuple[str, list[str]]]:
    """{image_id: (split, label lines)} for every label file in the layout."""
    labels = {}
    for split in SPLITS:
        for p in sorted((root / "labels" / split).glob("*.txt")):
            labels[p.stem] = (split, p.read_text().splitlines())
    return labels


def norm_boxes(lines: list[str]) -> list[Box]:
    """Boxes in normalised 0-1 units, so samples can be chosen without opening images."""
    return [from_yolo_line(line, 1, 1)[1] for line in lines]


def touches_edge(b: Box) -> bool:
    return b.xmin <= EDGE or b.ymin <= EDGE or b.xmax >= 1 - EDGE or b.ymax >= 1 - EDGE


def is_tiny(b: Box) -> bool:
    return min(b.xmax - b.xmin, b.ymax - b.ymin) < TINY


def choose(labels: dict, group_of: dict[str, int], seed: int) -> dict[str, str]:
    """{image_id: reason}: the reference image, then seeded picks from each risk category."""
    rng = random.Random(seed)
    defective = sorted(i for i, (_, lines) in labels.items() if lines)
    normal = sorted(i for i, (_, lines) in labels.items() if not lines)
    categories = [(f"group {g}", 2, [i for i in defective if group_of[i] == g]) for g in range(1, 6)]
    categories += [
        ("3+ boxes", 2, [i for i in defective if len(labels[i][1]) >= 3]),
        ("edge box", 2, [i for i in defective if any(map(touches_edge, norm_boxes(labels[i][1])))]),
        ("tiny box", 2, [i for i in defective if any(map(is_tiny, norm_boxes(labels[i][1])))]),
        ("normal", 3, normal),
    ]
    picked = {REFERENCE_ID: "reference"}
    for reason, k, pool in categories:
        pool = [i for i in pool if i not in picked]
        for image_id in rng.sample(pool, k):
            picked[image_id] = reason
    return picked


def mask_inside_boxes(mask: Image.Image, boxes: list[Box]) -> float:
    """Share of defect-mask pixels that fall inside at least one box (1.0 = fully enclosed)."""
    defect = np.asarray(mask) > 0
    covered = np.zeros_like(defect)
    for b in boxes:
        covered[round(b.ymin):round(b.ymax), round(b.xmin):round(b.xmax)] = True
    total = defect.sum()
    return float((defect & covered).sum() / total) if total else 1.0


def render(image_path: Path, mask_path: Path, lines: list[str], caption: str, font) -> tuple[Image.Image, float | None]:
    """One panel: image, red-tinted mask, green boxes from the label file, caption strip."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    boxes = [from_yolo_line(line, width, height)[1] for line in lines]
    inside = None
    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        red = Image.new("RGB", img.size, (255, 0, 0))
        img = Image.composite(red, img, mask.point(lambda v: 110 if v else 0))
        inside = mask_inside_boxes(mask, boxes)
    draw = ImageDraw.Draw(img)
    for b in boxes:
        draw.rectangle([b.xmin, b.ymin, b.xmax - 1, b.ymax - 1], outline=(0, 255, 0), width=2)
    panel = Image.new("RGB", (width, height + CAPTION_H), "white")
    panel.paste(img, (0, 0))
    text = caption + ("" if inside is None else f" | mask in boxes {inside:.0%}")
    ImageDraw.Draw(panel).text((4, height + 3), text, fill="black", font=font)
    return panel, inside


def main() -> None:
    parser = argparse.ArgumentParser(description="Contact sheet of converted ZJU-Leaper labels.")
    parser.add_argument("--root", help="layout folder (default: <data_root>/<derived_dir>/zju_yolo)")
    parser.add_argument("--seed", type=int, help="sampling seed (default: seed in config.yaml)")
    parser.add_argument("--out", help="PNG path (default: runs/label_check/zju_labels_seed<seed>.png)")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    root = Path(args.root) if args.root else (
        Path(cfg["data_root"]).expanduser() / cfg["derived_dir"] / "zju_yolo")
    out = Path(args.out) if args.out else (
        Path(cfg["repo_root"]) / cfg["outputs_dir"] / "label_check" / f"zju_labels_seed{seed}.png")
    mask_dir = dataset_path(cfg, "zju_leaper", "masks")
    group_of = {r.image_id: r.group_id for r in load_index(cfg)}

    labels = load_labels(root)
    picked = choose(labels, group_of, seed)
    font = ImageFont.load_default(size=14)

    panels, rows = [], []
    for image_id, reason in picked.items():
        split, lines = labels[image_id]
        caption = f"{image_id} {split} g{group_of[image_id]} | {reason} | boxes {len(lines)}"
        panel, inside = render(root / "images" / split / f"{image_id}.jpg",
                               mask_dir / f"{image_id}.png", lines, caption, font)
        panels.append(panel)
        rows.append([image_id, split, group_of[image_id], reason, len(lines),
                     "" if inside is None else f"{inside:.3f}"])

    pw, ph = panels[0].size
    n_rows = -(-len(panels) // COLS)
    sheet = Image.new("RGB", (COLS * pw, n_rows * ph), "white")
    for k, panel in enumerate(panels):
        sheet.paste(panel, ((k % COLS) * pw, (k // COLS) * ph))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_id", "split", "group", "reason", "boxes", "mask_inside_boxes"])
        w.writerows(rows)

    for r in rows:
        print(f"{r[0]}  {r[1]:5s}  g{r[2]}  {r[3]:10s}  boxes {r[4]}  mask-in {r[5] or '-'}")
    print(f"contact sheet: {out}")


if __name__ == "__main__":
    main()