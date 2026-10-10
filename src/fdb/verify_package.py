"""Verify an unpacked copy of a packaged YOLO layout and write its data.yaml (T88a).

STANDARD LIBRARY ONLY: this file ships inside the Kaggle package and runs there without this
repo, so it must not import fdb or any third-party module (a test enforces this).

    python verify_package.py --root /kaggle/input/<dataset> --out /kaggle/working/data.yaml

For every split in MANIFEST.json it checks one image and one label per ID, nothing extra, and
that the SHA-256 of the sorted IDs matches; then it writes data.yaml (train and val only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FORMAT = 1


def fingerprint(image_ids) -> str:
    """Same algorithm as fdb.thresholds.id_fingerprint (a test checks they agree)."""
    ids = sorted(image_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image IDs")
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def find_root(path: Path) -> Path:
    """The folder holding MANIFEST.json: the given folder, or one level below it."""
    for candidate in (path, *sorted(p for p in path.iterdir() if p.is_dir())):
        if (candidate / "MANIFEST.json").is_file():
            return candidate
    raise FileNotFoundError(f"no MANIFEST.json in {path} or its subfolders")


def read_manifest(root: Path) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"MANIFEST.json format {manifest.get('format')}, expected {FORMAT}")
    return manifest


def check(root: Path, manifest: dict) -> list[str]:
    """Compare the unpacked files with MANIFEST.json. An empty list means they match exactly."""
    problems = []
    for split, expected in manifest["splits"].items():
        stems = {}
        for kind in ("images", "labels"):
            folder = root / kind / split
            stems[kind] = [p.stem for p in folder.iterdir()] if folder.is_dir() else []
            if len(stems[kind]) != expected["images"]:
                problems.append(f"{kind}/{split}: {len(stems[kind])} files, expected {expected['images']}")
        if set(stems["images"]) != set(stems["labels"]):
            problems.append(f"{split}: image and label IDs differ")
        elif len(set(stems["images"])) != len(stems["images"]):
            problems.append(f"{split}: two images share an ID")
        elif fingerprint(stems["images"]) != expected["ids_sha256"]:
            problems.append(f"{split}: IDs differ from MANIFEST.json")
    return problems


def data_yaml_text(root: Path, manifest: dict) -> str:
    """Ultralytics data.yaml for training: train and val only, absolute path to this copy."""
    missing = {"train", "val"} - set(manifest["splits"])
    if missing:
        raise ValueError(f"the package has no {sorted(missing)} split: it cannot be trained on")
    names = "".join(f"  {k}: {v}\n" for k, v in sorted(manifest["names"].items(), key=lambda kv: int(kv[0])))
    return (f"path: {json.dumps(str(root))}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(manifest['names'])}\nnames:\n{names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an unpacked YOLO package and write data.yaml.")
    parser.add_argument("--root", required=True, help="the attached dataset, e.g. /kaggle/input/<dataset>")
    parser.add_argument("--out", required=True, help="where to write data.yaml, e.g. /kaggle/working/data.yaml")
    args = parser.parse_args()

    root = find_root(Path(args.root))
    manifest = read_manifest(root)
    for split, e in manifest["splits"].items():
        print(f"{split:5s}  {e['images']:6d} images  ids {e['ids_sha256'][:12]}")
    problems = check(root, manifest)
    if problems:
        raise SystemExit("the copy does not match MANIFEST.json:\n  " + "\n  ".join(problems))
    out = Path(args.out)
    out.write_text(data_yaml_text(root, manifest))
    print(f"verified: {root}  ({manifest['dataset']})")
    print(f"data.yaml written: {out}")


if __name__ == "__main__":
    main()