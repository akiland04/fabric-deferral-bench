"""Package a YOLO layout as one zip for a private Kaggle dataset (T88a).

The zip holds images/<split>/ and labels/<split>/ for the chosen splits only, MANIFEST.json (per
split: image count and SHA-256 of the sorted IDs) and verify_package.py, which checks the
unpacked copy on Kaggle without this repo. No data.yaml and no machine paths: the notebook
writes its own data.yaml for wherever Kaggle mounts the copy.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from . import verify_package
from .layout import check_layout
from .thresholds import id_fingerprint
from .yolo import CLASS_NAMES


def build_manifest(manifests: dict[str, list[str]], dataset: str) -> dict:
    return {
        "format": verify_package.FORMAT,
        "dataset": dataset,
        "layout": "yolo",
        "names": {str(k): v for k, v in CLASS_NAMES.items()},
        "splits": {split: {"images": len(ids), "ids_sha256": id_fingerprint(ids)}
                   for split, ids in manifests.items()},
    }


def write_package(zip_path: Path, layout_root: Path, manifests: dict[str, list[str]],
                  dataset: str, image_ext: str = ".jpg") -> dict:
    """Write the zip (refuses to overwrite); returns the manifest stored inside it."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        raise FileExistsError(f"{zip_path} already exists: delete it to re-package")
    problems = check_layout(layout_root, manifests)
    if problems:
        raise ValueError("layout does not match the manifests:\n  " + "\n  ".join(problems))

    manifest = build_manifest(manifests, dataset)
    part = zip_path.with_name(zip_path.name + ".part")     # renamed only when complete
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_STORED) as zf:   # JPEGs are already compressed
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
            zf.write(verify_package.__file__, "verify_package.py")
            for split, ids in manifests.items():
                for image_id in sorted(ids):
                    zf.write(layout_root / "images" / split / f"{image_id}{image_ext}",
                             f"images/{split}/{image_id}{image_ext}")
                    zf.write(layout_root / "labels" / split / f"{image_id}.txt",
                             f"labels/{split}/{image_id}.txt")
        part.rename(zip_path)
    finally:
        part.unlink(missing_ok=True)
    return manifest