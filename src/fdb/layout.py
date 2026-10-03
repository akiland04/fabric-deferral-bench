"""Build and check the on-disk dataset layout YOLO expects (T19b).

    <root>/images/<split>/<id>.<ext>      <root>/labels/<split>/<id>.txt

Dataset-agnostic: every adapter writes into the same layout.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def read_manifests(split_dir: Path, names: tuple[str, ...]) -> dict[str, list[str]]:
    """{split: [image IDs]} from <split_dir>/<split>.txt; fails if any ID is in two splits."""
    manifests = {name: (split_dir / f"{name}.txt").read_text().split() for name in names}
    seen: dict[str, str] = {}
    for name, ids in manifests.items():
        for image_id in ids:
            if image_id in seen:
                raise ValueError(f"{image_id} is in both {seen[image_id]} and {name}")
            seen[image_id] = name
    return manifests


def reset_dir(root: Path) -> None:
    """Delete a generated layout and recreate it empty, so no stale file survives a rebuild."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link src at dst (no extra disk space); copy if linking is impossible."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def check_layout(root: Path, manifests: dict[str, list[str]]) -> list[str]:
    """Compare what is on disk with the manifests. An empty list means they match exactly."""
    problems = []
    for split, ids in manifests.items():
        expected = set(ids)
        for kind in ("images", "labels"):
            on_disk = {p.stem for p in (root / kind / split).glob("*")}
            missing, unexpected = expected - on_disk, on_disk - expected
            if missing or unexpected:
                problems.append(f"{kind}/{split}: {len(missing)} missing, {len(unexpected)} unexpected")
    return problems