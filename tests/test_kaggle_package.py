"""Kaggle package and its verifier (T88a). Runs without the dataset."""
import ast
import json
import sys
import zipfile
from pathlib import Path

import pytest

from fdb import verify_package
from fdb.kaggle_package import write_package
from fdb.thresholds import id_fingerprint
from fdb.verify_package import check, data_yaml_text, find_root, fingerprint, read_manifest

MANIFESTS = {"train": ["000001", "000002", "000003"], "val": ["000004"], "test": ["000005"]}
STDLIB_ALLOWED = {"__future__", "argparse", "hashlib", "json", "pathlib"}


def build_layout(root, manifests):
    for split, ids in manifests.items():
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
            for image_id in ids:
                (root / kind / split / f"{image_id}{ext}").write_text(f"{kind} {image_id}")
    return root


def package_and_unpack(tmp_path, splits=("train", "val")):
    layout = build_layout(tmp_path / "layout", MANIFESTS)
    zip_path = tmp_path / "out" / "pkg.zip"
    write_package(zip_path, layout, {s: MANIFESTS[s] for s in splits}, dataset="test-set")
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unpacked)
    return zip_path, unpacked


def test_unpacked_package_verifies(tmp_path):
    _, unpacked = package_and_unpack(tmp_path)
    root = find_root(unpacked)
    assert check(root, read_manifest(root)) == []


def test_zip_holds_only_the_chosen_splits_and_no_paths(tmp_path):
    zip_path, _ = package_and_unpack(tmp_path)
    names = zipfile.ZipFile(zip_path).namelist()
    assert "MANIFEST.json" in names and "verify_package.py" in names
    assert not any("/test/" in n for n in names)          # not packaged
    assert "data.yaml" not in names
    assert all(not n.startswith("/") and ".." not in n for n in names)
    assert str(tmp_path) not in zipfile.ZipFile(zip_path).read("MANIFEST.json").decode()


def test_finds_the_manifest_one_folder_down(tmp_path):
    _, unpacked = package_and_unpack(tmp_path)
    nested = tmp_path / "mount"
    nested.mkdir()
    unpacked.rename(nested / "zju")
    assert find_root(nested) == nested / "zju"


def test_detects_a_missing_label(tmp_path):
    _, root = package_and_unpack(tmp_path)
    (root / "labels" / "train" / "000002.txt").unlink()
    assert check(root, read_manifest(root))


def test_detects_an_extra_file(tmp_path):
    _, root = package_and_unpack(tmp_path)
    (root / "images" / "val" / "999999.jpg").write_text("stray")
    assert check(root, read_manifest(root))


def test_detects_a_swapped_id_with_the_same_count(tmp_path):
    _, root = package_and_unpack(tmp_path)
    for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
        (root / kind / "train" / f"000003{ext}").rename(root / kind / "train" / f"000099{ext}")
    assert check(root, read_manifest(root)) == ["train: IDs differ from MANIFEST.json"]


def test_data_yaml_points_at_the_copy(tmp_path):
    _, root = package_and_unpack(tmp_path)
    text = data_yaml_text(root, read_manifest(root))
    assert text.startswith(f'path: "{root}"\n')
    assert "train: images/train\nval: images/val\nnc: 1\nnames:\n  0: defect\n" in text


def test_data_yaml_needs_train_and_val(tmp_path):
    _, root = package_and_unpack(tmp_path, splits=("train",))
    with pytest.raises(ValueError):
        data_yaml_text(root, read_manifest(root))


def test_refuses_to_overwrite_or_package_a_broken_layout(tmp_path):
    zip_path, _ = package_and_unpack(tmp_path)
    with pytest.raises(FileExistsError):
        write_package(zip_path, tmp_path / "layout", {"train": MANIFESTS["train"]}, dataset="x")
    (tmp_path / "layout" / "labels" / "train" / "000001.txt").unlink()
    broken = tmp_path / "out" / "broken.zip"
    with pytest.raises(ValueError):
        write_package(broken, tmp_path / "layout", {"train": MANIFESTS["train"]}, dataset="x")
    assert not broken.exists() and not broken.with_name("broken.zip.part").exists()


def test_verifier_fingerprint_matches_fdb():
    ids = ["000010", "000002", "000300"]
    assert fingerprint(ids) == id_fingerprint(ids)


def test_verifier_imports_only_the_standard_library():
    tree = ast.parse(Path(verify_package.__file__).read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module and node.level == 0}
    relative = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level > 0]
    assert imported <= STDLIB_ALLOWED and not relative