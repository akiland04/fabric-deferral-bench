"""Dataset layout helpers (T19b). Runs without the dataset."""
import pytest

from fdb.layout import check_layout, link_or_copy, read_manifests, reset_dir


def write_manifests(split_dir, manifests):
    split_dir.mkdir()
    for name, ids in manifests.items():
        (split_dir / f"{name}.txt").write_text("\n".join(ids) + "\n")


def build(root, manifests):
    for split, ids in manifests.items():
        for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
            for image_id in ids:
                (root / kind / split / f"{image_id}{ext}").write_text("")


def test_read_manifests(tmp_path):
    write_manifests(tmp_path / "s", {"train": ["001", "002"], "test": ["003"]})
    assert read_manifests(tmp_path / "s", ("train", "test")) == {"train": ["001", "002"], "test": ["003"]}


def test_read_manifests_rejects_an_id_in_two_splits(tmp_path):
    write_manifests(tmp_path / "s", {"train": ["001", "002"], "test": ["002"]})
    with pytest.raises(ValueError, match="002 is in both train and test"):
        read_manifests(tmp_path / "s", ("train", "test"))


def test_reset_dir_removes_stale_files(tmp_path):
    root = tmp_path / "layout"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "train" / "old.jpg").write_text("stale")
    reset_dir(root)
    assert root.exists() and list(root.iterdir()) == []


def test_link_or_copy_hard_links_on_the_same_disk(tmp_path):
    src, dst = tmp_path / "a.jpg", tmp_path / "b.jpg"
    src.write_bytes(b"pixels")
    assert link_or_copy(src, dst) == "link"
    assert dst.read_bytes() == b"pixels"
    assert dst.stat().st_ino == src.stat().st_ino  # same file on disk, not a copy


def test_check_layout_accepts_an_exact_match(tmp_path):
    manifests = {"train": ["001", "002"], "test": ["003"]}
    build(tmp_path, manifests)
    assert check_layout(tmp_path, manifests) == []


def test_check_layout_catches_ghost_and_missing_files(tmp_path):
    manifests = {"train": ["001", "002"], "test": ["003"]}
    build(tmp_path, manifests)
    (tmp_path / "images" / "train" / "003.jpg").write_text("")  # ghost: a test image also in train
    (tmp_path / "labels" / "test" / "003.txt").unlink()           # missing label
    assert check_layout(tmp_path, manifests) == [
        "images/train: 0 missing, 1 unexpected",
        "labels/test: 1 missing, 0 unexpected",
    ]