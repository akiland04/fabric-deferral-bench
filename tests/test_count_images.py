"""Counting the training images behind a data.yaml (T20b). Runs without the dataset."""
from fdb.yolo import count_train_images, write_data_yaml


def test_counts_a_txt_list(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "train.txt").write_text("/a/1.jpg\n/a/2.jpg\n\n/a/3.jpg\n")
    (tmp_path / "val.txt").write_text("/a/4.jpg\n")
    assert count_train_images(write_data_yaml(tmp_path, "train.txt", "val.txt")) == 3


def test_counts_a_folder(tmp_path):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
    for name in ("1.jpg", "2.jpg", "notes.txt"):
        (tmp_path / "images" / "train" / name).write_text("")
    assert count_train_images(write_data_yaml(tmp_path, "images/train", "images/val")) == 2