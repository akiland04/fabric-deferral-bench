"""The YOLO data.yaml (T19c): one class, train and val only, absolute paths. Runs without the dataset."""
import pytest
import yaml

from fdb.yolo import write_data_yaml


@pytest.fixture
def layout(tmp_path):
    for split in ("train", "val", "cal", "test"):
        (tmp_path / "images" / split).mkdir(parents=True)
    return tmp_path


def test_data_yaml_contents(layout):
    doc = yaml.safe_load(write_data_yaml(layout, "images/train", "images/val").read_text())
    assert doc == {
        "path": str(layout.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "defect"},
    }


def test_data_yaml_never_mentions_cal_or_test(layout):
    doc = yaml.safe_load(write_data_yaml(layout, "images/train", "images/val").read_text())
    assert "test" not in doc and "cal" not in doc


def test_data_yaml_path_is_absolute(layout):
    doc = yaml.safe_load(write_data_yaml(layout, "images/train", "images/val").read_text())
    assert doc["path"].startswith("/")


def test_data_yaml_refuses_a_missing_folder(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="images/val"):
        write_data_yaml(tmp_path, "images/train", "images/val")