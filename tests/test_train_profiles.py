"""Training profiles (T27a): config.yaml blocks are valid and split cleanly. Runs without the dataset."""
from pathlib import Path

import pytest
import yaml

from fdb.yolo import split_profile

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_split_profile_separates_model_and_data_from_training_arguments():
    model, view, kwargs = split_profile({"model": "yolo11s.pt", "data": "full", "epochs": 100, "flipud": 0.5})
    assert (model, view, kwargs) == ("yolo11s.pt", "full", {"epochs": 100, "flipud": 0.5})


def test_split_profile_rejects_settings_the_script_owns():
    with pytest.raises(ValueError, match="seed"):
        split_profile({"model": "yolo11s.pt", "data": "full", "seed": 7})


def test_split_profile_requires_model_and_data():
    with pytest.raises(ValueError, match="data"):
        split_profile({"model": "yolo11s.pt", "epochs": 10})


@pytest.mark.parametrize("name", yaml.safe_load(CONFIG.read_text())["train"])
def test_every_profile_in_config_is_valid(name):
    profile = yaml.safe_load(CONFIG.read_text())["train"][name]
    _, view, kwargs = split_profile(profile)
    assert view in ("smoke", "full")
    for key in ("epochs", "imgsz", "batch", "device"):
        assert key in kwargs, f"profile '{name}' has no '{key}'"