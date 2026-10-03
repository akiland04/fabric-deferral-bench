"""Load the shared config.yaml merged with the machine-specific config.local.yaml."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Environment variables override the config files (used on Kaggle and in tests).
ENV_OVERRIDES = {"data_root": "FDB_DATA_ROOT", "manifests_dir": "FDB_MANIFESTS_DIR"}


def load_config(repo_root: Path = REPO_ROOT) -> dict:
    cfg = yaml.safe_load((repo_root / "config.yaml").read_text())
    local = repo_root / "config.local.yaml"
    if local.exists():
        cfg.update(yaml.safe_load(local.read_text()) or {})
    for key, env in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    if not cfg.get("data_root"):
        raise RuntimeError("data_root is not set: create config.local.yaml or set FDB_DATA_ROOT")
    cfg["repo_root"] = str(repo_root)
    return cfg


def dataset_path(cfg: dict, dataset: str, part: str | None = None) -> Path:
    """Path to a dataset folder, or to one of its parts named in config.yaml."""
    ds = cfg["datasets"][dataset]
    base = Path(cfg["data_root"]).expanduser() / ds["dir"]
    return base / ds[part] if part else base
