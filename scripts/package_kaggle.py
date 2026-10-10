"""Zip the ZJU-Leaper YOLO layout for upload as a PRIVATE Kaggle dataset (T88a).

Default: train and val only. Training needs nothing else, and the calibration and test images
then never reach the training machine. The zip goes to <data_root>/<derived_dir>/kaggle/ (outside
the repo and the raw dataset). Upload it to Kaggle as a private dataset; ZJU-Leaper is
non-commercial, academic use only, and must not be redistributed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fdb.config import dataset_path, load_config  # noqa: E402
from fdb.kaggle_package import write_package  # noqa: E402
from fdb.layout import read_manifests  # noqa: E402

ALL_SPLITS = ("train", "val", "cal", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip the ZJU-Leaper YOLO layout for a private Kaggle dataset.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=ALL_SPLITS,
                        help="splits to include (default: train val)")
    parser.add_argument("--out", help="zip path (default: <data_root>/<derived_dir>/kaggle/zju_yolo_<splits>.zip)")
    args = parser.parse_args()

    cfg = load_config()
    derived = Path(cfg["data_root"]).expanduser() / cfg["derived_dir"]
    layout = derived / "zju_yolo"
    splits = tuple(s for s in ALL_SPLITS if s in args.splits)      # fixed order, no duplicates
    out = Path(args.out) if args.out else derived / "kaggle" / f"zju_yolo_{'_'.join(splits)}.zip"

    for protected in (dataset_path(cfg, "zju_leaper"), Path(cfg["repo_root"])):
        p, o = protected.resolve(), out.resolve()
        if o == p or p in o.parents:
            raise SystemExit(f"refusing to write the package inside {protected}")

    manifests = read_manifests(Path(cfg["repo_root"]) / "splits" / "zju", splits)
    manifest = write_package(out, layout, manifests, dataset="ZJU-Leaper")
    for split, e in manifest["splits"].items():
        print(f"{split:5s}  {e['images']:6d} images  ids {e['ids_sha256'][:12]}")
    print(f"package: {out}  ({out.stat().st_size / 1e9:.2f} GB)")
    print("upload it to Kaggle as a PRIVATE dataset, then run verify_package.py in the notebook")


if __name__ == "__main__":
    main()