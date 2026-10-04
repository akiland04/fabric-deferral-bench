# Changelog

All notable changes to fabric-deferral-bench are recorded here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [Semantic Versioning](https://semver.org/) (0.x until the first frozen experiment).
Task IDs (Txx) refer to the project tracker. Entries marked **Affects results** change data, splits or labels, so outputs from earlier versions are not comparable.

## [Unreleased]

### Added
- Smoke subset (T20a): `scripts/make_zju_smoke.py` writes seeded smoke lists stratified by fabric group and defective/normal (300 train / 100 val / 100 cal / 100 test) to `splits/zju/smoke_*.txt`, and a YOLO view under `<layout>/smoke/` (image-path lists plus a `data.yaml` with train and val only). Each smoke list is checked to lie inside its parent split.
- `fdb.sampling.stratified_sample`: exact-size proportional sampling (largest-remainder rounding), seeded and independent of input order.
- `smoke` sizes in `config.yaml`.
- Tests: `tests/test_sampling.py`, `tests/test_zju_smoke.py`.

## [0.1.0] - 2026-10-04 — ZJU-Leaper data pipeline

### Added
- Project scaffold: shared `config.yaml` with a git-ignored `config.local.yaml`, `FDB_DATA_ROOT` / `FDB_MANIFESTS_DIR` environment overrides, pinned `requirements.lock.txt`, pytest setup.
- Dataset manifest `datasets/zju.yaml` (version, source, counts, checksum, licence).
- Splits (T12): `scripts/make_zju_splits.py` carves val and cal (10% each) from the official ZJU-Leaper train split, stratified by pattern and defective, seed 42; the official test split is untouched. Writes `splits/zju/{train,val,cal,test}.txt` and `summary.csv`.
- ZJU adapter and YOLO label format (T19a): `fdb.zju.read_boxes`, `fdb.yolo.Box`, `fdb.yolo.to_yolo_line`. Single class `defect`; boxes clamped to the image; zero-area boxes dropped.
- YOLO layout (T19b): `scripts/convert_zju_yolo.py` builds `images/<split>/` (hard links to the raw images) and `labels/<split>/` from the split manifests, deleted and rebuilt on every run, with a reconciliation check against the manifests (`fdb.layout`).
- `data.yaml` (T19c): emitted at the end of the build by `fdb.yolo.write_data_yaml`; train and val only, one class, absolute path.
- Label preview (T19d): `scripts/preview_labels.py` draws a seeded, stratified contact sheet from the YOLO label files with the ZJU masks overlaid and a mask-inside-box ratio; `fdb.yolo.from_yolo_line` decodes label lines.

### Fixed
- `splits/zju/summary.csv` is written with LF line endings, so re-running the split script gives byte-identical output.

### Removed
- IDE settings (`.idea/`) from version control.