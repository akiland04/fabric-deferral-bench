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
- Training (T20b): `scripts/train_yolo.py` trains a YOLO detector from a named profile in `config.yaml` and writes the run to `runs/train/<name>/`, adding `run_info.json` (wall and training time, seconds per epoch and per image, Python/PyTorch/Ultralytics versions, machine). It is the only training code that imports Ultralytics (AGPL-3.0); the harness reads prediction files only.
- `train.smoke` profile in `config.yaml`: yolo11n COCO-pretrained, 10 epochs, 512 px, batch 8, MPS, seed 42.
- `fdb.yolo.count_train_images`, with `tests/test_count_images.py`.
- Predictions contract (T20c): `fdb.predictions` defines the per-image JSON Lines format read by the harness (one record per image, `"boxes": []` when nothing is detected, absolute-pixel boxes, no ground truth), with a strict reader and a completeness check against the manifest. It never imports a detector library.
- Inference adapter (T20c): `scripts/predict_yolo.py` runs trained weights over the smoke or full split manifests and writes `runs/predict/<name>/predictions_<split>.jsonl` plus `meta.json` (weights SHA-256, settings, versions). Refuses to overwrite an existing output folder; reads back and checks every file it writes.
- `predict` settings in `config.yaml` (conf 0.001, iou 0.7, imgsz 512, max_det 300, MPS); `conf` is part of the frame score s(x) and is shared by cal, test and target data.
- Tests: `tests/test_predictions.py`.
- Full training profile (T27a): `train.full` in `config.yaml` — yolo11s COCO-pretrained, up to 100 epochs with early stopping (patience 20), 512 px, batch 32, SGD lr0 0.01, Ultralytics default augmentations plus vertical flips, AMP, checkpoint every 5 epochs, CUDA device 0. Chosen from ZJU-Leaper only, before any target data is used.
- `fdb.yolo.split_profile`: splits a profile into model, data view and training arguments; refuses settings owned by the script (seed, run folder, resume). Tests: `tests/test_train_profiles.py` (also checks every profile in `config.yaml`).
- Frame score (T21a): `fdb.scoring.max_conf_score` implements s(x) = highest box confidence on a frame, or 0.0 when it has no boxes (M1; D1; D7 floor); higher means more likely defective. Rejects confidences outside [0, 1], including NaN. `fdb.scoring.score_frames` scores a whole predictions file with any scorer of the same shape and refuses duplicate images. Tests: `tests/test_scoring.py`.
- Score tables (T21b): `fdb.score_table` joins frame scores with ground truth by image ID (strict: an image missing on either side is an error) and reads and writes `scores_<split>.csv` (image_id, split, score, defective), sorted, 6-decimal scores, LF line endings, identical on re-run. The only point where ground truth meets model output.
- `scripts/score_frames.py`: for a predictions folder, scores every frame with s(x), joins ZJU-Leaper ground truth for the images in each split's manifest, writes the score tables beside the predictions and prints the mean score of defective vs normal frames. Tests: `tests/test_score_table.py`.
- Deferral metrics (T22a): `fdb.metrics` counts N, |A|, M and D at a threshold in one pass (a frame is auto-passed when s(x) < t; a score equal to t goes to a human) and computes coverage |A|/N, selective risk M/|A| (primary) and escape rate M/D (secondary), per D1, D2 and M2–M4. Undefined ratios (zero denominator) are returned as None, never 0. `scripts/show_metrics.py` prints the counts and metrics for a score table at chosen thresholds or, by default, at 0, the score quartiles and 1.0. Tests: `tests/test_metrics.py`.
- Risk–coverage curve (T22b): `fdb.curve.risk_coverage_curve` computes coverage, selective risk and escape rate at every distinct operating point (one sort plus running totals; tied scores move together; the last point uses the smallest float above the top score so everything is auto-passed), each point checked in tests against `fdb.metrics.counts_at`. `write_curve` writes `curve_<split>.csv` with exact thresholds and undefined ratios left empty (M8). `scripts/risk_coverage.py` writes the curves for a predictions folder and prints the first non-empty point and the all-auto-passed risk (= defect rate). Tests: `tests/test_curve.py`.
- AURC (T22c): `fdb.curve.aurc` computes the area under the risk–coverage curve step-wise and tie-aware (tied frames count at the end of their group, the only point a threshold can reach; without ties it equals the mean selective risk after accepting 1…N frames). `optimal_aurc` gives the best value any score can reach for a set's size and defect count, and `excess_aurc` (E-AURC = AURC − optimal) makes results comparable across defect rates. `scripts/risk_coverage.py` now prints AURC, optimal AURC, E-AURC and the defect rate (the useless-score reference) per split. Tests: `tests/test_aurc.py`.

### Changed
- `scripts/train_yolo.py` passes every setting in a profile to Ultralytics (previously only epochs, imgsz, batch, workers and device), stops early if a CUDA device is configured but unavailable, and records `epochs_configured`, `epochs_run` and the full settings in `run_info.json`.

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