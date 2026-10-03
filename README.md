# FYP Core PoC — Implementation Notes

## Compute

Development, the evaluation harness, tests, the smoke run and inference run locally on an Apple M2
(8 GB unified memory, PyTorch MPS backend). Full detector training runs on Kaggle GPU notebooks
(free tier, ~30 GPU-hours/week, T4/P100).

The interchange file (D5) is the boundary between the two: the training/inference side writes
per-image predictions, and every downstream computation reads only that file, so results do not
depend on where the detector ran.

## Licence note

Ultralytics YOLO is AGPL-3.0. The evaluation harness never imports it (it reads only the
interchange file), so the harness can be released under a permissive licence; only the YOLO
adapter carries AGPL obligations.
