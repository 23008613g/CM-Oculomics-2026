# Release preparation and verification

Prepared on 6 November 2026 from the current study scripts and the previously released public-cohort outputs. The manuscript and submission correspondence are kept separately.

## Checks performed

Using Python 3.11.9 in the existing project environment:

- Parsed all 73 Python files and all 24 released JSON files.
- Validated all 180 NPZ files using `allow_pickle=False`: only numeric `ys` and `pos` fields, matching lengths, binary labels, finite scores in [0, 1].
- Recomputed the 70 external per-model AUC and balanced-accuracy rows and 14 ensemble AUCs. These agreed with the released reference tables at their recorded precision.
- Checked consistent label ordering across models for the external cohorts and 11 same-task sets.
- Ran `75_estimator_baselines.py` and `77_same_task_estimators.py`. The regenerated JSON objects matched the released estimator snapshots.
- Ran the inference, Grad-CAM, and biomarker command-line `--help` entry points successfully.
- Confirmed that the demo stops with an explicit error when fine-tuned weights are absent, before allocating the classifier or returning a score.
- Inspected the release selection for submission documents, individual development-cohort records, checkpoints, cached files, private absolute paths, and common credential literals.

The analysis checks used NumPy 2.4.4, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.8.0, and Matplotlib 3.10.9. The command-line imports used PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and timm 1.0.27. Gradio 6.16.0 was used for the missing-checkpoint guard check; its `Blocks.launch` signature was checked against the demo's theme, CSS, and head arguments. The full demonstration interface was not run.

These checks verify packaging, selected executable paths, and consistency of released predictions. They do not establish independent clinical validity or reproduce model training.

## Changes made for this release

- Selected the current analysis and inference files; excluded older submission packages, screenshot-packaging utilities, the superseded finite-subset label-cost analysis, and the generic training example. Training entry points used by the study remain in `analysis/`.
- Used the more recent project versions of scripts 32, 40, 45, 46, 47, 74, and 77. Differences from the prior release concern figure layout and wording, not the calculations.
- Included the missing `06_delong.py` helper and declared previously omitted runtime dependencies.
- Added public-input preparation and consistency-check commands.
- Replaced machine-specific dataset/checkpoint locations with configurable paths and corrected the Grad-CAM script's repository-root lookup.
- Re-encoded `section26_numbers.json` from GB18030 to UTF-8; the parsed JSON values are unchanged.
- Made the demo require fine-tuned weights, and made its temporary biomarker image directory clean up when processing ends or raises an error.
- Updated project links and retained the existing Apache-2.0 license. The previous published source snapshot had commit `bf179521e45a1e5fea7d1e68ae8fc0b5a737e481`; local source files were preserved.

## Not rerun

Full training, checkpoint-based inference, analyses using restricted development data, lengthy bootstrap/resampling scripts, and the separately hosted Hugging Face application were not run or redeployed during release preparation. The installed environment was used as found; no dependency installation or upgrade was performed.

`SHA256SUMS` identifies the files in this release. Regenerated results and local execution logs are excluded from version control.
