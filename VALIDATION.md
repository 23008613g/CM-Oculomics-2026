# Release preparation and verification

Prepared on 6 November 2026 from the current study scripts and the previously released public-cohort outputs. The manuscript and submission correspondence are kept separately.

## Author-confirmed endpoint clarification on 10 November 2026

The study author confirmed that NVE/NVD was present before the first ranibizumab injection and at acquisition of the baseline photographs used for prediction. The development outcome is persistent or recurrent retinal/optic-disc neovascularization after six injections versus documented regression, rather than NPDR-to-PDR onset/progression or adverse-drug intolerance. Positive class 1 and negative class 0 retain their original assignments; `intolerant`/`PDR` and `tolerant`/`NPDR` are historical development-cohort identifiers, not baseline DR grades. Public-cohort PDR/NPDR proxy labels retain their cross-sectional meaning.

Repository documentation and local demonstration wording have been corrected to this definition. The manuscript title is now *Cohort dependence and metric-specific validation of retinal foundation models for diabetic retinopathy*. This clarification is based on the author's stated clinical definition; it is not an independent chart review. No data relabeling, retraining, weight update or numerical result change was performed for this wording correction. The hosted demonstration has not been redeployed.

Static AST comparisons confirmed that the `app.py` and `src/` changes are limited to documentation and presentation text: model builders, transformations, biomarker calculations, numerical constants, thresholds and output-key structure are preserved. The CLI retains the legacy `intolerance_risk` key with its corrected research-score meaning. The inference, Grad-CAM and biomarker `--help` commands completed successfully in the existing project environment. No server or new inference was run for this presentation update.

## Estimator correction and checks on 10 November 2026

Added `analysis/scripts/78_metric_matched_estimators.py` and the released `analysis/public_outputs/metric_matched/` snapshot. Primary ALine-D/S use the published probit-space equations and the author code's agreement-pair filter; ATC-MC uses maximum confidence without temperature scaling and exact empirical threshold matching. The regression-free comparator is named DoC-Feat. All primary estimators use ordinary source accuracy, with target ordinary accuracy as the primary evaluation metric. Source-BA substitution is retained separately as a sensitivity adaptation.

Executed in the existing Python 3.11.9 project environment with NumPy 2.4.4, SciPy 1.17.1 and pandas 3.0.3:

- Ran script 78's `--self-test`: synthetic checks passed for ALine equations, individual identifiability after rank deficiency, ATC ties and boundary objectives, and the fixed `p > 0.5` rule.
- Ran the full analysis using the original approved local clinical-anchor file: generated 70 main and 100 same-task aggregate rows. The clinical-anchor array was not distributed; public output includes only cohort/model aggregates and its file digest.
- Ran the public-only mode independently. Its 100-row CSV was byte-identical to the same-task CSV from the full run; corresponding summary objects matched exactly.
- Checked expected NPZ fields, binary labels, finite probabilities, lengths and exact label-order equality across models within each set. Prediction files lack sample identifiers, so these checks cannot independently authenticate image identity/order.
- Recorded input digests, script digest, dependency versions, formulas, finite-sample threshold conventions and availability diagnostics in `metadata.json` and `diagnostics.json`.

ALine-D is unavailable on all 10 main Messidor-2 rows: filtering leaves 9 pair equations of rank 9 for 10 unknown model quantities. Missing estimates remain in the CSV with status, pair counts and rank. Its MAE is 0.227 on 60 available rows, versus No-shift MAE 0.232 on those same rows. All 100 same-task rows are available; primary estimator MAEs are 0.179–0.200 on 60 balanced-target rows and 0.043–0.049 on 40 natural-prevalence rows (No-shift: 0.222 and 0.052).

Scripts 46, 75 and 77 and their prior outputs are historical. The old ALine-D formula was not the published joint estimator; the old unit-slope DoC corresponds to DoC-Feat; source BA was used where the original methods use ordinary accuracy. The 6 November numerical snapshot checks below establish consistency with those older calculations, not algorithmic fidelity. Use script 78 and `metric_matched/` for current estimator conclusions.

No model training or checkpoint-based inference was rerun for this correction. The full main estimator analysis was recomputed from stored predictions using the restricted local anchor; other restricted-data analyses were not rerun. These checks do not establish independent clinical validity or reproduce the original estimator papers' training experiments.

## Public resampling and packaging checks on 10 November 2026

Executed all four portable figure-generation scripts in `analysis/revision_20261110/`. The public-cohort resampling run reproduced the released aggregate exactly and all 16 cached error-curve arrays matched the reviewed run. Its seven generated figures were byte-identical to the corresponding reviewed figures. The simulation used the recorded `RandomState(0)`, 4,000 balanced-accuracy draws and 600 AUC draws; it did not use clinical treatment-outcome records.

Ran `analysis/verify_public_outputs.py` after distinguishing prediction archives from the aggregate curve cache. The check passed for 180 prediction files, 70 per-model metric rows, 14 external ensembles, 11 same-task sets and 16 aggregate error curves. The curve checks cover declared fields, numeric type, shape, error range, recorded nonfinite counts and input/reference digests. Unknown or missing NPZ files are rejected. Positive infinity and NaN quantiles from absent-class simulation draws remain explicitly recorded, rather than being treated as valid probability predictions.

## Figure 1 design update on 10 November 2026

The later Figure 1 design update is documented in `analysis/revision_20261110/figure1_design_validation.json`. The current schematic entrypoint reproduces the redesigned Figure 1 and preserves Fig. S1/S7 PNGs byte-for-byte. This update changes layout and clarifies the linked-subset partition and external-model seed count; it does not change numerical analyses. The earlier figure-bundle record remains unchanged.

## Historical release checks on 6 November 2026

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

## Not rerun during the 6 November release preparation

Full training, checkpoint-based inference, analyses using restricted development data, lengthy bootstrap/resampling scripts, and the separately hosted Hugging Face application were not run or redeployed during release preparation. The installed environment was used as found; no dependency installation or upgrade was performed.

`SHA256SUMS` identifies the files in this release. Regenerated results and local execution logs are excluded from version control.
