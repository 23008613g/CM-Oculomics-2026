# Figure sources for the 2026-11-10 revision

This directory contains the current figure generators and the public-cohort resampling calculation. It does not contain submission documents, review reports, private clinical images, or individual clinical-anchor predictions. Paths are resolved relative to this repository; commands below run from its root.

Use the dependencies in `requirements-analysis.txt`. The recorded validation environment was Python 3.11.9, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0 and Matplotlib 3.10.9. No dependency installation is performed by these scripts.

## Generate figures from the released inputs

```bash
python analysis/revision_20261110/build_estimator_figures.py
python analysis/revision_20261110/plot_local_validation_cost.py
python analysis/revision_20261110/draw_clinical_schematics.py
```

Each command accepts `--output-dir` and writes by default beneath `analysis/results/revision_20261110/`. The two quantitative plotting scripts also accept `--input-dir`. They do not overwrite their released input directories.

| Script | Inputs | Outputs and interpretation |
|---|---|---|
| `build_estimator_figures.py` | `analysis/public_outputs/metric_matched/` | Fig. 5: 70 clinical-anchor/public-target aggregate rows, with 60 available ALine-D estimates explicitly counted. Fig. S14: 40 natural-prevalence same-task rows, ordinary accuracy in the top row and the source-BA adaptation evaluated against BA in the bottom row. Raw DoC-Feat estimates above one are retained. |
| `plot_local_validation_cost.py` | `analysis/public_outputs/local_validation_revision/` | Fig. 6 and Fig. S6: median and interquartile range across seven public cohorts of the simulated 95th-percentile estimation error. Error tolerances are 0.10 and 0.05; no estimator-performance benchmark is plotted. |
| `draw_clinical_schematics.py` | Public study facts and code descriptions; `draw_figure1.py` for the current Fig. 1; selected repository source files for provenance hashes | Current Fig. 1 study-design flowchart, Fig. S1 and Fig. S7. These are documentation schematics, not new model inference, clinical-image exports, or a validation of treatment recommendations. |

The schematic entrypoint delegates Fig. 1 to `draw_figure1.build_figure1()` and preserves the existing Fig. S1/S7 rendering. The redesigned Fig. 1 separates the clinical timeline, linked-subset partition, internal modelling, independent public proxy and transfer assessment in a **183 × 126 mm** flowchart. It exports editable PDF/SVG, 600-dpi PNG/TIFF and a 300-dpi PNG preview. To generate only Fig. 1, run `python analysis/revision_20261110/draw_figure1.py`; its `--output-dir` and `--alignment-qa` options follow the same conventions.

Fig. 5 and Fig. S14 use the current outputs of `analysis/scripts/78_metric_matched_estimators.py`, not the superseded estimator implementations. `figure_sources.json` records their released source hashes, CSV columns and row counts. In the main analysis, ALine-D is unavailable for all ten Messidor-2 model/cohort rows after agreement filtering; the plot does not replace them with arbitrary minimum-norm values. Source-BA substitution is an explicit sensitivity adaptation, not a published BA estimator.

The quantitative generators export editable PDF/SVG and high-resolution PNG. Fig. 5/S14 include both 300- and 600-dpi PNG. Their CSV source-data exports retain the complete aggregate rows, including missing estimates. Generated PDFs and previews are not committed here.

## Recompute the public sampling simulation

```bash
python analysis/revision_20261110/recompute_local_validation_cost.py
python analysis/revision_20261110/plot_local_validation_cost.py --input-dir analysis/results/revision_20261110/local_validation
```

Recomputation reads the 70 released prediction files in `analysis/public_outputs/external_predictions/`. It preserves the five-seed ensemble calculation, `RandomState(0)`, four prevalence scenarios, 40 sample sizes, 4,000 BA draws and 600 AUC draws. Positive counts are binomial; scores are sampled with replacement. It checks the resulting aggregate exactly against `analysis/public_outputs/local_validation_revision/local_validation_cost.json` before saving.

The released curve cache contains 16 arrays, each with seven cohorts by 40 sample sizes. These are aggregate simulation curves, not clinical per-image arrays. Absent-class draws count as failures. At small sample sizes, percentiles can be nonfinite; the raw cache retains those values and the plot uses finite cohort quantiles for its median and IQR. NumPy can emit nonfinite-percentile warnings for these expected cases. A sample requirement is the first tested sample size meeting the stated criterion; it is not a guarantee between grid points or a clinical treatment-outcome sample-size validation.

The older estimator-MAE comparison is excluded from the revised aggregate and plotting code. The original historical output elsewhere in the repository remains provenance only.

## Validation and optional figure QA

`validation.json` is the preserved execution record for the earlier figure bundle, before the Fig. 1 redesign. The complete resampling calculation matched all released current aggregate values and all 16 curve arrays, including NaNs; the curve archive was also byte-identical. Those public renders were checked against the corresponding revision previews. `figure1_design_validation.json` separately records the updated schematic entrypoint's actual isolated execution, current source hashes, the new Fig. 1 PNG match, and byte-identical preservation of Fig. S1/S7. The earlier validation record is not rewritten to describe the new design. These checks establish reproducibility of the specified computations and exports, not model retraining, original-paper experimental reproduction, clinical validity or prospective generalization.

The figure generators do not require external skill packages. `--alignment-qa` optionally uses an already available `audit_panel_alignment` module on `PYTHONPATH`. Without that flag, an output status explicitly records `NOT RUN`; successful rendering must not be described as an automatic alignment, PDF collision or visual QA pass. No auditor code or review reports are bundled. The submitted revision figures underwent separate layout/font/collision and visual checks, retained outside this public repository.
