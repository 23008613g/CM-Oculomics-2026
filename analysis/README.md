# Analysis code and released inputs

Run commands below from the repository root. Numbered scripts retain the study's computational methods. `prepare_public_results.py` copies released inputs into the expected `analysis/results/` layout without overwriting existing working files. It never changes `public_outputs/`.

## Recompute from released inputs

```bash
python analysis/prepare_public_results.py
python analysis/verify_public_outputs.py
python analysis/scripts/75_estimator_baselines.py
python analysis/scripts/77_same_task_estimators.py
```

| Analysis | Script | Available inputs and requirements |
|---|---|---|
| External AUCs and per-model balanced accuracy | `verify_public_outputs.py` | All 70 external prediction files; compares with the stored per-model table and ensemble summaries |
| Estimators and no-shift baseline | `scripts/75_estimator_baselines.py` | Released estimator rows and model-level anchor summaries; does not regenerate the restricted anchor predictions |
| Same-task control estimators | `scripts/77_same_task_estimators.py` | 110 prediction files from DDR-trained models; balanced and natural-prevalence target settings |
| Cohort forest plot and fixed-effect pooling | `scripts/40_external_meta.py` | Public external predictions and released held-out aggregate reference; bootstrap computation |
| Random-effects pooling and variance components | `scripts/72_random_effects.py` | Public external predictions; bootstrap computation |
| Label-free signal plots | `scripts/45_shift_all_figure.py` | Released shift and agreement summaries; original feature extraction still requires restricted inputs |
| Local validation sampling analysis | `scripts/74_local_validation_cost.py` | Public external predictions; repeated resampling, potentially lengthy |

The latter four commands can be run after preparation. They were not rerun as part of the release checks. Script 74 estimates sampling requirements from public PDR/NPDR score distributions, not clinical sample requirements validated on treatment-outcome cohorts.

Same-task anchor files are predictions on the **public DDR** anchor split. They contain no in-house anchor observations. The in-house estimator anchors are provided only as model-level aggregate statistics.

## Training and analyses requiring additional inputs

| Purpose | Scripts and modules | Additional inputs |
|---|---|---|
| Backbone comparison, cross-validation, label efficiency | `27`, `28`, `tcm_retina/` | Approved development images and patient-level split manifest |
| External seed-model training and scoring | `38`, with DeLong helper `06` | Development data, public images, and pretrained backbones |
| Same-task DDR control training | `76`, importing `27` | Public cohort images and pretrained backbones |
| Shift metrics, agreement/confidence, performance estimation | `42`, `44`, `46`, `47` | Restricted anchor predictions or embeddings in addition to public results |
| Held-out and device-stratified internal analysis | `56`, `60`, `61`, `62`, `66` | Restricted development-cohort predictions and metadata |
| Per-model table and model provenance | `64`, `69` | Restricted anchor arrays/splits or deployment checkpoint; released summary tables remain available |
| Recalibration, uncertainty, supporting figures | `30`, `31`, `32`, `57`, `73` | Development predictions, images, or derived features, depending on the script |
| Vascular, syndrome, and cohort analyses | `49`–`55`, `67`, `68`, shared training modules | Restricted images and clinical metadata |
| Grad-CAM framing and figure generation | `05`, `59`, `63`, `70`, `71` | Images and weights for image-based figures; schematic scripts use study summaries |

For the numbered training scripts, the expected private layout is `analysis/data_anon/images/` with `analysis/data_anon/splits.csv`. Public images use `analysis/external_data/`. Do not add these directories to Git. `analysis/configs/` describes the corresponding training inputs. Run YAML-based `run*.py` entry points from inside `analysis/` so their relative paths resolve correctly.

Script 76 writes checkpoints to `analysis/results/same_task/checkpoints/`; `SAME_TASK_CKPT_DIR` overrides that location. The public DDR split is image-level because patient identifiers are unavailable. It should not be described as a verified patient-disjoint split.

Scripts `39` and `41` prepare datasets already obtained from their distributors. Script 39's optional Kaggle download route requires `kagglehub` and dataset access; it is not needed for the quick public-results workflow. Script 41 uses Windows directory junctions and reads source paths from `DEEPDRID_SOURCE` and `JSIEC_SOURCE`. On other platforms, arrange the corresponding image directories manually. Neither preparation step is run by the quickstart.

## Input records

- `external_predictions/*.npz`: `ys` is the binary proxy label and `pos` is the model score, in the order of the matching `subset_labels_*.csv`.
- `same_task/pred_*.npz`: the same two numeric arrays for the DDR anchor and target sets.
- `same_task/split.csv`: public DDR image identifier, grade, binary label, and split role.
- `aggregate/per_seed_table.csv`: one row per model and cohort, including aggregated in-house anchor metrics; no individual in-house observations.
- Other JSON/CSV files: released summaries. Preserve these as reference inputs; write regenerated results to `results/`.

Historical figure/section numbers remain in some original code comments. The purpose map above identifies the analyses independently of those earlier numbering conventions. The older finite-subset sampling script and screenshot-packaging helper are excluded from this release.
