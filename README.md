# CM-Oculomics 2026

Code and public-cohort analysis materials for **Cohort dependence and metric-specific validation of retinal foundation models for diabetic retinopathy**.

The study examines how a fundus-image model transfers between cohorts and whether unsupervised performance estimators can anticipate that transfer. DINOv2 is the primary backbone; RETFound is a comparator. The development endpoint is **persistent or recurrent retinal/optic-disc neovascularization after six ranibizumab injections versus documented regression**. Neovascularization elsewhere (NVE) and/or at the optic disc (NVD) was already present before the first injection and at acquisition of the baseline photographs used for prediction. This is a treatment-course neovascularization outcome, not incident PDR or adverse-drug intolerance. The seven public cohorts retain their separate cross-sectional endpoint of prevalent PDR versus NPDR, without treatment outcomes.

For the development cohort, positive class 1 denotes persistent/recurrent neovascularization and negative class 0 denotes documented regression. Historical names `intolerant`/`PDR` and `tolerant`/`NPDR` remain in stored records and computational identifiers to preserve the original mapping; they do not denote baseline DR grades or adverse-drug intolerance. Public-cohort PDR/NPDR labels retain their actual grading meaning. The endpoint clarification changes the interpretation and presentation, not the saved class assignments, scores, weights or numerical analyses.

This repository contains the current analysis code, inference and demonstration code, public-cohort predictions, and aggregate results. Manuscripts, submission correspondence, ethics documents, original fundus images, and individual development-cohort records are not included.

## Start with the public results

Python 3.11 is recommended. From the repository root, create and activate a separate Python environment, then run:

```bash
pip install -r requirements-analysis.txt
python analysis/verify_public_outputs.py
python analysis/scripts/78_metric_matched_estimators.py --self-test
python analysis/scripts/78_metric_matched_estimators.py
```

These commands require no GPU, image downloads, or restricted development data. The verifier checks 180 prediction files and 16 aggregate sampling-error curves, recomputes 70 per-model metric rows and 14 ensemble AUCs, and checks label ordering across the 11 same-task evaluation sets. Script 78 reads the released predictions directly and recomputes the 100 same-task model/cohort rows. Its outputs go to `analysis/results/metric_matched/`; the released input snapshots remain unchanged.

The current estimator analysis is in [analysis/public_outputs/metric_matched/](analysis/public_outputs/metric_matched/). Its primary methods are ALine-D, ALine-S, ATC-MC without temperature scaling, the regression-free DoC-Feat, and No-shift, all anchored to **ordinary accuracy** and evaluated against target ordinary accuracy. A separate `source_ba` sensitivity replaces only the source accuracy quantity with source balanced accuracy; it is an adaptation, not a published balanced-accuracy estimator. Predictions use the fixed rule `p > 0.5` throughout.

For the same-task control, the four estimators have ordinary-accuracy MAE of 0.179–0.200 on the 60 class-balanced target rows and 0.043–0.049 on the 40 natural-prevalence target rows; No-shift gives 0.222 and 0.052, respectively. These results must not be conflated with errors against balanced accuracy. In the 70-row main analysis, ALine-D is unavailable for all 10 Messidor-2 model rows because the author-code agreement filter leaves a rank-deficient system. Missing rows and reasons are retained; its MAE of 0.227 is compared with No-shift MAE of 0.232 on the same 60 available rows.

Script 78 and `public_outputs/metric_matched/` are the current estimator implementation and results. Superseded estimator implementations, their obsolete snapshots and the earlier estimator-MAE sampling benchmark have been removed from the current tree; prior versions remain in Git history.

See [analysis/README.md](analysis/README.md) for the analysis map, input requirements, and longer computations, and [MODEL_CARD.md](MODEL_CARD.md) for intended use and evaluation boundaries. [VALIDATION.md](VALIDATION.md) records what was actually checked for this release.

## Submission analysis map

This code supports the manuscript prepared for the Journal of Medical Systems collection *Clinical AI Governance, Safety, and Algorithmovigilance*. The contribution is an evaluation of transfer reliability and metric-specific local validation requirements. It does not introduce a new model architecture or report a deployed surveillance system.

| JMS figure | Original code/output numbering | Content |
|---|---|---|
| Fig. 1 | Fig. 1 | Study design and distinct clinical/proxy endpoints |
| Fig. 2 | Fig. 4 | External transfer across seven cohorts |
| Fig. 3 | Fig. 5 | Metric-matched performance estimation |
| Fig. 4 | Fig. 6 | Local-label sampling requirements |
| Fig. S15 | Fig. 2 | Internal model evaluation |
| Fig. S16 | Fig. 3 | Backbone and feature-model comparisons; DDR evaluation |
| Fig. S17 | Fig. 7 | Attention maps and imaging biomarkers |

Figs. S1–S14 retain their earlier numbering. The detailed methods and supporting results are in manuscript Online Resource 1, distributed with the submission rather than in this repository. Original figure filenames in the code remain stable; the table maps them to the JMS submission. The syndrome and fusion analyses remain because they support the reported negative supplementary findings.

## Files

| Location | Contents |
|---|---|
| `analysis/scripts/` | Training, evaluation, statistics, and figure scripts used for the study |
| `analysis/tcm_retina/`, `analysis/configs/`, `analysis/run*.py` | Shared training modules and configuration files |
| `analysis/public_outputs/external_predictions/` | Predictions for 2 backbones × 5 seeds × 7 public cohorts, plus subset image lists |
| `analysis/public_outputs/same_task/` | DDR training split and predictions for the same-task control |
| `analysis/public_outputs/aggregate/` | Study-level and model-level summary tables and JSON results |
| `analysis/public_outputs/metric_matched/` | Current accuracy-estimator results, source-BA sensitivity, availability diagnostics, and input/version metadata |
| `analysis/public_outputs/local_validation_revision/` | Current sampling-requirement aggregates, error curves, and simulation metadata |
| `analysis/revision_20261110/` | Current portable figure generators and public-cohort sampling simulation, with executed validation records |
| `src/` | DINOv2 inference, Grad-CAM, image transforms, and vascular measures |
| `app.py` | Local Gradio demonstration using the released DINOv2 checkpoint |
| `requirements*.txt`, `environment.yml` | Dependencies for the public analyses or full code |

## Inference and demonstration

For inference or training, use `requirements.txt` or `environment.yml`. These include the additional PyTorch and image-processing dependencies. Full model training was not rerun when preparing this release.

The fine-tuned DINOv2 checkpoint remains at [Hugging Face](https://huggingface.co/fc28/CM-Oculomics-weights), filename `dino_deploy.pth`. Obtain it separately and place it in `weights/`:

```bash
pip install -r requirements.txt
python -m src.inference --image path/to/fundus.jpg --weights weights/dino_deploy.pth
python -m src.gradcam --image path/to/fundus.jpg --weights weights/dino_deploy.pth --out cam.png
python -m src.biomarkers --image path/to/fundus.jpg
python app.py
```

The demo requires real fine-tuned weights and stops if none are available. Alternatively set `WEIGHTS_REPO=fc28/CM-Oculomics-weights` to download the checkpoint through Hugging Face, or set `WEIGHTS_PATH` to an existing local file. The existing [hosted demonstration](https://huggingface.co/spaces/fc28/CM-Oculomics) is maintained separately; this release does not update that deployment.

The score is a research model output, not a calibrated probability or a treatment recommendation. The default demonstration threshold is 0.5; `THRESHOLD` changes it. The CLI accepts `--threshold`. Threshold performance and image preprocessing must be considered when interpreting results. Vascular measures depend on the imaging device.

The demonstration presents the positive-class output as a persistent/recurrent neovascularization research score. The CLI's legacy `intolerance_risk` key is retained for compatibility and stores that same uncalibrated score; its displayed class names now describe persistence/recurrence or regression. The separately hosted demonstration has not been redeployed by this repository update.

## Data and reproducibility

Public cohorts are DDR, JSIEC, DeepDRiD, APTOS, IDRiD, Messidor-2, and EyePACS. Obtain images from their original distributors under the applicable terms. The repository distributes predictions and public image identifiers, not the source images.

Development-cohort images, patient identifiers, individual clinical records, held-out prediction arrays, and private split manifests are excluded. Development-cohort analyses require approved access to those records. Their aggregate summaries are included, but they cannot substitute for an independent rerun from the restricted data.

Script 78 defaults to the public same-task analysis. Recomputing its main-analysis estimates additionally requires an explicitly supplied, approved local clinical-anchor NPZ via `--clinical-anchor`; that original file is not distributed. The released main-analysis files contain cohort/model aggregates only. The metadata records input digests, dependency versions, threshold conventions, and unavailable estimates.

The public prediction analyses use fixed stored predictions. The repository does not claim that all original training checkpoints are released or that the full study can be rerun without restricted inputs. The released deployment checkpoint is one model; it is not the ten-model external-evaluation ensemble. Model provenance is documented in `analysis/public_outputs/aggregate/model_provenance.json`.

## License and attribution

The project code retains its Apache-2.0 license; see [LICENSE](LICENSE). The DINOv2 deployment weights are separately hosted under the license stated on their model page. RETFound is used as a comparator; its weights are not redistributed here and its own license applies. Public dataset terms remain with their distributors.

Please cite the manuscript title above and the DINOv2 or RETFound publications when using the corresponding models. This repository does not assign the manuscript a publication DOI.
