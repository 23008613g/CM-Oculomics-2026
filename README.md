# CM-Oculomics 2026

Code and public-cohort analysis materials for **Cohort dependence and limits of unsupervised performance estimation in deep learning for diabetic retinopathy**.

The study examines how a fundus-image model transfers between cohorts and whether unsupervised performance estimators can anticipate that transfer. DINOv2 is the primary backbone; RETFound is a comparator. The development endpoint is progression to proliferative diabetic retinopathy during an anti-VEGF treatment course. The seven public cohorts use prevalent PDR versus NPDR as a proxy, without treatment outcomes.

This repository contains the current analysis code, inference and demonstration code, public-cohort predictions, and aggregate results. Manuscripts, submission correspondence, ethics documents, original fundus images, and individual development-cohort records are not included.

## Start with the public results

Python 3.11 is recommended. From the repository root, create and activate a separate Python environment, then run:

```bash
pip install -r requirements-analysis.txt
python analysis/verify_public_outputs.py
python analysis/prepare_public_results.py
python analysis/scripts/75_estimator_baselines.py
python analysis/scripts/77_same_task_estimators.py
```

These commands require no GPU, image downloads, or restricted development data. The verifier checks 180 prediction files, recomputes 70 per-model metric rows and 14 ensemble AUCs, and checks label ordering across the 11 same-task evaluation sets. The other scripts recompute the no-shift comparison and same-task estimator analyses. Outputs go to `analysis/results/`; the released input snapshots remain unchanged.

See [analysis/README.md](analysis/README.md) for the analysis map, input requirements, and longer computations. [VALIDATION.md](VALIDATION.md) records what was actually checked for this release.

## Files

| Location | Contents |
|---|---|
| `analysis/scripts/` | Training, evaluation, statistics, and figure scripts used for the study |
| `analysis/tcm_retina/`, `analysis/configs/`, `analysis/run*.py` | Shared training modules and configuration files |
| `analysis/public_outputs/external_predictions/` | Predictions for 2 backbones × 5 seeds × 7 public cohorts, plus subset image lists |
| `analysis/public_outputs/same_task/` | DDR training split and predictions for the same-task control |
| `analysis/public_outputs/aggregate/` | Study-level and model-level summary tables and JSON results |
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

## Data and reproducibility

Public cohorts are DDR, JSIEC, DeepDRiD, APTOS, IDRiD, Messidor-2, and EyePACS. Obtain images from their original distributors under the applicable terms. The repository distributes predictions and public image identifiers, not the source images.

Development-cohort images, patient identifiers, individual clinical records, held-out prediction arrays, and private split manifests are excluded. Development-cohort analyses require approved access to those records. Their aggregate summaries are included, but they cannot substitute for an independent rerun from the restricted data.

The public prediction analyses use fixed stored predictions. The repository does not claim that all original training checkpoints are released or that the full study can be rerun without restricted inputs. The released deployment checkpoint is one model; it is not the ten-model external-evaluation ensemble. Model provenance is documented in `analysis/public_outputs/aggregate/model_provenance.json`.

## License and attribution

The project code retains its Apache-2.0 license; see [LICENSE](LICENSE). The DINOv2 deployment weights are separately hosted under the license stated on their model page. RETFound is used as a comparator; its weights are not redistributed here and its own license applies. Public dataset terms remain with their distributors.

Please cite the manuscript title above and the DINOv2 or RETFound publications when using the corresponding models. This repository does not assign the manuscript a publication DOI.
