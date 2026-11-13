# Model and evaluation record

This record accompanies *Cohort dependence and metric-specific validation of retinal foundation models for diabetic retinopathy*. It describes the released research code and checkpoint, not an approved clinical product.

## Model, input and output

The primary model fine-tunes DINOv2 ViT-L/14 on pretreatment color fundus photographs. RETFound is the retina-specific comparator. The local inference implementation resizes an RGB image to 224 × 224, applies ImageNet normalization and returns a two-class softmax score. The inference checkpoint is hosted separately at [CM-Oculomics-weights](https://huggingface.co/fc28/CM-Oculomics-weights); the public analysis uses multiple models and must not be equated with that single checkpoint. Model/run provenance is in `analysis/public_outputs/aggregate/model_provenance.json`.

In the clinical development cohort, NVE/NVD was present before the first injection and in the photographs used for prediction. Positive class 1 represents persistent or recurrent neovascularization after six monthly ranibizumab injections; negative class 0 represents documented regression. The legacy CLI key `intolerance_risk` stores this positive-class score. It does not measure adverse-drug intolerance or onset of PDR from baseline NPDR. The score is not a calibrated probability or a treatment recommendation.

## Evaluation domains

| Evaluation | Labels and unit | Interpretation |
|---|---|---|
| Clinical cross-validation and held-out evaluation | Post-treatment persistence/recurrence versus regression; patient-level and image-level analyses are distinguished | Retrospective evaluation on the development cohort; the clinical patient split and labels are restricted |
| Seven public external cohorts | Cross-sectional PDR versus NPDR; class-balanced image subsets | A separate grading proxy without treatment outcomes; it does not externally validate the clinical treatment-course endpoint |
| DDR-trained same-task control | Cross-sectional PDR versus NPDR on balanced and natural-prevalence image sets | Tests estimation with the task held fixed; patient separation cannot be established from the available DDR image identifiers |
| Local-label resampling | Public PDR/NPDR score distributions and prespecified prevalence scenarios | Conditional simulation of metric precision; an expected positive-image count is not a patient quota or prospective clinical guarantee |

Public-cohort results cover DDR, JSIEC, DeepDRiD, APTOS, IDRiD, Messidor-2 and EyePACS. All source images must be obtained from their distributors. Stored public-cohort predictions, labels and study/model-level aggregates are provided here; individual clinical records and clinical prediction arrays are excluded.

## Metric and operating-point boundaries

Current performance-estimator analysis uses ordinary source accuracy and evaluates ordinary target accuracy. Source balanced-accuracy substitution is a separately identified sensitivity adaptation. All current estimator class predictions use `p > 0.5`; no target-specific tuning is performed. Missing ALine-D estimates and unbounded DoC-Feat outputs are retained. The ordinary-accuracy estimate does not establish sensitivity to PDR, balanced accuracy or clinical safety. See `analysis/public_outputs/metric_matched/` for both metrics, subgroup counts and availability diagnostics.

The demonstration uses a configurable threshold, with 0.5 as its default. A threshold selected in one cohort need not preserve sensitivity in another. Cohort variation, prevalence, camera characteristics and annotation practices constrain transfer. The study does not identify their separate causal contributions. Clinical development estimates also have selection and small-held-out-sample limitations described in the manuscript; public consistency checks do not resolve these limitations.

## Intended use and implementation status

The release supports research inspection, reproduction of public prediction analyses and investigation of local validation requirements. Its governance contribution is evidence about evaluation and metric selection. No prospective clinical deployment, real-time drift alarm, patient-management workflow or operational algorithmovigilance system was evaluated. The separately hosted demonstration is maintained outside this repository and was not redeployed for this submission.

For the available executable checks, see `VALIDATION.md`. Reproducing a table from fixed predictions verifies computational consistency; it does not rerun model training or establish independent clinical validity. The code license is Apache-2.0; checkpoint and original dataset terms are stated by their respective hosts. No publication DOI is assigned by this repository.
