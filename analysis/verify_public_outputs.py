"""Verify released arrays and recompute public-cohort discrimination metrics.

This is a consistency check of released predictions, not a retraining experiment.
It does not access or validate the restricted development cohort.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public_outputs"
COHORTS = ["ddr", "jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]
KINDS = {"dinov2_l": "DINOv2", "retfound": "RETFound"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        check(set(archive.files) == {"ys", "pos"}, f"Unexpected fields: {path.name}")
        labels, scores = archive["ys"], archive["pos"]
    check(labels.ndim == scores.ndim == 1 and labels.shape == scores.shape, f"Shape: {path.name}")
    check(np.isin(labels, [0, 1]).all() and len(np.unique(labels)) == 2, f"Labels: {path.name}")
    check(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), f"Scores: {path.name}")
    return labels.astype(int), scores


def verify_curve_cache(directory):
    """Check the declared aggregate cache separately from individual predictions."""
    metadata = json.loads((directory / "cache_metadata.json").read_text(encoding="utf-8"))
    check(metadata["cohorts"] == COHORTS and metadata["backbones"] == KINDS, "Curve cohort/backbone metadata")
    check(metadata["prevalences"] == [0.5, 0.2, 0.1, 0.088], "Curve prevalence metadata")
    grid = np.asarray(metadata["grid"])
    check(grid.shape == (40,) and np.issubdtype(grid.dtype, np.integer), "Curve sample-size grid")
    check((grid > 0).all() and (np.diff(grid) > 0).all(), "Curve sample-size ordering")
    expected = {f"{metric}|{kind}|{prevalence}" for metric in ("ba", "auc")
                for kind in KINDS.values() for prevalence in metadata["prevalences"]}
    check(set(metadata["curve_shapes"]) == set(metadata["nonfinite_curve_counts"]) == expected,
          "Curve metadata fields")
    with np.load(directory / "local_validation_curves.npz", allow_pickle=False) as archive:
        check(set(archive.files) == expected, "Unexpected aggregate curve fields")
        for key in sorted(expected):
            values = archive[key]
            check(values.shape == (len(COHORTS), len(grid)), f"Curve shape: {key}")
            check(list(values.shape) == metadata["curve_shapes"][key], f"Curve shape metadata: {key}")
            check(values.dtype.kind in "fiu", f"Curve numeric dtype: {key}")
            finite = np.isfinite(values)
            check(((values[finite] >= 0) & (values[finite] <= 1)).all(), f"Curve error range: {key}")
            check(not np.isneginf(values).any(), f"Negative infinite curve error: {key}")
            # Absent-class draws count as failures; their quantiles can be inf/NaN.
            check(int((~finite).sum()) == metadata["nonfinite_curve_counts"][key],
                  f"Curve nonfinite count: {key}")
    expected_inputs = {f"pred_{kind}_seed{seed}_{cohort}.npz" for kind in KINDS
                       for seed in range(5) for cohort in COHORTS}
    check(set(metadata["source_inputs_sha256"]) == expected_inputs, "Curve source manifest")
    for name, digest in metadata["source_inputs_sha256"].items():
        source = PUBLIC / "external_predictions" / name
        check(hashlib.sha256(source.read_bytes()).hexdigest() == digest, f"Curve source digest: {name}")
    reference = directory / "local_validation_cost.json"
    check(hashlib.sha256(reference.read_bytes()).hexdigest() == metadata["reference_sha256"],
          "Curve aggregate reference digest")
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "public_verification.json")
    args = parser.parse_args()
    settings = ["anchor"] + ["bal_" + c for c in COHORTS if c != "ddr"] + ["nat_" + c for c in ["aptos", "eyepacs", "messidor2", "idrid"]]
    arrays = sorted(
        [PUBLIC / "external_predictions" / f"pred_{kind}_seed{seed}_{cohort}.npz"
         for kind in KINDS for seed in range(5) for cohort in COHORTS]
        + [PUBLIC / "same_task" / f"pred_{kind}_seed{seed}_{setting}.npz"
           for kind in KINDS for seed in range(5) for setting in settings])
    curve_directory = PUBLIC / "local_validation_revision"
    expected_npz = set(arrays) | {curve_directory / "local_validation_curves.npz"}
    check(set(PUBLIC.rglob("*.npz")) == expected_npz, "Missing or unexpected public NPZ files")
    for path in arrays:
        read_prediction(path)
    curve_count = verify_curve_cache(curve_directory)
    with (PUBLIC / "aggregate" / "per_seed_table.csv").open(newline="", encoding="utf-8-sig") as f:
        table = list(csv.DictReader(f))
    summary = json.loads((PUBLIC / "aggregate" / "external2_summary.json").read_text())
    meta = {x["key"]: x for x in json.loads((PUBLIC / "aggregate" / "meta_summary.json").read_text())["rows"]}
    rows = []
    checked = 0
    for cohort in COHORTS:
        reference_y = None
        for kind, display in KINDS.items():
            predictions = []
            for seed in range(5):
                y, p = read_prediction(PUBLIC / "external_predictions" / f"pred_{kind}_seed{seed}_{cohort}.npz")
                if reference_y is None:
                    reference_y = y
                check(np.array_equal(reference_y, y), f"Label ordering mismatch: {kind}/{cohort}/{seed}")
                auc = float(roc_auc_score(y, p))
                expected = summary["per_seed"][cohort][kind]["aucs"][seed]
                check(abs(auc - expected) < 1e-10, f"AUC snapshot mismatch: {kind}/{cohort}/{seed}")
                matches = [r for r in table if r["cohort"].lower().replace("-", "") == cohort and r["backbone"] == display and int(r["seed"]) == seed]
                check(len(matches) == 1, f"Per-model table lookup: {kind}/{cohort}/{seed}")
                r = matches[0]
                check(int(r["n"]) == len(y) and int(r["n_pdr"]) == int(y.sum()), "Per-model counts mismatch")
                ba = float(balanced_accuracy_score(y, p > 0.5))
                check(abs(auc - float(r["auc"])) <= 0.000051, "Per-model AUC mismatch")
                check(abs(ba - float(r["balanced_accuracy"])) <= 0.000051, "Per-model BA mismatch")
                predictions.append(p)
                checked += 1
            ensemble_auc = float(roc_auc_score(reference_y, np.mean(predictions, axis=0)))
            check(abs(ensemble_auc - meta[cohort]["dino" if kind == "dinov2_l" else "ret"]) < 1e-10, "Ensemble AUC mismatch")
            rows.append({"cohort": cohort, "backbone": display, "n": len(reference_y), "ensemble_auc": ensemble_auc})
    for setting in settings:
        labels = [read_prediction(PUBLIC / "same_task" / f"pred_{kind}_seed{seed}_{setting}.npz")[0] for kind in KINDS for seed in range(5)]
        check(all(np.array_equal(labels[0], y) for y in labels[1:]), f"Same-task label ordering: {setting}")
    report = {"status": "pass", "prediction_files_validated": len(arrays), "aggregate_curve_arrays_validated": curve_count, "per_model_metric_rows_verified": checked, "same_task_sets_checked": len(settings), "external_ensembles": rows, "scope": "Released predictions and aggregate consistency only; no retraining or restricted-cohort validation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {len(arrays)} prediction files, {curve_count} aggregate curves, {checked} per-model rows, {len(rows)} ensembles, {len(settings)} same-task sets.")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
