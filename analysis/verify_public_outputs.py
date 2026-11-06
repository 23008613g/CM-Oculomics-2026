"""Verify released arrays and recompute public-cohort discrimination metrics.

This is a consistency check of released predictions, not a retraining experiment.
It does not access or validate the restricted development cohort.
"""
from pathlib import Path
import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "public_verification.json")
    args = parser.parse_args()
    arrays = sorted(PUBLIC.rglob("*.npz"))
    for path in arrays:
        read_prediction(path)
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
    settings = ["anchor"] + ["bal_" + c for c in COHORTS if c != "ddr"] + ["nat_" + c for c in ["aptos", "eyepacs", "messidor2", "idrid"]]
    for setting in settings:
        labels = [read_prediction(PUBLIC / "same_task" / f"pred_{kind}_seed{seed}_{setting}.npz")[0] for kind in KINDS for seed in range(5)]
        check(all(np.array_equal(labels[0], y) for y in labels[1:]), f"Same-task label ordering: {setting}")
    report = {"status": "pass", "prediction_files_validated": len(arrays), "per_model_metric_rows_verified": checked, "same_task_sets_checked": len(settings), "external_ensembles": rows, "scope": "Released predictions and aggregate consistency only; no retraining or restricted-cohort validation."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {len(arrays)} prediction files, {checked} per-model rows, {len(rows)} ensembles, {len(settings)} same-task sets.")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
