# -*- coding: utf-8 -*-
"""Metric-matched, label-free accuracy estimators on stored predictions.

Primary: published ALine-D/S equations, ATC-MC (uncalibrated), DoC-Feat,
and No-shift, anchored to ordinary accuracy. Sensitivity: replace only the
source accuracy quantity with source balanced accuracy in the same formulas.
The latter is an adaptation, not a published balanced-accuracy estimator.

Default input is the released public same-task control (100 model/cohort rows).
--clinical-anchor enables the original-task analysis with an explicitly supplied
restricted local NPZ. Only cohort/model aggregate metrics are written; private
individual records, predictions, identifiers and paths are never copied.

No training, image loading, temperature fitting or target-label tuning occurs.
All predictions use p > 0.5, including exact ties assigned to class zero.

Sources: Baek et al. NeurIPS 2022, section 4 and author aline() notebook;
Guillory et al. ICCV 2021, sections 3-5 (DoC-Feat, not fitted DoC);
Garg et al. ICLR 2022, section 4, equations (1)-(2).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, spearmanr

ANALYSIS = Path(__file__).resolve().parents[1]
MODELS = [f"{kind}_seed{seed}" for kind in ("dinov2_l", "retfound") for seed in range(5)]
COHORTS = ["ddr", "jsiec", "deepdrid", "aptos", "idrid", "messidor2", "eyepacs"]
SETTINGS = {
    "same_task_balanced": ("bal_", COHORTS[1:]),
    "same_task_natural": ("nat_", ["aptos", "eyepacs", "messidor2", "idrid"]),
}
METHODS = ["ALine-D", "ALine-S", "ATC-MC", "DoC-Feat", "No-shift"]
PROTOCOLS = ("primary", "source_ba")
EPS = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(p: np.ndarray) -> np.ndarray:
    return (p > 0.5).astype(np.int64)


def check_labels(y: np.ndarray, context: str) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError(f"{context}: invalid label vector")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError(f"{context}: expected both binary classes")
    return y.astype(np.int64)


def check_probs(p: np.ndarray, n: int, context: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.shape != (n,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{context}: invalid probability vector")
    return p


def load_public(directory: Path, suffix: str, manifest: list) -> tuple[np.ndarray, np.ndarray]:
    y_ref = None
    probabilities = []
    for model in MODELS:
        path = directory / f"pred_{model}_{suffix}.npz"
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"ys", "pos"}:
                raise ValueError(f"{path.name}: unexpected NPZ keys")
            y = check_labels(z["ys"], path.name)
            p = check_probs(z["pos"], len(y), path.name)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"{path.name}: labels/order differ across models")
        probabilities.append(p)
        manifest.append({"input": f"{directory.name}/{path.name}", "sha256": sha256(path), "n": len(y)})
    return y_ref, np.stack(probabilities)


def load_clinical(path: Path, manifest: list) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"ys", *MODELS}:
            raise ValueError("Restricted anchor: unexpected model keys")
        y = check_labels(z["ys"], "restricted anchor")
        p = np.stack([check_probs(z[m], len(y), f"restricted anchor/{m}") for m in MODELS])
    manifest.append({"input": "restricted_clinical_anchor_not_distributed", "sha256": sha256(path), "n": len(y)})
    return y, p


def observed_metrics(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for one in predictions(p):
        sensitivity = float(np.mean(one[y == 1] == 1))
        specificity = float(np.mean(one[y == 0] == 0))
        rows.append({"actual_accuracy": float(np.mean(one == y)),
                     "balanced_accuracy": (sensitivity + specificity) / 2,
                     "sensitivity": sensitivity, "specificity": specificity})
    return rows


def agreement(p: np.ndarray) -> np.ndarray:
    pred = predictions(p)
    return np.mean(pred[:, None, :] == pred[None, :, :], axis=2)


def aline(source_quantity: np.ndarray, source_agreement: np.ndarray,
          target_agreement: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Paper equations (3)-(6), with author-code pair filtering.

    Individual identifiability is checked after least squares. A column's value
    is identifiable iff its unit vector has zero projection onto null(A).
    This avoids emitting an arbitrary minimum-norm value for excluded models.
    """
    n = len(source_quantity)
    pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)
             if 0.05 <= source_agreement[i, j] <= 0.98
             and 0.05 <= target_agreement[i, j] <= 0.98]
    x = np.array([norm.ppf(source_agreement[i, j]) for i, j in pairs])
    y = np.array([norm.ppf(target_agreement[i, j]) for i, j in pairs])
    diag = {"pairs_total": n * (n - 1) // 2, "pairs_retained": len(pairs),
            "retained_pairs": [[MODELS[i], MODELS[j]] for i, j in pairs],
            "source_quantity_boundary_clips": int(((source_quantity <= EPS) | (source_quantity >= 1 - EPS)).sum())}
    unavailable = np.full(n, np.nan)
    if len(pairs) < 2 or np.std(x) <= 1e-12:
        diag.update({"status": "agreement_regression_unidentifiable", "slope": None,
                     "intercept": None, "design_rank": 0,
                     "unidentifiable_models": MODELS[:n]})
        return unavailable.copy(), unavailable.copy(), diag
    line = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(line, y, rcond=None)[0]
    z = norm.ppf(np.clip(source_quantity, EPS, 1 - EPS))
    s = norm.cdf(slope * z + intercept)
    A = np.zeros((len(pairs), n))
    r = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        A[k, i] = A[k, j] = 0.5
        r[k] = y[k] + slope * ((z[i] + z[j]) / 2 - x[k])
    u, _, rank, singular_values = np.linalg.lstsq(A, r, rcond=None)
    null_projection = np.eye(n) - np.linalg.pinv(A) @ A
    identified = np.max(np.abs(null_projection), axis=1) < 1e-8
    d = norm.cdf(u)
    d[~identified] = np.nan
    residual = y - line @ np.array([slope, intercept])
    total = float(np.sum((y - y.mean()) ** 2))
    diag.update({"status": "full_rank" if rank == n else "some_model_accuracies_unidentifiable",
                 "slope": float(slope), "intercept": float(intercept),
                 "agreement_r_squared": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
                 "design_rank": int(rank), "singular_values": singular_values.tolist(),
                 "unidentifiable_models": [MODELS[i] for i in range(n) if not identified[i]],
                 "pair_equation_rmse_probit": float(np.sqrt(np.mean((A @ u - r) ** 2)))})
    return d, s, diag


def atc_threshold(scores: np.ndarray, source_quantity: float) -> tuple[float, float]:
    """Exact empirical objective in Garg et al. eq (1), including tied scores.

    The candidate at each unique score corresponds to scores >= threshold;
    nextafter(max, +inf) permits a zero-coverage partition. Among equal minima,
    choose the smallest threshold (highest retained source fraction).
    """
    values, counts = np.unique(scores, return_counts=True)
    retained = 1 - np.r_[0, np.cumsum(counts)] / len(scores)
    thresholds = np.r_[values, np.nextafter(values[-1], np.inf)]
    error = np.abs(retained - source_quantity)
    best = np.flatnonzero(np.isclose(error, error.min(), rtol=0, atol=1e-15))[0]
    return float(thresholds[best]), float(retained[best])


def estimate(source_quantity: np.ndarray, source_p: np.ndarray,
             target_p: np.ndarray) -> tuple[dict, dict]:
    """Intentionally accepts no target labels."""
    d, s, diagnostics = aline(source_quantity, agreement(source_p), agreement(target_p))
    cs = np.maximum(source_p, 1 - source_p)
    ct = np.maximum(target_p, 1 - target_p)
    thresholds = [atc_threshold(scores, q) for scores, q in zip(cs, source_quantity)]
    atc = np.array([np.mean(scores >= t[0]) for scores, t in zip(ct, thresholds)])
    diagnostics["atc_calibration"] = [
        {"model": model, "threshold": t, "source_fraction_at_or_above": matched,
         "source_quantity": float(q), "absolute_match_error": float(abs(matched - q))}
        for model, (t, matched), q in zip(MODELS, thresholds, source_quantity)]
    return {"ALine-D": d, "ALine-S": s, "ATC-MC": atc,
            "DoC-Feat": source_quantity + ct.mean(axis=1) - cs.mean(axis=1),
            "No-shift": source_quantity.copy()}, diagnostics


def self_test() -> None:
    """Small synthetic invariants, not scientific reproduction evidence."""
    q = np.array([0.60, 0.70, 0.80, 0.65])
    gs = np.eye(4)
    gt = np.eye(4)
    for (i, j), g in zip(itertools.combinations(range(4), 2), [.60, .65, .70, .75, .80, .85]):
        gs[i, j] = gs[j, i] = g
        gt[i, j] = gt[j, i] = norm.cdf(.7 * norm.ppf(g) - .1)
    d, s, _ = aline(q, gs, gt)
    expected = norm.cdf(.7 * norm.ppf(q) - .1)
    np.testing.assert_allclose(d, expected, atol=1e-12)
    np.testing.assert_allclose(s, expected, atol=1e-12)
    gs[3, :3] = gs[:3, 3] = 1
    d, _, diag = aline(q, gs, gt)
    assert np.isfinite(d[:3]).all() and np.isnan(d[3]) and diag["design_rank"] == 3
    scores = np.array([.5, .5, .8, .9])
    for target in [0, .25, .5, 1]:
        t, matched = atc_threshold(scores, target)
        assert np.mean(scores >= t) == matched == target
    t, matched = atc_threshold(scores, .75)
    assert matched == 1 and t == .5  # tie between attainable 1 and .5
    np.testing.assert_array_equal(predictions(np.array([.49, .5, .51])), [0, 0, 1])
    print("Synthetic invariant checks passed (ALine equations/rank, ATC ties/boundaries, threshold).")


def run_setting(name: str, prefix: str, cohorts: list[str], source_y: np.ndarray,
                source_p: np.ndarray, target_dir: Path, manifest: list) -> tuple[list, dict]:
    source_metrics = observed_metrics(source_y, source_p)
    quantities = {"primary": np.array([r["actual_accuracy"] for r in source_metrics]),
                  "source_ba": np.array([r["balanced_accuracy"] for r in source_metrics])}
    rows, diagnostics = [], {}
    for cohort in cohorts:
        target_y, target_p = load_public(target_dir, prefix + cohort, manifest)
        # Only the estimate() calls produce predictions. Labels are used below
        # for post-hoc scoring and descriptive aggregates, never to tune a rule.
        estimates = {}
        diagnostics[cohort] = {}
        for protocol, q in quantities.items():
            estimates[protocol], diagnostics[cohort][protocol] = estimate(q, source_p, target_p)
        target_metrics = observed_metrics(target_y, target_p)
        for i, model in enumerate(MODELS):
            aline_diag = diagnostics[cohort]["primary"]
            row = {"setting": name, "cohort": cohort, "model": model,
                   "backbone": model.rsplit("_seed", 1)[0], "n": len(target_y),
                   "n_positive": int(target_y.sum()), "prevalence": float(target_y.mean()),
                   "source_n": len(source_y), "source_n_positive": int(source_y.sum()),
                   "source_accuracy": source_metrics[i]["actual_accuracy"],
                   "source_balanced_accuracy": source_metrics[i]["balanced_accuracy"],
                   "aline_pairs_retained": aline_diag["pairs_retained"],
                   "aline_design_rank": aline_diag["design_rank"],
                   "aline_d_status": ("unidentifiable" if model in aline_diag["unidentifiable_models"] else "available"),
                   **target_metrics[i]}
            for protocol in PROTOCOLS:
                for method in METHODS:
                    row[f"{protocol}_{method}"] = float(estimates[protocol][method][i])
            rows.append(row)
    return rows, diagnostics


def errors(estimate_values: np.ndarray, truth: np.ndarray) -> dict:
    err = estimate_values - truth
    rho = None
    if len(err) > 1 and np.ptp(estimate_values) > 0 and np.ptp(truth) > 0:
        rho = float(spearmanr(estimate_values, truth).statistic)
    return {"n": len(err), "mae": float(np.mean(np.abs(err))) if len(err) else None,
            "bias": float(np.mean(err)) if len(err) else None,
            "worst_absolute_error": float(np.max(np.abs(err))) if len(err) else None,
            "spearman": rho}


def summarize(frame: pd.DataFrame) -> dict:
    out = {"n_model_cohort_rows": len(frame), "protocols": {}}
    for protocol in PROTOCOLS:
        out["protocols"][protocol] = {}
        for method in METHODS:
            est = frame[f"{protocol}_{method}"].to_numpy(float)
            valid = np.isfinite(est)
            base = frame[f"{protocol}_No-shift"].to_numpy(float)[valid]
            detail = {"n_available": int(valid.sum()), "n_unavailable": int((~valid).sum()),
                      "n_outside_unit_interval": int(((est[valid] < 0) | (est[valid] > 1)).sum())}
            for target in ("actual_accuracy", "balanced_accuracy"):
                truth = frame[target].to_numpy(float)[valid]
                detail[target] = errors(est[valid], truth)
                detail[f"no_shift_same_available_rows_{target}"] = errors(base, truth)
            out["protocols"][protocol][method] = detail
    return out


def json_dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=ANALYSIS / "public_outputs")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS / "results" / "metric_matched")
    parser.add_argument("--clinical-anchor", type=Path, default=None,
                        help="Restricted local NPZ; only aggregate statistics are exported")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic invariant checks and exit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.clinical_anchor is not None and not args.clinical_anchor.is_file():
        parser.error("--clinical-anchor must be an existing explicit local file")
    if args.clinical_anchor is None and (args.output_dir / "main_model_metrics.csv").exists():
        parser.error("Output directory contains a prior full run; use a fresh output directory for public-only mode")
    manifest, all_rows, diagnostics, anchor_summary = [], [], {}, {}
    source_y, source_p = load_public(args.public_root / "same_task", "anchor", manifest)
    anchor_summary["same_task_DDR"] = {
        "n": len(source_y), "n_positive": int(source_y.sum()),
        "per_model": dict(zip(MODELS, observed_metrics(source_y, source_p)))}
    for setting, (prefix, cohorts) in SETTINGS.items():
        rows, diag = run_setting(setting, prefix, cohorts, source_y, source_p,
                                 args.public_root / "same_task", manifest)
        all_rows.extend(rows)
        diagnostics[setting] = diag
    if args.clinical_anchor is not None:
        private_y, private_p = load_clinical(args.clinical_anchor, manifest)
        anchor_summary["clinical_restricted"] = {
            "n": len(private_y), "n_positive": int(private_y.sum()),
            "per_model": dict(zip(MODELS, observed_metrics(private_y, private_p)))}
        rows, diag = run_setting("main_balanced", "", COHORTS, private_y, private_p,
                                 args.public_root / "external_predictions", manifest)
        all_rows.extend(rows)
        diagnostics["main_balanced"] = diag
    frame = pd.DataFrame(all_rows)
    summary = {setting: summarize(group) for setting, group in frame.groupby("setting", sort=False)}
    summary["same_task_all"] = summarize(frame[frame.setting.str.startswith("same_task_")])
    expected_rows = 170 if args.clinical_anchor is not None else 100
    if len(frame) != expected_rows or frame.duplicated(["setting", "cohort", "model"]).any():
        raise ValueError("Unexpected output dimensions or duplicate rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame[frame.setting.str.startswith("same_task_")].to_csv(args.output_dir / "same_task_model_metrics.csv", index=False)
    if args.clinical_anchor is not None:
        frame[frame.setting == "main_balanced"].to_csv(args.output_dir / "main_model_metrics.csv", index=False)
    json_dump(args.output_dir / "summary.json", summary)
    json_dump(args.output_dir / "diagnostics.json", diagnostics)
    json_dump(args.output_dir / "anchor_aggregate.json", anchor_summary)
    metadata = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__, "randomness": "none",
        "models": MODELS, "input_manifest": manifest,
        "run_mode": "public_same_task_plus_restricted_anchor_aggregate" if args.clinical_anchor else "public_same_task_only",
        "reproduction_command": "python analysis/scripts/78_metric_matched_estimators.py --output-dir analysis/results/metric_matched"
                                + (" --clinical-anchor <approved-local-anchor.npz>" if args.clinical_anchor else ""),
        "protocols": {
            "primary": "Source ordinary accuracy; primary evaluation against target ordinary accuracy.",
            "source_ba": "Sensitivity adaptation: only source accuracy quantity replaced by source balanced accuracy; target confidence/agreement averages unchanged."},
        "evaluation": "Both target accuracy and BA scored after estimation. No target labels passed to estimator functions.",
        "prediction_threshold": "p > 0.5; equality assigned class 0; fixed for every cohort and method",
        "aline": {"pair_filter": "0.05 <= ID and OOD agreement <= 0.98, inclusive, as author notebook",
                  "source_probit_clip": EPS, "regression": "OLS with intercept; no target accuracy used",
                  "rank_policy": "ALine-D unavailable for individually unidentifiable coordinates; no minimum-norm arbitrary values reported"},
        "atc": {"variant": "Maximum confidence, without temperature scaling",
                "threshold_objective": "Minimize absolute difference between empirical source fraction score>=t and source quantity",
                "ties": "Score groups indivisible; among equal objectives choose smallest t (highest coverage)",
                "author_code_difference": "Follows paper equations (1)-(2), not helper's update-before-threshold finite-sample indexing"},
        "doc_feat": "Unit-slope, no regression; raw estimates retained even outside [0,1]",
        "data_alignment": "Each NPZ is checked for keys, shape, finite probabilities, binary labels; exact labels/order matched across all models within each cohort.",
        "alignment_limit": "No sample IDs in prediction NPZs; identical label order is checked but cannot independently authenticate image identity/order.",
        "privacy": "No individual restricted-anchor records/probabilities/labels or local path are exported; only cohort/model aggregates and file digest.",
        "sources": ["https://papers.nips.cc/paper_files/paper/2022/file/7a8d388b7a17df480856dff1cc079b08-Paper-Conference.pdf",
                    "https://raw.githubusercontent.com/kebaek/Agreement-on-the-line/main/agreement_trajectory.ipynb",
                    "https://arxiv.org/pdf/2107.03315", "https://arxiv.org/pdf/2201.04234",
                    "https://raw.githubusercontent.com/saurabhgarg1996/ATC_code/master/ATC_helper.py"],
        "source_access_date": "2026-11-10", "training_rerun": False,
        "interpretation": "Main analysis changes clinical endpoint to public PDR proxy; same-task control removes task change. Primary methods estimate ordinary accuracy, not balanced accuracy or clinical benefit."}
    json_dump(args.output_dir / "metadata.json", metadata)
    for setting, info in summary.items():
        print(setting, "rows", info["n_model_cohort_rows"])
        for method in METHODS:
            item = info["protocols"]["primary"][method]
            a = item["actual_accuracy"]
            mae = f"{a['mae']:.6f}" if a["mae"] is not None else "NA"
            bias = f"{a['bias']:+.6f}" if a["bias"] is not None else "NA"
            print(f"  {method:10s} n={a['n']:3d} MAE={mae} bias={bias} unavailable={item['n_unavailable']}")


if __name__ == "__main__":
    main()
