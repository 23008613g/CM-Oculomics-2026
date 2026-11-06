"""Copy released public inputs into the numbered scripts' working layout.

Run from any directory. The source snapshots in public_outputs are never modified.
Existing working files are left untouched, so rerunning preparation cannot erase
new computations. No images, weights, or development-cohort records are obtained.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public_outputs"


def copy_missing(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)
        return 1
    return 0


def main():
    count = 0
    for path in (PUBLIC / "external_predictions").glob("*.npz"):
        count += copy_missing(path, ROOT / "results" / "external2" / path.name)
    for path in (PUBLIC / "same_task").iterdir():
        if path.is_file():
            count += copy_missing(path, ROOT / "results" / "same_task" / path.name)
    external = {
        "agreement_confidence.json", "aline_atc_doc.json", "estimator_baselines.json",
        "external2_summary.json", "local_validation_cost.json", "meta_summary.json",
        "per_seed_table.csv", "random_effects.json", "shift_metrics.json", "subcohort.json",
    }
    for path in (PUBLIC / "aggregate").iterdir():
        if path.is_file():
            target = ROOT / "results" / ("external2" if path.name in external else "") / path.name
            count += copy_missing(path, target)
    print(f"Prepared {count} missing working files under analysis/results/.")
    print("Source snapshots in analysis/public_outputs/ were not modified.")


if __name__ == "__main__":
    main()
