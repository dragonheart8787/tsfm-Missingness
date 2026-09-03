"""Clean-prediction checksum audit — runbook step 4. A HARD STOP on mismatch.

Computes a canonical origin/step-sorted float32 SHA256 of the clean-condition
predictions in a new run and compares it against a reference run's.

Exits non-zero on any difference and prints full diagnostics. It deliberately
offers NO tolerance flag: selecting a tolerance after seeing a discrepancy is
exactly the move this audit exists to prevent.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def canonical_clean_hash(run_dir: Path) -> tuple[str, pd.DataFrame]:
    """sha256 over float32 predictions, sorted by (origin_id, step_index)."""
    frame = pd.read_csv(run_dir / "predictions_long.csv")
    clean = frame[frame["condition_id"] == "clean"].copy()
    if clean.empty:
        raise SystemExit(f"no clean predictions found in {run_dir}")
    clean = clean.sort_values(["origin_id", "step_index"], kind="mergesort")
    values = np.ascontiguousarray(
        clean["median_prediction"].to_numpy(dtype=np.float32)
    )
    return hashlib.sha256(values.tobytes()).hexdigest(), clean


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit clean predictions against a reference run.")
    parser.add_argument("--new-run-dir", required=True)
    parser.add_argument("--reference-run-dir", required=True)
    args = parser.parse_args()

    new_dir = REPO_ROOT / args.new_run_dir
    ref_dir = REPO_ROOT / args.reference_run_dir

    new_hash, new_clean = canonical_clean_hash(new_dir)
    ref_hash, ref_clean = canonical_clean_hash(ref_dir)

    print(f"new run  : {new_dir}\n  rows={len(new_clean)}  sha256={new_hash}")
    print(f"reference: {ref_dir}\n  rows={len(ref_clean)}  sha256={ref_hash}")

    if new_hash == ref_hash:
        print("\nPASS: clean predictions are byte-identical to the reference run.")
        return 0

    print("\nFAIL: clean predictions DIFFER from the reference run.")
    print("This is a HARD STOP. Do not proceed, and do not select a tolerance now.\n")

    if len(new_clean) != len(ref_clean):
        print(f"row count differs: {len(new_clean)} vs {len(ref_clean)}")
        return 1

    merged = new_clean.merge(
        ref_clean, on=["origin_id", "step_index"], suffixes=("_new", "_ref")
    )
    delta = (
        merged["median_prediction_new"].to_numpy(dtype=np.float64)
        - merged["median_prediction_ref"].to_numpy(dtype=np.float64)
    )
    differing = np.flatnonzero(delta != 0)
    print(f"differing cells      : {differing.size} / {len(merged)}")
    print(f"max |deviation|      : {np.abs(delta).max():.6e}")
    print(f"mean |deviation|     : {np.abs(delta).mean():.6e}")
    first = merged.iloc[differing[0]]
    print(
        f"first differing cell : origin {int(first['origin_id'])}, "
        f"step {int(first['step_index'])}: "
        f"new={first['median_prediction_new']!r} ref={first['median_prediction_ref']!r}"
    )
    per_origin = (
        merged.assign(diff=delta != 0).groupby("origin_id")["diff"].sum()
    )
    affected = per_origin[per_origin > 0]
    print(f"origins affected     : {len(affected)} / {per_origin.size}")
    print(f"worst origins        : {affected.sort_values(ascending=False).head(5).to_dict()}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
