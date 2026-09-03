"""Clean-prediction audit — runbook step 4. A HARD STOP on any difference.

Verifies IDENTITY BEFORE VALUES. Hashing predictions alone is insufficient:
numerically identical values would pass even if origin keys, timestamps, ground
truth, or target digests were wrong or shuffled underneath them. The value hash
answers "are the numbers the same"; it cannot answer "are they the same
numbers, for the same cells, of the same data".

Order of checks, each a hard stop:

  1. structural  — no duplicate or missing cells; row count exactly 178 x 96
  2. keys        — identical unique (origin_id, step_index) sets
  3. timestamps  — forecast_timestamp matches exactly, cell by cell
  4. ground truth— matches exactly, cell by cell
  5. provenance  — per-origin target_sha256 agrees with window_results.csv on
                   BOTH sides, and the two sides agree with each other
  6. environment — dataset checksum and model revision match
  7. values      — canonical origin/step-sorted float32 sha256 of the predictions

There is deliberately NO tolerance mechanism for ANY of these fields. Choosing a
tolerance after seeing a discrepancy is precisely what this audit exists to
prevent, and that applies to the identity checks as much as to the values.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_ORIGINS = 178
EXPECTED_HORIZON = 96
EXPECTED_ROWS = EXPECTED_ORIGINS * EXPECTED_HORIZON


class CleanAuditFailure(Exception):
    """Any audit failure. Always a hard stop; never downgraded to a warning."""


@dataclass
class AuditReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    new_hash: str = ""
    reference_hash: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def load_clean(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Clean predictions plus the run's window_results, when present."""
    frame = pd.read_csv(run_dir / "predictions_long.csv")
    clean = frame[frame["condition_id"] == "clean"].copy()
    if clean.empty:
        raise CleanAuditFailure(f"no clean predictions in {run_dir}")
    window_path = run_dir / "window_results.csv"
    windows = pd.read_csv(window_path) if window_path.exists() else None
    return clean, windows


def canonical_clean_hash(run_dir: Path) -> tuple[str, pd.DataFrame]:
    """Canonical origin/step-sorted float32 sha256 of the clean predictions."""
    clean, _ = load_clean(run_dir)
    clean = clean.sort_values(["origin_id", "step_index"], kind="mergesort")
    values = np.ascontiguousarray(clean["median_prediction"].to_numpy(dtype=np.float32))
    return hashlib.sha256(values.tobytes()).hexdigest(), clean


def _structural_failures(clean: pd.DataFrame, label: str, *, strict_size: bool) -> list[str]:
    failures: list[str] = []
    duplicated = clean.duplicated(subset=["origin_id", "step_index"])
    if duplicated.any():
        offenders = clean.loc[duplicated, ["origin_id", "step_index"]].head(5)
        failures.append(
            f"{label}: {int(duplicated.sum())} duplicate (origin_id, step_index) cells, "
            f"e.g. {offenders.to_dict('records')}"
        )
    if strict_size:
        if len(clean) != EXPECTED_ROWS:
            failures.append(
                f"{label}: {len(clean)} clean rows, expected "
                f"{EXPECTED_ORIGINS} x {EXPECTED_HORIZON} = {EXPECTED_ROWS}"
            )
        n_origins = clean["origin_id"].nunique()
        if n_origins != EXPECTED_ORIGINS:
            failures.append(f"{label}: {n_origins} origins, expected {EXPECTED_ORIGINS}")
        per_origin = sorted(clean.groupby("origin_id").size().unique().tolist())
        if per_origin != [EXPECTED_HORIZON]:
            failures.append(
                f"{label}: steps per origin {per_origin}, expected [{EXPECTED_HORIZON}]"
            )
    return failures


def _provenance(windows: pd.DataFrame | None, label: str) -> tuple[dict[int, str], list[str]]:
    """Per-origin target_sha256 from window_results, plus any inconsistency."""
    failures: list[str] = []
    if windows is None:
        return {}, [f"{label}: window_results.csv is absent; provenance cannot be verified"]
    per_origin = windows.groupby("origin_id")["target_sha256"].unique()
    digests: dict[int, str] = {}
    for origin, values in per_origin.items():
        if len(values) != 1:
            failures.append(
                f"{label}: origin {origin} has {len(values)} distinct target_sha256 values"
            )
        digests[int(origin)] = str(values[0])
    return digests, failures


def audit(new_dir: Path, reference_dir: Path, *, strict_size: bool = True) -> AuditReport:
    """Run every check. Returns a report; never raises on a mere mismatch."""
    report = AuditReport(passed=False)
    new_clean, new_windows = load_clean(new_dir)
    ref_clean, ref_windows = load_clean(reference_dir)

    # 1. Structural.
    report.failures += _structural_failures(new_clean, "new", strict_size=strict_size)
    report.failures += _structural_failures(ref_clean, "reference", strict_size=strict_size)

    # 2. Keys.
    new_keys = set(zip(new_clean["origin_id"], new_clean["step_index"]))
    ref_keys = set(zip(ref_clean["origin_id"], ref_clean["step_index"]))
    missing, extra = ref_keys - new_keys, new_keys - ref_keys
    if missing:
        report.failures.append(
            f"keys: {len(missing)} cells present in reference but MISSING from new, "
            f"e.g. {sorted(missing)[:5]}"
        )
    if extra:
        report.failures.append(
            f"keys: {len(extra)} cells present in new but ABSENT from reference, "
            f"e.g. {sorted(extra)[:5]}"
        )
    report.details["n_cells_new"] = len(new_keys)
    report.details["n_cells_reference"] = len(ref_keys)

    # The cell-by-cell comparison needs a well-formed one-to-one join. Duplicates
    # are already recorded as structural failures above; attempting the merge
    # anyway would raise instead of reporting them.
    has_duplicates = bool(
        new_clean.duplicated(subset=["origin_id", "step_index"]).any()
        or ref_clean.duplicated(subset=["origin_id", "step_index"]).any()
    )
    if not missing and not extra and not has_duplicates:
        merged = new_clean.merge(
            ref_clean, on=["origin_id", "step_index"], suffixes=("_new", "_ref"),
            validate="one_to_one",
        )
        # 3. Timestamps.
        ts_new = merged["forecast_timestamp_new"].astype(str)
        ts_ref = merged["forecast_timestamp_ref"].astype(str)
        ts_bad = merged.loc[ts_new != ts_ref, ["origin_id", "step_index"]]
        if not ts_bad.empty:
            report.failures.append(
                f"timestamps: {len(ts_bad)} cells differ, e.g. "
                f"{ts_bad.head(3).to_dict('records')}"
            )
        # 4. Ground truth — exact, no tolerance.
        gt_new = merged["ground_truth_new"].to_numpy(dtype=np.float64)
        gt_ref = merged["ground_truth_ref"].to_numpy(dtype=np.float64)
        gt_bad = np.flatnonzero(gt_new != gt_ref)
        if gt_bad.size:
            first = merged.iloc[gt_bad[0]]
            report.failures.append(
                f"ground_truth: {gt_bad.size} cells differ, first at origin "
                f"{int(first['origin_id'])} step {int(first['step_index'])}: "
                f"new={first['ground_truth_new']!r} ref={first['ground_truth_ref']!r}"
            )
        report.details["n_ground_truth_mismatches"] = int(gt_bad.size)

    # 5. Provenance: target_sha256 per origin, both sides.
    new_digests, f1 = _provenance(new_windows, "new")
    ref_digests, f2 = _provenance(ref_windows, "reference")
    report.failures += f1 + f2
    if new_digests and ref_digests:
        shared = set(new_digests) & set(ref_digests)
        mismatched = {o: (new_digests[o], ref_digests[o]) for o in sorted(shared)
                      if new_digests[o] != ref_digests[o]}
        if mismatched:
            report.failures.append(
                f"target_sha256: {len(mismatched)} origins disagree, e.g. "
                f"{dict(list(mismatched.items())[:3])}"
            )
        only_new, only_ref = set(new_digests) - shared, set(ref_digests) - shared
        if only_new or only_ref:
            report.failures.append(
                f"target_sha256: origin sets differ (new-only {sorted(only_new)[:5]}, "
                f"reference-only {sorted(only_ref)[:5]})"
            )
        report.details["n_origins_with_target_digest"] = len(shared)

    # 6. Environment: dataset checksum and model revision.
    for column, name in (("dataset_sha256", "dataset checksum"),
                         ("model_revision", "model revision")):
        for frame, label in ((new_windows, "new"), (ref_windows, "reference")):
            if frame is not None and column in frame.columns and frame[column].nunique() != 1:
                report.failures.append(
                    f"{name}: {label} run carries {frame[column].nunique()} distinct values"
                )
        if (
            new_windows is not None and ref_windows is not None
            and column in new_windows.columns and column in ref_windows.columns
        ):
            a, b = new_windows[column].iloc[0], ref_windows[column].iloc[0]
            report.details[f"{column}_new"], report.details[f"{column}_reference"] = a, b
            if a != b:
                report.failures.append(f"{name} differs: new={a!r} reference={b!r}")

    # 7. Values — only meaningful once identity holds, but always computed and
    #    reported so a value drift is visible alongside an identity failure.
    report.new_hash, _ = canonical_clean_hash(new_dir)
    report.reference_hash, _ = canonical_clean_hash(reference_dir)
    if report.new_hash != report.reference_hash:
        report.failures.append(
            f"predictions: canonical float32 sha256 differs "
            f"(new={report.new_hash}, reference={report.reference_hash})"
        )

    report.passed = not report.failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit clean predictions against a reference run. No tolerances."
    )
    parser.add_argument("--new-run-dir", required=True)
    parser.add_argument("--reference-run-dir", required=True)
    args = parser.parse_args()

    new_dir, ref_dir = REPO_ROOT / args.new_run_dir, REPO_ROOT / args.reference_run_dir
    try:
        report = audit(new_dir, ref_dir)
    except CleanAuditFailure as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"new run  : {new_dir}\n  sha256={report.new_hash}")
    print(f"reference: {ref_dir}\n  sha256={report.reference_hash}")
    for key, value in report.details.items():
        print(f"  {key}: {value}")

    if report.passed:
        print(
            "\nPASS: identity and values both match — keys, timestamps, ground truth, "
            "target digests, dataset checksum, model revision, and predictions."
        )
        return 0

    print(f"\nFAIL: {len(report.failures)} check(s) failed.")
    for failure in report.failures:
        print(f"  - {failure}")
    print(
        "\nThis is a HARD STOP. Do not proceed, and do not select a tolerance now — "
        "for the predictions or for any identity field."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
