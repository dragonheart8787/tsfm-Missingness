"""ETTh2 clean-reference audit — two independently generated runs, compared exactly.

    *** A HARD STOP. NO TOLERANCE FOR ANY FIELD. ***

ETTh1's clean audit compared a fresh clean pass against a PRIOR ETTh1 run. No
prior ETTh2 run exists, so the control is different in its reference and
identical in its discipline: two runs generated independently — separate
directories, separate processes, separate model loads — are required to agree
EXACTLY before any corrupted-condition forecast is produced.

Every check is REUSED from ``scripts.audit_clean_predictions.audit`` rather than
reimplemented, so the ETTh2 control cannot be a subtly weaker version of the
ETTh1 one. What this module adds is the part that genuinely differs: the
structural size check is parameterised by ETTh2's DERIVED origin count instead
of the ETTh1 module constant of 178. That the two numbers happen to coincide is
a consequence of the two files having the same length, and relying on the
coincidence would make the check accidentally correct rather than correct.

``clean_reference`` is QC ONLY
------------------------------
It exists to detect nondeterminism between independent model loads. It is
excluded from the statistical analysis; only the formal run's own clean-only
pass feeds ``R_g`` and ``B_g``. ``assert_reference_excluded_from_analysis``
makes that a checked property of a run directory rather than a convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_clean_predictions import (  # noqa: E402
    CleanAuditFailure,
    audit as shared_audit,
)

CLEAN_CONDITION_ID = "clean"


class CleanReferenceAuditFailure(RuntimeError):
    """The two runs disagree. Always a hard stop; never downgraded to a warning."""


@dataclass
class CleanReferenceReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    formal_hash: str = ""
    reference_hash: str = ""
    expected_origins: int = 0
    expected_horizon: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_origins": self.expected_origins,
            "expected_horizon": self.expected_horizon,
            "expected_clean_rows": self.expected_origins * self.expected_horizon,
            "formal_canonical_prediction_sha256": self.formal_hash,
            "reference_canonical_prediction_sha256": self.reference_hash,
            "n_failures": len(self.failures),
            "failures": self.failures,
            "details": self.details,
            "tolerance": "none",
            "note": (
                "clean_reference is QC-only and is EXCLUDED from the statistical "
                "analysis. Only the formal run's own clean-only pass feeds R_g/B_g."
            ),
        }


def _size_failures(
    frame: pd.DataFrame, label: str, *, expected_origins: int, expected_horizon: int
) -> list[str]:
    """ETTh2's own cardinality check, parameterised by the DERIVED origin count."""
    failures: list[str] = []
    expected_rows = expected_origins * expected_horizon
    if len(frame) != expected_rows:
        failures.append(
            f"{label}: {len(frame)} clean rows, expected {expected_origins} x "
            f"{expected_horizon} = {expected_rows}"
        )
    n_origins = int(frame["origin_id"].nunique())
    if n_origins != expected_origins:
        failures.append(
            f"{label}: {n_origins} origins, expected {expected_origins} "
            f"(DERIVED from ETTh2's row count, not inherited from ETTh1)"
        )
    per_origin = sorted(frame.groupby("origin_id").size().unique().tolist())
    if per_origin != [expected_horizon]:
        failures.append(
            f"{label}: steps per origin {per_origin}, expected [{expected_horizon}]"
        )
    return failures


def _clean_rows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "predictions_long.csv"
    if not path.exists():
        raise CleanReferenceAuditFailure(f"{path} is absent; nothing to audit")
    frame = pd.read_csv(path)
    clean = frame[frame["condition_id"] == CLEAN_CONDITION_ID]
    if clean.empty:
        raise CleanReferenceAuditFailure(f"no clean predictions in {run_dir}")
    return clean


def assert_reference_excluded_from_analysis(reference_dir: Path) -> None:
    """The reference run must contain ONLY clean forecasts.

    A reference that had run corrupted conditions could be mistaken for an
    analysable run, and its rows could reach R_g/B_g. Making that a checked
    property is cheaper than trusting the operator not to point the analysis at
    the wrong directory.
    """
    path = Path(reference_dir) / "window_results.csv"
    if not path.exists():
        raise CleanReferenceAuditFailure(f"{path} is absent")
    kinds = sorted(set(pd.read_csv(path, usecols=["kind"])["kind"].astype(str)))
    if kinds != [CLEAN_CONDITION_ID]:
        raise CleanReferenceAuditFailure(
            f"clean_reference at {reference_dir} contains non-clean conditions "
            f"{kinds}. It is QC-only and must hold clean forecasts and nothing "
            f"else, so it can never be mistaken for an analysable run."
        )


def audit_clean_reference(
    formal_dir: Path, reference_dir: Path, *, expected_origins: int, expected_horizon: int
) -> CleanReferenceReport:
    """Every shared check, plus ETTh2's own cardinality. Returns; never raises on mismatch."""
    formal_dir, reference_dir = Path(formal_dir), Path(reference_dir)
    report = CleanReferenceReport(
        passed=False,
        expected_origins=int(expected_origins),
        expected_horizon=int(expected_horizon),
    )

    # The shared audit owns keys, timestamps, ground truth, per-origin target
    # digests, dataset checksum, model revision and the canonical float32
    # prediction hash. strict_size=False because its size constants are ETTh1's;
    # the size check below is ETTh2's own.
    shared = shared_audit(formal_dir, reference_dir, strict_size=False)
    report.failures += list(shared.failures)
    report.formal_hash = shared.new_hash
    report.reference_hash = shared.reference_hash
    report.details.update(shared.details)

    report.failures += _size_failures(
        _clean_rows(formal_dir), "formal",
        expected_origins=expected_origins, expected_horizon=expected_horizon,
    )
    report.failures += _size_failures(
        _clean_rows(reference_dir), "reference",
        expected_origins=expected_origins, expected_horizon=expected_horizon,
    )

    try:
        assert_reference_excluded_from_analysis(reference_dir)
    except CleanReferenceAuditFailure as exc:
        report.failures.append(str(exc))

    report.passed = not report.failures
    return report


def require_clean_reference_match(
    formal_dir: Path, reference_dir: Path, *, expected_origins: int, expected_horizon: int
) -> CleanReferenceReport:
    """The gating form. Raises on ANY discrepancy — this is the hard stop."""
    report = audit_clean_reference(
        formal_dir, reference_dir,
        expected_origins=expected_origins, expected_horizon=expected_horizon,
    )
    if not report.passed:
        raise CleanReferenceAuditFailure(
            "HARD STOP: the clean_reference run and the formal run's clean pass "
            "do not match exactly.\n  - "
            + "\n  - ".join(report.failures)
            + "\n\nDo not proceed to any corrupted-condition forecast, and do not "
            "select a tolerance now — not for the predictions and not for any "
            "identity field. There is no tolerance mechanism, by design."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the ETTh2 formal clean pass against clean_reference. No tolerances."
    )
    parser.add_argument("--config", default="configs/etth2_config.yaml")
    parser.add_argument("--formal-run-dir", default=None)
    parser.add_argument("--reference-run-dir", default=None)
    parser.add_argument("--out", default=None, help="Write the report JSON here.")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    clean_control, design = config["clean_control"], config["design"]
    formal = Path(args.formal_run_dir or clean_control["formal_run_dir"])
    reference = Path(args.reference_run_dir or clean_control["reference_run_dir"])
    formal = formal if formal.is_absolute() else REPO_ROOT / formal
    reference = reference if reference.is_absolute() else REPO_ROOT / reference

    try:
        report = audit_clean_reference(
            formal, reference,
            expected_origins=int(design["expected_origins"]),
            expected_horizon=int(design["horizon"]),
        )
    except (CleanReferenceAuditFailure, CleanAuditFailure) as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"formal    : {formal}\n  sha256={report.formal_hash}")
    print(f"reference : {reference}\n  sha256={report.reference_hash}")
    print(f"expected  : {report.expected_origins} origins x {report.expected_horizon} steps")
    for key, value in report.details.items():
        print(f"  {key}: {value}")

    if args.out:
        out = Path(args.out)
        out = out if out.is_absolute() else REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
        print(f"  report written to {out}")

    if report.passed:
        print(
            "\nPASS: two independently generated runs agree exactly — cardinality, "
            "keys, timestamps, ground truth, target digests, dataset checksum, "
            "model revision, and predictions."
        )
        return 0

    print(f"\nFAIL: {len(report.failures)} check(s) failed.")
    for failure in report.failures:
        print(f"  - {failure}")
    print(
        "\nThis is a HARD STOP. Do not proceed to any corrupted-condition forecast, "
        "and do not select a tolerance now — for the predictions or for any "
        "identity field."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
