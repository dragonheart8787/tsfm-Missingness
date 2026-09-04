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
import hashlib
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

# Columns the audit compares. Their ABSENCE must be a failure, never a quietly
# skipped check — the shared audit skips a column it cannot find on both sides.
REQUIRED_PROVENANCE_COLUMNS = ("dataset_sha256", "model_revision")


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
    # sha256 of the exact clean-artifact bytes this audit read, per side. The
    # gate carries these so formal-full can prove the files have not moved.
    clean_artifact_digests: dict[str, dict[str, str]] = field(default_factory=dict)

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
            "clean_artifact_digests": self.clean_artifact_digests,
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


def clean_artifact_digest(run_dir: Path) -> dict[str, str]:
    """sha256 of the exact BYTES of the clean artifacts the audit read.

    The gate records these. At formal-full startup they are recomputed and
    compared, which closes a time-of-check-to-time-of-use gap: a gate that was
    valid when written must not authorise proceeding if the underlying clean
    files have changed since. Binding to file bytes rather than to a summary
    means any edit — a re-run, a hand-patched cell, a truncation — invalidates it.
    """
    digests: dict[str, str] = {}
    for name in ("predictions_long.csv", "window_results.csv"):
        path = Path(run_dir) / name
        if not path.exists():
            raise CleanReferenceAuditFailure(f"{path} is absent; cannot digest")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def assert_distinct_directories(formal_dir: Path, reference_dir: Path) -> None:
    """The two runs must be two runs. Refuse if they resolve to one directory.

    A control that compared a directory against itself would pass every check
    while proving nothing at all — the worst possible failure mode, because it
    looks like success. Resolved paths, so a symlink or a ``.`` cannot disguise
    the collision.
    """
    formal, reference = Path(formal_dir).resolve(), Path(reference_dir).resolve()
    if formal == reference:
        raise CleanReferenceAuditFailure(
            f"HARD STOP: the formal run and clean_reference resolve to the SAME "
            f"path ({formal}). The control compares two INDEPENDENTLY GENERATED "
            f"runs; comparing a directory with itself passes every check and "
            f"proves nothing."
        )


def assert_provenance_columns(run_dir: Path, label: str) -> pd.DataFrame:
    """window_results.csv must CARRY the provenance columns, not merely maybe.

    The shared audit skips a comparison whose column is absent on either side. A
    missing column would therefore read as a silently passed check. Requiring
    them here converts that into a failure.
    """
    path = Path(run_dir) / "window_results.csv"
    if not path.exists():
        raise CleanReferenceAuditFailure(f"{path} is absent")
    frame = pd.read_csv(path)
    missing = [c for c in REQUIRED_PROVENANCE_COLUMNS if c not in frame.columns]
    if missing:
        raise CleanReferenceAuditFailure(
            f"HARD STOP: {label} run at {run_dir} is missing provenance column(s) "
            f"{missing}. These are compared by the audit; absent, the comparison "
            f"would be skipped and reported as passing."
        )
    return frame


def assert_row_provenance_matches(
    run_dir: Path, label: str, *, dataset_sha256: str, model_revision: str
) -> None:
    """Row-level provenance must match the CURRENT dataset and model, not just
    the other side.

    Two runs can agree with each other and both be wrong — produced from a stale
    checkout, or against a dataset that has since been replaced. Anchoring each
    side to the checksum loaded right now, and to the revision the manifest says
    was loaded, makes agreement-with-each-other insufficient on its own.
    """
    frame = assert_provenance_columns(run_dir, label)
    for column, expected in (
        ("dataset_sha256", dataset_sha256),
        ("model_revision", model_revision),
    ):
        values = sorted(set(frame[column].astype(str)))
        if values != [str(expected)]:
            raise CleanReferenceAuditFailure(
                f"HARD STOP: {label} run at {run_dir} carries {column}={values}, "
                f"but this run has {column}={str(expected)!r}. The two runs "
                f"matching each other is not enough; both must match what is "
                f"loaded now."
            )


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
    formal_dir: Path, reference_dir: Path, *, expected_origins: int, expected_horizon: int,
    dataset_sha256: str | None = None, model_revision: str | None = None,
) -> CleanReferenceReport:
    """Every shared check, plus ETTh2's own cardinality and provenance anchoring.

    ``dataset_sha256`` / ``model_revision`` anchor both sides to what is loaded
    NOW. They are optional only so the audit can be exercised on synthetic
    fixtures; the runner always supplies them, and a test asserts it does.
    """
    formal_dir, reference_dir = Path(formal_dir), Path(reference_dir)
    report = CleanReferenceReport(
        passed=False,
        expected_origins=int(expected_origins),
        expected_horizon=int(expected_horizon),
    )

    # Misconfiguration checks come first: they invalidate everything downstream.
    for check in (
        lambda: assert_distinct_directories(formal_dir, reference_dir),
        # Both sides must hold clean conditions ONLY, before any comparison. A
        # formal directory that already contains corrupted conditions means the
        # audit is running too late to gate anything.
        lambda: assert_clean_conditions_only(formal_dir, "formal"),
        lambda: assert_clean_conditions_only(reference_dir, "reference"),
        lambda: assert_provenance_columns(formal_dir, "formal"),
        lambda: assert_provenance_columns(reference_dir, "reference"),
    ):
        try:
            check()
        except CleanReferenceAuditFailure as exc:
            report.failures.append(str(exc))
    if report.failures:
        return report

    if dataset_sha256 is not None and model_revision is not None:
        for run_dir, label in ((formal_dir, "formal"), (reference_dir, "reference")):
            try:
                assert_row_provenance_matches(
                    run_dir, label,
                    dataset_sha256=dataset_sha256, model_revision=model_revision,
                )
            except CleanReferenceAuditFailure as exc:
                report.failures.append(str(exc))
        report.details["anchored_dataset_sha256"] = dataset_sha256
        report.details["anchored_model_revision"] = model_revision

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

    # Bind the gate to the exact bytes audited, for the TOCTOU re-check.
    try:
        report.clean_artifact_digests = {
            "formal": clean_artifact_digest(formal_dir),
            "reference": clean_artifact_digest(reference_dir),
        }
    except CleanReferenceAuditFailure as exc:
        report.failures.append(str(exc))

    report.passed = not report.failures
    return report


def assert_clean_conditions_only(run_dir: Path, label: str) -> None:
    """A directory holding any non-clean condition is not a clean pass."""
    path = Path(run_dir) / "window_results.csv"
    if not path.exists():
        raise CleanReferenceAuditFailure(f"{path} is absent")
    kinds = sorted(set(pd.read_csv(path, usecols=["kind"])["kind"].astype(str)))
    if kinds != [CLEAN_CONDITION_ID]:
        raise CleanReferenceAuditFailure(
            f"HARD STOP: the {label} run at {run_dir} contains non-clean "
            f"condition(s) {kinds}. The audit gates the corrupted-condition "
            f"phase, so it must run while both sides hold clean forecasts only; "
            f"corrupted rows already present mean it is running too late to gate "
            f"anything."
        )


def verify_gate_still_binds(
    formal_dir: Path, reference_dir: Path, recorded: dict[str, dict[str, str]]
) -> None:
    """TOCTOU re-check: the clean artifacts must be the ones the gate audited.

    A gate is a statement about specific bytes. If those bytes changed after it
    was written — a re-run, a hand edit, a partially overwritten file — the gate
    no longer says anything about what is on disk now, and must not authorise
    proceeding.
    """
    if not recorded:
        raise CleanReferenceAuditFailure(
            "HARD STOP: the gate records no clean-artifact digests, so it cannot "
            "be bound to the files on disk. Re-run the audit."
        )
    for label, run_dir in (("formal", formal_dir), ("reference", reference_dir)):
        expected = recorded.get(label)
        if not expected:
            raise CleanReferenceAuditFailure(
                f"HARD STOP: the gate records no digests for the {label} run."
            )
        actual = clean_artifact_digest(run_dir)
        for name, digest in expected.items():
            if actual.get(name) != digest:
                raise CleanReferenceAuditFailure(
                    f"HARD STOP: {label}/{name} has CHANGED since the audit "
                    f"passed (gate recorded {digest}, file is now "
                    f"{actual.get(name)}). The gate no longer describes the data "
                    f"on disk. Re-run the clean passes and the audit; do not "
                    f"proceed on a stale authorisation."
                )


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
