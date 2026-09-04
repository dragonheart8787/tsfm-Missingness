"""ETTh2 replication runner — the two-pass clean design, phase by phase.

    *** NOT YET EXECUTED. No ETTh2 forecast has been produced. ***

Design principle: this module ORCHESTRATES, it does not re-implement. The
per-origin forecasting loop is ``experiments.run_trailing_gap.run_trailing_gap``
— the exact code that produced the ETTh1 numbers being replicated — called
with an ETTh2 dataset block and the UNMODIFIED
``configs/trailing_gap_config.yaml``. A replication whose execution path was a
parallel implementation could differ from the study it replicates in ways no
diff would show; reusing the path removes that class of difference entirely.

What is genuinely new here, and therefore lives here:

  1. the NATIVE-MISSINGNESS HARD STOP, run before anything else and able to
     halt the pipeline rather than warn;
  2. the effective-config merge, so ETTh2 restates no shared constant;
  3. the four phases of the two-pass clean control, one per process; and
  4. the audit gate that makes the corrupted-condition phase refuse to start
     until the two clean runs have been proved identical.

The four phases
---------------
Each is a SEPARATE invocation, so each gets its own process and its own model
load — that separation is the control, not an implementation detail.

  1. ``clean-reference``  all origins, clean only, in its own directory. QC ONLY:
                          excluded from the statistical analysis, never a data
                          source for R_g or B_g.
  2. ``formal-clean``     the formal run's own clean-only pass, in a second,
                          independent process.
  3. ``audit``            exact equality between them on cardinality, keys,
                          timestamps, ground truth, target digests, dataset
                          checksum, model revision and the canonical float32
                          prediction sha256. No tolerance of any kind. Writes the
                          gate file on success only.
  4. ``formal-full``      the remaining corrupted conditions. REFUSES to start
                          without a passing gate file that matches this run's own
                          dataset checksum and model revision.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.fetch_etth2 import load_series as load_etth2  # noqa: E402
from experiments.clean_reference_audit import (  # noqa: E402
    CleanReferenceAuditFailure,
    assert_clean_conditions_only,
    audit_clean_reference,
    verify_gate_still_binds,
)
from experiments.run_trailing_gap import (  # noqa: E402
    run_trailing_gap,
    write_run_summary,
)
from model.chronos2_runner import Forecaster, MockForecaster  # noqa: E402

GATE_FILENAME = "clean_reference_audit.json"

# The commit the Research Lead signed the frozen design off against. Recorded in
# every ETTh2 manifest so a run can be traced to the design it was authorised
# under, not merely to the tree it executed from.
DESIGN_SIGNOFF_COMMIT = "5b62949"
EXECUTION_ENTRYPOINT = "experiments/run_etth2.py"
UNAVAILABLE = "unavailable"

PHASE_CLEAN_REFERENCE = "clean-reference"
PHASE_FORMAL_CLEAN = "formal-clean"
PHASE_AUDIT = "audit"
PHASE_FORMAL_FULL = "formal-full"
PHASES = (PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN, PHASE_AUDIT, PHASE_FORMAL_FULL)

# One sentence, defined once, stamped on every artifact a mock run produces.
MOCK_NOT_A_FINDING = (
    "MOCK FORECASTER. This is a pipeline exercise. The numbers are arbitrary, "
    "they are NOT a Chronos-2 result, and they are NOT an ETTh2 finding. "
    "Nothing here may be reported, cited, or compared against ETTh1."
)

# Keys the ETTh2 config contributes to the effective config. Everything else
# comes from the pilot config untouched, so a shared constant cannot be
# redefined here by accident.
OVERRIDE_SECTIONS = ("dataset", "design")


class ReplicationGateError(RuntimeError):
    """A precondition for this phase is not satisfied. Always a hard stop."""


class CompletionInvariantViolation(RuntimeError):
    """The finished matrix violates a frozen completion invariant.

    Raised so the PROCESS fails. A failed cell recorded as a status value inside
    an output file, with the invocation still exiting 0, is invisible to any
    wrapper script and to anyone reading only the exit code — and the failure
    would be silently absorbed into a "resume later" path that never happens.
    """


def assert_completion_invariants(
    summary: dict[str, Any], *, expected_origins: int, conditions_per_origin: int
) -> None:
    """Every frozen completion invariant, checked on the finished matrix.

    NO CELL IS RETRIED. A failure is surfaced at the process level and left for
    a human to decide about; automatically re-running it would hide both the
    failure and its cause.
    """
    expected_cells = expected_origins * conditions_per_origin
    problems: list[str] = []

    if summary.get("result_rows") is None:
        problems.append(
            f"the run did not complete: only "
            f"{summary.get('origins_fully_completed')} of {expected_origins} "
            f"origins finished, so the completion invariants were never evaluated"
        )
    checks = (
        ("origins_fully_completed", expected_origins),
        ("result_rows", expected_cells),
        ("distinct_cells", expected_cells),
        ("max_distinct_target_digests_per_origin", 1),
    )
    for key, expected in checks:
        actual = summary.get(key)
        if actual is not None and actual != expected:
            problems.append(f"{key} is {actual}, expected {expected}")

    status = summary.get("rows_by_status")
    if status is not None and status != {"ok": expected_cells}:
        problems.append(
            f"rows_by_status is {status}, expected {{'ok': {expected_cells}}} — "
            f"every cell must have succeeded"
        )
    failed = summary.get("forecasts_failed_this_process")
    if failed:
        problems.append(f"{failed} forecast(s) failed in this process")

    if problems:
        raise CompletionInvariantViolation(
            "HARD STOP: the completed matrix violates a frozen completion "
            "invariant:\n  - " + "\n  - ".join(problems)
            + "\n\nNo cell was retried, deliberately: a failure must be visible "
            "at the process level rather than absorbed into a resume path. "
            "Investigate the recorded error before re-running anything."
        )


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def build_effective_config(
    pilot_config: dict[str, Any], etth2_config: dict[str, Any]
) -> dict[str, Any]:
    """Pilot config with ONLY ETTh2's dataset and design blocks substituted.

    The substitution is whole-section and explicit: statistics, decision, model,
    metrics and masks are carried through byte-for-byte from the pilot, so the
    SESOI, the bootstrap settings and the pinned model revision are structurally
    incapable of differing between the two datasets.

    ``design`` is substituted rather than merged because ETTh2 owns its origin
    count; ``tests/test_etth2_config.py`` asserts that every geometry key it
    restates (L, H, stride) is EQUAL to the pilot's, so the substitution cannot
    become a redefinition.
    """
    effective = copy.deepcopy(pilot_config)
    for section in OVERRIDE_SECTIONS:
        if section not in etth2_config:
            raise ReplicationGateError(f"etth2 config has no {section!r} section")
        effective[section] = copy.deepcopy(etth2_config[section])
    effective["run"] = copy.deepcopy(pilot_config.get("run", {}))
    effective["run"]["name"] = etth2_config["meta"]["name"]
    return effective


def load_configs(
    config_path: str, pilot_path: str, gap_path: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    etth2 = yaml.safe_load((REPO_ROOT / config_path).read_text(encoding="utf-8"))
    pilot = yaml.safe_load((REPO_ROOT / pilot_path).read_text(encoding="utf-8"))
    gap = yaml.safe_load((REPO_ROOT / gap_path).read_text(encoding="utf-8"))
    return etth2, pilot, gap


# --------------------------------------------------------------------------- #
# The hard stop
# --------------------------------------------------------------------------- #

def native_missingness_gate(effective_config: dict[str, Any]) -> dict[str, Any]:
    """Run the ETTh2 loader, which HARD-STOPS on any native missing value.

    Called at the top of every phase, before a forecaster is constructed and
    therefore before any model is loaded. It raises rather than returning a
    status, so no caller can proceed by ignoring a return value.

    It also re-derives the origin count from the file on disk and compares it to
    the config, so a swapped or truncated ETTh2 fails here rather than producing
    a differently shaped matrix.
    """
    _values, _stamps, validation = load_etth2(effective_config)
    return {
        "dataset_sha256": validation.sha256,
        "n_rows": validation.n_rows,
        "native_missing_in_target": validation.native_missing_in_target,
        "n_gaps_in_hourly_grid": validation.n_gaps_in_hourly_grid,
        "timestamp_first": validation.timestamp_first,
        "timestamp_last": validation.timestamp_last,
    }


# --------------------------------------------------------------------------- #
# The audit gate
# --------------------------------------------------------------------------- #

def _manifest_revision(formal_dir: Path) -> str:
    """The revision the formal run's clean pass actually loaded.

    Read from its own manifest rather than from the config, so the anchor is
    what ran, not what was intended to run.
    """
    manifest_path = Path(formal_dir) / "run_manifest.json"
    if not manifest_path.exists():
        raise ReplicationGateError(
            f"HARD STOP: {manifest_path} is absent. The formal run's clean pass "
            f"has not been run in this directory, so there is nothing to audit."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest["model_contract"]["revision"])


def write_gate(formal_dir: Path, report, *, dataset_sha256: str, model_revision: str) -> Path:
    """Record a PASSING audit, bound to the data and weights it was run on.

    Binding matters: an unbound gate file would authorise a later run against a
    different dataset or a different checkpoint, which is exactly the drift the
    audit exists to catch.
    """
    payload = report.as_dict()
    payload["dataset_sha256"] = dataset_sha256
    payload["model_revision"] = model_revision
    payload["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path = Path(formal_dir) / GATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def require_gate(
    formal_dir: Path, *, dataset_sha256: str, model_revision: str | None = None
) -> dict[str, Any]:
    """Refuse to run corrupted conditions without a passing, matching gate.

    ``model_revision`` is optional here so the cheap checks — the gate exists,
    it passed, and it was written for this dataset — can run BEFORE a model is
    loaded. The revision the run will actually use is only known after the load,
    and is checked by ``assert_gate_matches_model`` at that point.
    """
    path = Path(formal_dir) / GATE_FILENAME
    if not path.exists():
        raise ReplicationGateError(
            f"HARD STOP: {path} is absent. The clean_reference audit has not been "
            f"run and passed for this run directory, so no corrupted-condition "
            f"forecast may be produced. Run the '{PHASE_AUDIT}' phase first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise ReplicationGateError(
            f"HARD STOP: {path} records a FAILED clean_reference audit "
            f"({payload.get('n_failures')} failure(s)). Do not proceed, and do "
            f"not select a tolerance now."
        )
    _require_gate_field(path, payload, "dataset_sha256", dataset_sha256)
    if model_revision is not None:
        _require_gate_field(path, payload, "model_revision", model_revision)
    return payload


def _require_gate_field(path: Path, payload: dict[str, Any], key: str, actual: str) -> None:
    recorded = payload.get(key)
    if recorded != actual:
        raise ReplicationGateError(
            f"HARD STOP: {path} was written for {key}={recorded!r}, but this "
            f"run has {key}={actual!r}. The audit does not authorise this run. "
            f"Re-run the clean passes and the audit."
        )


def assert_gate_matches_model(
    formal_dir: Path, payload: dict[str, Any], *, model_revision: str
) -> None:
    """The loaded checkpoint must be the one the audit was run against.

    Separate from ``require_gate`` because it can only be answered once the model
    is loaded. A mock-produced gate therefore cannot authorise a real run, and a
    real gate cannot authorise a mock one — which is the intended behaviour, not
    an inconvenience to work around.
    """
    _require_gate_field(Path(formal_dir) / GATE_FILENAME, payload, "model_revision", model_revision)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

def git_provenance() -> dict[str, Any]:
    """The commit and dirty status AT RUN TIME, read from git, never assumed.

    A manifest that recorded only the intended commit would be silent about a
    run executed from a modified tree, which is exactly the case a reviewer most
    needs to see. Failures are recorded rather than swallowed: an unavailable
    git is a fact about the run, not a reason to omit the field.
    """
    def _git(*args: str, strip: bool = True) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{UNAVAILABLE}: {type(exc).__name__}"
        if out.returncode != 0:
            return f"{UNAVAILABLE}: git {' '.join(args)} exited {out.returncode}"
        # porcelain lines are position-significant: their first two characters
        # are the status code and the third is a separator, so a leading space
        # is data. Stripping it would shift every path on the first line.
        return out.stdout.strip() if strip else out.stdout.rstrip("\n")

    commit = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain", strip=False)
    available = not porcelain.startswith(UNAVAILABLE)
    return {
        "git_commit": commit,
        "git_dirty": bool(porcelain) if available else None,
        # Recorded VERBATIM, status codes included. Slicing off a fixed prefix
        # would mangle renames ("R  old -> new") and, as this code originally
        # did, drop a character whenever the output had been stripped.
        "git_status_porcelain": (
            porcelain.splitlines()[:50] if available else porcelain
        ),
    }


def build_run_metadata(
    etth2_config: dict[str, Any], *, phase: str, mock: bool
) -> dict[str, Any]:
    """ETTh2's OWN manifest identity. Explicitly overridden, never inherited.

    ``run_trailing_gap`` defaults ``experiment`` and ``preregistration`` to
    ``trailing_gap_config.meta``, which names the ETTh1 experiment. Letting that
    default through would put ETTh1's identity on an ETTh2 manifest — wrong in
    the single file a reviewer uses to establish what was executed. So both are
    set here, from the ETTh2 config, and the runner raises if any other manifest
    key is collided with.
    """
    return {
        "experiment": etth2_config["meta"]["name"],
        "preregistration": etth2_config["meta"]["preregistration"],
        "replication_rule_version": etth2_config["replication_rule"]["rule_version"],
        "design_signoff_commit": DESIGN_SIGNOFF_COMMIT,
        "execution_entrypoint": EXECUTION_ENTRYPOINT,
        "phase": phase,
        "mock_model": bool(mock),
        "mock_model_note": (
            MOCK_NOT_A_FINDING if mock
            else "Real forecaster. Subject to the runbook's preconditions."
        ),
        "etth2_config": etth2_config,
        **git_provenance(),
    }


def _forecaster(config: dict[str, Any], *, mock: bool) -> Forecaster:
    if mock:
        print(
            "WARNING: MOCK forecaster. A PIPELINE EXERCISE, NOT a Chronos-2 result "
            "and NOT an ETTh2 finding.",
            flush=True,
        )
        return MockForecaster()
    from model.chronos2_runner import Chronos2Forecaster

    return Chronos2Forecaster(config=config)


def run_phase(
    *,
    phase: str,
    etth2_config: dict[str, Any],
    pilot_config: dict[str, Any],
    gap_config: dict[str, Any],
    mock: bool = False,
    limit_origins: int | None = None,
    reference_dir: Path | None = None,
    formal_dir: Path | None = None,
) -> dict[str, Any]:
    """Run exactly ONE phase. One phase per process is the control, not a detail."""
    if phase not in PHASES:
        raise ReplicationGateError(f"unknown phase {phase!r}; expected one of {list(PHASES)}")

    effective = build_effective_config(pilot_config, etth2_config)

    # THE HARD STOP, before any model load, in every phase without exception.
    dataset_facts = native_missingness_gate(effective)

    clean_control = etth2_config["clean_control"]
    reference_dir = Path(reference_dir or REPO_ROOT / clean_control["reference_run_dir"])
    formal_dir = Path(formal_dir or REPO_ROOT / clean_control["formal_run_dir"])
    design = effective["design"]

    if phase == PHASE_AUDIT:
        revision_for_anchor = _manifest_revision(formal_dir)
        report = audit_clean_reference(
            formal_dir, reference_dir,
            expected_origins=int(design["expected_origins"]),
            expected_horizon=int(design["horizon"]),
            # Anchor both sides to what is loaded NOW, not merely to each other.
            dataset_sha256=dataset_facts["dataset_sha256"],
            model_revision=revision_for_anchor,
        )
        # The revision the formal run ACTUALLY loaded, read from its own
        # manifest rather than from the config, so the gate records what ran.
        revision = revision_for_anchor
        if not report.passed:
            raise CleanReferenceAuditFailure(
                "HARD STOP: the clean_reference run and the formal run's clean "
                "pass do not match exactly.\n  - " + "\n  - ".join(report.failures)
                + "\n\nDo not proceed to any corrupted-condition forecast, and do "
                "not select a tolerance now — not for the predictions and not for "
                "any identity field. There is no tolerance mechanism, by design."
            )
        gate = write_gate(
            formal_dir, report,
            dataset_sha256=dataset_facts["dataset_sha256"], model_revision=revision,
        )
        return {
            "phase": phase, "passed": True, "gate_file": str(gate),
            "dataset_facts": dataset_facts, **report.as_dict(),
        }

    if phase == PHASE_CLEAN_REFERENCE:
        run_dir, only = reference_dir, ["clean"]
    elif phase == PHASE_FORMAL_CLEAN:
        run_dir, only = formal_dir, ["clean"]
    else:  # PHASE_FORMAL_FULL
        run_dir, only = formal_dir, None
        # Cheap checks first, before paying for a model load.
        gate = require_gate(
            formal_dir, dataset_sha256=dataset_facts["dataset_sha256"]
        )
        # Then the time-of-check/time-of-use re-check: the gate is a statement
        # about specific bytes, and those bytes must still be the ones on disk.
        # A gate that was valid when written does not authorise proceeding if the
        # clean data has changed since.
        if "clean_content_digests" not in gate:
            raise ReplicationGateError(
                f"HARD STOP: the gate in {formal_dir} predates the clean-content "
                f"projection (it records whole-file digests, which could not tell "
                f"a tampered clean row from a legitimately appended corrupted "
                f"one). Re-run the '{PHASE_AUDIT}' phase to write a gate this "
                f"version can verify."
            )
        verify_gate_still_binds(
            formal_dir, reference_dir, gate["clean_content_digests"]
        )
        # And the clean side must still be clean-only: corrupted rows appearing
        # between the audit and now would mean this phase has already started.
        assert_clean_conditions_only(reference_dir, "reference")

    forecaster = _forecaster(effective, mock=mock)
    if phase == PHASE_FORMAL_FULL:
        # ...then the one check that needs the loaded checkpoint.
        assert_gate_matches_model(
            formal_dir, gate, model_revision=str(forecaster.contract.revision)
        )
    metadata = build_run_metadata(etth2_config, phase=phase, mock=mock)
    summary = run_trailing_gap(
        config=effective, gap_config=gap_config, forecaster=forecaster,
        run_dir=run_dir, limit_origins=limit_origins, only_conditions=only,
        run_metadata=metadata,
    )
    # The runner writes run_summary.json before returning; the wrapper knows
    # things the runner does not (phase, mock status, dataset facts, provenance).
    # Enrich and persist ATOMICALLY, so a crash mid-write cannot leave a summary
    # that is half the runner's and half the wrapper's.
    summary["phase"] = phase
    summary["dataset_facts"] = dataset_facts
    summary["mock_model"] = bool(mock)
    if mock:
        summary["scientifically_valid"] = False
        summary["authoritative_result"] = False
        summary["mock_model_note"] = MOCK_NOT_A_FINDING
    summary["experiment"] = metadata["experiment"]
    summary["preregistration"] = metadata["preregistration"]
    summary["replication_rule_version"] = metadata["replication_rule_version"]
    summary["design_signoff_commit"] = metadata["design_signoff_commit"]
    summary["execution_entrypoint"] = metadata["execution_entrypoint"]
    summary["git_commit"] = metadata["git_commit"]
    summary["git_dirty"] = metadata["git_dirty"]
    write_run_summary(run_dir, summary)

    # The completion invariants apply to a formal-full pass that was asked to do
    # the WHOLE matrix. A deliberately limited pass is a partial run by
    # instruction, not a violation. The summary is persisted first, so the
    # evidence survives the raise.
    if phase == PHASE_FORMAL_FULL and limit_origins is None:
        assert_completion_invariants(
            summary,
            expected_origins=int(design["expected_origins"]),
            conditions_per_origin=int(
                etth2_config["design"]["expected_conditions_per_origin"]
            ),
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one phase of the ETTh2 trailing-gap replication."
    )
    parser.add_argument("--phase", required=True, choices=list(PHASES))
    parser.add_argument("--config", default="configs/etth2_config.yaml")
    parser.add_argument("--pilot-config", default="configs/pilot_config.yaml")
    parser.add_argument("--gap-config", default="configs/trailing_gap_config.yaml")
    parser.add_argument("--reference-run-dir", default=None)
    parser.add_argument("--formal-run-dir", default=None)
    parser.add_argument("--limit-origins", type=int, default=None)
    parser.add_argument(
        "--mock-model", action="store_true",
        help="Pipeline exercise only. NOT a Chronos-2 result and NOT an ETTh2 finding.",
    )
    args = parser.parse_args()

    etth2, pilot, gap = load_configs(args.config, args.pilot_config, args.gap_config)
    try:
        summary = run_phase(
            phase=args.phase, etth2_config=etth2, pilot_config=pilot, gap_config=gap,
            mock=args.mock_model, limit_origins=args.limit_origins,
            reference_dir=Path(args.reference_run_dir) if args.reference_run_dir else None,
            formal_dir=Path(args.formal_run_dir) if args.formal_run_dir else None,
        )
    except CompletionInvariantViolation as exc:
        # The run_summary.json on disk already records what happened; this makes
        # the same fact visible to anything that only reads an exit code.
        print(str(exc), file=sys.stderr, flush=True)
        return 6
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
