"""The two-pass clean control: exact equality or a hard stop.

The audit compares two INDEPENDENTLY GENERATED runs. Every check is exercised by
perturbing exactly one field at a time and confirming the audit refuses — so a
passing audit is evidence, not decoration. There is deliberately no tolerance
mechanism anywhere, and these tests assert that a sub-ULP perturbation is
rejected just as a large one is.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.clean_reference_audit import (
    CleanReferenceAuditFailure,
    assert_reference_excluded_from_analysis,
    audit_clean_reference,
    require_clean_reference_match,
)

ORIGINS = 4          # small but structurally identical to the real 178
HORIZON = 96
SHA = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


def _write_run(root: Path, *, kinds=("clean",), seed: int = 0) -> Path:
    """A run directory shaped exactly like run_trailing_gap's output."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    predictions, windows = [], []
    for origin in range(ORIGINS):
        target = rng.normal(size=HORIZON).astype(np.float32)
        digest = f"digest_{origin:04d}"
        stamps = pd.date_range("2016-07-01", periods=HORIZON, freq="h") + pd.Timedelta(
            hours=96 * origin
        )
        for kind in kinds:
            windows.append({
                "origin_id": origin, "condition_id": kind, "kind": kind,
                "target_sha256": digest, "dataset_sha256": SHA,
                "model_revision": REVISION, "status": "ok", "mae": 1.0,
            })
        for step in range(HORIZON):
            predictions.append({
                "origin_id": origin, "condition_id": "clean", "step_index": step,
                "forecast_timestamp": str(stamps[step]),
                "ground_truth": float(target[step]),
                "median_prediction": float(target[step]) + 0.25,
            })
    pd.DataFrame(windows).to_csv(root / "window_results.csv", index=False)
    pd.DataFrame(predictions).to_csv(root / "predictions_long.csv", index=False)
    (root / "run_manifest.json").write_text(
        json.dumps({"model_contract": {"revision": REVISION}}, indent=2), encoding="utf-8"
    )
    return root


@pytest.fixture
def two_runs(tmp_path) -> tuple[Path, Path]:
    """Two runs generated independently from the same inputs — bit-identical."""
    formal = _write_run(tmp_path / "formal", seed=11)
    reference = _write_run(tmp_path / "clean_reference", seed=11)
    return formal, reference


def _audit(formal: Path, reference: Path):
    return audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )


def _edit_predictions(run: Path, fn):
    frame = pd.read_csv(run / "predictions_long.csv")
    fn(frame)
    frame.to_csv(run / "predictions_long.csv", index=False)


# --------------------------------------------------------------------------- #
# The passing case must actually pass, and must have compared something
# --------------------------------------------------------------------------- #

def test_two_identical_runs_pass_and_the_hashes_are_reported(two_runs):
    formal, reference = two_runs
    report = _audit(formal, reference)
    assert report.passed, report.failures
    assert report.formal_hash == report.reference_hash
    assert len(report.formal_hash) == 64
    assert report.expected_origins == ORIGINS
    assert report.as_dict()["tolerance"] == "none"
    assert report.as_dict()["expected_clean_rows"] == ORIGINS * HORIZON
    # Non-vacuity: it really compared the cells.
    assert report.details["n_cells_new"] == ORIGINS * HORIZON
    assert report.details["n_cells_reference"] == ORIGINS * HORIZON
    assert report.details["n_origins_with_target_digest"] == ORIGINS


def test_the_gating_form_returns_on_success(two_runs):
    formal, reference = two_runs
    report = require_clean_reference_match(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert report.passed


# --------------------------------------------------------------------------- #
# One field at a time. Each must be a hard stop.
# --------------------------------------------------------------------------- #

def test_a_single_perturbed_prediction_is_rejected(two_runs):
    formal, reference = two_runs
    _edit_predictions(formal, lambda f: f.__setitem__(
        "median_prediction", f["median_prediction"].mask(f.index == 0, 99.0)
    ))
    report = _audit(formal, reference)
    assert not report.passed
    assert any("canonical float32 sha256 differs" in v for v in report.failures)


def test_there_is_no_tolerance_a_sub_ulp_change_is_still_rejected(two_runs):
    """No tolerance means NO tolerance — not a small one."""
    formal, reference = two_runs
    frame = pd.read_csv(formal / "predictions_long.csv")
    original = np.float32(frame.loc[0, "median_prediction"])
    nudged = np.nextafter(original, np.float32(np.inf))
    assert nudged != original
    frame.loc[0, "median_prediction"] = float(nudged)
    frame.to_csv(formal / "predictions_long.csv", index=False)
    report = _audit(formal, reference)
    assert not report.passed, "a one-ULP difference was tolerated"
    assert any("canonical float32 sha256 differs" in v for v in report.failures)


def test_a_changed_ground_truth_is_rejected(two_runs):
    formal, reference = two_runs
    _edit_predictions(formal, lambda f: f.__setitem__(
        "ground_truth", f["ground_truth"].mask(f.index == 3, -7.5)
    ))
    report = _audit(formal, reference)
    assert not report.passed
    assert any("ground_truth" in v for v in report.failures)


def test_a_changed_forecast_timestamp_is_rejected(two_runs):
    formal, reference = two_runs
    _edit_predictions(formal, lambda f: f.__setitem__(
        "forecast_timestamp",
        f["forecast_timestamp"].mask(f.index == 5, "1999-01-01 00:00:00"),
    ))
    report = _audit(formal, reference)
    assert not report.passed
    assert any("timestamps" in v for v in report.failures)


def test_a_missing_cell_is_rejected(two_runs):
    formal, reference = two_runs
    frame = pd.read_csv(formal / "predictions_long.csv")
    frame.iloc[1:].to_csv(formal / "predictions_long.csv", index=False)
    report = _audit(formal, reference)
    assert not report.passed
    assert any("keys" in v for v in report.failures)
    assert any("clean rows, expected" in v for v in report.failures)


def test_a_changed_target_digest_is_rejected(two_runs):
    formal, reference = two_runs
    frame = pd.read_csv(formal / "window_results.csv")
    frame.loc[0, "target_sha256"] = "tampered"
    frame.to_csv(formal / "window_results.csv", index=False)
    report = _audit(formal, reference)
    assert not report.passed
    assert any("target_sha256" in v for v in report.failures)


@pytest.mark.parametrize("column,bad", [("dataset_sha256", "0" * 64),
                                        ("model_revision", "deadbeef")])
def test_a_changed_environment_field_is_rejected(two_runs, column, bad):
    formal, reference = two_runs
    frame = pd.read_csv(formal / "window_results.csv")
    frame[column] = bad
    frame.to_csv(formal / "window_results.csv", index=False)
    report = _audit(formal, reference)
    assert not report.passed
    assert any(column in v or "checksum" in v or "revision" in v for v in report.failures)


def test_a_wrong_origin_count_is_rejected_against_the_derived_expectation(two_runs):
    """The cardinality check uses ETTh2's DERIVED count, not ETTh1's constant."""
    formal, reference = two_runs
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS + 1, expected_horizon=HORIZON
    )
    assert not report.passed
    assert any("origins, expected" in v and "DERIVED" in v for v in report.failures)


def test_the_hard_stop_raises_and_says_not_to_pick_a_tolerance(two_runs):
    formal, reference = two_runs
    _edit_predictions(formal, lambda f: f.__setitem__(
        "median_prediction", f["median_prediction"] + 1.0
    ))
    with pytest.raises(CleanReferenceAuditFailure) as excinfo:
        require_clean_reference_match(
            formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
        )
    message = str(excinfo.value)
    assert "HARD STOP" in message
    assert "do not select a tolerance now" in message
    assert "There is no tolerance mechanism" in message


# --------------------------------------------------------------------------- #
# clean_reference is QC-only
# --------------------------------------------------------------------------- #

def test_a_reference_containing_corrupted_conditions_is_refused(tmp_path):
    """It must hold clean forecasts and nothing else, so it cannot be analysed."""
    reference = _write_run(tmp_path / "ref", kinds=("clean", "trailing_nan"), seed=3)
    with pytest.raises(CleanReferenceAuditFailure, match="QC-only"):
        assert_reference_excluded_from_analysis(reference)


def test_the_audit_itself_catches_a_non_qc_reference(tmp_path):
    formal = _write_run(tmp_path / "formal", seed=11)
    reference = _write_run(tmp_path / "ref", kinds=("clean", "internal_block"), seed=11)
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert not report.passed
    # The clean-conditions-only precheck now fires first and returns early, so
    # the refusal is reported as "contains non-clean condition(s)" rather than
    # by the later QC-only check. Either message is the same refusal.
    assert any("non-clean condition" in v for v in report.failures), report.failures
    assert "EXCLUDED from the statistical analysis" in report.as_dict()["note"]


def test_a_clean_only_reference_passes_the_qc_check(two_runs):
    _formal, reference = two_runs
    assert_reference_excluded_from_analysis(reference)   # must not raise


def test_an_absent_run_directory_is_a_failure_not_a_pass(tmp_path, two_runs):
    formal, _reference = two_runs
    with pytest.raises(CleanReferenceAuditFailure):
        assert_reference_excluded_from_analysis(tmp_path / "does_not_exist")


# --------------------------------------------------------------------------- #
# Gate hardening: staleness, misconfiguration, and time-of-check/time-of-use
# --------------------------------------------------------------------------- #

def test_the_same_directory_on_both_sides_is_refused(two_runs):
    """A control that compares a directory with itself passes everything and
    proves nothing. That is the worst failure mode, because it looks like success."""
    from experiments.clean_reference_audit import assert_distinct_directories

    formal, _reference = two_runs
    with pytest.raises(CleanReferenceAuditFailure, match="SAME"):
        assert_distinct_directories(formal, formal)
    report = audit_clean_reference(
        formal, formal, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert not report.passed
    assert any("SAME" in v for v in report.failures)


def test_a_symlinked_duplicate_is_also_refused(tmp_path, two_runs):
    """Resolved paths, so a symlink cannot disguise the collision."""
    from experiments.clean_reference_audit import assert_distinct_directories

    formal, _reference = two_runs
    link = tmp_path / "alias"
    link.symlink_to(formal, target_is_directory=True)
    with pytest.raises(CleanReferenceAuditFailure, match="SAME"):
        assert_distinct_directories(formal, link)


@pytest.mark.parametrize("column", ["dataset_sha256", "model_revision"])
def test_a_missing_provenance_column_is_a_failure_not_a_skipped_check(two_runs, column):
    """The shared audit SKIPS a column it cannot find. That must not read as a pass."""
    formal, reference = two_runs
    frame = pd.read_csv(formal / "window_results.csv")
    frame.drop(columns=[column]).to_csv(formal / "window_results.csv", index=False)
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert not report.passed, f"a missing {column} was silently skipped"
    assert any(column in v and "missing provenance column" in v for v in report.failures)


def test_both_sides_agreeing_with_each_other_is_not_enough(two_runs):
    """Two runs can agree and both be stale. Anchor to what is loaded NOW."""
    formal, reference = two_runs
    # They agree with each other...
    assert audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    ).passed
    # ...but not with the current dataset.
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON,
        dataset_sha256="0" * 64, model_revision=REVISION,
    )
    assert not report.passed
    assert any("matching each other is not enough" in v for v in report.failures)
    assert sum("dataset_sha256" in v for v in report.failures) >= 2, "both sides"


def test_anchoring_to_the_wrong_model_revision_is_refused(two_runs):
    formal, reference = two_runs
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON,
        dataset_sha256=SHA, model_revision="a-different-checkpoint",
    )
    assert not report.passed
    assert any("model_revision" in v for v in report.failures)


def test_anchoring_to_the_correct_values_passes_and_is_recorded(two_runs):
    """Non-vacuity: the anchor must not reject a correct pair."""
    formal, reference = two_runs
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON,
        dataset_sha256=SHA, model_revision=REVISION,
    )
    assert report.passed, report.failures
    assert report.details["anchored_dataset_sha256"] == SHA
    assert report.details["anchored_model_revision"] == REVISION


def test_a_formal_dir_already_holding_corrupted_conditions_is_refused(tmp_path):
    """The audit gates the corrupted phase, so it must run before those exist."""
    formal = _write_run(tmp_path / "formal", kinds=("clean", "trailing_nan"), seed=11)
    reference = _write_run(tmp_path / "ref", seed=11)
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert not report.passed
    assert any("running too late to gate" in v for v in report.failures)


def test_the_gate_records_digests_of_the_exact_bytes_it_audited(two_runs):
    from experiments.clean_reference_audit import clean_content_digest

    formal, reference = two_runs
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert report.passed
    assert set(report.clean_content_digests) == {"formal", "reference"}
    for side, run_dir in (("formal", formal), ("reference", reference)):
        assert report.clean_content_digests[side] == clean_content_digest(run_dir)
        assert set(report.clean_content_digests[side]) == {
            "predictions_long.csv", "window_results.csv"
        }
    assert report.as_dict()["clean_content_digests"] == report.clean_content_digests


def _gate(two_runs):
    formal, reference = two_runs
    report = audit_clean_reference(
        formal, reference, expected_origins=ORIGINS, expected_horizon=HORIZON
    )
    assert report.passed, report.failures
    return formal, reference, report.clean_content_digests


def test_appending_non_clean_rows_does_NOT_invalidate_the_gate(two_runs):
    """The resumability bug this projection fixes.

    formal-full appends corrupted-condition rows to the same files while
    building the 2,314-row matrix. Under a whole-file digest every one of those
    appends invalidated the gate, so an interrupted run found its own legitimate
    progress reported as tampering. They are outside the projection now.
    """
    from experiments.clean_reference_audit import verify_gate_still_binds

    formal, reference, digests = _gate(two_runs)

    # Append the way the runner does — open in append mode and write new rows,
    # leaving every existing byte untouched. Rewriting the whole file through
    # pandas would reformat the clean rows' floats, which IS a change to clean
    # content and is correctly rejected; the runner never does that.
    for filename, condition, kind in (
        ("predictions_long.csv", "trailing_nan_g16", None),
        ("window_results.csv", "internal_block_g64", "internal_block"),
    ):
        path = formal / filename
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = frame[frame["condition_id"] == "clean"].head(20).copy()
        rows["condition_id"] = condition
        if kind is not None:
            rows["kind"] = kind
        with path.open("a", encoding="utf-8") as handle:
            handle.write(rows.to_csv(index=False, header=False))

    verify_gate_still_binds(formal, reference, digests)   # must NOT raise

    # ...and the appended rows really are there.
    after = pd.read_csv(formal / "predictions_long.csv", dtype=str, keep_default_na=False)
    assert (after["condition_id"] == "trailing_nan_g16").sum() == 20


@pytest.mark.parametrize("side", ["formal", "reference"])
@pytest.mark.parametrize("filename", ["predictions_long.csv", "window_results.csv"])
def test_a_MUTATED_clean_row_still_hard_fails(two_runs, side, filename):
    """The fix must not become permissive toward what it protects."""
    from experiments.clean_reference_audit import verify_gate_still_binds

    formal, reference, digests = _gate(two_runs)
    verify_gate_still_binds(formal, reference, digests)   # binds before the edit

    target = (formal if side == "formal" else reference) / filename
    frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    clean_rows = frame.index[frame["condition_id"] == "clean"]
    column = "median_prediction" if filename == "predictions_long.csv" else "mae"
    frame.loc[clean_rows[0], column] = "999.0"
    frame.to_csv(target, index=False)

    with pytest.raises(CleanReferenceAuditFailure, match="CLEAN ROWS.*CHANGED"):
        verify_gate_still_binds(formal, reference, digests)


@pytest.mark.parametrize("side", ["formal", "reference"])
def test_a_REMOVED_clean_row_still_hard_fails(two_runs, side):
    from experiments.clean_reference_audit import verify_gate_still_binds

    formal, reference, digests = _gate(two_runs)
    target = (formal if side == "formal" else reference) / "predictions_long.csv"
    frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    frame.iloc[1:].to_csv(target, index=False)
    with pytest.raises(CleanReferenceAuditFailure, match="CLEAN ROWS.*CHANGED"):
        verify_gate_still_binds(formal, reference, digests)


@pytest.mark.parametrize("side", ["formal", "reference"])
def test_an_ADDED_clean_row_still_hard_fails(two_runs, side):
    from experiments.clean_reference_audit import verify_gate_still_binds

    formal, reference, digests = _gate(two_runs)
    target = (formal if side == "formal" else reference) / "predictions_long.csv"
    frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    extra = frame.head(1).copy()
    extra["origin_id"] = "9999"
    pd.concat([frame, extra]).to_csv(target, index=False)
    with pytest.raises(CleanReferenceAuditFailure, match="CLEAN ROWS.*CHANGED"):
        verify_gate_still_binds(formal, reference, digests)


def test_a_reordered_clean_file_still_binds(two_runs):
    """The projection sorts, so row ORDER is not part of the invariant."""
    from experiments.clean_reference_audit import verify_gate_still_binds

    formal, reference, digests = _gate(two_runs)
    target = formal / "predictions_long.csv"
    frame = pd.read_csv(target, dtype=str, keep_default_na=False)
    frame.iloc[::-1].to_csv(target, index=False)
    verify_gate_still_binds(formal, reference, digests)   # must NOT raise


def test_a_gate_lacking_clean_rows_entirely_is_refused(tmp_path):
    from experiments.clean_reference_audit import clean_content_digest

    run = _write_run(tmp_path / "corrupt_only", kinds=("trailing_nan",), seed=5)
    frame = pd.read_csv(run / "predictions_long.csv")
    frame["condition_id"] = "trailing_nan_g16"
    frame.to_csv(run / "predictions_long.csv", index=False)
    with pytest.raises(CleanReferenceAuditFailure, match="no clean rows to bind to"):
        clean_content_digest(run)


def test_a_gate_with_no_digests_cannot_bind(two_runs):
    from experiments.clean_reference_audit import verify_gate_still_binds

    formal, reference = two_runs
    with pytest.raises(CleanReferenceAuditFailure, match="records no clean-content"):
        verify_gate_still_binds(formal, reference, {})
    with pytest.raises(CleanReferenceAuditFailure, match="no digests for the formal"):
        verify_gate_still_binds(formal, reference, {"reference": {"a": "b"}})
