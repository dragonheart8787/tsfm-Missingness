"""Clean-prediction audit: identity before values (Blocker 2).

Every adversarial case below keeps the PREDICTIONS numerically identical and
alters exactly one identity field. Each must fail the audit — a value hash alone
would pass all of them.

Model-mocked: no weights required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit_clean_predictions import (
    CleanAuditFailure,
    audit,
    canonical_clean_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
N_ORIGINS, HORIZON = 4, 3


def _write_run(run_dir: Path, *, predictions=None, origins=None, timestamps=None,
               ground_truth=None, target_digests=None, dataset_sha="ds-abc",
               revision="rev-123", drop_last=False, duplicate_first=False):
    """A minimal run directory: predictions_long.csv + window_results.csv."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows, windows = [], []
    k = 0
    for i in range(N_ORIGINS):
        origin = origins[i] if origins else i
        digest = target_digests[i] if target_digests else f"digest{i:02d}"
        windows.append({
            "origin_id": origin, "condition_id": "clean", "kind": "clean",
            "target_sha256": digest, "dataset_sha256": dataset_sha,
            "model_revision": revision, "status": "ok",
        })
        for step in range(HORIZON):
            rows.append({
                "run_id": "r", "origin_id": origin, "condition_id": "clean",
                "step_index": step,
                "forecast_timestamp": (
                    timestamps[k] if timestamps else f"2016-07-01T{step:02d}:00:00"
                ),
                "ground_truth": ground_truth[k] if ground_truth else float(k),
                "median_prediction": (
                    predictions[k] if predictions is not None else float(k) * 1.5
                ),
            })
            k += 1
    frame = pd.DataFrame(rows)
    if drop_last:
        frame = frame.iloc[:-1]
    if duplicate_first:
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(run_dir / "predictions_long.csv", index=False)
    pd.DataFrame(windows).to_csv(run_dir / "window_results.csv", index=False)
    return run_dir


def _audit(tmp_path, **new_kwargs):
    """Reference is always the baseline; `new` differs by the given kwarg only."""
    _write_run(tmp_path / "ref")
    _write_run(tmp_path / "new", **new_kwargs)
    return audit(tmp_path / "new", tmp_path / "ref", strict_size=False)


# --------------------------------------------------------------------------- #
# Baseline: identical runs must pass
# --------------------------------------------------------------------------- #

def test_identical_runs_pass(tmp_path):
    report = _audit(tmp_path)
    assert report.passed, report.failures
    assert report.new_hash == report.reference_hash


# --------------------------------------------------------------------------- #
# Adversarial: predictions identical, exactly one identity field altered
# --------------------------------------------------------------------------- #

def _assert_values_identical_but_audit_fails(report, tmp_path, *, expect):
    """Every case here must show a matching value hash and still fail."""
    assert report.new_hash == report.reference_hash, (
        "the predictions must stay numerically identical, or the test proves nothing"
    )
    assert not report.passed
    assert any(expect in f for f in report.failures), report.failures


def test_changed_origin_id_fails(tmp_path):
    """Same numbers, different cell identity."""
    report = _audit(tmp_path, origins=[0, 1, 2, 99])
    _assert_values_identical_but_audit_fails(report, tmp_path, expect="keys")


def test_changed_forecast_timestamp_fails(tmp_path):
    stamps = [f"2016-07-01T{s:02d}:00:00" for _ in range(N_ORIGINS) for s in range(HORIZON)]
    stamps[5] = "1999-01-01T00:00:00"
    report = _audit(tmp_path, timestamps=stamps)
    _assert_values_identical_but_audit_fails(report, tmp_path, expect="timestamps")


def test_changed_ground_truth_fails(tmp_path):
    truth = [float(k) for k in range(N_ORIGINS * HORIZON)]
    truth[7] = truth[7] + 1.0
    report = _audit(tmp_path, ground_truth=truth)
    _assert_values_identical_but_audit_fails(report, tmp_path, expect="ground_truth")


def test_changed_target_sha256_fails(tmp_path):
    digests = [f"digest{i:02d}" for i in range(N_ORIGINS)]
    digests[2] = "TAMPERED"
    report = _audit(tmp_path, target_digests=digests)
    _assert_values_identical_but_audit_fails(report, tmp_path, expect="target_sha256")


def test_changed_dataset_checksum_fails(tmp_path):
    report = _audit(tmp_path, dataset_sha="ds-DIFFERENT")
    _assert_values_identical_but_audit_fails(report, tmp_path, expect="dataset checksum")


def test_changed_model_revision_fails(tmp_path):
    report = _audit(tmp_path, revision="rev-DIFFERENT")
    _assert_values_identical_but_audit_fails(report, tmp_path, expect="model revision")


def test_missing_cell_fails(tmp_path):
    report = _audit(tmp_path, drop_last=True)
    assert not report.passed
    assert any("MISSING from new" in f for f in report.failures), report.failures


def test_duplicate_cell_fails(tmp_path):
    report = _audit(tmp_path, duplicate_first=True)
    assert not report.passed
    assert any("duplicate" in f for f in report.failures), report.failures


def test_changed_prediction_value_still_fails(tmp_path):
    """The original value check must survive the identity additions."""
    preds = [float(k) * 1.5 for k in range(N_ORIGINS * HORIZON)]
    preds[4] = preds[4] + 1e-3
    report = _audit(tmp_path, predictions=preds)
    assert not report.passed
    assert report.new_hash != report.reference_hash
    assert any("canonical float32 sha256" in f for f in report.failures)


def test_a_value_hash_alone_would_have_passed_every_identity_case(tmp_path):
    """The point of Blocker 2, demonstrated rather than asserted.

    For each identity tamper, the prediction hash is UNCHANGED — so the previous
    hash-only audit would have returned PASS for all of them.
    """
    cases = {
        "origin_id": dict(origins=[0, 1, 2, 99]),
        "timestamp": dict(timestamps=[
            "1999-01-01T00:00:00" if k == 5 else f"2016-07-01T{k % HORIZON:02d}:00:00"
            for k in range(N_ORIGINS * HORIZON)
        ]),
        "ground_truth": dict(ground_truth=[
            float(k) + (1.0 if k == 7 else 0.0) for k in range(N_ORIGINS * HORIZON)
        ]),
        "target_sha256": dict(target_digests=["digest00", "digest01", "TAMPERED", "digest03"]),
        "dataset_sha256": dict(dataset_sha="ds-DIFFERENT"),
        "model_revision": dict(revision="rev-DIFFERENT"),
    }
    _write_run(tmp_path / "ref")
    baseline_hash, _ = canonical_clean_hash(tmp_path / "ref")
    for name, kwargs in cases.items():
        target = tmp_path / f"new_{name}"
        _write_run(target, **kwargs)
        tampered_hash, _ = canonical_clean_hash(target)
        assert tampered_hash == baseline_hash, (
            f"{name}: predictions changed, so this case would not prove the point"
        )
        report = audit(target, tmp_path / "ref", strict_size=False)
        assert not report.passed, f"{name}: identity tamper slipped through the audit"


# --------------------------------------------------------------------------- #
# Structural: exact 178 x 96 cardinality
# --------------------------------------------------------------------------- #

def test_row_count_must_be_178_x_96(tmp_path):
    """strict_size is what the runbook uses; the toy runs opt out explicitly."""
    _write_run(tmp_path / "ref")
    _write_run(tmp_path / "new")
    report = audit(tmp_path / "new", tmp_path / "ref", strict_size=True)
    assert not report.passed
    assert any("expected 178 x 96 = 17088" in f for f in report.failures), report.failures


def test_absent_clean_predictions_is_a_hard_stop(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    pd.DataFrame([{"origin_id": 0, "condition_id": "trailing_nan_g16",
                   "step_index": 0, "forecast_timestamp": "t", "ground_truth": 1.0,
                   "median_prediction": 1.0}]).to_csv(
        run_dir / "predictions_long.csv", index=False
    )
    _write_run(tmp_path / "ref")
    with pytest.raises(CleanAuditFailure, match="no clean predictions"):
        audit(run_dir, tmp_path / "ref", strict_size=False)


def test_missing_window_results_blocks_provenance(tmp_path):
    _write_run(tmp_path / "ref")
    _write_run(tmp_path / "new")
    (tmp_path / "new" / "window_results.csv").unlink()
    report = audit(tmp_path / "new", tmp_path / "ref", strict_size=False)
    assert not report.passed
    assert any("provenance cannot be verified" in f for f in report.failures)


# --------------------------------------------------------------------------- #
# No tolerance escape hatch — for values OR identity
# --------------------------------------------------------------------------- #

def test_no_tolerance_mechanism_anywhere_in_the_script():
    source = (REPO_ROOT / "scripts" / "audit_clean_predictions.py").read_text(
        encoding="utf-8"
    )
    for banned in ("--tolerance", "atol", "rtol", "np.isclose", "isclose", "allclose",
                   "approx", "rel_tol", "abs_tol", "math.isclose"):
        assert banned not in source, f"a tolerance escape hatch appeared: {banned}"


def test_comparisons_are_exact_equality_not_approximate():
    """Identity fields must be compared with ==/!=, never a nearness test."""
    source = (REPO_ROOT / "scripts" / "audit_clean_predictions.py").read_text(
        encoding="utf-8"
    )
    assert "gt_new != gt_ref" in source, "ground truth must be compared exactly"
    assert "ts_new != ts_ref" in source, "timestamps must be compared exactly"
    assert "report.new_hash != report.reference_hash" in source


def test_canonical_hash_is_still_order_independent(tmp_path):
    _write_run(tmp_path / "a")
    frame = pd.read_csv(tmp_path / "a" / "predictions_long.csv")
    (tmp_path / "b").mkdir()
    frame.sample(frac=1.0, random_state=0).to_csv(
        tmp_path / "b" / "predictions_long.csv", index=False
    )
    assert canonical_clean_hash(tmp_path / "a")[0] == canonical_clean_hash(tmp_path / "b")[0]


def test_canonical_hash_ignores_non_clean_conditions(tmp_path):
    _write_run(tmp_path / "a")
    frame = pd.read_csv(tmp_path / "a" / "predictions_long.csv")
    noise = frame.copy()
    noise["condition_id"] = "trailing_nan_g16"
    noise["median_prediction"] = 999.0
    (tmp_path / "b").mkdir()
    pd.concat([frame, noise]).to_csv(tmp_path / "b" / "predictions_long.csv", index=False)
    assert canonical_clean_hash(tmp_path / "a")[0] == canonical_clean_hash(tmp_path / "b")[0]
