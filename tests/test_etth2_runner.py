"""The ETTh2 runner: hard stop first, then the four phases, gated by the audit.

Two properties matter most here and are tested by construction rather than by
reading the source:

  1. the native-missingness check runs BEFORE any model is loaded, in every
     phase, and can actually halt the run; and
  2. the corrupted-condition phase REFUSES to start without a passing audit gate
     that matches this run's own dataset checksum and model revision.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data.fetch_etth2 import NativeMissingnessError
from experiments.run_etth2 import (
    GATE_FILENAME,
    PHASE_AUDIT,
    PHASE_CLEAN_REFERENCE,
    PHASE_FORMAL_CLEAN,
    PHASE_FORMAL_FULL,
    PHASES,
    ReplicationGateError,
    build_effective_config,
    native_missingness_gate,
    require_gate,
    run_phase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


@pytest.fixture(scope="module")
def etth2_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def gap_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture
def etth2_present(etth2_config) -> None:
    csv = REPO_ROOT / etth2_config["dataset"]["local_path"]
    if not csv.exists():
        pytest.skip("ETTh2 not fetched; run data/fetch_etth2.py")


# --------------------------------------------------------------------------- #
# The hard stop, before any model load
# --------------------------------------------------------------------------- #

class _ExplodingForecaster:
    """Constructing this is the failure. If a phase loads a model before the
    native-missingness gate has run, this raises and the test catches the wrong
    exception type — which is exactly the regression to detect."""

    def __init__(self, *args, **kwargs):
        raise AssertionError(
            "a forecaster was constructed before the native-missingness hard stop"
        )


@pytest.fixture(scope="module")
def series_with_a_native_nan(tmp_path_factory, etth2_config) -> Path:
    """An ETTh2-shaped file carrying one native NaN in OT. Built once."""
    root = tmp_path_factory.mktemp("nan_series")
    csv = root / "raw" / "ETTh2.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    n = int(etth2_config["dataset"]["expected_rows"])
    frame = pd.DataFrame({
        "date": pd.date_range("2016-07-01", periods=n, freq="h").astype(str),
        "OT": np.linspace(0.0, 50.0, n),
    })
    frame.loc[123, "OT"] = np.nan          # the violation
    frame.to_csv(csv, index=False)
    return root


@pytest.mark.parametrize("phase", [PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN,
                                   PHASE_FORMAL_FULL, PHASE_AUDIT])
def test_the_hard_stop_runs_before_any_model_load_in_every_phase(
    monkeypatch, tmp_path, series_with_a_native_nan, etth2_config, config,
    gap_config, phase
):
    """Every phase without exception, including the audit."""
    import data.fetch_etth2 as fetch
    import experiments.run_etth2 as mod

    monkeypatch.setattr(mod, "_forecaster", _ExplodingForecaster)
    monkeypatch.setattr(fetch, "REPO_ROOT", series_with_a_native_nan)

    broken = copy.deepcopy(etth2_config)
    broken["dataset"]["local_path"] = "raw/ETTh2.csv"
    broken["dataset"]["expected_sha256"] = None

    with pytest.raises(NativeMissingnessError):
        run_phase(
            phase=phase, etth2_config=broken, pilot_config=config,
            gap_config=gap_config, mock=True,
            reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
        )


def test_the_gate_returns_the_measured_dataset_facts(etth2_present, etth2_config, config):
    effective = build_effective_config(config, etth2_config)
    facts = native_missingness_gate(effective)
    assert facts["native_missing_in_target"] == 0
    assert facts["dataset_sha256"] == SHA
    assert facts["n_rows"] == 17420
    assert facts["n_gaps_in_hourly_grid"] == 0


def test_an_unknown_phase_is_refused(etth2_config, config, gap_config):
    with pytest.raises(ReplicationGateError, match="unknown phase"):
        run_phase(
            phase="whatever", etth2_config=etth2_config, pilot_config=config,
            gap_config=gap_config, mock=True,
        )


def test_there_are_exactly_four_phases():
    assert PHASES == (PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN,
                      PHASE_AUDIT, PHASE_FORMAL_FULL)


# --------------------------------------------------------------------------- #
# The audit gate
# --------------------------------------------------------------------------- #

def _write_gate(formal: Path, *, passed=True, sha=SHA, revision=REVISION) -> Path:
    formal.mkdir(parents=True, exist_ok=True)
    path = formal / GATE_FILENAME
    path.write_text(json.dumps({
        "passed": passed, "n_failures": 0 if passed else 3,
        "dataset_sha256": sha, "model_revision": revision,
    }, indent=2), encoding="utf-8")
    return path


def test_an_absent_gate_refuses_the_corrupted_condition_phase(tmp_path):
    with pytest.raises(ReplicationGateError, match="has not been run and passed"):
        require_gate(tmp_path / "formal", dataset_sha256=SHA, model_revision=REVISION)


def test_a_failed_gate_refuses(tmp_path):
    formal = tmp_path / "formal"
    _write_gate(formal, passed=False)
    with pytest.raises(ReplicationGateError, match="records a FAILED"):
        require_gate(formal, dataset_sha256=SHA, model_revision=REVISION)


def test_a_gate_written_for_a_different_dataset_does_not_authorise_this_run(tmp_path):
    """A stale gate must not authorise a run against different data."""
    formal = tmp_path / "formal"
    _write_gate(formal, sha="0" * 64)
    with pytest.raises(ReplicationGateError, match="does not authorise this run"):
        require_gate(formal, dataset_sha256=SHA, model_revision=REVISION)


def test_a_gate_written_for_a_different_model_does_not_authorise_this_run(tmp_path):
    formal = tmp_path / "formal"
    _write_gate(formal, revision="some-other-revision")
    with pytest.raises(ReplicationGateError, match="does not authorise this run"):
        require_gate(formal, dataset_sha256=SHA, model_revision=REVISION)
    # Omitting the revision defers that one check; it never waives it.
    assert require_gate(formal, dataset_sha256=SHA)["passed"] is True


def test_a_matching_passing_gate_authorises(tmp_path):
    formal = tmp_path / "formal"
    _write_gate(formal)
    payload = require_gate(formal, dataset_sha256=SHA, model_revision=REVISION)
    assert payload["passed"] is True


def test_the_corrupted_phase_refuses_without_a_gate_end_to_end(
    etth2_present, monkeypatch, tmp_path, etth2_config, config, gap_config
):
    """The whole phase, not just the helper: no gate -> no model load, no forecasting."""
    import experiments.run_etth2 as mod

    monkeypatch.setattr(mod, "_forecaster", _ExplodingForecaster)
    with pytest.raises(ReplicationGateError, match="clean_reference audit"):
        run_phase(
            phase=PHASE_FORMAL_FULL, etth2_config=etth2_config, pilot_config=config,
            gap_config=gap_config, mock=True,
            reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
        )


def test_a_gate_from_a_different_checkpoint_refuses_after_the_model_loads(
    etth2_present, tmp_path, etth2_config, config, gap_config
):
    """The revision check needs the loaded model, so it runs after the load.

    A gate written against one checkpoint must not authorise a run on another.
    """
    from experiments.run_etth2 import assert_gate_matches_model

    formal = tmp_path / "formal"
    _write_gate(formal, revision="a-different-checkpoint")
    payload = require_gate(formal, dataset_sha256=SHA)   # cheap checks pass
    assert payload["passed"] is True
    with pytest.raises(ReplicationGateError, match="does not authorise this run"):
        assert_gate_matches_model(formal, payload, model_revision=REVISION)


# --------------------------------------------------------------------------- #
# The clean passes actually run, on the shared execution path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phase", [PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN])
def test_a_clean_phase_writes_only_clean_rows(
    etth2_present, tmp_path, etth2_config, config, gap_config, phase
):
    summary = run_phase(
        phase=phase, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, mock=True, limit_origins=3,
        reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
    )
    assert summary["phase"] == phase
    assert summary["mock_model"] is True
    assert summary["dataset_facts"]["native_missing_in_target"] == 0
    run_dir = tmp_path / ("ref" if phase == PHASE_CLEAN_REFERENCE else "formal")
    frame = pd.read_csv(run_dir / "window_results.csv")
    assert sorted(set(frame["kind"])) == ["clean"]
    assert len(frame) == 3
    assert set(frame["dataset_sha256"]) == {SHA}


def test_two_independent_clean_passes_agree_exactly_and_the_gate_is_written(
    etth2_present, tmp_path, etth2_config, config, gap_config
):
    """The two-pass control end to end, on the mock forecaster.

    The mock is deterministic, so two independent passes MUST agree; if they did
    not, the audit would be the thing that caught it, which is the point.
    """
    reference, formal = tmp_path / "ref", tmp_path / "formal"
    # All 178 origins, because the audit's cardinality check is part of what is
    # being exercised. The origin-count guard refuses a shrunken config, which is
    # itself the guard working.
    for phase in (PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN):
        run_phase(
            phase=phase, etth2_config=etth2_config, pilot_config=config,
            gap_config=gap_config, mock=True,
            reference_dir=reference, formal_dir=formal,
        )
    result = run_phase(
        phase=PHASE_AUDIT, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, reference_dir=reference, formal_dir=formal,
    )
    assert result["expected_origins"] == 178
    assert result["passed"] is True
    assert result["formal_canonical_prediction_sha256"] == (
        result["reference_canonical_prediction_sha256"]
    )
    gate = json.loads((formal / GATE_FILENAME).read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["dataset_sha256"] == SHA
    assert gate["tolerance"] == "none"
    # The gate records the revision that ACTUALLY ran. These passes used the
    # mock forecaster, so it records the mock's revision — which is precisely
    # why a mock-produced gate can never authorise a real GPU run.
    assert gate["model_revision"] == "mock-revision"
    assert gate["model_revision"] != REVISION


def test_a_divergent_clean_pass_fails_the_audit_and_writes_no_gate(
    etth2_present, tmp_path, etth2_config, config, gap_config
):
    """Non-vacuity for the whole control: perturb one prediction, expect a stop."""
    from experiments.clean_reference_audit import CleanReferenceAuditFailure

    reference, formal = tmp_path / "ref", tmp_path / "formal"
    for phase in (PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN):
        run_phase(
            phase=phase, etth2_config=etth2_config, pilot_config=config,
            gap_config=gap_config, mock=True,
            reference_dir=reference, formal_dir=formal,
        )
    frame = pd.read_csv(formal / "predictions_long.csv")
    frame.loc[0, "median_prediction"] = float(frame.loc[0, "median_prediction"]) + 1e-3
    frame.to_csv(formal / "predictions_long.csv", index=False)

    with pytest.raises(CleanReferenceAuditFailure, match="HARD STOP"):
        run_phase(
            phase=PHASE_AUDIT, etth2_config=etth2_config, pilot_config=config,
            gap_config=gap_config, reference_dir=reference, formal_dir=formal,
        )
    assert not (formal / GATE_FILENAME).exists(), "a gate was written for a failed audit"


def test_the_runner_delegates_to_the_etth1_execution_path(etth2_present):
    """The replication must run the SAME loop that produced the ETTh1 numbers."""
    import experiments.run_etth2 as mod
    from experiments.run_trailing_gap import run_trailing_gap

    assert mod.run_trailing_gap is run_trailing_gap
    source = (REPO_ROOT / "experiments" / "run_etth2.py").read_text(encoding="utf-8")
    # It orchestrates; it does not re-implement the per-origin loop.
    assert "forecast_median" not in source
    assert "compute_metrics" not in source


# --------------------------------------------------------------------------- #
# Manifest and summary provenance — asserted on the FILES, not returned dicts
# --------------------------------------------------------------------------- #

def test_run_metadata_is_explicitly_overridden_not_inherited(etth2_config):
    """ETTh1's experiment identity must not default through into ETTh2."""
    from experiments.run_etth2 import build_run_metadata

    gap = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )
    metadata = build_run_metadata(etth2_config, phase=PHASE_FORMAL_CLEAN, mock=True)
    assert metadata["experiment"] == "etth2-trailing-gap-replication-v1"
    assert metadata["experiment"] != gap["meta"]["name"]
    assert metadata["preregistration"] == "docs/preregistration_etth2_replication_v1.md"
    assert metadata["preregistration"] != gap["meta"]["preregistration"]


def test_the_shared_runner_refuses_metadata_that_would_overwrite_a_manifest_key(
    etth2_present, tmp_path, config
):
    """Only experiment and preregistration are overridable; a collision is a bug."""
    from experiments.run_trailing_gap import run_trailing_gap
    from model.chronos2_runner import MockForecaster

    gap = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="may not overwrite manifest key"):
        run_trailing_gap(
            config=config, gap_config=gap, forecaster=MockForecaster(),
            run_dir=tmp_path / "x", limit_origins=1,
            run_metadata={"model_contract": {"revision": "spoofed"}},
        )


@pytest.mark.parametrize("phase", [PHASE_CLEAN_REFERENCE, PHASE_FORMAL_CLEAN])
def test_the_manifest_file_on_disk_carries_full_etth2_provenance(
    etth2_present, tmp_path, etth2_config, config, gap_config, phase
):
    """Read the FILE. An in-memory dict proves nothing about what was persisted."""
    run_phase(
        phase=phase, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, mock=True, limit_origins=2,
        reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
    )
    run_dir = tmp_path / ("ref" if phase == PHASE_CLEAN_REFERENCE else "formal")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest["experiment"] == "etth2-trailing-gap-replication-v1"
    assert manifest["preregistration"] == "docs/preregistration_etth2_replication_v1.md"
    assert manifest["replication_rule_version"] == "etth2-replication-v1"
    assert manifest["design_signoff_commit"] == "5b62949"
    assert manifest["execution_entrypoint"] == "experiments/run_etth2.py"
    assert manifest["phase"] == phase
    assert manifest["mock_model"] is True
    assert "NOT an ETTh2 finding" in manifest["mock_model_note"]
    # Real git state at run time, not an assumption.
    assert len(manifest["git_commit"]) == 40
    assert isinstance(manifest["git_dirty"], bool)
    assert isinstance(manifest["git_status_porcelain"], list)
    # The run_id derives from ETTh2's experiment, not ETTh1's.
    assert manifest["run_id"].startswith("etth2-trailing-gap-replication-v1")
    assert "trailing-gap-mechanism-v1" not in manifest["run_id"]


def test_the_persisted_run_summary_on_disk_carries_the_enrichment(
    etth2_present, tmp_path, etth2_config, config, gap_config
):
    """The wrapper enriches after the runner returns; the FILE must show it."""
    returned = run_phase(
        phase=PHASE_CLEAN_REFERENCE, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, mock=True, limit_origins=2,
        reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
    )
    on_disk = json.loads(
        (tmp_path / "ref" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert on_disk["phase"] == PHASE_CLEAN_REFERENCE
    assert on_disk["mock_model"] is True
    assert on_disk["scientifically_valid"] is False
    assert on_disk["authoritative_result"] is False
    assert "NOT an ETTh2 finding" in on_disk["mock_model_note"]
    assert on_disk["experiment"] == "etth2-trailing-gap-replication-v1"
    assert on_disk["design_signoff_commit"] == "5b62949"
    assert on_disk["execution_entrypoint"] == "experiments/run_etth2.py"
    assert on_disk["dataset_facts"]["native_missing_in_target"] == 0
    assert len(on_disk["git_commit"]) == 40
    # The file must agree with what the function returned.
    for key in ("phase", "mock_model", "experiment", "git_commit"):
        assert on_disk[key] == returned[key], key


def test_the_summary_is_written_atomically_leaving_no_temp_file(
    etth2_present, tmp_path, etth2_config, config, gap_config
):
    run_phase(
        phase=PHASE_CLEAN_REFERENCE, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, mock=True, limit_origins=2,
        reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
    )
    assert (tmp_path / "ref" / "run_summary.json").exists()
    assert not list((tmp_path / "ref").glob("*.tmp")), "a temp file survived"


def test_the_porcelain_first_line_is_not_mangled():
    """Regression: stripping the whole porcelain output ate the first path's
    leading character, because a porcelain line's first two chars are status."""
    from experiments.run_etth2 import git_provenance

    provenance = git_provenance()
    if not provenance["git_dirty"]:
        pytest.skip("clean tree; nothing to check")
    for line in provenance["git_status_porcelain"]:
        assert len(line) > 3
        assert line[2] == " ", repr(line)
        path = line[3:]
        assert not path.startswith("/"), repr(line)
        # A mangled path would have lost its first character.
        assert (REPO_ROOT / path).exists() or path.endswith("/"), repr(line)
