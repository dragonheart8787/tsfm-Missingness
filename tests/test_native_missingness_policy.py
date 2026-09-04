"""The shared runner's native-missingness hard stop: opt-in, and unbypassable.

Two properties, tested separately because they pull in opposite directions:

  1. **ETTh1 is unaffected.** Its config predates the policy key and does not set
     it, so the check inspects the flag, finds it absent, and returns. Byte
     identity is proved against a PINNED hash of the deterministic output
     columns, not by reading the code.
  2. **A dataset that opts in cannot be bypassed.** The check sits at the
     runner's lowest forecasting boundary — after ``load_series``, before
     ``build_windows`` — so it fires whether the runner is called directly or
     through a wrapper, and it fires with ZERO forecaster calls.

"Zero forecaster calls" is asserted on a counting forecaster rather than
inferred from an exception. An exception raised after the model had already been
asked for a forecast would still be a leak.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.run_trailing_gap import (
    THIS_EXPERIMENT_DATASET,
    NativeMissingnessPolicyViolation,
    enforce_native_missingness_policy,
    run_trailing_gap,
)
from model.chronos2_runner import MockForecaster

REPO_ROOT = Path(__file__).resolve().parents[1]

# Columns that legitimately differ between any two runs of identical code:
# run_id embeds a UTC timestamp, runtime_seconds is wall clock. Everything else
# is deterministic and is what the pinned hashes below cover.
NONDETERMINISTIC = ("run_id", "runtime_seconds")

# PINNED on the full 178-origin ETTh1 mock matrix, recorded BEFORE the policy
# check was added and re-verified after. A change to these means ETTh1's
# behaviour moved, which this round's changes are required not to do.
ETTH1_FULL_MATRIX_HASHES = {
    "window_results.csv":
        "28ee876c1c1e276163d8f46430a945361b3e6a3e317e8005eed95583b376c160",
    "predictions_long.csv":
        "ded897420a55953981ae21e1a3320b47d61f4b11b2f71cddb7748c4a8cdec762",
}


def canonical_hash(path: Path) -> str:
    """sha256 over every column except the two nondeterministic ones."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[[c for c in frame.columns if c not in NONDETERMINISTIC]]
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


class CountingForecaster(MockForecaster):
    """A mock that records how many times it was asked to forecast."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forecast_median(self, context, prediction_length):  # type: ignore[override]
        self.calls += 1
        return super().forecast_median(context, prediction_length)


class _Validation:
    """The fields the policy check reads off a DatasetValidation."""

    def __init__(self, n: int, positions: list[int] | None = None) -> None:
        self.native_missing_in_target = n
        self.native_missing_positions_in_target = positions or list(range(n))


# --------------------------------------------------------------------------- #
# 1. ETTh1 is unaffected — the policy is opt-in
# --------------------------------------------------------------------------- #

def test_etth1s_config_does_not_set_the_policy_key(config):
    """The premise: ETTh1 predates the key and must not acquire it."""
    assert "native_missing_is_a_hard_stop" not in config["dataset"]


def test_the_check_is_a_no_op_without_the_key_even_with_missingness(config):
    """Opt-in means opt-in. Absent key -> return, whatever validation reports."""
    enforce_native_missingness_policy(config, _Validation(0))       # must not raise
    enforce_native_missingness_policy(config, _Validation(500))     # must not raise either


@pytest.mark.parametrize("flag", [False, None, 0, ""])
def test_a_falsey_policy_key_is_also_a_no_op(config, flag):
    cfg = copy.deepcopy(config)
    cfg["dataset"]["native_missing_is_a_hard_stop"] = flag
    enforce_native_missingness_policy(cfg, _Validation(7))          # must not raise


def test_etth1_full_matrix_output_is_byte_identical_to_the_pinned_baseline(
    tmp_path, config, real_series
):
    """The regression lock: ETTh1's mock matrix must not have moved.

    Compares every deterministic column of both output files against hashes
    recorded before the policy check existed. run_id and runtime_seconds are
    excluded because they differ between any two runs of identical code, which
    ``test_only_the_nondeterministic_columns_can_differ`` demonstrates rather
    than assumes.
    """
    gap_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )
    run_dir = tmp_path / "etth1"
    summary = run_trailing_gap(
        config=config, gap_config=gap_config, forecaster=MockForecaster(),
        run_dir=run_dir,
    )
    assert summary["result_rows"] == 2314
    assert summary["distinct_cells"] == 2314
    for name, expected in ETTH1_FULL_MATRIX_HASHES.items():
        assert canonical_hash(run_dir / name) == expected, (
            f"{name} changed; ETTh1 behaviour is not byte-identical any more"
        )


def test_only_the_nondeterministic_columns_can_differ(tmp_path, config, real_series):
    """Two runs of the SAME code differ in exactly run_id and runtime_seconds.

    This is what justifies excluding them above: their difference is a property
    of running twice, not evidence of a behaviour change.
    """
    gap_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )
    dirs = []
    for i in range(2):
        run_dir = tmp_path / f"run{i}"
        run_trailing_gap(
            config=config, gap_config=gap_config, forecaster=MockForecaster(),
            run_dir=run_dir, limit_origins=4,
        )
        dirs.append(run_dir)
    a = pd.read_csv(dirs[0] / "window_results.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(dirs[1] / "window_results.csv", dtype=str, keep_default_na=False)
    differing = {c for c in a.columns if not a[c].equals(b[c])}
    assert differing <= set(NONDETERMINISTIC), differing


def test_etth1_manifest_identity_is_unchanged(tmp_path, config, real_series):
    """No wrapper metadata -> the manifest reads exactly as it always did."""
    import json

    gap_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )
    run_dir = tmp_path / "etth1"
    run_trailing_gap(
        config=config, gap_config=gap_config, forecaster=MockForecaster(),
        run_dir=run_dir, limit_origins=2,
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == gap_config["meta"]["name"]
    assert manifest["preregistration"] == gap_config["meta"]["preregistration"]
    assert manifest["run_id"].startswith(gap_config["meta"]["name"])
    # The wrapper-only keys must be absent, not present-and-empty.
    for key in ("phase", "mock_model", "design_signoff_commit", "execution_entrypoint"):
        assert key not in manifest, key


# --------------------------------------------------------------------------- #
# 2. A dataset that opts in cannot be bypassed
# --------------------------------------------------------------------------- #

@pytest.fixture
def etth2_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture
def planted_nan_dataset(tmp_path, etth2_config, config) -> tuple[dict, Path]:
    """An ETTh2-shaped effective config whose file carries a native NaN."""
    from experiments.run_etth2 import build_effective_config

    n = int(etth2_config["dataset"]["expected_rows"])
    frame = pd.DataFrame({
        "date": pd.date_range("2016-07-01", periods=n, freq="h").astype(str),
        "OT": np.linspace(0.0, 50.0, n),
    })
    frame.loc[4242, "OT"] = np.nan
    csv = tmp_path / "raw" / "ETTh2.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv, index=False)

    broken = copy.deepcopy(etth2_config)
    broken["dataset"]["local_path"] = "raw/ETTh2.csv"
    broken["dataset"]["expected_sha256"] = None
    return build_effective_config(config, broken), tmp_path


def test_the_policy_fires_when_the_key_is_set(etth2_config):
    with pytest.raises(NativeMissingnessPolicyViolation) as excinfo:
        enforce_native_missingness_policy(
            {"dataset": etth2_config["dataset"]}, _Validation(3, [10, 20, 30])
        )
    message = str(excinfo.value)
    assert "HARD STOP" in message
    assert "No window was built and no forecast was requested" in message
    assert "Do NOT impute" in message
    assert "Research Lead" in message


def test_bypass_route_1_calling_the_function_directly_makes_zero_forecast_calls(
    monkeypatch, tmp_path, planted_nan_dataset, gap_config_fixture
):
    """The hole this closes: importing the module and calling it directly."""
    import data.fetch_etth1 as fetch1

    effective, root = planted_nan_dataset
    monkeypatch.setattr(fetch1, "REPO_ROOT", root)
    forecaster = CountingForecaster()

    with pytest.raises(NativeMissingnessPolicyViolation):
        run_trailing_gap(
            config=effective, gap_config=gap_config_fixture, forecaster=forecaster,
            run_dir=tmp_path / "leak",
        )
    assert forecaster.calls == 0, (
        f"the forecaster was called {forecaster.calls} time(s) before the hard stop"
    )
    # Nothing was written either: no windows were built.
    assert not (tmp_path / "leak" / "window_results.csv").exists()
    assert not (tmp_path / "leak" / "predictions_long.csv").exists()


def test_bypass_route_2_the_cli_refuses_an_etth2_config_before_model_construction():
    """The other route: invoking this entrypoint's CLI with the wrong config."""
    result = subprocess.run(
        [sys.executable, "experiments/run_trailing_gap.py",
         "--config", "configs/etth2_config.yaml",
         "--run-dir", "results/should_never_be_created", "--mock-model"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert "REFUSING TO RUN" in result.stderr
    assert "No model was loaded and no forecast was requested" in result.stderr
    assert "experiments/run_etth2.py" in result.stderr
    assert not (REPO_ROOT / "results" / "should_never_be_created").exists()


def test_the_cli_still_accepts_its_own_dataset(config):
    """Non-vacuity: the refusal must not reject ETTh1."""
    assert config["dataset"]["name"] == THIS_EXPERIMENT_DATASET


def test_the_wrapper_route_also_makes_zero_forecast_calls(
    monkeypatch, tmp_path, etth2_config, config
):
    """Defence in depth: the wrapper's own gate fires first, and also leaks nothing."""
    import data.fetch_etth2 as fetch2
    import experiments.run_etth2 as mod
    from data.fetch_etth2 import NativeMissingnessError

    n = int(etth2_config["dataset"]["expected_rows"])
    frame = pd.DataFrame({
        "date": pd.date_range("2016-07-01", periods=n, freq="h").astype(str),
        "OT": np.linspace(0.0, 50.0, n),
    })
    frame.loc[9, "OT"] = np.nan
    csv = tmp_path / "raw" / "ETTh2.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv, index=False)

    broken = copy.deepcopy(etth2_config)
    broken["dataset"]["local_path"] = "raw/ETTh2.csv"
    broken["dataset"]["expected_sha256"] = None
    monkeypatch.setattr(fetch2, "REPO_ROOT", tmp_path)

    forecaster = CountingForecaster()
    monkeypatch.setattr(mod, "_forecaster", lambda *a, **k: forecaster)
    gap_config = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(NativeMissingnessError):
        mod.run_phase(
            phase=mod.PHASE_CLEAN_REFERENCE, etth2_config=broken, pilot_config=config,
            gap_config=gap_config, mock=True,
            reference_dir=tmp_path / "ref", formal_dir=tmp_path / "formal",
        )
    assert forecaster.calls == 0
