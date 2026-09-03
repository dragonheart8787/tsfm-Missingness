"""Model contract verification and q=0.5 discipline.

The offline tests here are model-mocked. ``requires_model`` tests need weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.chronos2_runner import ModelContract, verify_contract


def _contract(**overrides) -> ModelContract:
    base = dict(
        hf_model_id="amazon/chronos-2",
        revision="deadbeef",
        input_patch_size=16,
        input_patch_stride=16,
        output_patch_size=16,
        model_context_length=8192,
        # Matches configs/pilot_config.yaml model.verified_contract, and is
        # self-consistent: 64 * 16 == 1024, as the real pipeline computes it.
        max_output_patches=64,
        model_prediction_length=64 * 16,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        use_arcsinh=False,
        device="cpu",
        dtype="float32",
    )
    base.update(overrides)
    return ModelContract(**base)


def test_contract_passes_on_the_preregistered_grid(config):
    assert verify_contract(_contract(), config) == []


def test_contract_hard_fails_on_a_patch_size_mismatch(config):
    violations = verify_contract(_contract(input_patch_size=32), config)
    assert any("PATCH GRID MISMATCH" in v for v in violations)
    assert any("recomputed" in v for v in violations)


def test_contract_hard_fails_on_a_patch_stride_mismatch(config):
    violations = verify_contract(_contract(input_patch_stride=8), config)
    assert any("PATCH STRIDE MISMATCH" in v for v in violations)


def test_contract_hard_fails_when_q05_is_absent(config):
    violations = verify_contract(_contract(quantiles=[0.1, 0.9]), config)
    assert any("0.5" in v for v in violations)


def test_contract_flags_a_context_that_would_be_left_padded(config):
    """L must be a whole number of patches, else Chronos-2 prepends NaN padding
    and every block position shifts relative to the patch grid."""
    violations = verify_contract(_contract(input_patch_size=48), config)
    assert any("LEFT-PAD" in v for v in violations)


def test_contract_flags_a_truncated_context(config):
    violations = verify_contract(_contract(model_context_length=128), config)
    assert any("truncated" in v for v in violations)


def test_contract_flags_a_horizon_needing_autoregressive_unrolling(config):
    violations = verify_contract(_contract(model_prediction_length=48), config)
    assert any("unrolling" in v for v in violations)


def test_runner_refuses_to_start_on_a_contract_violation(config, tmp_path, synthetic_series):
    """The hard stop is wired into the runner, not only available as a helper."""
    import pandas as pd

    from model.chronos2_runner import MockForecaster
    from runner.run_pilot import PatchGridMismatch, run_pilot

    values, stamps = synthetic_series
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame({"date": stamps, "OT": values}).to_csv(csv, index=False)
    from data.fetch_etth1 import sha256_file

    cfg = json.loads(json.dumps(config))
    cfg["dataset"]["local_path"] = str(csv)
    cfg["dataset"]["expected_sha256"] = sha256_file(csv)

    forecaster = MockForecaster(contract=_contract(input_patch_size=32))
    with pytest.raises(PatchGridMismatch, match="PATCH GRID MISMATCH"):
        run_pilot(config=cfg, forecaster=forecaster, run_dir=tmp_path / "r", limit_origins=1)


def test_config_pins_float32_and_eval_only(config):
    assert config["model"]["dtype"] == "float32"
    assert config["model"]["quantile_level"] == 0.5
    assert config["model"]["missing_value_convention"] == "nan_passthrough"


def test_runner_refuses_an_unpinned_revision(config):
    """A floating tag is not acceptable provenance."""
    from model.chronos2_runner import Chronos2Forecaster

    cfg = json.loads(json.dumps(config))
    cfg["model"]["revision"] = None
    with pytest.raises(ValueError, match="not pinned"):
        Chronos2Forecaster(config=cfg)


def test_inference_wrapper_requests_q05_explicitly():
    """We must not accept predict_quantiles' convenience second return value.

    chronos-forecasting names it `mean` while in fact returning the q=0.5
    column; relying on that name would be relying on an implementation detail.
    """
    source = Path("model/chronos2_runner.py").read_text(encoding="utf-8")
    assert "quantile_levels=[self._quantile_level]" in source
    assert "_mean" in source, "the convenience return must be explicitly discarded"
    assert "cross_learning=self._cross_learning" in source


@pytest.mark.requires_model
def test_real_checkpoint_matches_the_preregistered_contract(config):
    """The live gate. Equivalent to scripts/verify_model_contract.py."""
    from model.chronos2_runner import Chronos2Forecaster

    forecaster = Chronos2Forecaster(config=config)
    assert verify_contract(forecaster.contract, config) == []
    assert forecaster.contract.input_patch_size == config["model"]["expected_patch_size"]


@pytest.mark.requires_model
def test_real_model_accepts_nan_context_and_returns_finite_forecasts(config):
    import numpy as np

    from model.chronos2_runner import Chronos2Forecaster

    forecaster = Chronos2Forecaster(config=config)
    context = (20 + 5 * np.sin(np.arange(320) / 12.0)).astype(np.float32)
    holed = context.copy()
    holed[256:] = np.nan
    out = forecaster.forecast_median(holed, 96)
    assert out.shape == (96,)
    assert np.all(np.isfinite(out))
