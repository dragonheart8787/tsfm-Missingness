"""Test 9 — evaluation origins are isolated; no cross-window leakage.

ADVERSARIAL: we deliberately corrupt one window's context and confirm that
every neighbouring window's forecast is BIT-IDENTICAL to a run without that
corruption. If Chronos-2's cross-series/grouped inference were leaking between
windows, this test fails.

Model-mocked for the CPU build (the mock genuinely reads only the context it is
handed, so it exercises the WIRING — that we submit one origin per call and
never batch windows together). ``test_9d`` repeats the identical adversarial
procedure against the real checkpoint and is marked ``requires_model``.
"""

from __future__ import annotations

import numpy as np
import pytest

from masks.base import apply_mask
from masks.plan import build_conditions
from model.chronos2_runner import MockForecaster
from runner.windows import build_windows


@pytest.fixture(scope="module")
def windows(synthetic_series, config):
    values, stamps = synthetic_series
    return build_windows(values=values, timestamps=stamps, config=config)


def _forecast_all(forecaster, contexts, horizon):
    return [forecaster.forecast_median(c, horizon) for c in contexts]


def test_9_corrupting_one_window_leaves_neighbours_bit_identical(windows, config, patch_size):
    """The adversarial leakage probe."""
    horizon = config["design"]["horizon"]
    selected = windows[:9]
    contexts = [np.array(w.context, copy=True) for w in selected]

    baseline = _forecast_all(MockForecaster(), contexts, horizon)

    victim = 4
    corrupted = [np.array(c, copy=True) for c in contexts]
    # A violent, obviously-detectable corruption of a single window.
    corrupted[victim] = corrupted[victim] * -1000.0 + 12345.0
    after = _forecast_all(MockForecaster(), corrupted, horizon)

    for i in range(len(selected)):
        if i == victim:
            assert not np.array_equal(baseline[i], after[i]), (
                "the corrupted window's own forecast did not change; the probe is not "
                "actually exercising anything"
            )
            continue
        assert np.array_equal(baseline[i], after[i]), (
            f"LEAKAGE: window {i}'s forecast changed when window {victim} was corrupted"
        )
        assert baseline[i].tobytes() == after[i].tobytes()


def test_9b_corruption_via_nan_also_does_not_leak(windows, config):
    """A NaN-flooded neighbour must not perturb anyone else either."""
    horizon = config["design"]["horizon"]
    selected = windows[:7]
    contexts = [np.array(w.context, copy=True) for w in selected]
    baseline = _forecast_all(MockForecaster(), contexts, horizon)

    victim = 2
    corrupted = [np.array(c, copy=True) for c in contexts]
    corrupted[victim][:] = np.nan
    after = _forecast_all(MockForecaster(), corrupted, horizon)

    for i in range(len(selected)):
        if i == victim:
            continue
        assert np.array_equal(baseline[i], after[i]), f"LEAKAGE into window {i}"


def test_9c_each_call_receives_exactly_one_window(windows, config, patch_size):
    """Wiring check: one origin per call, and the call sees only that context."""
    horizon = config["design"]["horizon"]
    forecaster = MockForecaster()
    window = windows[3]
    conditions = build_conditions(config=config, origin_id=window.origin_id, patch_size=patch_size)
    for condition in conditions:
        masked = apply_mask(window.context, condition.mask)
        forecaster.forecast_median(masked, horizon)

    assert len(forecaster.calls) == len(conditions)
    for recorded, condition in zip(forecaster.calls, conditions):
        assert recorded.shape == (config["design"]["context_length"],), (
            "a call received something other than exactly one context window"
        )
        assert np.array_equal(
            np.flatnonzero(np.isnan(recorded)), condition.mask.missing_indices
        )


def test_9d_config_forbids_cross_learning(config):
    """Chronos-2's cross_learning would share information across tasks in a call."""
    assert config["model"]["cross_learning"] is False
    assert int(config["model"]["forecasts_per_call"]) == 1


@pytest.mark.requires_model
def test_9e_real_model_isolation(config):
    """The same adversarial probe against the real checkpoint. Needs weights."""
    from model.chronos2_runner import Chronos2Forecaster

    horizon = config["design"]["horizon"]
    context_length = config["design"]["context_length"]
    rng = np.random.default_rng(0)
    contexts = [
        (10 + 5 * np.sin(np.arange(context_length) / 12.0 + k) + rng.normal(scale=0.3, size=context_length)).astype(np.float32)
        for k in range(5)
    ]
    forecaster = Chronos2Forecaster(config=config)
    baseline = _forecast_all(forecaster, contexts, horizon)

    victim = 2
    corrupted = [np.array(c, copy=True) for c in contexts]
    corrupted[victim] = corrupted[victim] * -1000.0 + 12345.0
    after = _forecast_all(forecaster, corrupted, horizon)

    for i in range(len(contexts)):
        if i == victim:
            assert not np.array_equal(baseline[i], after[i])
            continue
        assert np.array_equal(baseline[i], after[i]), (
            f"LEAKAGE: real-model window {i} changed when window {victim} was corrupted"
        )
