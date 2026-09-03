"""`limit_prediction_length=True` must reach the REAL pipeline call.

The config declared it and the docs claimed it, but ``forecast_median`` did not
pass it — so Chronos-2's default (``False``) applied, under which an over-long
``prediction_length`` only warns and then silently unrolls autoregressively.

A config-only or arithmetic-only check cannot catch that class of bug. These
tests exercise the actual call chain:

    Chronos2Forecaster.forecast_median
        -> Chronos2Pipeline.predict_quantiles   (the REAL method)
            -> pipeline.predict                 (spied, kwargs recorded)

Model-mocked: the spy stands in for the weights, but every line of the wrapper
and of the library's own ``predict_quantiles`` is really executed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chronos.chronos2.pipeline import Chronos2Pipeline
from model.chronos2_runner import Chronos2Forecaster

TRAINING_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class SpyPipeline:
    """A stand-in carrying the REAL predict_quantiles over a spied predict."""

    def __init__(self):
        self.predict_calls: list[dict] = []

    @property
    def quantiles(self) -> list[float]:
        return TRAINING_QUANTILES

    def predict(self, inputs, prediction_length=None, **kwargs):
        # Record exactly what the library forwarded to predict.
        self.predict_calls.append(
            {"prediction_length": prediction_length, "n_inputs": len(inputs), **kwargs}
        )
        # (n_variates, n_quantiles, horizon), as the real predict returns.
        return [torch.zeros(1, len(TRAINING_QUANTILES), prediction_length)]

    # The real method, bound to this object.
    def predict_quantiles(self, *args, **kwargs):
        return Chronos2Pipeline.predict_quantiles(self, *args, **kwargs)


def _forecaster(spy: SpyPipeline, *, limit: bool = True) -> Chronos2Forecaster:
    """Build the wrapper without loading weights, then run its real method."""
    forecaster = object.__new__(Chronos2Forecaster)
    forecaster._pipeline = spy
    forecaster._torch = torch
    forecaster._device = "cpu"
    forecaster._cross_learning = False
    forecaster._quantile_level = 0.5
    forecaster._limit_prediction_length = limit
    return forecaster


def test_limit_prediction_length_reaches_the_pipeline_predict_call():
    """THE test for item 4: the kwarg arrives at predict, not just at config."""
    spy = SpyPipeline()
    forecaster = _forecaster(spy)

    out = forecaster.forecast_median(np.arange(320, dtype=np.float32), 96)

    assert out.shape == (96,)
    assert len(spy.predict_calls) == 1
    call = spy.predict_calls[0]
    assert "limit_prediction_length" in call, (
        "limit_prediction_length never reached Chronos2Pipeline.predict — the library's "
        "default of False would apply and an over-long horizon would silently unroll"
    )
    assert call["limit_prediction_length"] is True


def test_the_other_isolation_kwargs_also_reach_predict():
    """Same call path; guards against a fix that drops a neighbour."""
    spy = SpyPipeline()
    _forecaster(spy).forecast_median(np.arange(320, dtype=np.float32), 96)
    call = spy.predict_calls[0]
    assert call["cross_learning"] is False
    assert call["batch_size"] == 1
    assert call["n_inputs"] == 1, "exactly one origin per call"
    assert call["prediction_length"] == 96


def test_the_forwarded_value_follows_the_config_flag():
    """Not hardcoded True: it is the configured value that travels."""
    spy = SpyPipeline()
    _forecaster(spy, limit=False).forecast_median(np.arange(320, dtype=np.float32), 96)
    assert spy.predict_calls[0]["limit_prediction_length"] is False


def test_the_spy_would_have_caught_the_original_bug():
    """Guard against a vacuous test: the pre-fix call really omitted the kwarg."""

    class PreFixForecaster(Chronos2Forecaster):
        def forecast_median(self, context, horizon):
            # The call exactly as it stood before this round's fix.
            quantiles, _mean = self._pipeline.predict_quantiles(
                [torch.as_tensor(np.asarray(context, dtype=np.float32))],
                prediction_length=horizon,
                quantile_levels=[self._quantile_level],
                cross_learning=self._cross_learning,
                batch_size=1,
            )
            return quantiles[0][..., 0].reshape(-1).numpy()

    spy = SpyPipeline()
    broken = object.__new__(PreFixForecaster)
    broken._pipeline = spy
    broken._torch = torch
    broken._quantile_level = 0.5
    broken._cross_learning = False
    broken.forecast_median(np.arange(320, dtype=np.float32), 96)

    assert "limit_prediction_length" not in spy.predict_calls[0], (
        "the pre-fix call should NOT carry the kwarg; if it does, this test proves nothing"
    )


def test_config_declares_the_flag(config):
    """Necessary but NOT sufficient — the tests above are what actually gate it."""
    assert config["model"]["limit_prediction_length"] is True


def test_default_is_safe_when_the_key_is_absent():
    """A config missing the key must not silently fall back to the unsafe default."""
    import inspect

    source = inspect.getsource(Chronos2Forecaster.__init__)
    assert 'model_cfg.get("limit_prediction_length", True)' in source, (
        "the default must be True; the library's own default of False is the unsafe one"
    )
