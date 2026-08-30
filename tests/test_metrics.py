"""Test 12 — metrics match hand-calculated toy examples.

Model-mocked: no weights required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics.aggregate import aggregate_mae, aggregate_rmse, relative_error_degradation
from metrics.pointwise import compute_metrics, seasonal_naive_denominator


def test_12_mae_mse_rmse_against_hand_calculation():
    """errors = [+1, -2, +3, 0]  ->  MAE = 6/4 = 1.5, MSE = 14/4 = 3.5."""
    truth = np.array([10.0, 20.0, 30.0, 40.0])
    prediction = np.array([11.0, 18.0, 33.0, 40.0])
    m = compute_metrics(truth=truth, prediction=prediction)
    assert m.mae == pytest.approx(1.5)
    assert m.mse == pytest.approx(3.5)
    assert m.rmse == pytest.approx(math.sqrt(3.5))
    assert m.n_steps == 4


def test_12b_perfect_forecast_is_exactly_zero_error():
    truth = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(truth=truth, prediction=truth.copy())
    assert m.mae == 0.0 and m.mse == 0.0 and m.rmse == 0.0


def test_12c_constant_offset_forecast():
    """A uniform +2 offset gives MAE 2, MSE 4, RMSE 2."""
    truth = np.arange(96, dtype=np.float64)
    m = compute_metrics(truth=truth, prediction=truth + 2.0)
    assert m.mae == pytest.approx(2.0)
    assert m.mse == pytest.approx(4.0)
    assert m.rmse == pytest.approx(2.0)


def test_12d_rmse_aggregates_through_mse_not_through_rmse():
    """The reason MSE is retained: sqrt(mean(MSE)) != mean(RMSE)."""
    per_origin_mse = np.array([1.0, 9.0])       # RMSEs are 1 and 3
    correct = aggregate_rmse(per_origin_mse)
    naive_mean_of_rmse = np.mean(np.sqrt(per_origin_mse))
    assert correct == pytest.approx(math.sqrt(5.0))   # sqrt((1+9)/2)
    assert naive_mean_of_rmse == pytest.approx(2.0)
    assert correct != pytest.approx(naive_mean_of_rmse)


def test_12e_red_uses_aggregate_errors_not_a_mean_of_ratios():
    """RED from aggregates is stable where a mean of per-window ratios explodes."""
    clean = np.array([1.0, 1.0, 1e-9])          # one near-zero clean error
    missing = np.array([1.5, 1.5, 2e-9])

    red = relative_error_degradation(aggregate_mae(missing), aggregate_mae(clean))
    assert red == pytest.approx(0.5, rel=1e-6)

    # The forbidden alternative: a mean of per-window ratios. Here it happens to
    # agree, so make the divergence explicit with a window whose clean error is
    # near zero but whose absolute degradation is tiny.
    clean2 = np.array([1.0, 1.0, 1e-9])
    missing2 = np.array([1.0, 1.0, 1e-3])
    red2 = relative_error_degradation(aggregate_mae(missing2), aggregate_mae(clean2))
    mean_of_ratios = float(np.mean(missing2 / clean2))
    assert red2 < 0.001
    assert mean_of_ratios > 100_000, "the per-window-ratio pathology must be real"


def test_12f_red_sign_and_zero_cases():
    assert relative_error_degradation(1.5, 1.0) == pytest.approx(0.5)
    assert relative_error_degradation(0.5, 1.0) == pytest.approx(-0.5)
    assert relative_error_degradation(1.0, 1.0) == 0.0
    assert math.isnan(relative_error_degradation(1.0, 0.0))


def test_12g_mase_denominator_from_the_clean_context_only():
    """Seasonal-naive denominator, hand-computed on a period-4 saw."""
    context = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    # |x[t] - x[t-4]| = |3-1|,|4-2|,|5-3|,|6-4| = 2,2,2,2 -> mean 2.0
    denominator = seasonal_naive_denominator(context, seasonal_period=4, min_denominator=1e-8)
    assert denominator == pytest.approx(2.0)

    truth = np.array([10.0, 20.0])
    m = compute_metrics(truth=truth, prediction=truth + 1.0, mase_denominator=denominator)
    assert m.mae == pytest.approx(1.0)
    assert m.mase == pytest.approx(0.5)


def test_12h_mase_zero_denominator_is_guarded_not_infinite():
    """A flat seasonal profile yields a null MASE, never inf or a crash."""
    flat = np.tile([1.0, 2.0, 3.0, 4.0], 5).astype(np.float64)
    assert seasonal_naive_denominator(flat, seasonal_period=4, min_denominator=1e-8) is None
    short = np.arange(3, dtype=np.float64)
    assert seasonal_naive_denominator(short, seasonal_period=4, min_denominator=1e-8) is None
    m = compute_metrics(truth=np.array([1.0]), prediction=np.array([2.0]), mase_denominator=None)
    assert m.mase is None


def test_12i_mase_is_excluded_from_the_decision_rule(config):
    """The decision function must not consume MASE anywhere."""
    from pathlib import Path

    source = Path("stats/decision.py").read_text().lower()
    assert "mase" not in source
    fields = set(__import__("stats.decision", fromlist=["ContrastStat"]).ContrastStat.__dataclass_fields__)
    assert not any("mase" in f for f in fields)


def test_12j_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(truth=np.zeros(96), prediction=np.zeros(95))
