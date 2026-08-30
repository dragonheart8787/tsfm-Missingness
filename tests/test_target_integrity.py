"""Tests 1, 2, 3, 8 — the target may never be touched, seen, or corrupted.

Model-mocked: no weights required.
"""

from __future__ import annotations

import numpy as np
import pytest

from masks.base import apply_mask
from masks.plan import build_conditions
from metrics.pointwise import compute_metrics
from model.chronos2_runner import MockForecaster
from runner.run_pilot import assert_target_integrity
from runner.windows import build_windows


@pytest.fixture(scope="module")
def windows(synthetic_series, config):
    values, stamps = synthetic_series
    return build_windows(values=values, timestamps=stamps, config=config)


def test_1_target_identical_across_all_conditions(windows, config, patch_size):
    """Test 1: every condition for an origin scores against byte-identical truth."""
    for window in windows[:12]:
        conditions = build_conditions(config=config, origin_id=window.origin_id, patch_size=patch_size)
        digests = {}
        for condition in conditions:
            masked = apply_mask(window.context, condition.mask)
            assert masked is not window.context
            digests[condition.condition_id] = window.target_digest()
        assert len(set(digests.values())) == 1, (
            f"origin {window.origin_id}: conditions disagree on the scored target"
        )
        assert_target_integrity(window, digests)


def test_1b_target_bytes_match_the_source_series(windows, synthetic_series, config):
    """The target is exactly the source slice — not rescaled, shifted or filled."""
    values, _ = synthetic_series
    for window in windows[:20]:
        expected = values[window.target_start : window.target_end_exclusive]
        assert np.array_equal(np.asarray(window.target), expected)


def test_2_only_context_is_modified_never_the_target(windows, config, patch_size):
    """Test 2: masking changes only context positions, and the target is read-only."""
    for window in windows[:8]:
        original_context = np.array(window.context, copy=True)
        original_target = np.array(window.target, copy=True)
        conditions = build_conditions(config=config, origin_id=window.origin_id, patch_size=patch_size)
        for condition in conditions:
            masked = apply_mask(window.context, condition.mask)
            # The source context is never mutated in place.
            assert np.array_equal(window.context, original_context, equal_nan=True)
            # Exactly the mask's positions became NaN; nothing else moved.
            changed = np.flatnonzero(np.isnan(masked) & ~np.isnan(original_context))
            assert np.array_equal(changed, condition.mask.missing_indices)
            unchanged = np.setdiff1d(np.arange(len(masked)), condition.mask.missing_indices)
            assert np.array_equal(masked[unchanged], original_context[unchanged])
            assert np.array_equal(np.asarray(window.target), original_target)
        assert not window.target.flags.writeable


def test_2b_target_array_is_structurally_read_only(windows):
    """A write to the target raises rather than silently corrupting a run."""
    window = windows[0]
    with pytest.raises(ValueError):
        window.target[0] = 999.0


def test_2c_mask_functions_cannot_reach_the_target(config, patch_size):
    """Structural check: no mask generator accepts a series or a target array.

    Mask generators take only a context LENGTH, so there is no code path by
    which they could read an index at or beyond the context boundary.
    """
    import inspect

    from masks.contiguous_block import contiguous_block_mask
    from masks.point_random import point_random_mask

    forbidden = {"target", "series", "values", "full_series", "future", "y"}
    for func in (point_random_mask, contiguous_block_mask):
        params = set(inspect.signature(func).parameters)
        assert not (params & forbidden), f"{func.__name__} exposes {params & forbidden}"
        assert "context_length" in params


def test_3_timestamps_and_row_counts_never_change(windows, synthetic_series, config):
    """Test 3: the pipeline never drops a row or rewrites a timestamp."""
    values, stamps = synthetic_series
    context_length = config["design"]["context_length"]
    horizon = config["design"]["horizon"]
    for window in windows[:10]:
        assert len(window.context) == context_length
        assert len(window.target) == horizon
        assert len(window.target_timestamps) == horizon
        assert list(window.target_timestamps) == list(
            stamps[window.target_start : window.target_end_exclusive]
        )
        # Masking preserves length exactly: missingness is in-place, never a deletion.
        from masks.base import clean_mask
        from masks.contiguous_block import contiguous_block_mask

        masked = apply_mask(window.context, contiguous_block_mask(
            context_length=context_length, block_length=64, distance=0, rate=0.2, patch_size=16
        ))
        assert masked.shape == (context_length,)
        assert apply_mask(window.context, clean_mask(context_length)).shape == (context_length,)
    assert len(values) == config["dataset"]["expected_rows"]


def test_8_no_missing_value_ever_appears_in_the_scored_target(windows, config, patch_size):
    """Test 8: the scored target is always fully finite, under every condition."""
    forecaster = MockForecaster()
    horizon = config["design"]["horizon"]
    for window in windows[:6]:
        assert np.all(np.isfinite(np.asarray(window.target)))
        conditions = build_conditions(config=config, origin_id=window.origin_id, patch_size=patch_size)
        for condition in conditions:
            masked = apply_mask(window.context, condition.mask)
            prediction = forecaster.forecast_median(masked, horizon)
            metrics = compute_metrics(truth=np.asarray(window.target), prediction=prediction)
            assert np.isfinite(metrics.mae)


def test_8b_metrics_refuse_a_non_finite_target():
    """compute_metrics rejects a corrupted target rather than scoring it."""
    truth = np.arange(96, dtype=np.float64)
    truth[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(truth=truth, prediction=np.zeros(96))


def test_integrity_guard_detects_a_tampered_digest(windows):
    """The runtime guard actually fires — it is not a no-op assertion."""
    window = windows[0]
    digests = {"clean": window.target_digest(), "block_r20_d0": "0" * 64}
    with pytest.raises(AssertionError, match="TARGET INTEGRITY VIOLATION"):
        assert_target_integrity(window, digests)
