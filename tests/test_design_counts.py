"""Test 10 — the design produces exactly the preregistered number of rows.

Model-mocked: no weights required.
"""

from __future__ import annotations

import pytest

from masks.plan import build_conditions
from runner.windows import build_windows, enumerate_origins


def test_10_origin_count_is_exactly_178(config):
    """178 origins for ETTh1's canonical 17,420 rows at L=320, H=96, stride=96."""
    design = config["design"]
    starts = enumerate_origins(
        n_rows=int(config["dataset"]["expected_rows"]),
        context_length=int(design["context_length"]),
        horizon=int(design["horizon"]),
        stride=int(design["stride"]),
    )
    assert len(starts) == 178 == int(design["expected_origins"])
    assert starts[0] == 0
    # The last origin's target must end inside the series.
    assert starts[-1] + design["context_length"] + design["horizon"] <= config["dataset"]["expected_rows"]
    # ...and one more stride would not fit.
    assert (
        starts[-1] + design["stride"] + design["context_length"] + design["horizon"]
        > config["dataset"]["expected_rows"]
    )


def test_10b_conditions_per_origin_is_exactly_17(config, patch_size):
    conditions = build_conditions(config=config, origin_id=0, patch_size=patch_size)
    assert len(conditions) == 17 == int(config["design"]["expected_conditions_per_origin"])
    by_pattern = {}
    for condition in conditions:
        by_pattern.setdefault(condition.pattern, []).append(condition.condition_id)
    assert len(by_pattern["clean"]) == 1
    assert len(by_pattern["point_random"]) == 6          # 2 rates x 3 seeds
    assert len(by_pattern["contiguous_block"]) == 10     # 2 rates x 5 positions


def test_10c_total_forecast_count_is_exactly_3026(config):
    design = config["design"]
    total = int(design["expected_origins"]) * int(design["expected_conditions_per_origin"])
    assert total == 3026 == int(design["expected_total_forecasts"])


def test_10d_per_seed_and_per_position_breakdown(config, patch_size):
    """Per-seed breakdown: each seed contributes 178 x 2 rates = 356 forecasts."""
    n_origins = int(config["design"]["expected_origins"])
    seeds = config["masks"]["point_random_seeds"]
    rates = config["masks"]["rates"]

    conditions = build_conditions(config=config, origin_id=0, patch_size=patch_size)
    for seed in seeds:
        per_seed = [c for c in conditions if c.seed == seed]
        assert len(per_seed) == len(rates)
        assert len(per_seed) * n_origins == 356

    assert 1 * n_origins == 178                                  # clean
    assert len(seeds) * len(rates) * n_origins == 1068           # point-random
    assert sum(len(r["block_distances"]) for r in rates) * n_origins == 1780  # blocks
    assert 178 + 1068 + 1780 == 3026


def test_10e_builder_rejects_a_wrong_condition_count(config, patch_size):
    """The count assertion is live, not decorative."""
    broken = {**config, "design": {**config["design"], "expected_conditions_per_origin": 16}}
    with pytest.raises(AssertionError):
        build_conditions(config=broken, origin_id=0, patch_size=patch_size)


def test_10f_origin_count_assertion_fires_on_a_wrong_series_length(config, synthetic_series):
    values, stamps = synthetic_series
    with pytest.raises(AssertionError, match="evaluation origins"):
        build_windows(values=values[:10000], timestamps=stamps[:10000], config=config)


def test_10g_real_etth1_yields_178_origins(real_series, config):
    """The count against the ACTUAL fetched dataset, not just its declared length."""
    values, stamps, validation = real_series
    assert validation.n_rows == 17420
    windows = build_windows(values=values, timestamps=stamps, config=config)
    assert len(windows) == 178
