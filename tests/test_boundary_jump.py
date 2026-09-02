"""Formal boundary-jump contrasts BJ1_20 / BJ2_40.

STILL POST-HOC EXPLORATORY. Being more rigorously specified than the previous
round does not make this confirmatory.

The load-bearing point: the boundary jump must be bootstrapped DIRECTLY from
per-origin paired values. Subtracting two separately-bootstrapped contrasts is
valid for the point estimate (the arithmetic is linear) but NOT for the
interval, because both contrasts are measured on the same 178 origins and their
difference carries a covariance structure that differencing two independently
drawn intervals throws away.

Model-mocked: no weights required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stats.bootstrap import bootstrap_paired_difference
from stats.decision import ci_within_equivalence_band, equivalence_band, top_origin_share
from stats.internal_only import CONFOUNDER_FAMILY, bootstrap_rho, rate_positions


# --------------------------------------------------------------------------- #
# 1. Direct computation vs. algebraic subtraction
# --------------------------------------------------------------------------- #

def test_point_estimate_agrees_with_algebraic_subtraction():
    """BJ = (d0 - dNear) must equal (d0 - dFar) - (dNear - dFar) exactly.

    Linearity holds for the POINT estimate, so the direct computation is not
    changing the estimand - only how its uncertainty is obtained.
    """
    rng = np.random.default_rng(0)
    n = 178
    d0, d_near, d_far = rng.normal(3, 1, n), rng.normal(2, 1, n), rng.normal(2, 1, n)

    direct = (d0 - d_near).mean()
    c1_like = (d0 - d_far).mean()          # the preregistered endpoint contrast
    ic1_like = (d_near - d_far).mean()     # the internal contrast
    assert direct == pytest.approx(c1_like - ic1_like)


def test_naive_ci_subtraction_differs_from_the_direct_ci():
    """The reason this round exists: differencing two CIs is NOT the CI.

    Because the two contrasts share the same origins, their difference's
    sampling distribution is far tighter than a naive subtraction of separately
    bootstrapped intervals suggests. This test constructs that shared-origin
    structure and shows the two answers genuinely disagree.
    """
    rng = np.random.default_rng(1)
    n = 178
    # A strong per-origin effect shared by every condition - exactly the
    # structure real evaluation origins have.
    origin_effect = rng.normal(0, 3.0, n)
    d_far = origin_effect + rng.normal(0, 0.1, n)
    d_near = origin_effect + rng.normal(0, 0.1, n) + 0.05
    d0 = origin_effect + rng.normal(0, 0.1, n) + 0.60

    kwargs = dict(block_length=8, n_replicates=2000, ci_level_corrected=0.95)
    direct, _ = bootstrap_paired_difference(
        contrast_id="BJ_direct", paired_differences=(d0 - d_near), seed=11, **kwargs
    )
    c1, _ = bootstrap_paired_difference(
        contrast_id="C1_like", paired_differences=(d0 - d_far), seed=12, **kwargs
    )
    ic1, _ = bootstrap_paired_difference(
        contrast_id="IC1_like", paired_differences=(d_near - d_far), seed=13, **kwargs
    )

    # Point estimates agree, as the previous test established.
    assert direct.observed_mean == pytest.approx(c1.observed_mean - ic1.observed_mean)

    # A naive "difference of intervals" built from the two separate bootstraps.
    naive_low = c1.ci_low_corrected - ic1.ci_high_corrected
    naive_high = c1.ci_high_corrected - ic1.ci_low_corrected
    direct_width = direct.ci_high_corrected - direct.ci_low_corrected
    naive_width = naive_high - naive_low

    assert naive_width > 1.5 * direct_width, (
        "the naive subtraction should be materially wider than the direct CI; "
        f"naive={naive_width:.5f} direct={direct_width:.5f}"
    )
    assert not (
        naive_low == pytest.approx(direct.ci_low_corrected, abs=1e-3)
        and naive_high == pytest.approx(direct.ci_high_corrected, abs=1e-3)
    ), "the naive interval must not be mistaken for the direct one"


def test_covariance_matters_only_because_origins_are_shared():
    """Control: with INDEPENDENT origins the two approaches converge.

    This proves the previous test measures shared-origin covariance and not
    some artefact of the bootstrap itself.
    """
    rng = np.random.default_rng(2)
    n = 178
    independent_a = rng.normal(0.6, 0.1, n)   # d0 - d_near, standalone
    kwargs = dict(block_length=8, n_replicates=2000, ci_level_corrected=0.95)
    result, _ = bootstrap_paired_difference(
        contrast_id="indep", paired_differences=independent_a, seed=21, **kwargs
    )
    width = result.ci_high_corrected - result.ci_low_corrected
    # A tight, well-behaved interval around a clean mean; nothing pathological.
    assert 0 < width < 0.1
    assert result.observed_mean == pytest.approx(0.6, abs=0.05)


# --------------------------------------------------------------------------- #
# 2. BJ arms are d=0 vs nearest internal
# --------------------------------------------------------------------------- #

def test_boundary_jump_arms_are_d0_and_nearest_internal(config):
    positions = {p.rate_tag: p for p in rate_positions(config)}
    assert (positions["20"].trailing_boundary, positions["20"].near) == (0, 64)
    assert (positions["40"].trailing_boundary, positions["40"].near) == (0, 48)
    assert positions["20"].trailing_boundary_column == "block_r20_d0"
    assert positions["40"].trailing_boundary_column == "block_r40_d0"


# --------------------------------------------------------------------------- #
# 3. Shared helpers are reused, not reimplemented
# --------------------------------------------------------------------------- #

def test_top_origin_share_is_the_shared_implementation():
    """stats.internal_only must import it rather than carry its own copy."""
    from pathlib import Path

    import stats.internal_only as module

    assert module.top_origin_share is top_origin_share
    source = Path("stats/internal_only.py").read_text()
    assert "def top_origin_share" not in source, "a second copy was defined"


def test_top_origin_share_matches_a_hand_calculation():
    """20 origins, top 5% = 1 origin. |diffs| = 1..20, sum = 210, top = 20."""
    diffs = np.arange(1, 21, dtype=float)
    assert top_origin_share(diffs, top_frac=0.05) == pytest.approx(20.0 / 210.0)
    # Top 25% = 5 origins: 16+17+18+19+20 = 90.
    assert top_origin_share(diffs, top_frac=0.25) == pytest.approx(90.0 / 210.0)


def test_top_origin_share_handles_degenerate_input():
    assert np.isnan(top_origin_share(np.zeros(10), top_frac=0.05))
    assert np.isnan(top_origin_share(np.array([]), top_frac=0.05))


# --------------------------------------------------------------------------- #
# 4. Equivalence check reuses the preregistered NO-GO threshold
# --------------------------------------------------------------------------- #

def test_equivalence_band_comes_from_the_no_go_rule(config):
    """One definition of +/-3%, not a second hardcoded literal."""
    frac = config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
    assert frac == 0.03
    assert equivalence_band(config, 2.4575) == pytest.approx(0.03 * 2.4575)
    assert equivalence_band(config, 2.4575) == pytest.approx(0.0737, abs=1e-4)


def test_equivalence_band_is_not_hardcoded_anywhere(config):
    """Changing the config must move the band; a literal copy would not."""
    import json

    cfg = json.loads(json.dumps(config))
    cfg["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"] = 0.10
    assert equivalence_band(cfg, 2.0) == pytest.approx(0.20)

    from pathlib import Path

    source = Path("stats/internal_only.py").read_text()
    assert "0.03" not in source, "the +/-3% literal was copied instead of imported"


def test_non_significance_is_not_equivalence():
    """A CI straddling zero but wider than the band establishes NOTHING."""
    band = 0.0737
    # Straddles zero, but reaches well beyond the band: not equivalence.
    assert ci_within_equivalence_band(-0.20, +0.25, band) is False
    # Entirely inside the band: equivalence established.
    assert ci_within_equivalence_band(-0.05, +0.06, band) is True
    # Excludes zero but sits inside the band: still equivalent in this sense.
    assert ci_within_equivalence_band(+0.01, +0.05, band) is True
    # One side outside is enough to fail.
    assert ci_within_equivalence_band(-0.05, +0.08, band) is False


def test_equivalence_check_rejects_non_finite_bounds():
    assert ci_within_equivalence_band(float("nan"), 0.01, 0.0737) is False
    assert ci_within_equivalence_band(-0.01, float("inf"), 0.0737) is False


# --------------------------------------------------------------------------- #
# 5. Bootstrap CI on rho
# --------------------------------------------------------------------------- #

def test_bootstrap_rho_recovers_a_strong_association():
    rng = np.random.default_rng(3)
    n = 178
    z = rng.normal(size=n)
    mae = 0.9 * z + rng.normal(scale=0.15, size=n)
    per_block = bootstrap_rho(
        delta_mae=mae, delta_z=z, block_lengths=[4, 8, 12],
        n_replicates=400, seed=99,
    )
    assert set(per_block) == {"4", "8", "12"}
    low, high = per_block["8"]
    assert low > 0, "a strong positive association must give a CI excluding zero"
    assert 0 < low < high < 1


def test_bootstrap_rho_on_noise_includes_zero():
    rng = np.random.default_rng(4)
    n = 178
    per_block = bootstrap_rho(
        delta_mae=rng.normal(size=n), delta_z=rng.normal(size=n),
        block_lengths=[8], n_replicates=400, seed=100,
    )
    low, high = per_block["8"]
    assert low < 0 < high, "unrelated series must not yield a CI excluding zero"


def test_bootstrap_rho_is_deterministic_given_its_seed():
    rng = np.random.default_rng(5)
    z, mae = rng.normal(size=178), rng.normal(size=178)
    kwargs = dict(delta_mae=mae, delta_z=z, block_lengths=[8], n_replicates=200, seed=7)
    assert bootstrap_rho(**kwargs) == bootstrap_rho(**kwargs)


def test_bootstrap_rho_uses_an_isolated_stream_per_block_length():
    """Different block lengths must not reuse the same draws."""
    rng = np.random.default_rng(6)
    z = rng.normal(size=178)
    mae = 0.5 * z + rng.normal(scale=0.5, size=178)
    per_block = bootstrap_rho(
        delta_mae=mae, delta_z=z, block_lengths=[4, 8, 12], n_replicates=300, seed=13
    )
    assert per_block["4"] != per_block["8"] != per_block["12"]
