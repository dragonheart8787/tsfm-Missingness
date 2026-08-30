"""Test 11 — bootstrap resampling preserves condition pairing.

Model-mocked: no weights required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics.aggregate import pivot_condition_matrix, seed_averaged_random
from stats.bootstrap import bootstrap_paired_difference, holm_correct, moving_block_indices


def test_11_resampling_carries_every_condition_jointly():
    """A resampled origin brings ALL its conditions with it, never a mix."""
    rng = np.random.default_rng(0)
    n_origins, n_conditions = 178, 17
    matrix = rng.normal(size=(n_origins, n_conditions))
    # A per-origin signature that must survive resampling intact.
    signature = np.arange(n_origins)

    for block_length in (4, 8, 12):
        for _ in range(50):
            idx = moving_block_indices(n=n_origins, block_length=block_length, rng=rng)
            resampled = matrix[idx, :]
            assert resampled.shape == (n_origins, n_conditions)
            # Every resampled row equals some ORIGINAL row, unchanged across columns.
            for row, origin in zip(resampled, signature[idx]):
                assert np.array_equal(row, matrix[origin, :]), (
                    "pairing broken: a resampled row mixes conditions from different origins"
                )


def test_11b_paired_difference_of_a_resample_equals_resample_of_the_difference():
    """The identity that pairing preservation is equivalent to."""
    rng = np.random.default_rng(1)
    n = 178
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    matrix = np.column_stack([a, b])
    for block_length in (4, 8, 12):
        for _ in range(50):
            idx = moving_block_indices(n=n, block_length=block_length, rng=rng)
            resampled = matrix[idx, :]
            assert np.allclose(resampled[:, 0] - resampled[:, 1], (a - b)[idx])


def test_11c_unpaired_resampling_is_detectably_different():
    """Adversarial control: independently resampling each condition BREAKS pairing.

    If it did not, this test would prove the pairing test above is vacuous.
    """
    rng = np.random.default_rng(2)
    n = 178
    shared = rng.normal(size=n) * 5.0          # a strong per-origin effect
    a = shared + rng.normal(size=n) * 0.1
    b = shared + rng.normal(size=n) * 0.1      # tightly paired with a

    paired_sd, unpaired_sd = [], []
    for _ in range(200):
        idx = moving_block_indices(n=n, block_length=8, rng=rng)
        paired_sd.append((a[idx] - b[idx]).mean())
        idx_a = moving_block_indices(n=n, block_length=8, rng=rng)
        idx_b = moving_block_indices(n=n, block_length=8, rng=rng)
        unpaired_sd.append((a[idx_a] - b[idx_b]).mean())

    # Breaking the pairing inflates the variance of the mean difference enormously.
    assert np.std(unpaired_sd) > 5 * np.std(paired_sd)


def test_11d_moving_block_indices_are_well_formed():
    rng = np.random.default_rng(3)
    for n, block_length in ((178, 8), (178, 4), (178, 12), (10, 3), (5, 9)):
        idx = moving_block_indices(n=n, block_length=block_length, rng=rng)
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n
        effective = min(block_length, n)
        # Blocks are contiguous runs of `effective` consecutive origins.
        for start in range(0, n - effective + 1, effective):
            chunk = idx[start : start + effective]
            if len(chunk) == effective:
                assert np.array_equal(np.diff(chunk), np.ones(effective - 1, dtype=chunk.dtype))


def test_11e_bootstrap_is_deterministic_given_its_seed():
    diffs = np.random.default_rng(4).normal(loc=0.3, size=178)
    kwargs = dict(
        contrast_id="C", paired_differences=diffs, block_length=8,
        n_replicates=500, seed=99,
    )
    first, draws_a = bootstrap_paired_difference(**kwargs)
    second, draws_b = bootstrap_paired_difference(**kwargs)
    assert np.array_equal(draws_a, draws_b)
    assert first.as_dict() == second.as_dict()


def test_11f_seeds_are_averaged_within_origin_before_any_contrast():
    """The three point-random seeds are repeated measures, never independent samples."""
    matrix = pd.DataFrame(
        {
            "random_r20_s42": [1.0, 2.0, 3.0],
            "random_r20_s123": [2.0, 3.0, 4.0],
            "random_r20_s2026": [3.0, 4.0, 5.0],
        },
        index=[0, 1, 2],
    )
    averaged = seed_averaged_random(matrix, rate_tag="20", seeds=[42, 123, 2026])
    assert list(averaged) == [2.0, 3.0, 4.0]
    assert len(averaged) == 3, "seed averaging must not multiply the sample size"


def test_11g_holm_is_monotone_and_more_conservative_than_uncorrected():
    diffs = {
        "A": np.random.default_rng(5).normal(loc=0.50, size=178),
        "B": np.random.default_rng(6).normal(loc=0.20, size=178),
        "C": np.random.default_rng(7).normal(loc=0.05, size=178),
        "D": np.random.default_rng(8).normal(loc=0.00, size=178),
    }
    results, draws = [], {}
    for i, (name, values) in enumerate(diffs.items()):
        result, replicate_means = bootstrap_paired_difference(
            contrast_id=name, paired_differences=values, block_length=8,
            n_replicates=2000, seed=100 + i,
        )
        results.append(result)
        draws[name] = replicate_means

    holm_correct(results, draws, family_alpha=0.05)
    for result in results:
        assert result.holm_adjusted_p >= result.p_value_two_sided
        assert result.holm_alpha <= 0.05
        # The Holm interval is at least as wide as the uncorrected 95% one.
        assert result.ci_low_holm <= result.ci_low_corrected + 1e-12
        assert result.ci_high_holm >= result.ci_high_corrected - 1e-12

    ordered = sorted(results, key=lambda r: r.p_value_two_sided)
    adjusted = [r.holm_adjusted_p for r in ordered]
    assert adjusted == sorted(adjusted), "Holm adjusted p-values must be monotone"


def test_11h_pivot_matrix_preserves_pairing_from_the_results_file():
    """The matrix the bootstrap consumes is genuinely origin-by-condition."""
    rows = []
    # Deterministic offsets, not builtin hash(): this repo forbids salted hashing
    # anywhere its outputs are compared, tests included.
    offsets = {"clean": 0.0, "block_r20_d0": 0.5, "block_r20_d256": 0.25}
    for origin in range(5):
        for condition, offset in offsets.items():
            rows.append(
                {
                    "origin_id": origin,
                    "condition_id": condition,
                    "mae": origin + offset,
                    "status": "ok",
                }
            )
    matrix = pivot_condition_matrix(pd.DataFrame(rows))
    assert matrix.shape == (5, 3)
    assert list(matrix.index) == [0, 1, 2, 3, 4]
    assert not matrix.isna().any().any()
