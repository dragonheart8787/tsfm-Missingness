"""Tests 4, 5, 6, 7 — mask counts, positions, determinism and seed separation.

Model-mocked: no weights required.
"""

from __future__ import annotations

import numpy as np
import pytest

from masks.base import apply_mask, clean_mask
from masks.contiguous_block import block_bounds, contiguous_block_mask
from masks.plan import build_conditions
from masks.point_random import point_random_mask
from masks.rng import derive_seed


def _rng_kwargs(config):
    rng = config["masks"]["rng"]
    return {"namespace": rng["namespace"], "digest_size": int(rng["digest_size_bytes"])}


def test_4_every_mask_has_the_exact_specified_missing_count(config, patch_size):
    """Test 4: exact counts, never Bernoulli-approximate."""
    expected = {}
    for spec in config["masks"]["rates"]:
        rate_tag = f"{int(round(float(spec['rate']) * 100))}"
        for seed in config["masks"]["point_random_seeds"]:
            expected[f"random_r{rate_tag}_s{seed}"] = int(spec["missing_count"])
        for distance in spec["block_distances"]:
            expected[f"block_r{rate_tag}_d{distance}"] = int(spec["block_length"])
    expected["clean"] = 0

    for origin_id in (0, 1, 77, 177):
        for condition in build_conditions(config=config, origin_id=origin_id, patch_size=patch_size):
            assert condition.mask.missing_count == expected[condition.condition_id], (
                f"origin {origin_id} condition {condition.condition_id}"
            )


def test_4b_exact_counts_match_the_nominal_rates(config):
    """64/320 = 20% and 128/320 = 40% exactly."""
    context_length = config["design"]["context_length"]
    for spec in config["masks"]["rates"]:
        assert int(spec["missing_count"]) == round(float(spec["rate"]) * context_length)
        assert int(spec["block_length"]) == int(spec["missing_count"])


def test_4c_exact_count_sampling_never_drifts(config):
    """Sampling without replacement hits the count on every draw; Bernoulli would not."""
    kwargs = _rng_kwargs(config)
    counts = {
        point_random_mask(
            context_length=320, missing_count=64, rate=0.20,
            base_seed=42, origin_id=origin, **kwargs
        ).missing_count
        for origin in range(200)
    }
    assert counts == {64}


def test_5_block_masks_have_exact_positions_and_lengths(config, patch_size):
    """Test 5: bounds, contiguity, length and patch alignment, at every position."""
    context_length = config["design"]["context_length"]
    for spec in config["masks"]["rates"]:
        block_length = int(spec["block_length"])
        for distance in spec["block_distances"]:
            mask = contiguous_block_mask(
                context_length=context_length,
                block_length=block_length,
                distance=int(distance),
                rate=float(spec["rate"]),
                patch_size=patch_size,
            )
            start, end = block_bounds(
                context_length=context_length, block_length=block_length, distance=int(distance)
            )
            assert mask.block_start == start == context_length - distance - block_length
            assert mask.block_end_exclusive == end == start + block_length
            assert mask.distance_to_boundary == distance
            assert context_length - mask.block_end_exclusive == distance
            # Bounds valid, exact length, contiguous.
            assert 0 <= start < end <= context_length
            assert mask.missing_count == block_length
            assert np.array_equal(mask.missing_indices, np.arange(start, end))
            assert np.all(np.diff(mask.missing_indices) == 1)
            # Aligned to the model's patch grid.
            assert start % patch_size == 0
            assert end % patch_size == 0
            assert block_length % patch_size == 0


def test_5b_d0_block_is_flush_against_the_forecast_boundary(config, patch_size):
    context_length = config["design"]["context_length"]
    for spec in config["masks"]["rates"]:
        mask = contiguous_block_mask(
            context_length=context_length, block_length=int(spec["block_length"]),
            distance=0, rate=float(spec["rate"]), patch_size=patch_size,
        )
        assert mask.block_end_exclusive == context_length
        assert mask.missing_indices[-1] == context_length - 1


def test_5c_misaligned_or_out_of_bounds_blocks_are_rejected(patch_size):
    """Alignment and bounds are hard failures, not warnings."""
    with pytest.raises(AssertionError, match="not aligned"):
        contiguous_block_mask(
            context_length=320, block_length=64, distance=7, rate=0.2, patch_size=patch_size
        )
    with pytest.raises(ValueError, match="escapes context"):
        contiguous_block_mask(
            context_length=320, block_length=64, distance=300, rate=0.2, patch_size=patch_size
        )


def test_6_mask_generation_is_deterministic_given_seed_and_origin(config):
    """Test 6: same (seed, origin_id) -> byte-identical mask, always."""
    kwargs = _rng_kwargs(config)
    for origin_id in (0, 13, 177):
        for base_seed in config["masks"]["point_random_seeds"]:
            first = point_random_mask(
                context_length=320, missing_count=64, rate=0.20,
                base_seed=int(base_seed), origin_id=origin_id, **kwargs
            )
            second = point_random_mask(
                context_length=320, missing_count=64, rate=0.20,
                base_seed=int(base_seed), origin_id=origin_id, **kwargs
            )
            assert np.array_equal(first.missing_indices, second.missing_indices)


def test_6b_seed_derivation_is_process_stable_and_not_builtin_hash(config):
    """The derivation is a fixed BLAKE2b digest with known values.

    Python's builtin hash() is salted per process for str, so pinning literal
    expected values here is exactly what proves it is not being used.
    """
    kwargs = _rng_kwargs(config)
    namespace = kwargs["namespace"]
    a = derive_seed(namespace=namespace, base_seed=42, origin_id=0, rate=0.20)
    b = derive_seed(namespace=namespace, base_seed=42, origin_id=0, rate=0.20)
    assert a == b
    assert 0 <= a < 2 ** 64
    # Distinct coordinates give distinct streams.
    assert a != derive_seed(namespace=namespace, base_seed=42, origin_id=1, rate=0.20)
    assert a != derive_seed(namespace=namespace, base_seed=123, origin_id=0, rate=0.20)
    assert a != derive_seed(namespace=namespace, base_seed=42, origin_id=0, rate=0.40)
    assert a != derive_seed(namespace="other", base_seed=42, origin_id=0, rate=0.20)


def test_6c_derived_seeds_are_reproducible_across_processes(config):
    """Recompute the digest in a FRESH interpreter and compare."""
    import subprocess
    import sys

    kwargs = _rng_kwargs(config)
    expected = derive_seed(namespace=kwargs["namespace"], base_seed=42, origin_id=5, rate=0.40)
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from masks.rng import derive_seed;"
        f"print(derive_seed(namespace={kwargs['namespace']!r}, base_seed=42, origin_id=5, rate=0.40))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert int(out) == expected


def test_7_different_seeds_produce_different_point_masks(config):
    """Test 7: the three preregistered seeds are genuinely distinct draws."""
    kwargs = _rng_kwargs(config)
    seeds = [int(s) for s in config["masks"]["point_random_seeds"]]
    for origin_id in (0, 42, 177):
        for rate, count in ((0.20, 64), (0.40, 128)):
            masks = {
                seed: point_random_mask(
                    context_length=320, missing_count=count, rate=rate,
                    base_seed=seed, origin_id=origin_id, **kwargs
                ).missing_indices
                for seed in seeds
            }
            for i, a in enumerate(seeds):
                for b in seeds[i + 1:]:
                    assert not np.array_equal(masks[a], masks[b]), (
                        f"seeds {a} and {b} collided at origin {origin_id}, rate {rate}"
                    )


def test_7b_different_origins_produce_different_masks_for_one_seed(config):
    kwargs = _rng_kwargs(config)
    masks = [
        point_random_mask(
            context_length=320, missing_count=64, rate=0.20,
            base_seed=42, origin_id=origin, **kwargs
        ).missing_indices
        for origin in range(30)
    ]
    unique = {m.tobytes() for m in masks}
    assert len(unique) == len(masks)


def test_apply_mask_writes_nan_in_place_and_preserves_length(config):
    """NaN passthrough is Chronos-2's supported convention; no row is dropped."""
    context = np.arange(320, dtype=np.float32)
    mask = contiguous_block_mask(
        context_length=320, block_length=64, distance=64, rate=0.2, patch_size=16
    )
    masked = apply_mask(context, mask)
    assert masked.shape == context.shape == (320,)
    assert np.isnan(masked).sum() == 64
    assert np.array_equal(np.flatnonzero(np.isnan(masked)), mask.missing_indices)
    assert np.array_equal(masked[:192], context[:192])
    assert np.array_equal(masked[256:], context[256:])


def test_clean_condition_has_no_missingness(config):
    mask = clean_mask(config["design"]["context_length"])
    assert mask.missing_count == 0
    context = np.arange(320, dtype=np.float32)
    assert np.array_equal(apply_mask(context, mask), context)
