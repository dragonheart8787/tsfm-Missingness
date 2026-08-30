"""Exact-count point-random missingness.

Exactly ``missing_count`` context positions are removed — sampled WITHOUT
replacement, never by independent Bernoulli draws (Bernoulli would only hit the
target count in expectation, and the pilot preregisters exact counts of 64 and
128 out of L=320).
"""

from __future__ import annotations

import numpy as np

from .base import Mask
from .rng import derive_seed, rng_for


def point_random_mask(
    *,
    context_length: int,
    missing_count: int,
    rate: float,
    base_seed: int,
    origin_id: int,
    namespace: str,
    digest_size: int = 8,
) -> Mask:
    """Build one exact-count point-random mask for a single context window.

    Receives only the context *length*: no series, no target, no absolute index.
    """
    if not 0 <= missing_count <= context_length:
        raise ValueError(f"missing_count {missing_count} outside [0, {context_length}]")

    rng = rng_for(
        namespace=namespace,
        base_seed=base_seed,
        origin_id=origin_id,
        rate=rate,
        digest_size=digest_size,
    )
    # Sampling without replacement guarantees the exact preregistered count.
    indices = rng.choice(context_length, size=missing_count, replace=False)

    mask = Mask(
        pattern="point_random",
        rate=rate,
        context_length=context_length,
        missing_indices=np.asarray(indices, dtype=np.int64),
        seed=base_seed,
        extra={
            "derived_seed": derive_seed(
                namespace=namespace,
                base_seed=base_seed,
                origin_id=origin_id,
                rate=rate,
                digest_size=digest_size,
            ),
            "rng_namespace": namespace,
            "origin_id": origin_id,
        },
    )
    if mask.missing_count != missing_count:
        raise AssertionError(
            f"exact-count violation: requested {missing_count}, produced {mask.missing_count}"
        )
    return mask
