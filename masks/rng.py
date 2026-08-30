"""Deterministic, process-stable RNG construction for mask generation.

Python's builtin ``hash()`` is salted per process for ``str`` inputs and is
therefore NEVER used here. Instead a seed is derived by a fixed, documented
BLAKE2b digest over a canonical UTF-8 payload, so the same
(namespace, base_seed, origin_id, rate) always yields the same mask on any
machine, in any process, in any Python build.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(
    *,
    namespace: str,
    base_seed: int,
    origin_id: int,
    rate: float,
    digest_size: int = 8,
) -> int:
    """Derive a 64-bit seed integer from the mask's identifying coordinates.

    The rate is quantised to parts-per-million before hashing so that the
    payload is an exact decimal string and never depends on float repr.
    """
    rate_ppm = int(round(rate * 1_000_000))
    payload = f"{namespace}|{base_seed}|{origin_id}|{rate_ppm}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=digest_size).digest()
    return int.from_bytes(digest, "big")


def rng_for(
    *,
    namespace: str,
    base_seed: int,
    origin_id: int,
    rate: float,
    digest_size: int = 8,
) -> np.random.Generator:
    """Return the numpy Generator for one (origin, rate, seed) mask."""
    return np.random.default_rng(
        derive_seed(
            namespace=namespace,
            base_seed=base_seed,
            origin_id=origin_id,
            rate=rate,
            digest_size=digest_size,
        )
    )
