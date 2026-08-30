"""Contiguous-block missingness, positioned by distance to the forecast boundary.

Parameterisation (preregistered)
--------------------------------
    d                   = context_length - block_end_exclusive
    block_start         = context_length - d - block_length
    block_end_exclusive = block_start + block_length

so d=0 places the block flush against the forecast boundary and larger d pushes
it back into history.

Patch-grid alignment is asserted against the model's REAL patch size, which the
caller must read from the loaded checkpoint (``Chronos2ForecastingConfig``);
the value is never hardcoded here.
"""

from __future__ import annotations

import numpy as np

from .base import Mask


def block_bounds(*, context_length: int, block_length: int, distance: int) -> tuple[int, int]:
    """Return (block_start, block_end_exclusive) for a given distance-to-boundary."""
    block_start = context_length - distance - block_length
    block_end_exclusive = block_start + block_length
    return block_start, block_end_exclusive


def contiguous_block_mask(
    *,
    context_length: int,
    block_length: int,
    distance: int,
    rate: float,
    patch_size: int,
) -> Mask:
    """Build one contiguous-block mask for a single context window.

    Receives only the context *length*: no series, no target, no absolute index.
    Asserts valid bounds, contiguity, the exact requested length, and alignment
    to the supplied (model-verified) patch grid.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    block_start, block_end_exclusive = block_bounds(
        context_length=context_length, block_length=block_length, distance=distance
    )

    if block_start < 0 or block_end_exclusive > context_length:
        raise ValueError(
            f"block [{block_start}, {block_end_exclusive}) escapes context [0, {context_length}) "
            f"for block_length={block_length}, d={distance}"
        )
    if block_end_exclusive - block_start != block_length:
        raise AssertionError("block length invariant violated")

    # Patch-grid alignment against the model's real patch size. A misaligned
    # block would confound "distance to boundary" with "how many patches are
    # only partially missing", so this is a hard failure, not a warning.
    if block_start % patch_size != 0:
        raise AssertionError(
            f"block_start={block_start} is not aligned to the model patch grid "
            f"(patch_size={patch_size}); d={distance}, block_length={block_length}"
        )
    if block_length % patch_size != 0:
        raise AssertionError(
            f"block_length={block_length} is not a whole number of patches "
            f"(patch_size={patch_size})"
        )

    indices = np.arange(block_start, block_end_exclusive, dtype=np.int64)
    if indices.size != block_length:
        raise AssertionError("contiguity/length invariant violated")
    if indices.size > 1 and not np.all(np.diff(indices) == 1):
        raise AssertionError("block indices are not contiguous")

    return Mask(
        pattern="contiguous_block",
        rate=rate,
        context_length=context_length,
        missing_indices=indices,
        block_start=block_start,
        block_end_exclusive=block_end_exclusive,
        distance_to_boundary=distance,
        extra={"patch_size": patch_size, "block_length": block_length},
    )
