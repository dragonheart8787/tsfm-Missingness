"""Mask representation shared by both missingness patterns.

Structural target-integrity guarantee
-------------------------------------
Nothing in this package accepts a target array, a full series, or an origin's
absolute index range. Mask generators receive only ``context_length`` (an int)
and return positions *within the context*. ``apply_mask`` receives only the
context array. There is therefore no code path by which a masking function can
read or write an index at or beyond the context boundary — the target is not
reachable from here, rather than merely conventionally untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Mask:
    """A set of missing positions inside a single context window."""

    pattern: str                 # "clean" | "point_random" | "contiguous_block"
    rate: float                  # nominal missing rate (0.0 for clean)
    context_length: int
    missing_indices: np.ndarray  # sorted int64 positions in [0, context_length)
    seed: int | None = None      # point-random only
    block_start: int | None = None
    block_end_exclusive: int | None = None
    distance_to_boundary: int | None = None  # d = context_length - block_end_exclusive
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        idx = np.asarray(self.missing_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("missing_indices must be 1-dimensional")
        if idx.size and (idx.min() < 0 or idx.max() >= self.context_length):
            raise ValueError(
                f"mask indices must lie in [0, {self.context_length}); "
                f"got [{idx.min()}, {idx.max()}]"
            )
        if np.unique(idx).size != idx.size:
            raise ValueError("mask indices must be unique")
        object.__setattr__(self, "missing_indices", np.sort(idx))

    @property
    def missing_count(self) -> int:
        return int(self.missing_indices.size)

    @property
    def realised_rate(self) -> float:
        return self.missing_count / self.context_length

    def boolean(self) -> np.ndarray:
        """Boolean mask over the context: True where the observation is removed."""
        flags = np.zeros(self.context_length, dtype=bool)
        flags[self.missing_indices] = True
        return flags

    def descriptor(self) -> str:
        """Compact, reproducible serialisation for the results file."""
        if self.pattern == "clean":
            return "clean"
        if self.pattern == "contiguous_block":
            return f"block[{self.block_start}:{self.block_end_exclusive})d={self.distance_to_boundary}"
        return f"random(n={self.missing_count},seed={self.seed},rate={self.rate})"


def apply_mask(context: np.ndarray, mask: Mask) -> np.ndarray:
    """Return a copy of the context with masked positions set to NaN.

    NaN is Chronos-2's officially supported representation of a missing history
    observation: ``Chronos2Model._prepare_patched_context`` derives its
    ``context_mask`` as ``~torch.isnan(context)`` and zeroes those positions
    after a NaN-aware instance normalisation. No timestamp is dropped and no
    row is removed — the array keeps its exact length.
    """
    if context.ndim != 1:
        raise ValueError("context must be 1-dimensional")
    if context.shape[0] != mask.context_length:
        raise ValueError(
            f"context length {context.shape[0]} != mask context_length {mask.context_length}"
        )
    out = np.array(context, dtype=np.float32, copy=True)
    out[mask.missing_indices] = np.nan
    return out


def clean_mask(context_length: int) -> Mask:
    """The no-missingness reference condition."""
    return Mask(
        pattern="clean",
        rate=0.0,
        context_length=context_length,
        missing_indices=np.empty(0, dtype=np.int64),
    )
