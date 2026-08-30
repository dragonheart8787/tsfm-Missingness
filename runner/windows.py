"""Evaluation-origin slicing with a structural target-integrity guarantee.

The forecast target is exposed as a READ-ONLY numpy view. Masking, scaling and
any other preprocessing operate on ``context`` only; they are never handed the
target array, and the target they *would* have to reach is not writeable even
if some future code tried.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OriginWindow:
    origin_id: int
    context_start: int
    context_end_exclusive: int   # == target_start
    target_start: int
    target_end_exclusive: int
    origin_timestamp: pd.Timestamp     # timestamp of the first forecast step
    context: np.ndarray                # float32, length L, writeable copy
    target: np.ndarray                 # float32, length H, READ-ONLY
    target_timestamps: pd.DatetimeIndex

    def target_digest(self) -> str:
        """sha256 over the raw target bytes — the runtime integrity witness."""
        return hashlib.sha256(np.ascontiguousarray(self.target, dtype=np.float32).tobytes()).hexdigest()


def enumerate_origins(*, n_rows: int, context_length: int, horizon: int, stride: int) -> list[int]:
    """All context start indices whose full context+target fits inside the series."""
    last_start = n_rows - context_length - horizon
    if last_start < 0:
        return []
    return list(range(0, last_start + 1, stride))


def build_windows(
    *,
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    config: dict[str, Any],
) -> list[OriginWindow]:
    """Slice the series into the preregistered evaluation origins."""
    design = config["design"]
    context_length = int(design["context_length"])
    horizon = int(design["horizon"])
    stride = int(design["stride"])

    starts = enumerate_origins(
        n_rows=len(values), context_length=context_length, horizon=horizon, stride=stride
    )
    expected = int(design["expected_origins"])
    if len(starts) != expected:
        raise AssertionError(
            f"enumerated {len(starts)} evaluation origins, preregistered {expected} "
            f"(n_rows={len(values)}, L={context_length}, H={horizon}, stride={stride})"
        )

    windows: list[OriginWindow] = []
    for origin_id, start in enumerate(starts):
        boundary = start + context_length
        end = boundary + horizon
        target = np.array(values[boundary:end], dtype=np.float32, copy=True)
        target.flags.writeable = False  # structural: the target cannot be mutated
        windows.append(
            OriginWindow(
                origin_id=origin_id,
                context_start=start,
                context_end_exclusive=boundary,
                target_start=boundary,
                target_end_exclusive=end,
                origin_timestamp=timestamps[boundary],
                context=np.array(values[start:boundary], dtype=np.float32, copy=True),
                target=target,
                target_timestamps=timestamps[boundary:end],
            )
        )
    return windows
