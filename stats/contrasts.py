"""The four preregistered contrasts. No other comparison feeds the decision rule.

Contrasts 1-2 (family "location") compare two block conditions at the same rate
and are within-pattern.

Contrasts 3-4 (family "geometry") are PIPELINE COMPARISONS, not pure causal
estimates of contiguity: the random and block arms differ in more than
contiguity alone (position of the removed observations, patch occupancy, and
which specific observations are removed all vary together). This label travels
with the contrast into every output row and every figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from metrics.aggregate import seed_averaged_random

PIPELINE_COMPARISON_NOTE = (
    "PIPELINE COMPARISON, not a pure causal estimate of contiguity: the random and "
    "block arms differ in position, patch occupancy and which observations are "
    "removed, in addition to contiguity."
)


@dataclass(frozen=True)
class ContrastSeries:
    contrast_id: str
    family: str
    label: str
    kind: str
    rate: float
    minuend_label: str
    subtrahend_label: str
    paired_differences: pd.Series   # indexed by origin_id, chronological
    minuend_values: pd.Series
    subtrahend_values: pd.Series

    @property
    def is_pipeline_comparison(self) -> bool:
        return self.kind == "pipeline_comparison"

    @property
    def interpretation_note(self) -> str:
        return PIPELINE_COMPARISON_NOTE if self.is_pipeline_comparison else ""


def _arm_series(
    matrix: pd.DataFrame, arm: dict[str, Any], *, rate: float, seeds: list[int]
) -> tuple[pd.Series, str]:
    rate_tag = f"{int(round(rate * 100))}"
    pattern = arm["pattern"]
    if pattern == "random_seed_mean":
        return seed_averaged_random(matrix, rate_tag=rate_tag, seeds=seeds), (
            f"random_r{rate_tag}_seedmean"
        )
    if pattern == "block":
        column = f"block_r{rate_tag}_d{arm['distance']}"
        if column not in matrix.columns:
            raise KeyError(f"condition column {column} absent")
        return matrix[column], column
    raise ValueError(f"unknown contrast arm pattern {pattern!r}")


def build_contrasts(*, matrix: pd.DataFrame, config: dict[str, Any]) -> list[ContrastSeries]:
    """Build the four preregistered paired-difference series from the MAE matrix."""
    seeds = [int(s) for s in config["masks"]["point_random_seeds"]]
    series: list[ContrastSeries] = []
    for spec in config["contrasts"]:
        rate = float(spec["rate"])
        minuend, minuend_label = _arm_series(matrix, spec["minuend"], rate=rate, seeds=seeds)
        subtrahend, subtrahend_label = _arm_series(matrix, spec["subtrahend"], rate=rate, seeds=seeds)
        series.append(
            ContrastSeries(
                contrast_id=spec["id"],
                family=spec["family"],
                label=spec["label"],
                kind=spec["kind"],
                rate=rate,
                minuend_label=minuend_label,
                subtrahend_label=subtrahend_label,
                paired_differences=(minuend - subtrahend).sort_index(),
                minuend_values=minuend.sort_index(),
                subtrahend_values=subtrahend.sort_index(),
            )
        )
    if len(series) != 4:
        raise AssertionError(f"expected exactly 4 preregistered contrasts, built {len(series)}")
    return series
