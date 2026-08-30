"""Per origin x condition forecast error metrics.

MSE is retained alongside RMSE because RMSE does not aggregate linearly:
pooling across origins requires averaging MSE and taking the square root once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PointwiseMetrics:
    mae: float
    mse: float
    rmse: float
    mase: float | None
    mase_denominator: float | None
    n_steps: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def seasonal_naive_denominator(
    clean_context: np.ndarray, *, seasonal_period: int, min_denominator: float
) -> float | None:
    """MASE denominator computed EXCLUSIVELY from the clean historical context.

    Secondary metric only: MASE is documented, zero-guarded, and explicitly
    excluded from the GO/PIVOT/NO-GO decision logic.
    """
    series = np.asarray(clean_context, dtype=np.float64)
    if series.size <= seasonal_period:
        return None
    diffs = np.abs(series[seasonal_period:] - series[:-seasonal_period])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return None
    denominator = float(diffs.mean())
    if not np.isfinite(denominator) or denominator < min_denominator:
        return None  # zero-denominator guard: MASE is reported as null, never inf
    return denominator


def compute_metrics(
    *,
    truth: np.ndarray,
    prediction: np.ndarray,
    mase_denominator: float | None = None,
) -> PointwiseMetrics:
    """Errors of one forecast against its ground truth."""
    y = np.asarray(truth, dtype=np.float64)
    yhat = np.asarray(prediction, dtype=np.float64)
    if y.shape != yhat.shape:
        raise ValueError(f"shape mismatch: truth {y.shape} vs prediction {yhat.shape}")
    if y.size == 0:
        raise ValueError("empty forecast window")
    if not np.all(np.isfinite(y)):
        # The scored target must never contain a missing value.
        raise ValueError("non-finite value in the scored ground-truth target")

    errors = yhat - y
    abs_errors = np.abs(errors)
    mae = float(abs_errors.mean())
    mse = float(np.square(errors).mean())
    rmse = float(np.sqrt(mse))
    mase = float(mae / mase_denominator) if mase_denominator else None
    return PointwiseMetrics(
        mae=mae, mse=mse, rmse=rmse, mase=mase,
        mase_denominator=mase_denominator, n_steps=int(y.size),
    )
