"""Aggregation across evaluation origins, and relative error degradation (RED).

RED is computed from AGGREGATE errors:

    RED = (mean_error_missing - mean_error_clean) / mean_error_clean

never as the mean of per-window ratios, which explodes when a window's clean
error is near zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_mae(per_origin_mae: np.ndarray) -> float:
    return float(np.mean(np.asarray(per_origin_mae, dtype=np.float64)))


def aggregate_rmse(per_origin_mse: np.ndarray) -> float:
    """Pool MSE across origins, then take the root once."""
    return float(np.sqrt(np.mean(np.asarray(per_origin_mse, dtype=np.float64))))


def relative_error_degradation(mean_error_missing: float, mean_error_clean: float) -> float:
    if mean_error_clean == 0:
        return float("nan")
    return (float(mean_error_missing) - float(mean_error_clean)) / float(mean_error_clean)


def pivot_condition_matrix(window_results: pd.DataFrame, *, value: str = "mae") -> pd.DataFrame:
    """origins (rows, chronological) x condition_id (columns) matrix of a metric.

    This is the object the bootstrap resamples: resampling ROWS resamples every
    condition jointly and so preserves the pairing by construction.
    """
    ok = window_results[window_results["status"] == "ok"]
    matrix = ok.pivot(index="origin_id", columns="condition_id", values=value)
    return matrix.sort_index()


def seed_averaged_random(matrix: pd.DataFrame, *, rate_tag: str, seeds: list[int]) -> pd.Series:
    """Average the three point-random seeds WITHIN each origin, before any contrast.

    The seeds are repeated measurements of one origin, never independent samples.
    """
    columns = [f"random_r{rate_tag}_s{seed}" for seed in seeds]
    missing = [c for c in columns if c not in matrix.columns]
    if missing:
        raise KeyError(f"missing random-seed columns: {missing}")
    return matrix[columns].mean(axis=1)


def condition_summary(window_results: pd.DataFrame) -> pd.DataFrame:
    """Per-condition aggregate errors and RED against the clean condition."""
    matrix_mae = pivot_condition_matrix(window_results, value="mae")
    matrix_mse = pivot_condition_matrix(window_results, value="mse")
    clean_mae = aggregate_mae(matrix_mae["clean"].to_numpy())
    clean_mse_pooled = aggregate_rmse(matrix_mse["clean"].to_numpy())

    rows = []
    meta = (
        window_results[["condition_id", "pattern", "rate", "seed", "distance_to_boundary"]]
        .drop_duplicates("condition_id")
        .set_index("condition_id")
    )
    for condition in matrix_mae.columns:
        mae_values = matrix_mae[condition].to_numpy()
        mse_values = matrix_mse[condition].to_numpy()
        paired_diff = mae_values - matrix_mae["clean"].to_numpy()
        info = meta.loc[condition]
        rows.append(
            {
                "condition_id": condition,
                "pattern": info["pattern"],
                "rate": info["rate"],
                "seed": info["seed"],
                "distance_to_boundary": info["distance_to_boundary"],
                "n_origins": int(len(mae_values)),
                "mean_mae": aggregate_mae(mae_values),
                "pooled_rmse": aggregate_rmse(mse_values),
                "red_mae": relative_error_degradation(aggregate_mae(mae_values), clean_mae),
                "red_rmse": relative_error_degradation(aggregate_rmse(mse_values), clean_mse_pooled),
                "mean_abs_paired_diff_from_clean": float(np.mean(np.abs(paired_diff))),
                "mean_paired_diff_from_clean": float(np.mean(paired_diff)),
            }
        )
    return pd.DataFrame(rows).sort_values("condition_id").reset_index(drop=True)
