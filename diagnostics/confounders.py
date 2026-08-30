"""Per-mask confounder diagnostics, computed for EVERY mask at generation time.

These exist to support the report's confounder-explanation section. No severity
metric is fitted from them in this pilot — any weighting parameter is out of
scope pending Research Lead audit.

Like everything in the masking path, this function receives only the CLEAN
CONTEXT and a mask; it never receives the forecast target.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    if flags.size == 0 or not flags.any():
        return []
    padded = np.concatenate(([False], flags, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2].tolist(), changes[1::2].tolist()))


def _slope(values: np.ndarray) -> float:
    """OLS slope per step; NaN when fewer than two finite points."""
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return float("nan")
    x = np.flatnonzero(finite).astype(np.float64)
    y = values[finite].astype(np.float64)
    return float(np.polyfit(x, y, 1)[0])


def mask_diagnostics(
    *,
    clean_context: np.ndarray,
    mask,
    context_timestamps: pd.DatetimeIndex,
    trailing_windows: list[int],
    patch_size: int,
    seasonal_period: int = 24,
) -> dict[str, Any]:
    """Compute the full diagnostic row for one mask."""
    context = np.asarray(clean_context, dtype=np.float64)
    length = context.shape[0]
    flags = mask.boolean()
    missing_idx = mask.missing_indices
    n_missing = int(flags.sum())

    runs = _runs(flags)
    run_lengths = [end - start for start, end in runs]
    # A run "touches the context end" if it is flush against the forecast boundary.
    trailing_run_length = 0
    for start, end in runs:
        if end == length:
            trailing_run_length = end - start

    if n_missing:
        # Distance measured from each missing position to the forecast boundary,
        # where the last context step (index length-1) sits at distance 1.
        distances = (length - missing_idx).astype(np.float64)
        nearest_missing_to_boundary = float(distances.min())
        mean_distance_to_boundary = float(distances.mean())
    else:
        nearest_missing_to_boundary = float("nan")
        mean_distance_to_boundary = float("nan")

    diagnostics: dict[str, Any] = {
        "missing_count": n_missing,
        "missing_rate_realised": float(n_missing / length),
        "n_missing_runs": len(runs),
        "longest_missing_run": int(max(run_lengths)) if run_lengths else 0,
        "trailing_run_length": int(trailing_run_length),
        "nearest_missing_to_boundary": nearest_missing_to_boundary,
        "mean_missing_distance_to_boundary": mean_distance_to_boundary,
    }

    # Missing fraction in the final k context steps.
    for window in trailing_windows:
        tail = flags[-window:] if window <= length else flags
        diagnostics[f"missing_frac_last_{window}"] = float(tail.mean())

    # Patch occupancy on the model's real grid: fully vs. partially missing patches.
    n_patches = length // patch_size
    usable = n_patches * patch_size
    patched = flags[:usable].reshape(n_patches, patch_size)
    per_patch_missing = patched.sum(axis=1)
    diagnostics["n_patches"] = int(n_patches)
    diagnostics["n_patches_fully_missing"] = int((per_patch_missing == patch_size).sum())
    diagnostics["n_patches_partially_missing"] = int(
        ((per_patch_missing > 0) & (per_patch_missing < patch_size)).sum()
    )
    diagnostics["n_patches_any_missing"] = int((per_patch_missing > 0).sum())

    # Retained-context statistics and their change vs. the clean context. These
    # are the direct handle on Chronos-2's NaN-aware instance normalisation,
    # which recomputes loc/scale from the RETAINED observations only.
    retained = context[~flags]
    clean_mean = float(np.mean(context))
    clean_std = float(np.std(context))
    if retained.size:
        retained_mean = float(np.mean(retained))
        retained_std = float(np.std(retained))
    else:
        retained_mean = float("nan")
        retained_std = float("nan")
    diagnostics.update(
        {
            "clean_context_mean": clean_mean,
            "clean_context_std": clean_std,
            "retained_context_mean": retained_mean,
            "retained_context_std": retained_std,
            "retained_mean_change": retained_mean - clean_mean,
            "retained_std_change": retained_std - clean_std,
            "retained_mean_change_frac_of_clean_std": (
                (retained_mean - clean_mean) / clean_std if clean_std else float("nan")
            ),
            "retained_std_ratio": (retained_std / clean_std if clean_std else float("nan")),
        }
    )

    # Statistics of the REMOVED observations themselves.
    removed = context[flags]
    if removed.size:
        diagnostics.update(
            {
                "removed_mean": float(np.mean(removed)),
                "removed_variance": float(np.var(removed)),
                "removed_min": float(np.min(removed)),
                "removed_max": float(np.max(removed)),
                # Slope of the removed segment in its original positions, so a
                # scattered point-random mask and a block are comparable.
                "removed_slope": _slope(np.where(flags, context, np.nan)),
                "removed_abs_diff_mean": (
                    float(np.mean(np.abs(np.diff(removed)))) if removed.size > 1 else float("nan")
                ),
            }
        )
    else:
        diagnostics.update(
            {
                "removed_mean": float("nan"),
                "removed_variance": float("nan"),
                "removed_min": float("nan"),
                "removed_max": float("nan"),
                "removed_slope": float("nan"),
                "removed_abs_diff_mean": float("nan"),
            }
        )

    # Seasonal (hour-of-day) composition of the removed observations. For a
    # contiguous block whose length is a multiple of the period this is uniform
    # by construction, which is itself worth recording.
    hours = context_timestamps.hour.to_numpy()
    if n_missing:
        removed_hours = hours[flags]
        counts = np.bincount(removed_hours % seasonal_period, minlength=seasonal_period)
        share = counts / counts.sum()
        # Normalised entropy: 1.0 = removed observations spread evenly over the
        # daily cycle, lower = concentrated in particular hours.
        nonzero = share[share > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(seasonal_period))
        diagnostics["removed_hour_entropy_normalised"] = entropy
        diagnostics["removed_hour_max_share"] = float(share.max())
    else:
        diagnostics["removed_hour_entropy_normalised"] = float("nan")
        diagnostics["removed_hour_max_share"] = float("nan")

    return diagnostics
