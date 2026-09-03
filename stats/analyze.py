"""Full statistical analysis: contrasts -> bootstrap -> Holm -> decision.

Reads a completed run directory and writes machine-readable analysis outputs
plus the confounder correlations the report's explanation section needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics.aggregate import (
    aggregate_mae,
    condition_summary,
    pivot_condition_matrix,
    seed_averaged_random,
)
from stats.bootstrap import bootstrap_paired_difference, holm_correct
from stats.contrasts import ContrastSeries, build_contrasts
from stats.decision import (
    DECISION_RULE_VERSION,
    ContrastStat,
    Decision,
    DecisionInputs,
    _spearman,
    decide,
    top_origin_share,
)

# Confounder columns correlated against each contrast's paired differences.
CONFOUNDER_COLUMNS = [
    "removed_variance",
    "removed_slope",
    "removed_abs_diff_mean",
    "removed_mean",
    "removed_hour_entropy_normalised",
    "n_patches_fully_missing",
    "n_patches_partially_missing",
    "trailing_run_length",
    "nearest_missing_to_boundary",
    "mean_missing_distance_to_boundary",
    "missing_frac_last_24",
    "missing_frac_last_96",
]
SCALING_COLUMNS = ["retained_mean_change_frac_of_clean_std", "retained_std_ratio"]


def _minuend_diagnostics(
    diagnostics: pd.DataFrame, contrast: ContrastSeries, seeds: list[int]
) -> pd.DataFrame:
    """Per-origin diagnostics of the contrast's minuend arm.

    For the seed-averaged random arm the three seeds' diagnostics are averaged
    within origin, mirroring how their MAEs are averaged.
    """
    rate_tag = f"{int(round(contrast.rate * 100))}"
    if contrast.minuend_label.endswith("seedmean"):
        columns = [f"random_r{rate_tag}_s{seed}" for seed in seeds]
        subset = diagnostics[diagnostics["condition_id"].isin(columns)]
        numeric = subset.select_dtypes(include=[np.number])
        numeric = numeric.assign(origin_id=subset["origin_id"].to_numpy())
        return numeric.groupby("origin_id").mean(numeric_only=True)
    subset = diagnostics[diagnostics["condition_id"] == contrast.minuend_label]
    return subset.set_index("origin_id").select_dtypes(include=[np.number])


def analyse(
    run_dir: Path,
    config: dict[str, Any],
    *,
    out_dir: Path | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Analyse a completed run. Reads from ``run_dir``, writes to ``out_dir``.

    ``out_dir`` defaults to ``run_dir``. Pass a separate directory to perform a
    POST-HOC re-analysis without disturbing the preregistered outputs of the
    original run — those are the record of what the run actually produced under
    the decision rule in force at the time, and must not be rewritten.

    Writing over an existing ``decision.json`` that was produced by a DIFFERENT
    decision rule version is refused unless ``allow_overwrite`` is set.
    """
    out_dir = run_dir if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = out_dir / "decision.json"
    if existing.exists() and not allow_overwrite:
        previous = json.loads(existing.read_text(encoding="utf-8")).get("decision_rule_version", "v1")
        if previous != DECISION_RULE_VERSION:
            raise FileExistsError(
                f"{existing} was produced by decision rule {previous}, but this run uses "
                f"{DECISION_RULE_VERSION}. Overwriting it would destroy the record of what "
                f"the earlier rule produced. Pass an --out-dir pointing somewhere else for a "
                f"post-hoc re-analysis, or --allow-overwrite if you genuinely mean to replace it."
            )

    window_results = pd.read_csv(run_dir / "window_results.csv")
    diagnostics = pd.read_csv(run_dir / "mask_diagnostics.csv")

    stats_cfg = config["statistics"]
    boot_cfg = stats_cfg["bootstrap"]
    seeds = [int(s) for s in config["masks"]["point_random_seeds"]]
    main_block = int(boot_cfg["main_block_length"])
    all_blocks = [main_block] + [int(b) for b in boot_cfg["sensitivity_block_lengths"]]
    n_replicates = int(boot_cfg["n_replicates"])
    base_seed = int(boot_cfg["seed"])

    failed = window_results[window_results["status"] != "ok"]
    matrix = pivot_condition_matrix(window_results, value="mae")
    if matrix.isna().any().any():
        raise ValueError(
            "the MAE matrix has holes: some origin x condition cells are missing or failed. "
            f"{len(failed)} failed rows are recorded in window_results.csv."
        )

    mean_clean_mae = aggregate_mae(matrix["clean"].to_numpy())
    contrasts = build_contrasts(matrix=matrix, config=config)

    # ---- bootstrap at every block length, Holm at the main block length ---- #
    per_block: dict[int, dict[str, Any]] = {}
    for block_length in all_blocks:
        results = []
        draws: dict[str, np.ndarray] = {}
        for offset, contrast in enumerate(contrasts):
            result, replicate_means = bootstrap_paired_difference(
                contrast_id=contrast.contrast_id,
                paired_differences=contrast.paired_differences.to_numpy(),
                block_length=block_length,
                n_replicates=n_replicates,
                # Distinct but deterministic stream per (block length, contrast).
                seed=base_seed + 1000 * block_length + offset,
                ci_level_corrected=float(stats_cfg["ci_level_corrected"]),
                ci_level_equivalence=float(stats_cfg["ci_level_equivalence"]),
                joint_matrix=matrix.to_numpy(),
            )
            results.append(result)
            draws[contrast.contrast_id] = replicate_means
        holm_correct(results, draws, family_alpha=float(stats_cfg["family_alpha"]))
        per_block[block_length] = {r.contrast_id: r for r in results}

    # ---- confounder correlations, per contrast ---------------------------- #
    dominance_frac = float(config["decision"]["pivot"]["origin_dominance_top_frac"])
    contrast_stats: list[ContrastStat] = []
    contrast_rows: list[dict[str, Any]] = []

    for contrast in contrasts:
        main = per_block[main_block][contrast.contrast_id]
        diffs = contrast.paired_differences.to_numpy()

        arm = _minuend_diagnostics(diagnostics, contrast, seeds).reindex(
            contrast.paired_differences.index
        )
        rho_confounders = {
            column: _spearman(diffs, arm[column].to_numpy())
            for column in CONFOUNDER_COLUMNS
            if column in arm.columns
        }
        rho_scaling = max(
            (
                abs(_spearman(diffs, arm[column].to_numpy()))
                for column in SCALING_COLUMNS
                if column in arm.columns
            ),
            default=float("nan"),
        )
        # Distance is only defined for a block minuend; for the pipeline
        # comparisons the block arm sits in the subtrahend, so we correlate
        # against the subtrahend arm's boundary distance instead.
        distance_source = (
            diagnostics[diagnostics["condition_id"] == contrast.subtrahend_label]
            .set_index("origin_id")
            .reindex(contrast.paired_differences.index)
        )
        # Retained so the decision function can tell "distance is constant, so
        # its correlation is undefined" apart from "the correlation was computed
        # and came out small". For the preregistered two-arm contrasts the
        # subtrahend arm is a single fixed condition, so this is constant across
        # all origins and rho_distance is legitimately NaN.
        distance_values = (
            distance_source["mean_missing_distance_to_boundary"].to_numpy(dtype=float).tolist()
            if "mean_missing_distance_to_boundary" in distance_source.columns
            else []
        )
        rho_distance = (
            _spearman(diffs, distance_source["mean_missing_distance_to_boundary"].to_numpy())
            if "mean_missing_distance_to_boundary" in distance_source.columns
            else float("nan")
        )

        # Origin dominance, via the shared implementation.
        top_share = top_origin_share(diffs, top_frac=dominance_frac)

        partial_patches = 0
        for label in (contrast.minuend_label, contrast.subtrahend_label):
            subset = diagnostics[diagnostics["condition_id"] == label]
            if not subset.empty and "n_patches_partially_missing" in subset.columns:
                partial_patches = max(partial_patches, int(subset["n_patches_partially_missing"].max()))

        spec = next(c for c in config["contrasts"] if c["id"] == contrast.contrast_id)
        contrast_stats.append(
            ContrastStat(
                contrast_id=contrast.contrast_id,
                family=contrast.family,
                kind=contrast.kind,
                rate=contrast.rate,
                mean_difference=main.observed_mean,
                median_difference=main.observed_median,
                frac_positive=main.frac_positive,
                holm_rejects=bool(main.holm_rejects),
                ci_low_holm=float(main.ci_low_holm),
                ci_high_holm=float(main.ci_high_holm),
                ci_low_equivalence=main.ci_low_equivalence,
                ci_high_equivalence=main.ci_high_equivalence,
                mean_clean_mae=mean_clean_mae,
                block_length_estimates={
                    b: per_block[b][contrast.contrast_id].observed_mean for b in all_blocks
                },
                block_length_rejects={
                    b: bool(per_block[b][contrast.contrast_id].holm_rejects) for b in all_blocks
                },
                block_length_ci_low={
                    b: float(per_block[b][contrast.contrast_id].ci_low_holm) for b in all_blocks
                },
                block_length_ci_high={
                    b: float(per_block[b][contrast.contrast_id].ci_high_holm) for b in all_blocks
                },
                top_origin_share=top_share,
                rho_distance=rho_distance,
                distance_values=distance_values,
                rho_confounders=rho_confounders,
                rho_scaling=rho_scaling,
                n_partially_missing_patches_max=partial_patches,
            )
        )

        contrast_rows.append(
            {
                "contrast_id": contrast.contrast_id,
                "family": contrast.family,
                "kind": contrast.kind,
                "is_pipeline_comparison": contrast.is_pipeline_comparison,
                "interpretation_note": contrast.interpretation_note,
                "label": spec["label"],
                "rate": contrast.rate,
                "minuend": contrast.minuend_label,
                "subtrahend": contrast.subtrahend_label,
                "n_origins": main.n_origins,
                "raw_mae_difference": main.observed_mean,
                "pct_of_mean_clean_mae": 100.0 * main.observed_mean / mean_clean_mae,
                "median_paired_difference": main.observed_median,
                "frac_origins_positive": main.frac_positive,
                "holm_ci95_low": main.ci_low_holm,
                "holm_ci95_high": main.ci_high_holm,
                "holm_adjusted_p": main.holm_adjusted_p,
                "holm_rejects": main.holm_rejects,
                "uncorrected_ci90_low": main.ci_low_equivalence,
                "uncorrected_ci90_high": main.ci_high_equivalence,
                "uncorrected_ci95_low": main.ci_low_corrected,
                "uncorrected_ci95_high": main.ci_high_corrected,
                "p_value_two_sided": main.p_value_two_sided,
                **{
                    f"estimate_block_len_{b}": per_block[b][contrast.contrast_id].observed_mean
                    for b in all_blocks
                },
                **{
                    f"holm_rejects_block_len_{b}": per_block[b][contrast.contrast_id].holm_rejects
                    for b in all_blocks
                },
                **{
                    f"holm_ci95_low_block_len_{b}": per_block[b][contrast.contrast_id].ci_low_holm
                    for b in all_blocks
                },
                **{
                    f"holm_ci95_high_block_len_{b}": per_block[b][contrast.contrast_id].ci_high_holm
                    for b in all_blocks
                },
                "top_origin_share": top_share,
                "rho_distance": rho_distance,
                "n_unique_finite_distance": int(
                    len({v for v in distance_values if v == v and abs(v) != float("inf")})
                ),
                "rho_scaling_max_abs": rho_scaling,
                **{f"rho_{k}": v for k, v in rho_confounders.items()},
            }
        )

    # ---- location ordering and seed sensitivity, per rate ------------------ #
    location_ordering: dict[float, float] = {}
    seed_sensitivity: dict[float, float] = {}
    for spec in config["masks"]["rates"]:
        rate = float(spec["rate"])
        rate_tag = f"{int(round(rate * 100))}"
        distances = [int(d) for d in spec["block_distances"]]
        mean_by_distance = np.array(
            [aggregate_mae(matrix[f"block_r{rate_tag}_d{d}"].to_numpy()) for d in distances]
        )
        location_ordering[rate] = _spearman(np.array(distances, dtype=float), mean_by_distance)

        seed_means = np.array(
            [aggregate_mae(matrix[f"random_r{rate_tag}_s{s}"].to_numpy()) for s in seeds]
        )
        seed_sensitivity[rate] = float(seed_means.max() - seed_means.min()) / mean_clean_mae

    decision: Decision = decide(
        DecisionInputs(
            contrasts=contrast_stats,
            mean_clean_mae=mean_clean_mae,
            location_ordering_spearman=location_ordering,
            seed_sensitivity_frac=seed_sensitivity,
            config=config,
        )
    )

    summary = condition_summary(window_results)
    contrast_frame = pd.DataFrame(contrast_rows)

    paired_frame = pd.DataFrame(
        {c.contrast_id: c.paired_differences for c in contrasts}
    ).sort_index()
    for spec in config["masks"]["rates"]:
        rate_tag = f"{int(round(float(spec['rate']) * 100))}"
        paired_frame[f"random_r{rate_tag}_seedmean"] = seed_averaged_random(
            matrix, rate_tag=rate_tag, seeds=seeds
        )

    bootstrap_frame = pd.DataFrame(
        [
            {"block_length": b, **per_block[b][cid].as_dict()}
            for b in all_blocks
            for cid in per_block[b]
        ]
    ).drop(columns=["extra"])

    payload: dict[str, Any] = {
        "mean_clean_mae": mean_clean_mae,
        "n_origins": int(matrix.shape[0]),
        "n_conditions": int(matrix.shape[1]),
        "n_failed_forecasts": int(len(failed)),
        "failed_rows": failed[["origin_id", "condition_id", "error_message"]].to_dict("records"),
        "location_ordering_spearman": {str(k): v for k, v in location_ordering.items()},
        "seed_sensitivity_frac_of_clean_mae": {str(k): v for k, v in seed_sensitivity.items()},
        "contrasts": contrast_rows,
        "decision": decision.as_dict(),
        "decision_rule_version": decision.decision_rule_version,
        # Surfaced at the top level as well as inside decision.criteria.pivot,
        # because "this check could not be asked" is the kind of thing a reader
        # must not have to dig for.
        "distance_relative_check_status": {
            name: decision.criteria["pivot"][name]
            for name in (
                "confounder_out_tracks_distance_status",
                "internal_scaling_distance_branch_status",
                "patch_occupancy_out_tracking_distance_status",
            )
        },
        "bootstrap": {
            "block_lengths": all_blocks,
            "main_block_length": main_block,
            "n_replicates": n_replicates,
        },
    }

    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    contrast_frame.to_csv(out_dir / "contrast_results.csv", index=False)
    paired_frame.to_csv(out_dir / "paired_differences.csv")
    bootstrap_frame.to_csv(out_dir / "bootstrap_results.csv", index=False)
    matrix.to_csv(out_dir / "mae_matrix.csv")
    (out_dir / "analysis.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "decision.json").write_text(json.dumps(decision.as_dict(), indent=2, default=str), encoding="utf-8")
    return payload
