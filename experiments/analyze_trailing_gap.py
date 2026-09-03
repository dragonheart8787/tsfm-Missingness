"""Analysis for the trailing-gap experiment: R_g, B_g, dose-response, classification.

Mirrors ``scripts/analyze_run.py``'s CLI shape. Reuses the existing moving-block
bootstrap and Holm implementation; nothing statistical is reimplemented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.aggregate import aggregate_mae  # noqa: E402
from stats.bootstrap import bootstrap_paired_difference, holm_correct  # noqa: E402
from stats.mechanism_decision import (  # noqa: E402
    INTERPRETATION,
    DoseResponse,
    GapStat,
    MechanismInputs,
    classify,
    sesoi_delta,
)


def per_origin_slopes(diffs: pd.DataFrame, *, x: np.ndarray) -> pd.Series:
    """OLS slope of R_g against x (MISSING PATCHES), per origin."""
    x_centred = x - x.mean()
    denominator = float(np.sum(x_centred**2))
    values = diffs.to_numpy(dtype=np.float64)
    y_centred = values - values.mean(axis=1, keepdims=True)
    return pd.Series((y_centred @ x_centred) / denominator, index=diffs.index)


def analyse(run_dir: Path, config: dict[str, Any], gap_config: dict[str, Any],
            *, out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = (run_dir / "analysis") if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(run_dir / "window_results.csv")
    ok = frame[frame["status"] == "ok"]
    matrix = ok.pivot(index="origin_id", columns="condition_id", values="mae").sort_index()
    if matrix.isna().any().any():
        raise ValueError("the MAE matrix has holes; cannot analyse")

    mean_clean_mae = aggregate_mae(matrix["clean"].to_numpy())
    delta = sesoi_delta(config, mean_clean_mae)

    boot = config["statistics"]["bootstrap"]
    all_blocks = [int(boot["main_block_length"])] + [
        int(b) for b in boot["sensitivity_block_lengths"]
    ]
    main_block = int(boot["main_block_length"])
    n_replicates = int(boot["n_replicates"])
    base_seed = int(boot["seed"]) + 91_000          # isolated stream
    alpha = float(config["statistics"]["family_alpha"])
    gaps = [int(g) for g in gap_config["design"]["gap_lengths"]]
    patch = int(config["model"]["verified_contract"]["input_patch_size"])

    def family(series: dict[str, np.ndarray], *, offset: int):
        per_block = {}
        for block_length in all_blocks:
            results, draws = [], {}
            for i, (cid, values) in enumerate(series.items()):
                result, means = bootstrap_paired_difference(
                    contrast_id=cid, paired_differences=values,
                    block_length=block_length, n_replicates=n_replicates,
                    seed=base_seed + offset + 1000 * block_length + i,
                    ci_level_corrected=float(config["statistics"]["ci_level_corrected"]),
                    ci_level_equivalence=float(config["statistics"]["ci_level_equivalence"]),
                    joint_matrix=matrix.to_numpy(),
                )
                results.append(result)
                draws[cid] = means
            holm_correct(results, draws, family_alpha=alpha)
            per_block[block_length] = {r.contrast_id: r for r in results}
        return per_block

    # R_g and B_g: two independent Holm families of 4.
    r_series = {
        f"R_{g}": (matrix[f"trailing_nan_g{g}"] - matrix[f"truncated_long_g{g}"]).to_numpy()
        for g in gaps
    }
    b_series = {
        f"B_{g}": (matrix[f"trailing_nan_g{g}"] - matrix[f"internal_block_g{g}"]).to_numpy()
        for g in gaps
    }
    r_blocks = family(r_series, offset=0)
    b_blocks = family(b_series, offset=500)

    def to_stats(blocks, prefix):
        out = []
        for g in gaps:
            r = blocks[main_block][f"{prefix}_{g}"]
            out.append(
                GapStat(
                    contrast_id=f"{prefix}_{g}", gap=g,
                    mean_difference=r.observed_mean, median_difference=r.observed_median,
                    ci_low_holm=float(r.ci_low_holm), ci_high_holm=float(r.ci_high_holm),
                    ci_low_equivalence=r.ci_low_equivalence,
                    ci_high_equivalence=r.ci_high_equivalence,
                    holm_rejects=bool(r.holm_rejects), mean_clean_mae=mean_clean_mae,
                )
            )
        return out

    r_stats, b_stats = to_stats(r_blocks, "R"), to_stats(b_blocks, "B")

    # Dose-response against x = g / patch_size (missing patches).
    x = np.array([g / patch for g in gaps], dtype=np.float64)
    r_frame = pd.DataFrame({f"R_{g}": r_series[f"R_{g}"] for g in gaps}, index=matrix.index)
    slopes = per_origin_slopes(r_frame, x=x)
    slope_blocks = family({"SLOPE": slopes.to_numpy()}, offset=900)
    slope_main = slope_blocks[main_block]["SLOPE"]
    # Descriptive-only curvature: per-origin quadratic coefficient.
    quad = np.polyfit(x, r_frame.to_numpy().T, 2)[0]
    dose = DoseResponse(
        mean_slope_per_patch=slope_main.observed_mean,
        ci95_low=slope_main.ci_low_corrected, ci95_high=slope_main.ci_high_corrected,
        ci95_excludes_zero=bool(
            slope_main.ci_low_corrected > 0 or slope_main.ci_high_corrected < 0
        ),
        quadratic_term=float(np.mean(quad)),
        quadratic_ci95_low=float(np.percentile(quad, 2.5)),
        quadratic_ci95_high=float(np.percentile(quad, 97.5)),
    )

    decision = classify(
        MechanismInputs(
            r_gaps=r_stats, b_gaps=b_stats, dose_response=dose,
            mean_clean_mae=mean_clean_mae, config=gap_config, pilot_config=config,
        )
    )

    def rows(stats, blocks, prefix):
        return [
            {
                "contrast_id": s.contrast_id, "family": prefix, "gap": s.gap,
                "reading": s.reading(delta), "flags": ";".join(s.flags(delta)),
                "interpretation": INTERPRETATION[s.reading(delta)],
                "raw_mae_difference": s.mean_difference,
                "pct_of_mean_clean_mae": 100.0 * s.mean_difference / mean_clean_mae,
                "median_paired_difference": s.median_difference,
                "holm_ci95_low": s.ci_low_holm, "holm_ci95_high": s.ci_high_holm,
                "holm_adjusted_p": blocks[main_block][s.contrast_id].holm_adjusted_p,
                "holm_rejects": s.holm_rejects,
                "ci90_low": s.ci_low_equivalence, "ci90_high": s.ci_high_equivalence,
                "frac_origins_positive": blocks[main_block][s.contrast_id].frac_positive,
                "sesoi_delta": delta,
                **{
                    f"estimate_block_len_{b}": blocks[b][s.contrast_id].observed_mean
                    for b in all_blocks
                },
                **{
                    f"holm_ci95_low_block_len_{b}": blocks[b][s.contrast_id].ci_low_holm
                    for b in all_blocks
                },
                **{
                    f"holm_ci95_high_block_len_{b}": blocks[b][s.contrast_id].ci_high_holm
                    for b in all_blocks
                },
            }
            for s in stats
        ]

    r_rows, b_rows = rows(r_stats, r_blocks, "R"), rows(b_stats, b_blocks, "B")
    payload = {
        "experiment": gap_config["meta"]["name"],
        "mean_clean_mae": mean_clean_mae,
        "sesoi_delta": delta,
        "n_origins": int(matrix.shape[0]),
        "n_conditions": int(matrix.shape[1]),
        "n_failed": int((frame["status"] != "ok").sum()),
        "r_contrasts": r_rows,
        "b_contrasts": b_rows,
        "dose_response": dose.as_dict(),
        "classification": decision.as_dict(),
    }

    pd.DataFrame(r_rows).to_csv(out_dir / "R_contrasts.csv", index=False)
    pd.DataFrame(b_rows).to_csv(out_dir / "B_contrasts.csv", index=False)
    r_frame.to_csv(out_dir / "R_paired_differences.csv")
    slopes.rename("slope_per_missing_patch").to_csv(out_dir / "per_origin_slopes.csv")
    (out_dir / "dose_response.json").write_text(
        json.dumps(dose.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "classification.json").write_text(
        json.dumps(decision.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse the trailing-gap experiment.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--gap-config", default="configs/trailing_gap_config.yaml")
    parser.add_argument("--run-dir", default="results/trailing_gap_v1")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    gap_config = yaml.safe_load((REPO_ROOT / args.gap_config).read_text(encoding="utf-8"))
    payload = analyse(
        REPO_ROOT / args.run_dir, config, gap_config,
        out_dir=REPO_ROOT / args.out_dir if args.out_dir else None,
    )

    print(f"mean clean MAE: {payload['mean_clean_mae']:.6f}   "
          f"SESOI delta: +/-{payload['sesoi_delta']:.6f}")
    print("\n--- R_g (primary evidence) ---")
    for row in payload["r_contrasts"]:
        print(
            f"  {row['contrast_id']:<7} {row['raw_mae_difference']:+.6f} "
            f"({row['pct_of_mean_clean_mae']:+6.2f}%)  Holm CI "
            f"[{row['holm_ci95_low']:+.5f},{row['holm_ci95_high']:+.5f}]  "
            f"90% [{row['ci90_low']:+.5f},{row['ci90_high']:+.5f}]  -> {row['reading']}"
            + (f"  [{row['flags']}]" if row["flags"] else "")
        )
    print("\n--- B_g (reported alongside; never determines the classification) ---")
    for row in payload["b_contrasts"]:
        print(f"  {row['contrast_id']:<7} {row['raw_mae_difference']:+.6f} -> {row['reading']}")
    dose = payload["dose_response"]
    print(f"\n--- Dose-response (sub-question only; x = {dose['x_units']}) ---")
    print(f"  slope {dose['mean_slope_per_patch']:+.6f} per patch  "
          f"95% CI [{dose['ci95_low']:+.6f}, {dose['ci95_high']:+.6f}]  "
          f"excludes zero: {dose['ci95_excludes_zero']}")
    print(f"  {dose['non_significant_means']}")
    cls = payload["classification"]
    print(f"\n--- CLASSIFICATION ({cls['rule_version']}) ---")
    print(f"  {cls['label']}" + (f"  reason={cls['reason']}" if cls["reason"] else ""))
    print(f"  {cls['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
