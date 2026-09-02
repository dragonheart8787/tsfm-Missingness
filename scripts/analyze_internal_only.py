"""CLI for the POST-HOC EXPLORATORY internal-only block-proximity analysis.

Reads an already-completed run's CSVs. No model inference, no GPU, no new data.

    *** POST-HOC AND EXPLORATORY. Not part of the original preregistration. ***
    *** Feeds no GO/PIVOT/NO-GO decision function. ***
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from stats.internal_only import analyse_internal_only  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-hoc exploratory internal-only block-proximity analysis."
    )
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--run-dir", default="results/pilot_v1")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to <run-dir>/post_hoc_internal_only_analysis/.",
    )
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    run_dir = REPO_ROOT / args.run_dir
    out_dir = REPO_ROOT / args.out_dir if args.out_dir else None
    payload = analyse_internal_only(run_dir, config, out_dir=out_dir)

    print("=" * 78)
    print("POST-HOC EXPLORATORY ANALYSIS - not preregistered, feeds no decision rule")
    print("=" * 78)
    print(f"mean clean MAE: {payload['mean_clean_mae']:.6f}   origins: {payload['n_origins']}")
    print(f"internal positions: {payload['internal_positions']}")

    print("\n--- 1. PRIMARY: internal-near vs internal-far "
          "(effective forecast distance held constant) ---")
    print("    Holm family: {IC1_20, IC2_40}, separate from the original {C1..C4}.")
    for row in payload["primary_contrasts"]:
        print(
            f"\n{row['contrast_id']}  {row['label']}\n"
            f"  raw MAE diff      : {row['raw_mae_difference']:+.6f}"
            f" ({row['pct_of_mean_clean_mae']:+.2f}% of mean clean MAE)\n"
            f"  Holm 95% CI       : [{row['holm_ci95_low']:+.6f}, {row['holm_ci95_high']:+.6f}]"
            f"  adj p={row['holm_adjusted_p']:.4g}  rejects={row['holm_rejects']}\n"
            f"  90% CI (uncorr.)  : [{row['uncorrected_ci90_low']:+.6f}, "
            f"{row['uncorrected_ci90_high']:+.6f}]\n"
            f"  median paired diff: {row['median_paired_difference']:+.6f}"
            f"   frac origins > 0: {row['frac_origins_positive']:.3f}\n"
            f"  block-length sens.: "
            + ", ".join(
                f"b={b}:{row[f'estimate_block_len_{b}']:+.6f}"
                for b in (4, 8, 12) if f"estimate_block_len_{b}" in row
            )
        )

    print("\n--- 2a. SECONDARY: pairwise internal contrasts (diagnostic only) ---")
    print("    Holm applied within each rate's 6-contrast family, independently.")
    for row in payload["pairwise_contrasts"]:
        print(
            f"  {row['contrast_id']:<26} {row['raw_mae_difference']:+.6f} "
            f"({row['pct_of_mean_clean_mae']:+6.2f}%)  "
            f"Holm CI [{row['holm_ci95_low']:+.5f},{row['holm_ci95_high']:+.5f}]  "
            f"adj p={row['holm_adjusted_p']:.4g}  rejects={row['holm_rejects']}"
        )

    print("\n--- 2b. SECONDARY: ordered trend slope across the four internal positions ---")
    for row in payload["trend_slopes"]:
        print(
            f"  rate {row['rate']:.2f}  d={row['internal_distances']}\n"
            f"    mean slope : {row['mean_slope_mae_per_step']:+.8f} MAE per step of d\n"
            f"    95% CI     : [{row['ci95_low']:+.8f}, {row['ci95_high']:+.8f}]"
            f"   excludes zero: {row['ci95_excludes_zero']}\n"
            f"    median     : {row['median_slope']:+.8f}"
            f"   frac origins with positive slope: {row['frac_origins_positive_slope']:.3f}\n"
            f"    block-len sensitivity (excludes zero): "
            + ", ".join(
                f"b={b}:{row[f'slope_ci95_excludes_zero_block_len_{b}']}"
                for b in (4, 8, 12)
                if f"slope_ci95_excludes_zero_block_len_{b}" in row
            )
        )

    print("\n--- 3. TRAILING-BOUNDARY CONDITION (d=0) - reported separately, NEVER pooled ---")
    for row in payload["trailing_boundary"]:
        print(
            f"  rate {row['rate']:.2f}  {row['condition_id']}  "
            f"(nearest real observation {row['nearest_real_observation_steps_back']} steps back)\n"
            f"    mean paired diff from clean: {row['mean_paired_diff_from_clean']:+.6f} "
            f"({row['pct_of_mean_clean_mae']:+.2f}% of mean clean MAE)\n"
            f"    median: {row['median_paired_diff_from_clean']:+.6f}   "
            f"95% CI [{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}]   "
            f"frac > 0: {row['frac_origins_positive']:.3f}"
        )

    print("\n--- 4. Confounder-difference correlations rho(dMAE, dz), primary contrasts ---")
    print("    Holm within each contrast's pre-specified 4-item family.")
    print(f"  {'contrast':<12} {'confounder':<42} {'status':<14} {'rho':>8} {'adj p':>10}")
    for row in payload["confounder_difference_correlations"]:
        rho = f"{row['rho']:+.4f}" if row["rho"] == row["rho"] else "n/a"
        adj = f"{row['holm_adjusted_p']:.4g}" if row["holm_adjusted_p"] == row["holm_adjusted_p"] else "n/a"
        print(
            f"  {row['contrast_id']:<12} {row['confounder']:<42} "
            f"{row['status']:<14} {rho:>8} {adj:>10}   {row['reason']}"
        )

    print("\nInterpretation is deliberately NOT offered here: these are numbers for the "
          "Research Lead to weigh, not a decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
