"""CLI entry point for the statistical analysis of a completed pilot run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from stats.analyze import analyse  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a completed pilot run.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--run-dir", default="results/pilot_v1")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    payload = analyse(REPO_ROOT / args.run_dir, config)

    print(json.dumps({
        "mean_clean_mae": payload["mean_clean_mae"],
        "n_origins": payload["n_origins"],
        "n_failed_forecasts": payload["n_failed_forecasts"],
        "decision": payload["decision"]["label"],
        "triggers": payload["decision"]["triggers"],
    }, indent=2))
    for row in payload["contrasts"]:
        print(
            f"\n{row['contrast_id']} [{row['kind']}] {row['label']}\n"
            f"  raw MAE diff      : {row['raw_mae_difference']:+.6f}"
            f" ({row['pct_of_mean_clean_mae']:+.2f}% of mean clean MAE)\n"
            f"  Holm 95% CI       : [{row['holm_ci95_low']:+.6f}, {row['holm_ci95_high']:+.6f}]"
            f"  adj p={row['holm_adjusted_p']:.4g} rejects={row['holm_rejects']}\n"
            f"  90% CI (uncorr.)  : [{row['uncorrected_ci90_low']:+.6f}, {row['uncorrected_ci90_high']:+.6f}]\n"
            f"  median paired diff: {row['median_paired_difference']:+.6f}"
            f"   frac origins > 0: {row['frac_origins_positive']:.3f}\n"
            f"  block-length sens.: "
            + ", ".join(
                f"b={b}:{row[f'estimate_block_len_{b}']:+.6f}"
                for b in (4, 8, 12) if f"estimate_block_len_{b}" in row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
