"""Audit-oriented figures. No dramatic framing; uncertainty is always shown.

Every panel displays confidence intervals and/or per-origin variability. Axes
are not truncated to exaggerate differences, and the geometry contrasts carry
their pipeline-comparison caveat in the title.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def fig_mae_by_block_distance(matrix, config, out: Path) -> None:
    """Clean-vs-corrupted MAE by block distance, with per-origin spread."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    clean = matrix["clean"].to_numpy()
    clean_mean = clean.mean()

    for ax, spec in zip(axes, config["masks"]["rates"]):
        rate_tag = f"{int(round(float(spec['rate']) * 100))}"
        distances = [int(d) for d in spec["block_distances"]]
        means, los, his = [], [], []
        for d in distances:
            values = matrix[f"block_r{rate_tag}_d{d}"].to_numpy()
            means.append(values.mean())
            # Per-origin variability, shown as the interquartile range.
            los.append(np.percentile(values, 25))
            his.append(np.percentile(values, 75))
            ax.scatter(
                np.full(values.size, d) + np.random.default_rng(d).normal(scale=3, size=values.size),
                values, s=3, alpha=0.12, color="tab:blue", zorder=1,
            )
        ax.errorbar(
            distances, means,
            yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
            fmt="o-", capsize=4, color="tab:blue", zorder=3, label="block mean (bars: per-origin IQR)",
        )
        ax.axhline(clean_mean, color="tab:green", ls="--", lw=1.5, zorder=2,
                   label=f"clean mean MAE = {clean_mean:.3f}")
        ax.set_title(f"{rate_tag}% missing — block length {spec['block_length']}")
        ax.set_xlabel("d  (distance from block end to forecast boundary)")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("MAE")
    fig.suptitle("MAE by contiguous-block distance, against the clean baseline", y=1.0)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_paired_difference_distributions(paired, contrasts, out: Path) -> None:
    """Per-origin paired difference distributions for the four contrasts."""
    ids = [c["contrast_id"] for c in contrasts]
    fig, axes = plt.subplots(1, len(ids), figsize=(4 * len(ids), 3.8), sharey=False)
    for ax, cid in zip(np.atleast_1d(axes), ids):
        row = next(c for c in contrasts if c["contrast_id"] == cid)
        values = paired[cid].to_numpy()
        ax.hist(values, bins=30, color="tab:blue", alpha=0.65, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        ax.axvline(values.mean(), color="tab:red", lw=1.6, label=f"mean {values.mean():+.4f}")
        ax.axvline(np.median(values), color="tab:orange", lw=1.6, ls="--",
                   label=f"median {np.median(values):+.4f}")
        ax.axvspan(row["holm_ci95_low"], row["holm_ci95_high"], color="tab:red", alpha=0.12,
                   label="Holm 95% CI (of the mean)")
        caveat = "\nPIPELINE COMPARISON" if row["is_pipeline_comparison"] else ""
        ax.set_title(f"{cid}{caveat}\nfrac origins > 0: {row['frac_origins_positive']:.3f}", fontsize=8)
        ax.set_xlabel("paired MAE difference")
        ax.legend(fontsize=6.5)
    np.atleast_1d(axes)[0].set_ylabel("origins")
    fig.suptitle("Per-origin paired differences (n = 178 origins, paired throughout)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_random_vs_block(matrix, config, out: Path) -> None:
    """Random (seed-averaged within origin) vs. the centred block, per rate."""
    from metrics.aggregate import seed_averaged_random

    seeds = [int(s) for s in config["masks"]["point_random_seeds"]]
    centred = {"20": 128, "40": 96}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, spec in zip(axes, config["masks"]["rates"]):
        rate_tag = f"{int(round(float(spec['rate']) * 100))}"
        random_arm = seed_averaged_random(matrix, rate_tag=rate_tag, seeds=seeds).to_numpy()
        block_arm = matrix[f"block_r{rate_tag}_d{centred[rate_tag]}"].to_numpy()
        lim = [min(random_arm.min(), block_arm.min()), max(random_arm.max(), block_arm.max())]
        ax.scatter(block_arm, random_arm, s=9, alpha=0.5)
        ax.plot(lim, lim, color="black", lw=1, ls="--", label="parity")
        ax.set_xlabel(f"centred block MAE (d={centred[rate_tag]})")
        ax.set_ylabel("point-random MAE (seed mean within origin)")
        ax.set_title(
            f"{rate_tag}% missing — PIPELINE COMPARISON,\nnot a pure causal estimate of contiguity",
            fontsize=8.5,
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_block_length_sensitivity(bootstrap, out: Path) -> None:
    """Sensitivity of the intervals across bootstrap block lengths 4, 8, 12."""
    ids = sorted(bootstrap["contrast_id"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    lengths = sorted(bootstrap["block_length"].unique())
    offsets = np.linspace(-0.22, 0.22, len(lengths))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(lengths)))
    for offset, block_length, color in zip(offsets, lengths, colors):
        subset = bootstrap[bootstrap["block_length"] == block_length].set_index("contrast_id")
        x = np.arange(len(ids)) + offset
        means = subset.loc[ids, "observed_mean"].to_numpy()
        lo = subset.loc[ids, "ci_low_holm"].to_numpy()
        hi = subset.loc[ids, "ci_high_holm"].to_numpy()
        ax.errorbar(x, means, yerr=[means - lo, hi - means], fmt="o", capsize=4,
                    color=color, label=f"block length {block_length}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_ylabel("mean paired MAE difference")
    ax.set_title(
        "Moving-block bootstrap sensitivity (Holm-corrected 95% intervals).\n"
        "The point estimate is bootstrap-invariant; the interval is what moves.",
        fontsize=9,
    )
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_effect_vs_scaling(paired, diagnostics, contrasts, out: Path) -> None:
    """Effect size against the retained-context scaling change.

    Chronos-2 normalises the context with a NaN-aware instance norm, so removing
    observations moves loc/scale. This panel is the direct look at whether that
    tracks the effect.
    """
    rows = [c for c in contrasts if not c["is_pipeline_comparison"]]
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.0))
    for ax, row in zip(np.atleast_1d(axes), rows):
        arm = diagnostics[diagnostics["condition_id"] == row["minuend"]].set_index("origin_id")
        arm = arm.reindex(paired.index)
        x = arm["retained_mean_change_frac_of_clean_std"].to_numpy()
        y = paired[row["contrast_id"]].to_numpy()
        ax.scatter(x, y, s=9, alpha=0.5)
        ax.axhline(0, color="black", lw=1)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("retained-context mean shift  (in clean-context SD units)")
        ax.set_ylabel("paired MAE difference")
        ax.set_title(
            f"{row['contrast_id']}: effect vs. internal-scaling change\n"
            f"Spearman |rho| = {row['rho_scaling_max_abs']:.3f}  "
            f"(distance rho = {row['rho_distance']:.3f})",
            fontsize=8.5,
        )
    fig.suptitle("Association only — no causal claim is made by this panel.", y=1.02, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the pilot's audit figures.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--run-dir", default="results/pilot_v1")
    parser.add_argument("--out-dir", default="report/figures")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    run_dir = REPO_ROOT / args.run_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(run_dir / "mae_matrix.csv", index_col=0)
    paired = pd.read_csv(run_dir / "paired_differences.csv", index_col=0)
    bootstrap = pd.read_csv(run_dir / "bootstrap_results.csv")
    diagnostics = pd.read_csv(run_dir / "mask_diagnostics.csv")
    contrasts = json.loads((run_dir / "analysis.json").read_text())["contrasts"]

    fig_mae_by_block_distance(matrix, config, out_dir / "fig1_mae_by_block_distance.png")
    fig_paired_difference_distributions(paired, contrasts, out_dir / "fig2_paired_differences.png")
    fig_random_vs_block(matrix, config, out_dir / "fig3_random_vs_centred_block.png")
    fig_block_length_sensitivity(bootstrap, out_dir / "fig4_bootstrap_block_length_sensitivity.png")
    fig_effect_vs_scaling(paired, diagnostics, contrasts, out_dir / "fig5_effect_vs_scaling.png")

    print(f"Wrote 5 figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
