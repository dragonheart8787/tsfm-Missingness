"""POST-HOC EXPLORATORY analysis: block proximity among INTERNAL positions only.

    *** THIS ANALYSIS IS POST-HOC AND EXPLORATORY. ***
    *** It was NOT part of the original preregistration. ***
    *** It feeds NO decision function and carries no confirmatory weight. ***

Why it exists
-------------
The preregistered C1/C2 contrasts compare d=0 against a far position. That
conflates two things that move together:

  * block position within the context, and
  * "effective forecast distance" - how far back the model's nearest REAL
    observation sits from the forecast start.

At d=0 the block abuts the forecast boundary, so the nearest real observation is
``block_length + 1`` steps back. At every other preregistered position there is
at least one real observation between the block and the boundary, so the nearest
real observation is always exactly 1 step back, whichever internal position the
block occupies.

Restricting to those internal positions therefore holds effective forecast
distance CONSTANT, isolating block-proximity variation from that confound -
using data already collected. d=0 is not discarded: it is reported separately
and explicitly as a distinct trailing-boundary condition, and is NEVER pooled
into any internal-position aggregate.

Everything here reuses the existing moving-block bootstrap machinery in
``stats/bootstrap.py``; nothing is reimplemented. Rates are analysed separately
and never pooled. All comparisons are paired by evaluation origin.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metrics.aggregate import aggregate_mae, pivot_condition_matrix
from stats.bootstrap import bootstrap_paired_difference, holm_correct, moving_block_indices
from stats.decision import ci_within_equivalence_band, equivalence_band, top_origin_share

ANALYSIS_KIND = "post_hoc_exploratory"
ANALYSIS_NOTE = (
    "POST-HOC EXPLORATORY analysis, not part of the original preregistration. "
    "It feeds no GO/PIVOT/NO-GO decision function and carries no confirmatory weight."
)
TRAILING_BOUNDARY_NOTE = (
    "TRAILING-BOUNDARY CONDITION (d=0), reported separately. The block abuts the "
    "forecast boundary, so the nearest real observation is block_length+1 steps back "
    "rather than 1. It is NEVER pooled into the internal-position statistics."
)

# Pre-specified BEFORE looking at any result, per the pilot's own discipline.
# Do not extend this list after seeing which confounders correlate.
CONFOUNDER_FAMILY: tuple[str, ...] = (
    "removed_variance",
    "removed_slope",
    "removed_abs_diff_mean",
    "retained_mean_change_frac_of_clean_std",
)

# The boundary-jump contrasts: d=0 minus the NEAREST INTERNAL position.
#
# These MUST be bootstrapped directly from the per-origin paired MAE values at
# those two conditions. Subtracting two separately-bootstrapped contrasts
# (C1 - IC1_20) is valid for the POINT ESTIMATE - the arithmetic is linear - but
# NOT for the interval: both contrasts are measured on the same 178 origins, so
# their difference's sampling distribution carries a covariance structure that
# differencing two independently-drawn intervals does not represent.
BOUNDARY_JUMP_NOTE = (
    "BOUNDARY-JUMP CONTRAST: d=0 minus the nearest internal position, bootstrapped "
    "DIRECTLY from per-origin paired values. Not derived by subtracting other "
    "contrasts' outputs, whose intervals cannot be differenced."
)

# Status typing carried over from the v1->v2 decision-rule patch, so this
# diagnostic cannot regress into the NaN-to-zero collapse that was fixed there.
NOT_EVALUABLE_CONSTANT_DELTA = "delta_z_constant_across_origins"
INVALID_NON_FINITE_DELTA = "delta_z_non_finite"
INVALID_NON_FINITE_RESPONSE = "delta_mae_non_finite"


# --------------------------------------------------------------------------- #
# Position bookkeeping
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RatePositions:
    """The block positions available at one missing rate, split by kind."""

    rate: float
    rate_tag: str
    block_length: int
    trailing_boundary: int          # always d=0
    internal: tuple[int, ...]       # every position with d > 0, ascending
    near: int                       # nearest internal position to the boundary
    far: int                        # furthest internal position

    def column(self, distance: int) -> str:
        return f"block_r{self.rate_tag}_d{distance}"

    @property
    def internal_columns(self) -> list[str]:
        return [self.column(d) for d in self.internal]

    @property
    def trailing_boundary_column(self) -> str:
        return self.column(self.trailing_boundary)


def rate_positions(config: dict[str, Any]) -> list[RatePositions]:
    """Split each rate's preregistered block positions into d=0 vs. internal.

    Derived from the config rather than hardcoded, so the internal set cannot
    silently drift out of step with the design.
    """
    out: list[RatePositions] = []
    for spec in config["masks"]["rates"]:
        rate = float(spec["rate"])
        distances = sorted(int(d) for d in spec["block_distances"])
        if 0 not in distances:
            raise ValueError(f"rate {rate}: expected a d=0 trailing-boundary position")
        internal = tuple(d for d in distances if d > 0)
        if len(internal) < 2:
            raise ValueError(f"rate {rate}: need >= 2 internal positions, found {internal}")
        out.append(
            RatePositions(
                rate=rate,
                rate_tag=f"{int(round(rate * 100))}",
                block_length=int(spec["block_length"]),
                trailing_boundary=0,
                internal=internal,
                near=internal[0],
                far=internal[-1],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Confounder-difference diagnostic
# --------------------------------------------------------------------------- #

@dataclass
class DeltaConfounderResult:
    """rho(dMAE, dz) for one contrast x one confounder, with explicit status.

    status:
      not_evaluable - dz is constant across origins, so the rank correlation is
                      undefined. Should not occur here by construction.
      invalid_input - a non-finite value appeared where a real difference of
                      real per-origin values was expected. A DATA PROBLEM, and
                      surfaced rather than suppressed.
      false         - evaluated; Holm did not reject at the family alpha.
      true          - evaluated; Holm rejected.
    """

    contrast_id: str
    confounder: str
    status: str
    reason: str
    n_origins: int
    rho: float = float("nan")
    p_value: float = float("nan")
    holm_adjusted_p: float = float("nan")
    holm_alpha: float = float("nan")
    holm_rejects: bool = False
    n_unique_finite_delta: int = 0
    # Bootstrap CI on rho itself. The parametric Spearman p-value assumes
    # independent observations; evaluation origins are NOT independent (stride
    # 96 < context 320), so the parametric value must never stand alone.
    rho_ci95_low: float = float("nan")
    rho_ci95_high: float = float("nan")
    rho_ci95_excludes_zero: bool = False
    rho_bootstrap_by_block_length: dict[str, list[float]] = field(default_factory=dict)
    delta_z_source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _delta_confounder(
    *, contrast_id: str, confounder: str, delta_mae: np.ndarray, delta_z: np.ndarray
) -> DeltaConfounderResult:
    """One rho(dMAE, dz), refusing to invent a value when the input is broken."""
    from scipy.stats import spearmanr

    n = int(delta_mae.size)
    base = dict(contrast_id=contrast_id, confounder=confounder, n_origins=n)

    if not np.all(np.isfinite(delta_mae)):
        return DeltaConfounderResult(
            status="invalid_input", reason=INVALID_NON_FINITE_RESPONSE, **base
        )
    if not np.all(np.isfinite(delta_z)):
        # dz is a real difference of real per-origin values, so a non-finite
        # entry means something upstream is wrong. This is NOT the structurally
        # undefined case, and must not be quietly treated as one.
        return DeltaConfounderResult(
            status="invalid_input", reason=INVALID_NON_FINITE_DELTA, **base
        )

    n_unique = int(np.unique(delta_z).size)
    if n_unique < 2 or np.std(delta_z) == 0:
        return DeltaConfounderResult(
            status="not_evaluable",
            reason=NOT_EVALUABLE_CONSTANT_DELTA,
            n_unique_finite_delta=n_unique,
            **base,
        )

    rho, p_value = spearmanr(delta_mae, delta_z)
    return DeltaConfounderResult(
        status="false",  # provisional; Holm decides below
        reason="evaluated",
        rho=float(rho),
        p_value=float(p_value),
        n_unique_finite_delta=n_unique,
        **base,
    )


def bootstrap_rho(
    *,
    delta_mae: np.ndarray,
    delta_z: np.ndarray,
    block_lengths: list[int],
    n_replicates: int,
    seed: int,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Moving-block bootstrap CI on rho(dMAE, dz), respecting origin dependence.

    The SAME resampled origin index set is applied to both series in every
    replicate, so the pairing that defines rho is preserved by construction -
    exactly the discipline the contrast bootstrap already follows.
    """
    from scipy.stats import rankdata

    n = int(delta_mae.size)
    alpha = 1.0 - ci_level
    per_block: dict[str, list[float]] = {}

    for block_length in block_lengths:
        rng = np.random.default_rng(seed + 1000 * block_length)
        draws = np.empty(n_replicates, dtype=np.float64)
        for r in range(n_replicates):
            idx = moving_block_indices(n=n, block_length=block_length, rng=rng)
            x, y = delta_mae[idx], delta_z[idx]
            # Spearman = Pearson on ranks; rank within the replicate.
            rx, ry = rankdata(x), rankdata(y)
            if np.std(rx) == 0 or np.std(ry) == 0:
                draws[r] = np.nan
                continue
            draws[r] = float(np.corrcoef(rx, ry)[0, 1])
        finite = draws[np.isfinite(draws)]
        if finite.size == 0:
            per_block[str(block_length)] = [float("nan"), float("nan")]
            continue
        low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        per_block[str(block_length)] = [float(low), float(high)]

    return per_block


def _holm_over_confounders(
    results: list[DeltaConfounderResult], *, family_alpha: float
) -> list[DeltaConfounderResult]:
    """Holm-Bonferroni within ONE contrast's confounder family.

    Only evaluated members participate; not_evaluable and invalid_input results
    are excluded from the family and can never become ``true``.
    """
    evaluated = [r for r in results if r.status in {"false", "true"}]
    m = len(evaluated)
    if m == 0:
        return results

    order = sorted(range(m), key=lambda i: evaluated[i].p_value)
    running_max = 0.0
    for step, i in enumerate(order, start=1):
        alpha_k = family_alpha / (m - step + 1)
        adjusted = min(1.0, (m - step + 1) * evaluated[i].p_value)
        adjusted = max(adjusted, running_max)
        running_max = adjusted
        evaluated[i].holm_alpha = float(alpha_k)
        evaluated[i].holm_adjusted_p = float(adjusted)
        evaluated[i].holm_rejects = bool(adjusted < family_alpha)
        evaluated[i].status = "true" if evaluated[i].holm_rejects else "false"
        evaluated[i].reason = (
            "confounder_difference_correlated" if evaluated[i].holm_rejects
            else "no_significant_correlation"
        )
    return results


# --------------------------------------------------------------------------- #
# Per-origin trend slope
# --------------------------------------------------------------------------- #

def per_origin_trend_slopes(
    *, paired_diff_matrix: pd.DataFrame, distances: tuple[int, ...], columns: list[str]
) -> pd.Series:
    """OLS slope of paired-diff-from-clean against d, per origin, over 4 points.

    Note the clean term cancels out of the slope: the per-origin clean MAE is a
    constant offset across the four positions, so slope(MAE_d - MAE_clean) is
    identically slope(MAE_d). Using the paired difference keeps the response on
    the same scale as everything else reported here.
    """
    x = np.asarray(distances, dtype=np.float64)
    if x.size < 2:
        raise ValueError("a trend slope needs at least two positions")
    x_centred = x - x.mean()
    denominator = float(np.sum(x_centred**2))
    if denominator == 0:
        raise ValueError("distances are constant; a trend slope is undefined")

    values = paired_diff_matrix[columns].to_numpy(dtype=np.float64)
    if values.shape[1] != x.size:
        raise ValueError("column count does not match the number of distances")
    y_centred = values - values.mean(axis=1, keepdims=True)
    slopes = (y_centred @ x_centred) / denominator
    return pd.Series(slopes, index=paired_diff_matrix.index, name="trend_slope")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def _summarise(result, *, mean_clean_mae: float, extra: dict[str, Any]) -> dict[str, Any]:
    """Flatten one BootstrapResult into the same shape the C1-C4 table uses."""
    return {
        **extra,
        "n_origins": result.n_origins,
        "raw_mae_difference": result.observed_mean,
        "pct_of_mean_clean_mae": 100.0 * result.observed_mean / mean_clean_mae,
        "median_paired_difference": result.observed_median,
        "frac_origins_positive": result.frac_positive,
        "holm_ci95_low": result.ci_low_holm,
        "holm_ci95_high": result.ci_high_holm,
        "holm_adjusted_p": result.holm_adjusted_p,
        "holm_rejects": result.holm_rejects,
        # NOTE: ci_*_corrected are the UNCORRECTED percentile intervals at the
        # configured levels; the Holm-adjusted ones are ci_*_holm above.
        "uncorrected_ci95_low": result.ci_low_corrected,
        "uncorrected_ci95_high": result.ci_high_corrected,
        "uncorrected_ci90_low": result.ci_low_equivalence,
        "uncorrected_ci90_high": result.ci_high_equivalence,
        "p_value_two_sided": result.p_value_two_sided,
    }


def analyse_internal_only(
    run_dir: Path, config: dict[str, Any], *, out_dir: Path | None = None
) -> dict[str, Any]:
    """Run the whole post-hoc internal-only analysis against an existing run."""
    out_dir = (run_dir / "post_hoc_internal_only_analysis") if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    window_results = pd.read_csv(run_dir / "window_results.csv")
    diagnostics = pd.read_csv(run_dir / "mask_diagnostics.csv")

    stats_cfg = config["statistics"]
    boot_cfg = stats_cfg["bootstrap"]
    main_block = int(boot_cfg["main_block_length"])
    all_blocks = [main_block] + [int(b) for b in boot_cfg["sensitivity_block_lengths"]]
    n_replicates = int(boot_cfg["n_replicates"])
    base_seed = int(boot_cfg["seed"]) + 77_000  # distinct stream from the main analysis
    family_alpha = float(stats_cfg["family_alpha"])
    ci_corrected = float(stats_cfg["ci_level_corrected"])
    ci_equivalence = float(stats_cfg["ci_level_equivalence"])

    matrix = pivot_condition_matrix(window_results, value="mae")
    if matrix.isna().any().any():
        raise ValueError("the MAE matrix has holes; cannot run the internal-only analysis")
    mean_clean_mae = aggregate_mae(matrix["clean"].to_numpy())
    # Paired-diff-from-clean, per origin x condition.
    paired_diff = matrix.sub(matrix["clean"], axis=0)

    positions = rate_positions(config)
    joint = matrix.to_numpy()

    def run_family(
        series_by_id: dict[str, np.ndarray], *, seed_offset: int, apply_holm: bool
    ):
        """Bootstrap a set of paired series at every block length; Holm at main."""
        per_block: dict[int, dict[str, Any]] = {}
        for block_length in all_blocks:
            results, draws = [], {}
            for offset, (cid, values) in enumerate(series_by_id.items()):
                result, replicate_means = bootstrap_paired_difference(
                    contrast_id=cid,
                    paired_differences=values,
                    block_length=block_length,
                    n_replicates=n_replicates,
                    seed=base_seed + seed_offset + 1000 * block_length + offset,
                    ci_level_corrected=ci_corrected,
                    ci_level_equivalence=ci_equivalence,
                    joint_matrix=joint,
                )
                results.append(result)
                draws[cid] = replicate_means
            if apply_holm:
                holm_correct(results, draws, family_alpha=family_alpha)
            per_block[block_length] = {r.contrast_id: r for r in results}
        return per_block

    # ---- 1. PRIMARY: internal-near vs internal-far, per rate ---------------- #
    # Multiplicity family: {IC1_20, IC2_40} only. Separate from the original
    # {C1,C2,C3,C4} family, which is NOT re-corrected.
    primary_ids = {}
    primary_series: dict[str, np.ndarray] = {}
    for pos in positions:
        cid = f"IC{1 if pos.rate_tag == '20' else 2}_{pos.rate_tag}"
        primary_ids[cid] = pos
        primary_series[cid] = (
            matrix[pos.column(pos.near)] - matrix[pos.column(pos.far)]
        ).to_numpy()
    primary_blocks = run_family(primary_series, seed_offset=0, apply_holm=True)

    primary_rows = []
    for cid, pos in primary_ids.items():
        main = primary_blocks[main_block][cid]
        primary_rows.append(
            _summarise(
                main,
                mean_clean_mae=mean_clean_mae,
                extra={
                    "contrast_id": cid,
                    "analysis_kind": ANALYSIS_KIND,
                    "note": ANALYSIS_NOTE,
                    "position_kind": "internal_positions",
                    "holm_family": "internal_primary{IC1_20,IC2_40}",
                    "rate": pos.rate,
                    "label": (
                        f"{pos.rate_tag}% internal-near (d={pos.near}) - "
                        f"internal-far (d={pos.far}); effective forecast distance held constant"
                    ),
                    "minuend": pos.column(pos.near),
                    "subtrahend": pos.column(pos.far),
                    "near_distance": pos.near,
                    "far_distance": pos.far,
                },
            )
            | {
                f"estimate_block_len_{b}": primary_blocks[b][cid].observed_mean
                for b in all_blocks
            }
            | {
                f"holm_ci95_low_block_len_{b}": primary_blocks[b][cid].ci_low_holm
                for b in all_blocks
            }
            | {
                f"holm_ci95_high_block_len_{b}": primary_blocks[b][cid].ci_high_holm
                for b in all_blocks
            }
            | {
                f"holm_rejects_block_len_{b}": primary_blocks[b][cid].holm_rejects
                for b in all_blocks
            }
        )

    # ---- 1b. BOUNDARY JUMP: d=0 vs the nearest internal position ----------- #
    # Its OWN Holm family {BJ1_20, BJ2_40}, independent of every other family.
    # Computed DIRECTLY from per-origin paired MAE values, never by subtracting
    # C1/IC1 outputs: the point estimate would agree, but the interval would not.
    dominance_frac = float(config["decision"]["pivot"]["origin_dominance_top_frac"])
    bj_ids: dict[str, RatePositions] = {}
    bj_series: dict[str, np.ndarray] = {}
    for pos in positions:
        cid = f"BJ{1 if pos.rate_tag == '20' else 2}_{pos.rate_tag}"
        bj_ids[cid] = pos
        bj_series[cid] = (
            matrix[pos.trailing_boundary_column] - matrix[pos.column(pos.near)]
        ).to_numpy()
    bj_blocks = run_family(bj_series, seed_offset=700, apply_holm=True)

    bj_rows = []
    for cid, pos in bj_ids.items():
        main = bj_blocks[main_block][cid]
        bj_rows.append(
            _summarise(
                main,
                mean_clean_mae=mean_clean_mae,
                extra={
                    "contrast_id": cid,
                    "analysis_kind": ANALYSIS_KIND,
                    "note": ANALYSIS_NOTE + " " + BOUNDARY_JUMP_NOTE,
                    "position_kind": "boundary_jump",
                    "holm_family": "boundary_jump{BJ1_20,BJ2_40}",
                    "rate": pos.rate,
                    "label": (
                        f"{pos.rate_tag}% trailing-boundary (d=0) - "
                        f"nearest internal (d={pos.near})"
                    ),
                    "minuend": pos.trailing_boundary_column,
                    "subtrahend": pos.column(pos.near),
                    "near_distance": 0,
                    "far_distance": pos.near,
                    "directly_bootstrapped": True,
                    "derived_by_subtracting_other_contrasts": False,
                },
            )
            | {
                # Reused from stats/decision.py, not reimplemented here.
                "top_origin_share": top_origin_share(
                    bj_series[cid], top_frac=dominance_frac
                ),
                "top_origin_frac": dominance_frac,
            }
            | {f"estimate_block_len_{b}": bj_blocks[b][cid].observed_mean for b in all_blocks}
            | {f"holm_ci95_low_block_len_{b}": bj_blocks[b][cid].ci_low_holm for b in all_blocks}
            | {f"holm_ci95_high_block_len_{b}": bj_blocks[b][cid].ci_high_holm for b in all_blocks}
            | {f"holm_rejects_block_len_{b}": bj_blocks[b][cid].holm_rejects for b in all_blocks}
        )

    # ---- 1c. Equivalence check on the internal contrasts -------------------- #
    # Non-significance is NOT equivalence. The preregistered NO-GO rule's
    # convention is the only thing that establishes equivalence here: the 90% CI
    # must lie entirely inside +/- equivalence_band_frac_of_clean_mae. That
    # threshold is imported, never re-hardcoded.
    band = equivalence_band(config, mean_clean_mae)
    equivalence_rows = []
    for cid, pos in primary_ids.items():
        main = primary_blocks[main_block][cid]
        established = ci_within_equivalence_band(
            main.ci_low_equivalence, main.ci_high_equivalence, band
        )
        equivalence_rows.append(
            {
                "contrast_id": cid,
                "analysis_kind": ANALYSIS_KIND,
                "rate": pos.rate,
                "equivalence_band_frac_of_clean_mae": float(
                    config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
                ),
                "equivalence_band_abs": band,
                "ci90_low": main.ci_low_equivalence,
                "ci90_high": main.ci_high_equivalence,
                "equivalence_established": bool(established),
                "holm_rejects": bool(main.holm_rejects),
                "interpretation": (
                    "90% CI lies entirely within the band: equivalence established "
                    "under the preregistered NO-GO convention."
                    if established
                    else "90% CI is NOT entirely within the band: this is non-significance, "
                         "NOT equivalence. The data do not rule out an effect of up to the "
                         "CI's magnitude."
                ),
            }
        )

    # ---- 2a. SECONDARY: all 6 pairwise internal contrasts, per rate --------- #
    # Two independent Holm families of 6 (one per rate), never one family of 12.
    pairwise_rows = []
    for pos in positions:
        series: dict[str, np.ndarray] = {}
        meta: dict[str, tuple[int, int]] = {}
        for near, far in itertools.combinations(pos.internal, 2):
            cid = f"IP_{pos.rate_tag}_d{near}_vs_d{far}"
            series[cid] = (matrix[pos.column(near)] - matrix[pos.column(far)]).to_numpy()
            meta[cid] = (near, far)
        blocks = run_family(series, seed_offset=100 + int(pos.rate_tag), apply_holm=True)
        for cid, (near, far) in meta.items():
            main = blocks[main_block][cid]
            pairwise_rows.append(
                _summarise(
                    main,
                    mean_clean_mae=mean_clean_mae,
                    extra={
                        "contrast_id": cid,
                        "analysis_kind": ANALYSIS_KIND,
                        "note": ANALYSIS_NOTE + " Diagnostic only; feeds no decision rule.",
                        "position_kind": "internal_positions",
                        "holm_family": f"internal_pairwise_rate{pos.rate_tag}(6)",
                        "rate": pos.rate,
                        "label": f"{pos.rate_tag}% d={near} - d={far}",
                        "minuend": pos.column(near),
                        "subtrahend": pos.column(far),
                        "near_distance": near,
                        "far_distance": far,
                    },
                )
                | {f"estimate_block_len_{b}": blocks[b][cid].observed_mean for b in all_blocks}
            )

    # ---- 2b. SECONDARY: ordered trend slope across the 4 internal positions -- #
    trend_rows, per_origin_slopes = [], {}
    for pos in positions:
        slopes = per_origin_trend_slopes(
            paired_diff_matrix=paired_diff,
            distances=pos.internal,
            columns=pos.internal_columns,
        )
        per_origin_slopes[pos.rate_tag] = slopes
        cid = f"TREND_{pos.rate_tag}"
        # No multiplicity correction: one descriptive slope per rate, not a
        # test family. Uncorrected percentile intervals are reported.
        blocks = run_family(
            {cid: slopes.to_numpy()}, seed_offset=300 + int(pos.rate_tag), apply_holm=False
        )
        main = blocks[main_block][cid]
        trend_rows.append(
            {
                "trend_id": cid,
                "analysis_kind": ANALYSIS_KIND,
                "note": ANALYSIS_NOTE,
                "position_kind": "internal_positions",
                "multiplicity_correction": "none (single descriptive slope per rate)",
                "rate": pos.rate,
                "internal_distances": list(pos.internal),
                "n_origins": main.n_origins,
                "mean_slope_mae_per_step": main.observed_mean,
                "median_slope": main.observed_median,
                "frac_origins_positive_slope": main.frac_positive,
                "ci95_low": main.ci_low_corrected,
                "ci95_high": main.ci_high_corrected,
                "ci95_excludes_zero": bool(
                    main.ci_low_corrected > 0 or main.ci_high_corrected < 0
                ),
                "ci90_low": main.ci_low_equivalence,
                "ci90_high": main.ci_high_equivalence,
                "p_value_two_sided": main.p_value_two_sided,
                **{f"slope_ci95_low_block_len_{b}": blocks[b][cid].ci_low_corrected for b in all_blocks},
                **{f"slope_ci95_high_block_len_{b}": blocks[b][cid].ci_high_corrected for b in all_blocks},
                **{
                    f"slope_ci95_excludes_zero_block_len_{b}": bool(
                        blocks[b][cid].ci_low_corrected > 0 or blocks[b][cid].ci_high_corrected < 0
                    )
                    for b in all_blocks
                },
            }
        )

    # ---- 3. d=0 reported SEPARATELY, never pooled --------------------------- #
    trailing_rows = []
    for pos in positions:
        cid = f"TRAILING_{pos.rate_tag}"
        series = paired_diff[pos.trailing_boundary_column].to_numpy()
        blocks = run_family({cid: series}, seed_offset=500 + int(pos.rate_tag), apply_holm=False)
        main = blocks[main_block][cid]
        trailing_rows.append(
            {
                "condition_id": pos.trailing_boundary_column,
                "analysis_kind": ANALYSIS_KIND,
                "position_kind": "trailing_boundary_condition",
                "note": TRAILING_BOUNDARY_NOTE,
                "multiplicity_correction": "none (descriptive, single condition per rate)",
                "rate": pos.rate,
                "distance_to_boundary": 0,
                "nearest_real_observation_steps_back": pos.block_length + 1,
                "n_origins": main.n_origins,
                "mean_paired_diff_from_clean": main.observed_mean,
                "pct_of_mean_clean_mae": 100.0 * main.observed_mean / mean_clean_mae,
                "median_paired_diff_from_clean": main.observed_median,
                "frac_origins_positive": main.frac_positive,
                "ci95_low": main.ci_low_corrected,
                "ci95_high": main.ci_high_corrected,
                "ci90_low": main.ci_low_equivalence,
                "ci90_high": main.ci_high_equivalence,
                **{f"estimate_block_len_{b}": blocks[b][cid].observed_mean for b in all_blocks},
            }
        )

    # ---- 4. Confounder-difference correlations, primary contrasts only ------ #
    # rho(dMAE, dz) where dz = z(near) - z(far) per origin. Well-posed: dz varies
    # across origins because the removed content differs by origin, even though
    # the two positions being compared are fixed.
    def confounder_family_for(
        *, cid: str, delta_mae: np.ndarray, minuend_column: str, subtrahend_column: str,
        seed_offset: int, delta_z_source: str,
    ) -> list[DeltaConfounderResult]:
        """rho(dMAE, dz) for one contrast, across the pre-specified 4 confounders."""
        a = (
            diagnostics[diagnostics["condition_id"] == minuend_column]
            .set_index("origin_id").reindex(matrix.index)
        )
        b = (
            diagnostics[diagnostics["condition_id"] == subtrahend_column]
            .set_index("origin_id").reindex(matrix.index)
        )
        family: list[DeltaConfounderResult] = []
        for i, confounder in enumerate(CONFOUNDER_FAMILY):
            if confounder not in a.columns or confounder not in b.columns:
                family.append(
                    DeltaConfounderResult(
                        contrast_id=cid, confounder=confounder, status="invalid_input",
                        reason=f"column_absent_from_diagnostics:{confounder}",
                        n_origins=int(delta_mae.size), delta_z_source=delta_z_source,
                    )
                )
                continue
            delta_z = (
                a[confounder].to_numpy(dtype=np.float64)
                - b[confounder].to_numpy(dtype=np.float64)
            )
            result = _delta_confounder(
                contrast_id=cid, confounder=confounder,
                delta_mae=np.asarray(delta_mae, dtype=np.float64), delta_z=delta_z,
            )
            result.delta_z_source = delta_z_source
            if result.status in {"false", "true"}:
                # Bootstrap CI on rho, on the established isolated-stream
                # convention. The parametric p-value assumes independent
                # observations; origins are not independent.
                per_block = bootstrap_rho(
                    delta_mae=np.asarray(delta_mae, dtype=np.float64),
                    delta_z=delta_z,
                    block_lengths=all_blocks,
                    n_replicates=n_replicates,
                    seed=base_seed + seed_offset + 10 * i,
                    ci_level=ci_corrected,
                )
                result.rho_bootstrap_by_block_length = per_block
                low, high = per_block[str(main_block)]
                result.rho_ci95_low, result.rho_ci95_high = low, high
                result.rho_ci95_excludes_zero = bool(
                    np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
                )
            family.append(result)
        # Holm within THIS contrast's 4-item family; never one pooled family.
        _holm_over_confounders(family, family_alpha=family_alpha)
        return family

    # PRIMARY confounder diagnostic: against the BOUNDARY-JUMP contrasts.
    # The previous round ran this against IC1_20/IC2_40, which tests what
    # explains the small internal-position variation and says nothing about the
    # boundary jump. dz here is (d=0 minus nearest internal), matching BJ.
    confounder_results: list[DeltaConfounderResult] = []
    for cid, pos in bj_ids.items():
        confounder_results.extend(
            confounder_family_for(
                cid=cid,
                delta_mae=bj_series[cid],
                minuend_column=pos.trailing_boundary_column,
                subtrahend_column=pos.column(pos.near),
                seed_offset=900 + int(pos.rate_tag),
                delta_z_source=f"d0_minus_nearest_internal(d={pos.near})",
            )
        )

    # RETAINED from the previous round, explicitly relabelled: these describe
    # association with the small INTERNAL-POSITION variation. They are NOT a
    # test of what drives the boundary jump, and are kept only so the earlier
    # result stays on the record rather than disappearing.
    internal_confounder_results: list[DeltaConfounderResult] = []
    for cid, pos in primary_ids.items():
        internal_confounder_results.extend(
            confounder_family_for(
                cid=cid,
                delta_mae=primary_series[cid],
                minuend_column=pos.column(pos.near),
                subtrahend_column=pos.column(pos.far),
                seed_offset=1100 + int(pos.rate_tag),
                delta_z_source=f"internal_near(d={pos.near})_minus_internal_far(d={pos.far})",
            )
        )

    # ---- persist ------------------------------------------------------------ #
    bj_frame = pd.DataFrame(bj_rows)
    equivalence_frame = pd.DataFrame(equivalence_rows)
    internal_confounder_frame = pd.DataFrame(
        [r.as_dict() for r in internal_confounder_results]
    )
    primary_frame = pd.DataFrame(primary_rows)
    pairwise_frame = pd.DataFrame(pairwise_rows)
    trend_frame = pd.DataFrame(trend_rows)
    trailing_frame = pd.DataFrame(trailing_rows)
    confounder_frame = pd.DataFrame([r.as_dict() for r in confounder_results])
    slope_frame = pd.DataFrame(
        {f"trend_slope_rate{tag}": s for tag, s in per_origin_slopes.items()}
    )

    bj_frame.to_csv(out_dir / "boundary_jump_contrasts.csv", index=False)
    equivalence_frame.to_csv(out_dir / "internal_equivalence_check.csv", index=False)
    internal_confounder_frame.to_csv(
        out_dir / "confounder_difference_internal_contrasts.csv", index=False
    )
    primary_frame.to_csv(out_dir / "internal_primary_contrasts.csv", index=False)
    pairwise_frame.to_csv(out_dir / "internal_pairwise_contrasts.csv", index=False)
    trend_frame.to_csv(out_dir / "internal_trend_slopes.csv", index=False)
    slope_frame.to_csv(out_dir / "per_origin_trend_slopes.csv")
    trailing_frame.to_csv(out_dir / "trailing_boundary_d0.csv", index=False)
    confounder_frame.to_csv(out_dir / "confounder_difference_correlations.csv", index=False)

    payload: dict[str, Any] = {
        "analysis_kind": ANALYSIS_KIND,
        "note": ANALYSIS_NOTE,
        "preregistered": False,
        "feeds_decision_function": False,
        "mean_clean_mae": mean_clean_mae,
        "n_origins": int(matrix.shape[0]),
        "rates_analysed_separately": [p.rate for p in positions],
        "internal_positions": {p.rate_tag: list(p.internal) for p in positions},
        "trailing_boundary_position": {p.rate_tag: p.trailing_boundary for p in positions},
        "equivalence_convention": {
            "source": "preregistered NO-GO rule: decision.no_go.equivalence_band_frac_of_clean_mae",
            "band_frac": float(
                config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
            ),
            "band_abs": band,
            "rule": "the 90% CI must lie ENTIRELY within +/- band_abs",
            "note": (
                "Non-significance is not equivalence. A CI that merely includes zero "
                "establishes nothing about the absence of an effect."
            ),
        },
        "holm_families": {
            "boundary_jump": sorted(bj_ids),
            "internal_primary": sorted(primary_ids),
            "internal_pairwise_per_rate": {
                p.rate_tag: sorted(
                    f"IP_{p.rate_tag}_d{a}_vs_d{b}"
                    for a, b in itertools.combinations(p.internal, 2)
                )
                for p in positions
            },
            "confounder_difference_per_boundary_jump_contrast": {
                cid: list(CONFOUNDER_FAMILY) for cid in sorted(bj_ids)
            },
            "confounder_difference_per_internal_contrast": {
                cid: list(CONFOUNDER_FAMILY) for cid in sorted(primary_ids)
            },
            "original_preregistered_family_not_recorrected": [
                c["id"] for c in config["contrasts"]
            ],
        },
        "confounder_family_prespecified": list(CONFOUNDER_FAMILY),
        "boundary_jump_contrasts": bj_rows,
        "internal_equivalence_check": equivalence_rows,
        "primary_contrasts": primary_rows,
        "pairwise_contrasts": pairwise_rows,
        "trend_slopes": trend_rows,
        "trailing_boundary": trailing_rows,
        # PRIMARY: what is associated with the BOUNDARY JUMP.
        "confounder_difference_correlations": [r.as_dict() for r in confounder_results],
        "confounder_difference_correlations_target": "boundary_jump_contrasts (BJ1_20, BJ2_40)",
        # RETAINED and RELABELLED: association with the small internal-position
        # variation only. NOT a test of what drives the boundary jump.
        "confounder_difference_internal_contrasts": [
            r.as_dict() for r in internal_confounder_results
        ],
        "confounder_difference_internal_contrasts_caption": (
            "Association (or its absence) with the SMALL INTERNAL-POSITION variation "
            "measured by IC1_20/IC2_40. This is NOT a test of what drives the boundary "
            "jump; the previous round reported it as though it were. See "
            "confounder_difference_correlations for the boundary-jump diagnostic."
        ),
        "bootstrap": {
            "block_lengths": all_blocks,
            "main_block_length": main_block,
            "n_replicates": n_replicates,
        },
    }
    (out_dir / "internal_only_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload
