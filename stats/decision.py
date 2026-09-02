"""Deterministic GO / PIVOT / NO-GO decision function.

This is a pure function over computed statistics. It emits one label plus the
specific conditions that triggered it. It contains no narrative judgement, and
nothing downstream is permitted to override it.

Precedence (fixed, and stated here because the preregistration defines three
labels whose criteria are not mutually exclusive):

  1. NO-GO   — all four equivalence intervals inside the band, no stable
               location ordering, and small seed sensitivity. A clean null.
  2. PIVOT   — any PIVOT trigger fires. PIVOT deliberately OUTRANKS GO: an
               effect that is real but confounded is a design problem, not a
               green light.
  3. GO      — location GO or geometry GO criteria met and no PIVOT trigger.
  4. PIVOT   — residual: neither a clean null nor a clean effect. Recorded with
               the trigger "inconclusive_no_criteria_met".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

# Version of the decision logic that produced a given output.
#   v1 - original. Coerced an undefined rho_distance (NaN) to 0.0 before
#        comparing confounder correlations against it, which manufactured a
#        comparison baseline that does not exist.
#   v2 - current. Distance-relative comparisons report an explicit status and
#        are `not_evaluable` when distance is constant within the contrast.
DECISION_RULE_VERSION = "v2"


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN-safe, returns NaN when undegenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import rankdata

    rx = rankdata(x[ok])
    ry = rankdata(y[ok])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# Outcome of a "does confounder X out-track distance?" comparison.
#   not_evaluable - distance is constant within this contrast, so its
#                   correlation is mathematically undefined. NOT a null result.
#   invalid_input - distance genuinely varies but rho_distance is non-finite;
#                   something upstream is wrong. NOT a null result either.
#   false         - evaluated; no confounder exceeded distance by the margin.
#   true          - evaluated; at least one confounder did.
DistanceComparisonStatus = Literal["not_evaluable", "invalid_input", "false", "true"]

NOT_EVALUABLE_CONSTANT_DISTANCE = "distance_constant_within_contrast"
INVALID_RHO_DISTANCE = "rho_distance_non_finite_despite_varying_distance"


@dataclass
class DistanceComparison:
    """A single confounder-vs-distance comparison and why it came out that way.

    Only ``status == "true"`` may contribute a PIVOT trigger. ``not_evaluable``
    and ``invalid_input`` are structurally distinct from ``false``: they mean
    the question could not be asked, not that the answer was no.
    """

    status: DistanceComparisonStatus
    reason: str
    n_unique_finite_distance: int
    rho_distance: float
    compared: list[str] = field(default_factory=list)
    exceeding: list[str] = field(default_factory=list)

    @property
    def is_trigger(self) -> bool:
        return self.status == "true"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_finite_count(values) -> int:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0
    return int(np.unique(finite).size)


def compare_confounders_to_distance(
    *,
    rho_distance: float,
    distance_values,
    candidates: dict[str, float],
    margin: float,
) -> DistanceComparison:
    """Ask whether any candidate confounder out-tracks distance, honestly.

    The preregistered contrasts are fixed two-arm comparisons: every origin's
    subtrahend arm sits at the SAME distance to the forecast boundary. A rank
    correlation against a constant is undefined, not zero. Coercing it to zero
    (as decision rule v1 did) invents a baseline of "distance explains nothing",
    against which almost any confounder wins — so the check fired essentially
    independently of the data. This function refuses to answer instead.
    """
    n_unique = _unique_finite_count(distance_values)

    if n_unique < 2:
        return DistanceComparison(
            status="not_evaluable",
            reason=NOT_EVALUABLE_CONSTANT_DISTANCE,
            n_unique_finite_distance=n_unique,
            rho_distance=float(rho_distance),
        )

    if not np.isfinite(rho_distance):
        # Distance varies, so this SHOULD have been computable. Do not let it
        # share the constant-distance path, and never treat it as a null.
        return DistanceComparison(
            status="invalid_input",
            reason=INVALID_RHO_DISTANCE,
            n_unique_finite_distance=n_unique,
            rho_distance=float(rho_distance),
        )

    # Both sides finite: the comparison is meaningful.
    compared = [name for name, rho in candidates.items() if np.isfinite(rho)]
    exceeding = [
        name
        for name in compared
        if abs(candidates[name]) > abs(rho_distance) + margin
    ]
    return DistanceComparison(
        status="true" if exceeding else "false",
        reason="confounder_exceeds_distance" if exceeding else "no_confounder_exceeds_distance",
        n_unique_finite_distance=n_unique,
        rho_distance=float(rho_distance),
        compared=sorted(compared),
        exceeding=sorted(exceeding),
    )


def top_origin_share(paired_differences, *, top_frac: float) -> float:
    """Share of the summed |paired difference| held by the top `top_frac` origins.

    The origin-dominance diagnostic. Shared so every caller measures dominance
    identically rather than each re-deriving the same few lines.
    """
    magnitude = np.abs(np.asarray(paired_differences, dtype=np.float64))
    magnitude = magnitude[np.isfinite(magnitude)]
    if magnitude.size == 0:
        return float("nan")
    total = float(magnitude.sum())
    if total <= 0:
        return float("nan")
    k = max(1, int(np.ceil(top_frac * magnitude.size)))
    return float(np.sort(magnitude)[-k:].sum() / total)


def equivalence_band(config: dict[str, Any], mean_clean_mae: float) -> float:
    """The +/- band, in MAE units, that the preregistered NO-GO rule uses.

    Single definition of the threshold, so an equivalence assessment anywhere in
    the codebase cannot drift from the one the decision rule applies.
    """
    frac = float(config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"])
    return frac * float(mean_clean_mae)


def ci_within_equivalence_band(ci_low: float, ci_high: float, band: float) -> bool:
    """Whether an interval lies entirely inside +/- band.

    Note this is the ONLY way equivalence is ever established here: a CI that
    merely includes zero is non-significance, which is not equivalence.
    """
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return False
    return bool(ci_low >= -band and ci_high <= band)


@dataclass
class ContrastStat:
    """Everything the decision function needs about one contrast."""

    contrast_id: str
    family: str          # "location" | "geometry"
    kind: str            # "causal_contrast_within_pattern" | "pipeline_comparison"
    rate: float
    mean_difference: float
    median_difference: float
    frac_positive: float
    holm_rejects: bool
    ci_low_holm: float
    ci_high_holm: float
    ci_low_equivalence: float
    ci_high_equivalence: float
    mean_clean_mae: float
    # Point estimates under each bootstrap block length, keyed by block length.
    # NOTE: the observed mean paired difference does NOT depend on the bootstrap,
    # so these are identical by construction and are kept only for reporting.
    # What actually moves with the block length is the interval and the
    # rejection decision, so the instability trigger is based on those.
    block_length_estimates: dict[int, float] = field(default_factory=dict)
    block_length_rejects: dict[int, bool] = field(default_factory=dict)
    block_length_ci_low: dict[int, float] = field(default_factory=dict)
    block_length_ci_high: dict[int, float] = field(default_factory=dict)
    # Share of the summed paired difference held by the top origins.
    top_origin_share: float = float("nan")
    # |Spearman(paired difference, X)| across origins.
    rho_distance: float = float("nan")
    # The per-origin distance-to-boundary values rho_distance was computed
    # against. Needed to tell "distance is constant, so the correlation is
    # undefined" apart from "the correlation was computed and came out small".
    distance_values: list[float] = field(default_factory=list)
    rho_confounders: dict[str, float] = field(default_factory=dict)
    rho_scaling: float = float("nan")
    n_partially_missing_patches_max: int = 0

    @property
    def magnitude_frac(self) -> float:
        if not self.mean_clean_mae:
            return float("nan")
        return abs(self.mean_difference) / self.mean_clean_mae

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionInputs:
    contrasts: list[ContrastStat]
    mean_clean_mae: float
    # Spearman(mean MAE across the 5 block positions, d), per rate.
    location_ordering_spearman: dict[float, float]
    # Aggregate seed spread (max-min of across-origin mean MAE), per rate,
    # as a fraction of mean clean MAE.
    seed_sensitivity_frac: dict[float, float]
    config: dict[str, Any]


@dataclass
class Decision:
    label: str                              # "GO" | "PIVOT" | "NO-GO"
    go_family: str | None                   # "location" | "geometry" | None
    triggers: list[str]
    criteria: dict[str, Any]
    # Which decision logic produced this output. See DECISION_RULE_VERSION.
    decision_rule_version: str = DECISION_RULE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _family_go(
    stats: list[ContrastStat], *, family: str, min_magnitude_frac: float
) -> tuple[bool, dict[str, Any]]:
    """GO criteria for one contrast family, required at BOTH rates."""
    members = [s for s in stats if s.family == family]
    detail: dict[str, Any] = {
        "family": family,
        "contrast_ids": [s.contrast_id for s in members],
        "n_contrasts": len(members),
    }
    if len(members) != 2:
        detail["failed"] = f"expected 2 contrasts in family {family}, found {len(members)}"
        return False, detail

    signs = {np.sign(s.mean_difference) for s in members}
    same_sign = len(signs) == 1 and 0 not in signs
    magnitudes_ok = all(s.magnitude_frac >= min_magnitude_frac for s in members)
    ci_ok = all(s.holm_rejects for s in members)
    median_ok = all(
        np.sign(s.median_difference) == np.sign(s.mean_difference) and s.median_difference != 0
        for s in members
    )

    detail.update(
        {
            "same_sign_at_both_rates": bool(same_sign),
            "signs": {s.contrast_id: float(np.sign(s.mean_difference)) for s in members},
            "magnitude_ge_threshold_at_both_rates": bool(magnitudes_ok),
            "magnitudes_frac_of_clean_mae": {s.contrast_id: s.magnitude_frac for s in members},
            "min_magnitude_frac_required": min_magnitude_frac,
            "holm_ci_excludes_zero_at_both_rates": bool(ci_ok),
            "holm_rejects": {s.contrast_id: bool(s.holm_rejects) for s in members},
            "median_sign_matches_mean_at_both_rates": bool(median_ok),
            "median_differences": {s.contrast_id: s.median_difference for s in members},
        }
    )
    passed = bool(same_sign and magnitudes_ok and ci_ok and median_ok)
    detail["passed"] = passed
    return passed, detail


def _no_go(inputs: DecisionInputs) -> tuple[bool, dict[str, Any]]:
    cfg = inputs.config["decision"]["no_go"]
    band = equivalence_band(inputs.config, inputs.mean_clean_mae)
    ordering_threshold = float(cfg["location_ordering_abs_spearman"])
    seed_threshold = float(cfg["seed_sensitivity_frac_of_clean_mae"])

    within_band = {
        s.contrast_id: ci_within_equivalence_band(
            s.ci_low_equivalence, s.ci_high_equivalence, band
        )
        for s in inputs.contrasts
    }
    all_within_band = all(within_band.values())

    rhos = inputs.location_ordering_spearman
    strong = [abs(r) >= ordering_threshold for r in rhos.values() if np.isfinite(r)]
    finite_rhos = [r for r in rhos.values() if np.isfinite(r)]
    same_sign = (
        len({float(np.sign(r)) for r in finite_rhos}) == 1
        and len(finite_rhos) == len(rhos)
        and 0.0 not in {float(np.sign(r)) for r in finite_rhos}
    )
    stable_ordering = bool(len(strong) == len(rhos) and all(strong) and same_sign)

    seed_small = all(
        v <= seed_threshold for v in inputs.seed_sensitivity_frac.values() if np.isfinite(v)
    )

    detail = {
        "equivalence_band_abs": band,
        "equivalence_band_frac": float(cfg["equivalence_band_frac_of_clean_mae"]),
        "ci90_within_band": within_band,
        "all_ci90_within_band": all_within_band,
        "location_ordering_spearman": {str(k): v for k, v in rhos.items()},
        "location_ordering_threshold": ordering_threshold,
        "stable_location_ordering": stable_ordering,
        "no_stable_location_ordering": not stable_ordering,
        "seed_sensitivity_frac": {str(k): v for k, v in inputs.seed_sensitivity_frac.items()},
        "seed_sensitivity_threshold": seed_threshold,
        "seed_sensitivity_small": bool(seed_small),
    }
    triggered = bool(all_within_band and not stable_ordering and seed_small)
    detail["triggered"] = triggered
    return triggered, detail


def _pivot_triggers(inputs: DecisionInputs) -> tuple[list[str], dict[str, Any]]:
    cfg = inputs.config["decision"]["pivot"]
    min_mag = float(inputs.config["decision"]["go"]["min_magnitude_frac_of_clean_mae"])
    triggers: list[str] = []
    detail: dict[str, Any] = {}

    # (a) effect appears only at 40%.
    per_family: dict[str, dict[float, bool]] = {}
    for stat in inputs.contrasts:
        strong = bool(stat.holm_rejects and stat.magnitude_frac >= min_mag)
        per_family.setdefault(stat.family, {})[stat.rate] = strong
    only_40: list[str] = []
    for family, by_rate in per_family.items():
        if by_rate.get(0.40) and not by_rate.get(0.20):
            only_40.append(family)
    detail["effect_strong_by_family_and_rate"] = {
        f: {str(r): v for r, v in d.items()} for f, d in per_family.items()
    }
    if only_40:
        triggers.append(f"effect_only_at_40pct:{','.join(sorted(only_40))}")
    detail["families_with_effect_only_at_40pct"] = sorted(only_40)

    # (b) results shift materially across bootstrap block lengths.
    #
    # The observed mean paired difference is invariant to the bootstrap, so
    # comparing point estimates across block lengths would be a no-op that could
    # never fire. What the block length genuinely changes is how much serial
    # dependence between overlapping origins the resample preserves, and hence
    # the interval width and the rejection decision. Sensitivity is therefore
    # assessed on the Holm interval bounds and on rejection/sign agreement.
    instability_threshold = float(cfg["block_length_instability_frac_of_clean_mae"]) * inputs.mean_clean_mae
    unstable: dict[str, Any] = {}
    for stat in inputs.contrasts:
        lows = list(stat.block_length_ci_low.values())
        highs = list(stat.block_length_ci_high.values())
        rejects = list(stat.block_length_rejects.values())
        if not lows or not highs:
            continue
        low_spread = float(max(lows) - min(lows))
        high_spread = float(max(highs) - min(highs))
        ci_spread = max(low_spread, high_spread)
        # A sign flip that matters is an interval that excludes zero on one side
        # under one block length and on the other side (or not at all) elsewhere.
        excludes_zero = [bool(lo > 0 or hi < 0) for lo, hi in zip(lows, highs)]
        signs = {float(np.sign(lo + hi)) for lo, hi in zip(lows, highs)}
        sign_flip = len(signs) > 1
        reject_flip = len(set(rejects)) > 1 or len(set(excludes_zero)) > 1
        flagged = bool(ci_spread > instability_threshold or sign_flip or reject_flip)
        unstable[stat.contrast_id] = {
            "estimates_by_block_length": {str(k): v for k, v in stat.block_length_estimates.items()},
            "point_estimate_is_bootstrap_invariant": True,
            "ci_low_by_block_length": {str(k): v for k, v in stat.block_length_ci_low.items()},
            "ci_high_by_block_length": {str(k): v for k, v in stat.block_length_ci_high.items()},
            "rejects_by_block_length": {str(k): v for k, v in stat.block_length_rejects.items()},
            "ci_excludes_zero_by_block_length": excludes_zero,
            "ci_bound_spread": ci_spread,
            "spread_threshold": instability_threshold,
            "sign_flip": sign_flip,
            "reject_flip": reject_flip,
            "flagged": flagged,
        }
    detail["block_length_sensitivity"] = unstable
    flagged_bl = [k for k, v in unstable.items() if v["flagged"]]
    if flagged_bl:
        triggers.append(f"bootstrap_block_length_instability:{','.join(sorted(flagged_bl))}")

    # (c) a handful of origins dominate the effect.
    share_threshold = float(cfg["origin_dominance_share"])
    dominated = {
        s.contrast_id: s.top_origin_share
        for s in inputs.contrasts
        if np.isfinite(s.top_origin_share) and s.top_origin_share > share_threshold
    }
    detail["top_origin_share"] = {s.contrast_id: s.top_origin_share for s in inputs.contrasts}
    detail["top_origin_share_threshold"] = share_threshold
    detail["top_origin_frac"] = float(cfg["origin_dominance_top_frac"])
    if dominated:
        triggers.append(f"origin_dominance:{','.join(sorted(dominated))}")

    # (d) effect tracks removed-segment properties more than distance.
    #
    # Entirely distance-relative: there is no distance-free branch here, so when
    # distance is constant within the contrast this check is `not_evaluable` as
    # a whole and contributes nothing to the decision.
    margin = float(cfg["confounder_margin_over_distance"])
    confounder_checks = {
        stat.contrast_id: compare_confounders_to_distance(
            rho_distance=stat.rho_distance,
            distance_values=stat.distance_values,
            candidates=stat.rho_confounders,
            margin=margin,
        )
        for stat in inputs.contrasts
    }
    out_tracked = {
        cid: check.exceeding for cid, check in confounder_checks.items() if check.is_trigger
    }
    detail["rho_distance"] = {s.contrast_id: s.rho_distance for s in inputs.contrasts}
    detail["rho_confounders"] = {s.contrast_id: s.rho_confounders for s in inputs.contrasts}
    detail["confounder_margin_over_distance"] = margin
    detail["confounders_out_tracking_distance"] = out_tracked
    detail["confounder_out_tracks_distance_status"] = {
        cid: check.as_dict() for cid, check in confounder_checks.items()
    }
    if out_tracked:
        triggers.append(f"confounder_out_tracks_distance:{','.join(sorted(out_tracked))}")

    # (e) Chronos-2's internal scaling behaviour explains most of the pattern.
    #
    # This check has TWO independent branches. The first is an absolute
    # threshold on |rho_scaling| and involves distance not at all, so it stays
    # fully live. The second is distance-relative and gets the same honest
    # status treatment as (d): when distance is constant it is `not_evaluable`
    # and contributes nothing. The trigger can still fire on the absolute
    # branch alone.
    scaling_threshold = float(cfg["scaling_abs_spearman"])
    scaling_absolute = {
        s.contrast_id: s.rho_scaling
        for s in inputs.contrasts
        if np.isfinite(s.rho_scaling) and abs(s.rho_scaling) >= scaling_threshold
    }
    scaling_distance_checks = {
        stat.contrast_id: compare_confounders_to_distance(
            rho_distance=stat.rho_distance,
            distance_values=stat.distance_values,
            candidates={"rho_scaling": stat.rho_scaling},
            margin=margin,
        )
        for stat in inputs.contrasts
    }
    scaling_distance_relative = {
        cid: check.exceeding for cid, check in scaling_distance_checks.items() if check.is_trigger
    }
    scaling_flagged = {
        s.contrast_id: s.rho_scaling
        for s in inputs.contrasts
        if s.contrast_id in scaling_absolute or s.contrast_id in scaling_distance_relative
    }
    detail["rho_scaling"] = {s.contrast_id: s.rho_scaling for s in inputs.contrasts}
    detail["scaling_threshold"] = scaling_threshold
    detail["scaling_flagged_by_absolute_threshold"] = scaling_absolute
    detail["scaling_flagged_by_distance_comparison"] = scaling_distance_relative
    detail["internal_scaling_distance_branch_status"] = {
        cid: check.as_dict() for cid, check in scaling_distance_checks.items()
    }
    if scaling_flagged:
        triggers.append(f"internal_scaling_explains_pattern:{','.join(sorted(scaling_flagged))}")

    # (f) patch position is a plausible confound. All preregistered blocks are
    # patch-aligned, so any partially-missing patch inside a BLOCK arm means the
    # alignment assumption broke; and patch occupancy out-tracking distance is
    # itself the confound.
    patch_flagged = {
        s.contrast_id: s.n_partially_missing_patches_max
        for s in inputs.contrasts
        if s.kind == "causal_contrast_within_pattern" and s.n_partially_missing_patches_max > 0
    }
    # The partial-patch branch above is distance-free and stays live. The
    # patch-occupancy-vs-distance branch below is distance-relative and gets
    # the same status treatment.
    patch_distance_checks = {
        stat.contrast_id: compare_confounders_to_distance(
            rho_distance=stat.rho_distance,
            distance_values=stat.distance_values,
            candidates={
                "n_patches_fully_missing": stat.rho_confounders.get(
                    "n_patches_fully_missing", float("nan")
                )
            },
            margin=margin,
        )
        for stat in inputs.contrasts
    }
    patch_rho_flagged = {
        cid: next(
            s.rho_confounders.get("n_patches_fully_missing")
            for s in inputs.contrasts
            if s.contrast_id == cid
        )
        for cid, check in patch_distance_checks.items()
        if check.is_trigger
    }
    detail["block_arms_with_partial_patches"] = patch_flagged
    detail["patch_occupancy_out_tracking_distance"] = patch_rho_flagged
    detail["patch_occupancy_out_tracking_distance_status"] = {
        cid: check.as_dict() for cid, check in patch_distance_checks.items()
    }
    if patch_flagged or patch_rho_flagged:
        names = sorted(set(patch_flagged) | set(patch_rho_flagged))
        triggers.append(f"patch_position_confound:{','.join(names)}")

    return triggers, detail


def decide(inputs: DecisionInputs) -> Decision:
    """Return the pilot's GO / PIVOT / NO-GO label and what triggered it."""
    min_mag = float(inputs.config["decision"]["go"]["min_magnitude_frac_of_clean_mae"])

    location_go, location_detail = _family_go(
        inputs.contrasts, family="location", min_magnitude_frac=min_mag
    )
    geometry_go, geometry_detail = _family_go(
        inputs.contrasts, family="geometry", min_magnitude_frac=min_mag
    )
    no_go, no_go_detail = _no_go(inputs)
    pivot_triggers, pivot_detail = _pivot_triggers(inputs)

    criteria: dict[str, Any] = {
        "decision_rule_version": DECISION_RULE_VERSION,
        "mean_clean_mae": inputs.mean_clean_mae,
        "location_go": location_detail,
        "geometry_go": geometry_detail,
        "no_go": no_go_detail,
        "pivot": pivot_detail,
        "precedence": ["NO-GO", "PIVOT", "GO", "PIVOT(residual)"],
        "note_distance_relative_checks": (
            "Checks that compare a confounder's rank correlation against distance's are "
            "reported with an explicit status. For the preregistered two-arm contrasts "
            "distance is constant across origins, so its correlation is undefined and the "
            "status is 'not_evaluable' - which is not the same as 'evaluated, no effect' "
            "and carries no evidentiary weight either way."
        ),
        "note_geometry_family_is_pipeline_comparison": (
            "Contrasts C3/C4 compare a random-mask arm with a block arm and are pipeline "
            "comparisons, not pure causal estimates of contiguity."
        ),
    }

    if no_go:
        return Decision(
            label="NO-GO",
            go_family=None,
            triggers=["no_go_all_criteria_met"],
            criteria=criteria,
        )
    if pivot_triggers:
        return Decision(label="PIVOT", go_family=None, triggers=pivot_triggers, criteria=criteria)
    if location_go or geometry_go:
        families = [f for f, ok in (("location", location_go), ("geometry", geometry_go)) if ok]
        return Decision(
            label="GO",
            go_family="+".join(families),
            triggers=[f"{f}_go_criteria_met" for f in families],
            criteria=criteria,
        )
    return Decision(
        label="PIVOT",
        go_family=None,
        triggers=["inconclusive_no_criteria_met"],
        criteria=criteria,
    )
