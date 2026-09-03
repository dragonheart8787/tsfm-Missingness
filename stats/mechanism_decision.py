"""Preregistered mechanism classification rule for the trailing-gap experiment.

    *** RULE FROZEN BEFORE GPU EXECUTION. rule_version: preregistered-v1 ***

``authoritative: True`` means **the decision rule itself was preregistered** —
its per-gap thresholds and its precedence were fixed before any forecast was
produced, so the classification cannot have been tuned to the data. It does
**NOT** mean any resulting causal interpretation is authoritative. The labels
below describe an observed pattern of error differences; none of them isolates
masking, normalization, positional handling, or any other internal component.

What R_g measures
-----------------
    R_g = MAE(trailing_nan(g)) - MAE(truncated_long(g))

Both arms lose the same ``g`` most-recent observations and are scored against
the same target over the same timestamps. They differ only in whether the model
is SHOWN a trailing run of NaNs, or handed a shorter context and asked for a
longer single-shot horizon. No autoregressive unrolling occurs anywhere in this
design (confirmed against the library's control flow), so the comparison arm is
the **shorter-context/longer-horizon single-shot framing**. It must never be
described as autoregressive: that would not be imprecise, it would be false.

Per-gap classification is FOUR-way against the SESOI
----------------------------------------------------
With ``delta = 0.03 * mean_clean_mae``:

  EQUIVALENT        the complete 90% equivalence CI lies inside [-delta, +delta]
  MATERIAL_POSITIVE Holm-adjusted CI lower bound > +delta   (NOT merely > 0)
  MATERIAL_NEGATIVE Holm-adjusted CI upper bound < -delta
  UNRESOLVED        everything else

``ci_low_holm > 0`` is NOT sufficient for MATERIAL_POSITIVE. A difference can be
statistically distinguishable from zero while its practical magnitude remains
unresolved against the SESOI; that case is UNRESOLVED, and is flagged as such
rather than counted as a directional finding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from stats.decision import ci_within_equivalence_band

MECHANISM_RULE_VERSION = "preregistered-v1"
RULE_STATUS = (
    "Rule frozen before GPU execution. authoritative=True means the DECISION RULE "
    "was preregistered — thresholds and precedence fixed before any forecast — and "
    "does NOT mean any resulting causal interpretation is authoritative."
)

# --------------------------------------------------------------------------- #
# Per-gap readings
# --------------------------------------------------------------------------- #

GapReading = Literal["EQUIVALENT", "MATERIAL_POSITIVE", "MATERIAL_NEGATIVE", "UNRESOLVED"]

EQUIVALENT = "EQUIVALENT"
MATERIAL_POSITIVE = "MATERIAL_POSITIVE"
MATERIAL_NEGATIVE = "MATERIAL_NEGATIVE"
UNRESOLVED = "UNRESOLVED"

MATERIAL_READINGS = (MATERIAL_POSITIVE, MATERIAL_NEGATIVE)

# Secondary diagnostic flags. Recorded for visibility; they can NEVER promote a
# gap to a material directional reading.
FLAG_SMALL_BUT_DETECTABLE = "statistically_detectable_but_practically_small"
FLAG_SESOI_UNRESOLVED = "statistically_detectable_but_sesoi_unresolved"

INTERPRETATION: dict[str, str] = {
    EQUIVALENT: (
        "no practically meaningful performance difference was established... "
        "consistent with a horizon-dominant explanation, but does NOT demonstrate "
        "an 'effective-horizon-only' mechanism."
    ),
    MATERIAL_POSITIVE: (
        "explicit trailing-gap encoding has higher mean error than "
        "shorter-context/longer-horizon single-shot forecasting by more than the "
        "SESOI. It does not isolate masking, normalization, positional handling, "
        "or another internal component."
    ),
    MATERIAL_NEGATIVE: (
        "lower error was observed with explicit trailing NaNs. Do not call this a "
        "beneficial causal mechanism."
    ),
    UNRESOLVED: (
        "neither equivalence nor a material directional difference was established "
        "at this gap length."
    ),
}

# --------------------------------------------------------------------------- #
# Classification labels and INCONCLUSIVE reasons
# --------------------------------------------------------------------------- #

DIRECTION_REVERSAL = "DIRECTION_REVERSAL_ACROSS_GAPS"
NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_DIFFERENCE_VS_TRUNCATION"
CONSISTENT_PENALTY = "CONSISTENT_EXTRA_TRAILING_GAP_PENALTY"
CONSISTENT_LOWER_ERROR = "CONSISTENT_LOWER_ERROR_WITH_TRAILING_NAN"
INCONCLUSIVE = "INCONCLUSIVE"

REASON_SPARSE = "sparse_or_gap_dependent_directional_evidence"
REASON_MIXED = "mixed_equivalent_and_unresolved"
REASON_INSUFFICIENT = "insufficient_precision"


@dataclass
class GapStat:
    """One R_g (or B_g) as the classification rule consumes it."""

    contrast_id: str
    gap: int
    mean_difference: float
    median_difference: float
    ci_low_holm: float
    ci_high_holm: float
    ci_low_equivalence: float
    ci_high_equivalence: float
    holm_rejects: bool
    mean_clean_mae: float

    def reading(self, delta: float) -> str:
        """Four-way reading against the SESOI. See the module docstring."""
        if ci_within_equivalence_band(self.ci_low_equivalence, self.ci_high_equivalence, delta):
            return EQUIVALENT
        if np.isfinite(self.ci_low_holm) and self.ci_low_holm > delta:
            return MATERIAL_POSITIVE
        if np.isfinite(self.ci_high_holm) and self.ci_high_holm < -delta:
            return MATERIAL_NEGATIVE
        return UNRESOLVED

    def flags(self, delta: float) -> list[str]:
        """Secondary diagnostic flags. Never promote a gap to a material reading."""
        found: list[str] = []
        excludes_zero = bool(
            np.isfinite(self.ci_low_holm)
            and np.isfinite(self.ci_high_holm)
            and (self.ci_low_holm > 0 or self.ci_high_holm < 0)
        )
        if not excludes_zero:
            return found
        reading = self.reading(delta)
        if reading == EQUIVALENT:
            found.append(FLAG_SMALL_BUT_DETECTABLE)
        elif reading == UNRESOLVED:
            found.append(FLAG_SESOI_UNRESOLVED)
        return found

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoseResponse:
    """Slope of R_g against x = g / patch_size (MISSING PATCHES, not raw g).

    Secondary to the four per-gap readings. It is the primary statistic for the
    DOSE-RESPONSE SUB-QUESTION ONLY, and is never primary evidence for the
    experiment as a whole. It NEVER feeds ``classify``.

    A non-significant slope means only "no detected linear trend". It does not
    mean the relationship is flat, does not establish equivalence, and does not
    support an effective-horizon-only explanation.
    """

    x_units: str = "missing_patches (g / patch_size)"
    mean_slope_per_patch: float = float("nan")
    ci95_low: float = float("nan")
    ci95_high: float = float("nan")
    ci95_excludes_zero: bool = False
    # Descriptive only. Cannot trigger any confirmatory output.
    quadratic_term: float = float("nan")
    quadratic_ci95_low: float = float("nan")
    quadratic_ci95_high: float = float("nan")
    descriptive_only: bool = True
    non_significant_means: str = (
        "no detected linear trend; NOT flatness, NOT equivalence, and NOT support "
        "for an effective-horizon-only explanation"
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismDecision:
    label: str
    reason: str | None
    triggers: list[str]
    criteria: dict[str, Any]
    rule_version: str = MECHANISM_RULE_VERSION
    status: str = RULE_STATUS
    # The RULE was preregistered. Causal interpretation is NOT authoritative.
    authoritative: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismInputs:
    r_gaps: list[GapStat]
    b_gaps: list[GapStat] = field(default_factory=list)
    dose_response: DoseResponse | None = None
    mean_clean_mae: float = float("nan")
    config: dict[str, Any] = field(default_factory=dict)
    pilot_config: dict[str, Any] = field(default_factory=dict)


def sesoi_delta(pilot_config: dict[str, Any], mean_clean_mae: float) -> float:
    """delta = equivalence_band_frac_of_clean_mae * mean_clean_mae.

    Inherited from the original pilot's NO-GO rule; never redefined here.
    """
    frac = float(
        pilot_config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
    )
    return frac * float(mean_clean_mae)


def classify(inputs: MechanismInputs) -> MechanismDecision:
    """Classify the trailing-gap result. Precedence is fixed and exhaustive.

    1. >=1 MATERIAL_POSITIVE and >=1 MATERIAL_NEGATIVE -> DIRECTION_REVERSAL_ACROSS_GAPS
    2. all 4 EQUIVALENT                                -> NO_MATERIAL_DIFFERENCE_VS_TRUNCATION
    3. >=3 MATERIAL_POSITIVE and 0 MATERIAL_NEGATIVE   -> CONSISTENT_EXTRA_TRAILING_GAP_PENALTY
    4. >=3 MATERIAL_NEGATIVE and 0 MATERIAL_POSITIVE   -> CONSISTENT_LOWER_ERROR_WITH_TRAILING_NAN
    5. otherwise                                        -> INCONCLUSIVE + exactly one reason

    ``min_consistent_gaps`` (3) governs ONLY rules 3-4. ``equivalence_required_gaps``
    (4) governs ONLY rule 2: family-wise equivalence is an INTERSECTION across all
    four gaps. A 3-of-4 equivalent subset is NOT family-wise equivalence without a
    separately adjusted procedure, and falls through to INCONCLUSIVE.

    Neither ``b_gaps`` nor ``dose_response`` participates in any branch below.
    """
    delta = sesoi_delta(inputs.pilot_config, inputs.mean_clean_mae)
    rule_cfg = inputs.config["mechanism_rule"]
    min_consistent = int(rule_cfg["min_consistent_gaps"])
    equivalence_required = int(rule_cfg["equivalence_required_gaps"])

    readings = {stat.contrast_id: stat.reading(delta) for stat in inputs.r_gaps}
    flags = {stat.contrast_id: stat.flags(delta) for stat in inputs.r_gaps}
    counts = {
        value: sum(1 for r in readings.values() if r == value)
        for value in (EQUIVALENT, MATERIAL_POSITIVE, MATERIAL_NEGATIVE, UNRESOLVED)
    }
    n_gaps = len(readings)

    criteria: dict[str, Any] = {
        "rule_version": MECHANISM_RULE_VERSION,
        "status": RULE_STATUS,
        "authoritative_meaning": (
            "the decision rule was preregistered; NOT a claim that any causal "
            "interpretation is authoritative"
        ),
        "sesoi_delta": delta,
        "sesoi_source": (
            "pilot_config decision.no_go.equivalence_band_frac_of_clean_mae x "
            "mean_clean_mae — inherited, never redefined"
        ),
        "min_consistent_gaps": min_consistent,
        "min_consistent_gaps_applies_to": [CONSISTENT_PENALTY, CONSISTENT_LOWER_ERROR],
        "equivalence_required_gaps": equivalence_required,
        "equivalence_required_gaps_applies_to": [NO_MATERIAL_DIFFERENCE],
        "equivalence_is_an_intersection_note": (
            "Family-wise equivalence is an INTERSECTION: it requires ALL gaps equivalent. "
            "A subset (e.g. 3 of 4) "
            "is not family-wise equivalence without a separately adjusted procedure and "
            "falls through to INCONCLUSIVE."
        ),
        "r_readings": readings,
        "r_reading_counts": counts,
        "r_secondary_flags": flags,
        "secondary_flags_note": (
            "Diagnostic visibility only. A flag can never promote a gap to a material "
            "directional reading."
        ),
        "interpretation": {
            cid: INTERPRETATION[reading] for cid, reading in readings.items()
        },
        "b_readings": {
            stat.contrast_id: stat.reading(delta) for stat in inputs.b_gaps
        },
        "b_note": (
            "B_g is CONFOUNDED WITH DIFFERENCES IN REMOVED CONTENT between its arms. "
            "Reported alongside, never used to determine the classification."
        ),
        "dose_response": inputs.dose_response.as_dict() if inputs.dose_response else None,
        "dose_response_note": (
            "Primary statistic for the DOSE-RESPONSE SUB-QUESTION only; not primary "
            "evidence for the experiment, and never an input to this classification. "
            "A non-significant slope means only 'no detected linear trend'."
        ),
        "hypothesis_provenance": (
            "R_g/B_g were designed in response to the boundary jump, a POST-HOC finding "
            "identified after inspecting the original preregistered results. Even a clean "
            "result here tests a hypothesis generated from prior data and is one level "
            "removed from a fully independent confirmatory test."
        ),
    }

    # 1. Direction reversal outranks everything, including a 3-of-4 majority.
    if counts[MATERIAL_POSITIVE] >= 1 and counts[MATERIAL_NEGATIVE] >= 1:
        return MechanismDecision(
            label=DIRECTION_REVERSAL, reason=None,
            triggers=[
                f"{counts[MATERIAL_POSITIVE]}_material_positive_and_"
                f"{counts[MATERIAL_NEGATIVE]}_material_negative"
            ],
            criteria=criteria,
        )

    # 2. Family-wise equivalence: an intersection across ALL gaps.
    if counts[EQUIVALENT] == n_gaps and n_gaps >= equivalence_required:
        return MechanismDecision(
            label=NO_MATERIAL_DIFFERENCE, reason=None,
            triggers=[f"all_{n_gaps}_gaps_equivalent"],
            criteria=criteria,
        )

    # 3-4. Consistent directional readings.
    if counts[MATERIAL_POSITIVE] >= min_consistent and counts[MATERIAL_NEGATIVE] == 0:
        return MechanismDecision(
            label=CONSISTENT_PENALTY, reason=None,
            triggers=[f"{counts[MATERIAL_POSITIVE]}_of_{n_gaps}_material_positive"],
            criteria=criteria,
        )
    if counts[MATERIAL_NEGATIVE] >= min_consistent and counts[MATERIAL_POSITIVE] == 0:
        return MechanismDecision(
            label=CONSISTENT_LOWER_ERROR, reason=None,
            triggers=[f"{counts[MATERIAL_NEGATIVE]}_of_{n_gaps}_material_negative"],
            criteria=criteria,
        )

    # 5. INCONCLUSIVE, with exactly one machine-readable reason.
    n_material = counts[MATERIAL_POSITIVE] + counts[MATERIAL_NEGATIVE]
    if n_material >= 1:
        reason = REASON_SPARSE
    elif counts[UNRESOLVED] == n_gaps:
        reason = REASON_INSUFFICIENT
    else:
        reason = REASON_MIXED
    return MechanismDecision(
        label=INCONCLUSIVE, reason=reason, triggers=[reason], criteria=criteria
    )
