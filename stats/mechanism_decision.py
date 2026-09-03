"""DRAFT mechanism classification rule for the trailing-gap experiment.

    *** PROPOSAL. NOT AUTHORITATIVE. AWAITING RESEARCH LEAD SIGN-OFF. ***

    The thresholds below are a starting point for review, not a decision the
    pilot has adopted. Nothing has been executed against them.

Unlike the original pilot's GO/PIVOT/NO-GO rule, this classifies a MECHANISM.
It is deterministic and emits one label plus the specific conditions that
produced it, in the same style as ``stats/decision.py`` so it can be audited
the same way.

What R_g means
--------------
    R_g = MAE(trailing_nan(g)) - MAE(truncated_long(g))

Both arms lose the same g most-recent observations and both are scored against
the same target over the same timestamps. They differ only in whether the model
is SHOWN a trailing run of NaNs or simply handed a shorter context and asked for
a longer horizon.

  * R_g inside the equivalence band  -> consistent with an EFFECTIVE-HORIZON-ONLY
    story: the trailing NaNs cost nothing beyond the lost recency.
  * R_g > 0 and outside the band     -> consistent with an EXTRA PENALTY specific
    to the explicit trailing-NaN representation, masking, or position handling.
  * R_g < 0 and outside the band     -> the explicit NaN framing OUTPERFORMS the
    truncated-autoregressive framing.

"Consistent with" is the strongest available reading. None of these labels is a
demonstrated mechanism, and the rule's output must not be reported as one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from stats.decision import ci_within_equivalence_band

MECHANISM_RULE_VERSION = "draft-v1"
RULE_STATUS = "PROPOSAL — thresholds await Research Lead sign-off; not authoritative"

MechanismLabel = Literal[
    "EFFECTIVE_HORIZON_ONLY",
    "EXTRA_TRAILING_NAN_PENALTY",
    "NAN_FRAMING_OUTPERFORMS",
    "INCONSISTENT_ACROSS_GAPS",
    "INCONCLUSIVE",
]

# Per-gap readings.
GAP_EQUIVALENT = "equivalent"        # CI inside the band
GAP_POSITIVE = "positive"            # significant and above the band
GAP_NEGATIVE = "negative"            # significant and below the band
GAP_INDETERMINATE = "indeterminate"  # neither equivalent nor significant


@dataclass
class GapStat:
    """One R_g (or B_g), as the classification rule consumes it."""

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

    def reading(self, band: float) -> str:
        """Classify this gap: equivalent, positive, negative, or indeterminate.

        Equivalence is decided by the SAME convention the pilot uses everywhere
        — the 90% CI entirely inside the band — and is checked FIRST, so a
        result that is both statistically significant and practically negligible
        reads as equivalent rather than as an effect.
        """
        if ci_within_equivalence_band(self.ci_low_equivalence, self.ci_high_equivalence, band):
            return GAP_EQUIVALENT
        if not self.holm_rejects:
            return GAP_INDETERMINATE
        if self.ci_low_holm > 0:
            return GAP_POSITIVE
        if self.ci_high_holm < 0:
            return GAP_NEGATIVE
        return GAP_INDETERMINATE

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoseResponse:
    """Slope of R_g against g, bootstrapped across origins."""

    mean_slope: float
    ci95_low: float
    ci95_high: float
    ci95_excludes_zero: bool
    # Secondary, descriptive only: is a straight line an adequate summary?
    quadratic_term: float = float("nan")
    quadratic_ci95_low: float = float("nan")
    quadratic_ci95_high: float = float("nan")
    quadratic_ci95_excludes_zero: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MechanismDecision:
    label: str
    triggers: list[str]
    criteria: dict[str, Any]
    rule_version: str = MECHANISM_RULE_VERSION
    status: str = RULE_STATUS
    authoritative: bool = False

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


def classify(inputs: MechanismInputs) -> MechanismDecision:
    """Classify the trailing-gap mechanism. DRAFT — not authoritative.

    Precedence, fixed in advance:
      1. Readings disagree in DIRECTION across gaps -> INCONSISTENT_ACROSS_GAPS.
         A mechanism that reverses sign with gap length is not one mechanism.
      2. Enough gaps agree on one reading -> that label.
      3. Otherwise -> INCONCLUSIVE.
    """
    # The SESOI is the pilot's, never redefined here.
    band = float(
        inputs.pilot_config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
    ) * float(inputs.mean_clean_mae)
    min_consistent = int(inputs.config["mechanism_rule"]["min_consistent_gaps"])

    readings = {stat.contrast_id: stat.reading(band) for stat in inputs.r_gaps}
    counts = {
        value: sum(1 for r in readings.values() if r == value)
        for value in (GAP_EQUIVALENT, GAP_POSITIVE, GAP_NEGATIVE, GAP_INDETERMINATE)
    }

    b_readings = {stat.contrast_id: stat.reading(band) for stat in inputs.b_gaps}

    criteria: dict[str, Any] = {
        "rule_version": MECHANISM_RULE_VERSION,
        "status": RULE_STATUS,
        "authoritative": False,
        "equivalence_band_abs": band,
        "equivalence_band_source": (
            "pilot_config decision.no_go.equivalence_band_frac_of_clean_mae — "
            "the SESOI is inherited, never redefined"
        ),
        "min_consistent_gaps": min_consistent,
        "r_readings": readings,
        "r_reading_counts": counts,
        "b_readings": b_readings,
        "b_interpretation_note": (
            "B_g tests a boundary-specific penalty but remains CONFOUNDED WITH "
            "DIFFERENCES IN REMOVED CONTENT between the two arms. It is not a pure "
            "mechanism test and never determines this label."
        ),
        "dose_response": inputs.dose_response.as_dict() if inputs.dose_response else None,
        "hypothesis_provenance": (
            "R_g/B_g were designed in response to the boundary jump, a POST-HOC finding "
            "identified after inspecting the original preregistered results. Even a clean "
            "result here tests a hypothesis generated from prior data and is one level "
            "removed from a fully independent confirmatory test."
        ),
    }

    directional = {counts[GAP_POSITIVE] > 0, counts[GAP_NEGATIVE] > 0}
    if directional == {True}:
        return MechanismDecision(
            label="INCONSISTENT_ACROSS_GAPS",
            triggers=["r_readings_disagree_in_direction"],
            criteria=criteria,
        )

    for value, label in (
        (GAP_EQUIVALENT, "EFFECTIVE_HORIZON_ONLY"),
        (GAP_POSITIVE, "EXTRA_TRAILING_NAN_PENALTY"),
        (GAP_NEGATIVE, "NAN_FRAMING_OUTPERFORMS"),
    ):
        if counts[value] >= min_consistent:
            return MechanismDecision(
                label=label,
                triggers=[f"{counts[value]}_of_{len(readings)}_gaps_read_{value}"],
                criteria=criteria,
            )

    return MechanismDecision(
        label="INCONCLUSIVE",
        triggers=["no_reading_reached_the_consistency_threshold"],
        criteria=criteria,
    )
