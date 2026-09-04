"""Preregistered three-way replication rule for the ETTh2 trailing-gap replication.

    *** RULE FROZEN BEFORE EXECUTION. rule_version: etth2-replication-v1 ***

``authoritative: True`` means **the decision rule itself was preregistered** —
its components, its precedence and its multiplicity treatment were fixed before
any ETTh2 forecast was produced, so the outcome cannot have been tuned to the
data. It does **NOT** mean the replication's substantive interpretation is
authoritative, and it does not upgrade the hypothesis's provenance: the primary
claim was SELECTED after seeing ETTh1's per-gap readings, so even a clean
`REPLICATED` establishes that an exploratorily selected pattern held on a second
dataset — not that a pattern specified before any data were seen did.

The three outcomes
------------------
Decided from the readings of the three PRIMARY gaps ``R_16``, ``R_32``, ``R_64``
under the frozen four-way per-gap rule (``stats.mechanism_decision.GapStat``),
which is reused here rather than reimplemented:

  REPLICATED    all three read EQUIVALENT
  CONTRADICTED  at least one reads MATERIAL_POSITIVE or MATERIAL_NEGATIVE
  INCONCLUSIVE  no material directional reading, but at least one UNRESOLVED

Precedence
----------
``CONTRADICTED`` outranks ``INCONCLUSIVE``. A material directional reading
contradicts the equivalence claim whatever the other gaps do, so a run with one
material and one unresolved gap is CONTRADICTED, not INCONCLUSIVE. The three
outcomes are mutually exclusive and exhaustive over the four possible readings,
which ``assert_exhaustive`` proves rather than asserts in prose.

What is NOT in this decision
----------------------------
``R_128`` is a secondary stress condition. It is measured, reported, and
excluded from this rule — it is not one of the three gaps that read EQUIVALENT
on ETTh1, so including it would silently change the hypothesis being replicated.
``B_g`` remains secondary and confounded with differences in removed content
between its arms; it is reported alongside and never determines the outcome.

Multiplicity
------------
Each primary component is checked against its own NOMINAL 90% equivalence CI.
No additional Holm or Bonferroni correction is applied to the intersection.
This is an intersection-union test: under the global null at least one component
equivalence null is true, and rejecting the global null requires rejecting every
component null including that true one, so the probability of a false global
rejection is no greater than the size of any true component test, regardless of
dependence. Holm across the four ``R`` gaps is retained for MATERIAL directional
readings only, exactly as on ETTh1, and is what ``GapStat.reading`` already
consumes through ``ci_low_holm`` / ``ci_high_holm``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from stats.mechanism_decision import (
    EQUIVALENT,
    MATERIAL_NEGATIVE,
    MATERIAL_POSITIVE,
    MATERIAL_READINGS,
    UNRESOLVED,
    GapStat,
    sesoi_delta,
)

REPLICATION_RULE_VERSION = "etth2-replication-v1"
RULE_STATUS = (
    "Rule frozen before ETTh2 execution. authoritative=True means the DECISION "
    "RULE was preregistered — its components, precedence and multiplicity "
    "treatment fixed before any forecast — and does NOT mean the replication's "
    "substantive interpretation is authoritative."
)

# The PRIMARY intersection, in the order the preregistration names it. R_128 is
# absent by design; see the module docstring.
PRIMARY_CONTRASTS = ("R_16", "R_32", "R_64")
SECONDARY_STRESS_CONTRAST = "R_128"

REPLICATED = "REPLICATED"
CONTRADICTED = "CONTRADICTED"
INCONCLUSIVE = "INCONCLUSIVE"

OUTCOMES = (REPLICATED, CONTRADICTED, INCONCLUSIVE)

OUTCOME_INTERPRETATION: dict[str, str] = {
    REPLICATED: (
        "all three primary gaps read EQUIVALENT on ETTh2, as they did on ETTh1. "
        "An exploratorily selected pattern held on a second dataset. This does "
        "NOT establish a mechanism, does not demonstrate an "
        "'effective-horizon-only' explanation, and does not retrospectively "
        "upgrade the ETTh1 classification."
    ),
    CONTRADICTED: (
        "at least one primary gap read a material directional difference on "
        "ETTh2, by more than the SESOI. The ETTh1 equivalence pattern did not "
        "hold. This is a finding about the pattern's generality, not evidence "
        "for any particular internal component."
    ),
    INCONCLUSIVE: (
        "no primary gap read a material directional difference, but at least one "
        "was UNRESOLVED, so equivalence was not established across all three. "
        "Non-significance is not equivalence; this is a failure to resolve, not "
        "a replication and not a contradiction."
    ),
}

MULTIPLICITY_NOTE = (
    "Each primary component is checked against its own nominal 90% equivalence "
    "CI; no additional correction is applied to the intersection. "
    "Intersection-union test: under the global null at least one component "
    "equivalence null is true, and rejecting the global null requires rejecting "
    "every component null including that true one, so the probability of a false "
    "global rejection is no greater than the size of any true component test, "
    "regardless of dependence."
)

B_FAMILY_NOTE = (
    "B_g is CONFOUNDED WITH DIFFERENCES IN REMOVED CONTENT between its arms, is "
    "corrected as its own separate Holm family, and never determines this "
    "outcome. R and B are NOT assumed statistically independent — they share "
    "origins and the trailing_nan arm — and no sign is claimed for that "
    "dependence."
)

R128_NOTE = (
    "R_128 is a secondary stress condition: measured and reported, excluded from "
    "the primary decision. It did not read EQUIVALENT on ETTh1 and is not part of "
    "the hypothesis being replicated; including it would change that hypothesis."
)

HYPOTHESIS_PROVENANCE = (
    "The primary claim (R_16 AND R_32 AND R_64 equivalent) was SELECTED after "
    "seeing ETTh1's per-gap readings. It is a confirmatory hypothesis derived "
    "from an exploratory finding, one level further removed than the trailing-gap "
    "experiment itself, and no report may present it as an independent a priori "
    "claim."
)


class ReplicationRuleError(RuntimeError):
    """The inputs cannot be classified as specified."""


@dataclass
class ReplicationDecision:
    label: str
    reason: str | None
    triggers: list[str]
    criteria: dict[str, Any]
    rule_version: str = REPLICATION_RULE_VERSION
    status: str = RULE_STATUS
    # The RULE was preregistered. The substantive interpretation is NOT
    # authoritative, and neither is the hypothesis's provenance improved by it.
    authoritative: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplicationInputs:
    """Everything the rule consumes. Only ``r_gaps`` can change the outcome."""

    r_gaps: list[GapStat]
    b_gaps: list[GapStat] = field(default_factory=list)
    mean_clean_mae: float = float("nan")
    pilot_config: dict[str, Any] = field(default_factory=dict)


def assert_exhaustive() -> None:
    """Prove the three outcomes partition every possible triple of readings.

    Called by the tests rather than trusted from the prose above: with four
    possible readings per gap and three gaps there are 4**3 = 64 combinations,
    and every one must land in exactly one outcome.
    """
    from itertools import product

    readings = (EQUIVALENT, MATERIAL_POSITIVE, MATERIAL_NEGATIVE, UNRESOLVED)
    for triple in product(readings, repeat=3):
        matched = [
            outcome
            for outcome, holds in (
                (REPLICATED, all(r == EQUIVALENT for r in triple)),
                (CONTRADICTED, any(r in MATERIAL_READINGS for r in triple)),
                (
                    INCONCLUSIVE,
                    not any(r in MATERIAL_READINGS for r in triple)
                    and any(r == UNRESOLVED for r in triple),
                ),
            )
            if holds
        ]
        if len(matched) != 1:
            raise AssertionError(
                f"readings {triple} match {matched}; the outcomes must partition "
                f"every combination into exactly one"
            )


def classify_replication(inputs: ReplicationInputs) -> ReplicationDecision:
    """Decide REPLICATED / CONTRADICTED / INCONCLUSIVE. Precedence is fixed.

    1. >=1 primary gap MATERIAL_POSITIVE or MATERIAL_NEGATIVE -> CONTRADICTED
    2. all 3 primary gaps EQUIVALENT                          -> REPLICATED
    3. otherwise (>=1 UNRESOLVED, none material)              -> INCONCLUSIVE

    Rule 1 is written first because CONTRADICTED outranks INCONCLUSIVE: a
    material directional reading contradicts the equivalence claim even when
    another gap is merely unresolved. Rules 1 and 2 are mutually exclusive, so
    their order between themselves does not matter; ``assert_exhaustive`` proves
    the partition holds over all 64 reading combinations.

    Neither ``b_gaps`` nor ``R_128`` participates in any branch.
    """
    delta = sesoi_delta(inputs.pilot_config, inputs.mean_clean_mae)

    by_id = {stat.contrast_id: stat for stat in inputs.r_gaps}
    missing = [cid for cid in PRIMARY_CONTRASTS if cid not in by_id]
    if missing:
        raise ReplicationRuleError(
            f"the primary intersection needs {list(PRIMARY_CONTRASTS)}; missing "
            f"{missing}. A replication decided on a subset of its own primary "
            f"contrasts is not the preregistered decision."
        )

    primary = {cid: by_id[cid].reading(delta) for cid in PRIMARY_CONTRASTS}
    all_r = {stat.contrast_id: stat.reading(delta) for stat in inputs.r_gaps}
    material = [cid for cid, r in primary.items() if r in MATERIAL_READINGS]
    equivalent = [cid for cid, r in primary.items() if r == EQUIVALENT]
    unresolved = [cid for cid, r in primary.items() if r == UNRESOLVED]

    criteria: dict[str, Any] = {
        "rule_version": REPLICATION_RULE_VERSION,
        "status": RULE_STATUS,
        "authoritative_meaning": (
            "the decision rule was preregistered; NOT a claim that the "
            "replication's substantive interpretation is authoritative"
        ),
        "sesoi_delta": delta,
        "sesoi_source": (
            "pilot_config decision.no_go.equivalence_band_frac_of_clean_mae x "
            "mean_clean_mae — inherited from the ETTh1 pilot, never redefined"
        ),
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "primary_readings": primary,
        "primary_reading_counts": {
            value: sum(1 for r in primary.values() if r == value)
            for value in (EQUIVALENT, MATERIAL_POSITIVE, MATERIAL_NEGATIVE, UNRESOLVED)
        },
        "primary_secondary_flags": {
            cid: by_id[cid].flags(delta) for cid in PRIMARY_CONTRASTS
        },
        "secondary_flags_note": (
            "Diagnostic visibility only. A flag can never promote a gap to a "
            "material directional reading, and never changes the outcome."
        ),
        "all_r_readings": all_r,
        "secondary_stress_contrast": SECONDARY_STRESS_CONTRAST,
        "secondary_stress_reading": all_r.get(SECONDARY_STRESS_CONTRAST),
        "r128_note": R128_NOTE,
        "b_readings": {stat.contrast_id: stat.reading(delta) for stat in inputs.b_gaps},
        "b_note": B_FAMILY_NOTE,
        "multiplicity": MULTIPLICITY_NOTE,
        "precedence": (
            "CONTRADICTED outranks INCONCLUSIVE. A material directional reading "
            "contradicts the equivalence claim even if another gap is unresolved."
        ),
        "outcome_interpretations": OUTCOME_INTERPRETATION,
        "hypothesis_provenance": HYPOTHESIS_PROVENANCE,
        "etth1_result_being_replicated": "INCONCLUSIVE / mixed_equivalent_and_unresolved",
        "etth1_result_note": (
            "This replication does not alter, relabel or reinterpret the ETTh1 "
            "classification, whatever it finds."
        ),
    }

    # 1. CONTRADICTED outranks INCONCLUSIVE.
    if material:
        return ReplicationDecision(
            label=CONTRADICTED,
            reason="material_directional_reading_in_the_primary_intersection",
            triggers=[f"{cid}={primary[cid]}" for cid in material],
            criteria=criteria,
        )

    # 2. The intersection claim: ALL three, not a majority.
    if len(equivalent) == len(PRIMARY_CONTRASTS):
        return ReplicationDecision(
            label=REPLICATED,
            reason=None,
            triggers=[f"all_{len(PRIMARY_CONTRASTS)}_primary_gaps_equivalent"],
            criteria=criteria,
        )

    # 3. No material reading, at least one unresolved.
    return ReplicationDecision(
        label=INCONCLUSIVE,
        reason="unresolved_primary_gap_without_material_direction",
        triggers=[f"{cid}={primary[cid]}" for cid in unresolved],
        criteria=criteria,
    )
