"""The frozen three-way replication rule: REPLICATED / CONTRADICTED / INCONCLUSIVE.

One worked example per outcome, plus the precedence case that distinguishes the
rule from a naive reading, plus an exhaustive proof that the three outcomes
partition all 64 combinations of three four-way readings. The rule must be
deterministic, must ignore R_128 and B_g entirely, and must not be reachable
with a missing primary contrast.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

from itertools import product

import pytest

from stats.mechanism_decision import (
    EQUIVALENT,
    MATERIAL_NEGATIVE,
    MATERIAL_POSITIVE,
    UNRESOLVED,
    GapStat,
)
from stats.replication_decision import (
    CONTRADICTED,
    INCONCLUSIVE,
    OUTCOMES,
    PRIMARY_CONTRASTS,
    REPLICATED,
    REPLICATION_RULE_VERSION,
    SECONDARY_STRESS_CONTRAST,
    ReplicationInputs,
    ReplicationRuleError,
    assert_exhaustive,
    classify_replication,
)

MEAN_CLEAN_MAE = 2.0
# delta = 0.03 * 2.0 = 0.06 under the inherited SESOI.
DELTA = 0.06
PILOT = {"decision": {"no_go": {"equivalence_band_frac_of_clean_mae": 0.03}}}


def _stat(contrast_id: str, reading: str, *, gap: int | None = None) -> GapStat:
    """A GapStat whose CIs really produce the requested four-way reading.

    Built from intervals rather than by stubbing ``reading()``, so the tests
    exercise the frozen per-gap rule instead of trusting a label.
    """
    gap = gap or int(contrast_id.split("_")[1])
    if reading == EQUIVALENT:
        # Whole 90% CI inside +/-delta.
        lo90, hi90, lo95, hi95 = -0.5 * DELTA, 0.5 * DELTA, -0.5 * DELTA, 0.5 * DELTA
    elif reading == MATERIAL_POSITIVE:
        # Holm lower bound strictly above +delta, 90% CI outside the band.
        lo90, hi90, lo95, hi95 = 2 * DELTA, 4 * DELTA, 2 * DELTA, 4 * DELTA
    elif reading == MATERIAL_NEGATIVE:
        lo90, hi90, lo95, hi95 = -4 * DELTA, -2 * DELTA, -4 * DELTA, -2 * DELTA
    elif reading == UNRESOLVED:
        # Straddles the band: not equivalent, and neither Holm bound clears it.
        lo90, hi90, lo95, hi95 = -3 * DELTA, 3 * DELTA, -3 * DELTA, 3 * DELTA
    else:
        raise AssertionError(f"unknown reading {reading!r}")
    stat = GapStat(
        contrast_id=contrast_id, gap=gap,
        mean_difference=0.5 * (lo95 + hi95), median_difference=0.5 * (lo95 + hi95),
        ci_low_holm=lo95, ci_high_holm=hi95,
        ci_low_equivalence=lo90, ci_high_equivalence=hi90,
        holm_rejects=bool(lo95 > 0 or hi95 < 0), mean_clean_mae=MEAN_CLEAN_MAE,
    )
    assert stat.reading(DELTA) == reading, (contrast_id, reading, stat.reading(DELTA))
    return stat


def _decide(primary: dict[str, str], *, r128: str = UNRESOLVED, b: str | None = None):
    r_gaps = [_stat(cid, primary[cid]) for cid in PRIMARY_CONTRASTS]
    r_gaps.append(_stat(SECONDARY_STRESS_CONTRAST, r128))
    b_gaps = (
        [_stat(f"B_{g}", b, gap=g) for g in (16, 32, 64, 128)] if b else []
    )
    return classify_replication(
        ReplicationInputs(
            r_gaps=r_gaps, b_gaps=b_gaps,
            mean_clean_mae=MEAN_CLEAN_MAE, pilot_config=PILOT,
        )
    )


# --------------------------------------------------------------------------- #
# Worked example per outcome
# --------------------------------------------------------------------------- #

def test_all_three_equivalent_is_REPLICATED():
    decision = _decide({c: EQUIVALENT for c in PRIMARY_CONTRASTS})
    assert decision.label == REPLICATED
    assert decision.reason is None
    assert decision.triggers == ["all_3_primary_gaps_equivalent"]
    assert decision.rule_version == REPLICATION_RULE_VERSION
    assert decision.authoritative is True


def test_one_material_positive_is_CONTRADICTED():
    decision = _decide({"R_16": EQUIVALENT, "R_32": MATERIAL_POSITIVE, "R_64": EQUIVALENT})
    assert decision.label == CONTRADICTED
    assert decision.triggers == [f"R_32={MATERIAL_POSITIVE}"]


def test_one_material_negative_is_also_CONTRADICTED():
    decision = _decide({"R_16": MATERIAL_NEGATIVE, "R_32": EQUIVALENT, "R_64": EQUIVALENT})
    assert decision.label == CONTRADICTED
    assert decision.triggers == [f"R_16={MATERIAL_NEGATIVE}"]


def test_unresolved_without_a_material_reading_is_INCONCLUSIVE():
    decision = _decide({"R_16": EQUIVALENT, "R_32": EQUIVALENT, "R_64": UNRESOLVED})
    assert decision.label == INCONCLUSIVE
    assert decision.reason == "unresolved_primary_gap_without_material_direction"
    assert decision.triggers == [f"R_64={UNRESOLVED}"]


def test_two_of_three_equivalent_is_not_a_partial_success():
    """The claim is an INTERSECTION. Two of three is a failure to replicate."""
    decision = _decide({"R_16": EQUIVALENT, "R_32": EQUIVALENT, "R_64": UNRESOLVED})
    assert decision.label != REPLICATED


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #

def test_CONTRADICTED_outranks_INCONCLUSIVE():
    """One material and one unresolved gap is CONTRADICTED, not INCONCLUSIVE.

    This is the case a naive 'any unresolved -> inconclusive' reading gets wrong.
    """
    decision = _decide(
        {"R_16": MATERIAL_POSITIVE, "R_32": UNRESOLVED, "R_64": EQUIVALENT}
    )
    assert decision.label == CONTRADICTED
    assert decision.criteria["precedence"].startswith("CONTRADICTED outranks")


def test_material_readings_in_both_directions_are_still_CONTRADICTED():
    decision = _decide(
        {"R_16": MATERIAL_POSITIVE, "R_32": MATERIAL_NEGATIVE, "R_64": EQUIVALENT}
    )
    assert decision.label == CONTRADICTED
    assert len(decision.triggers) == 2


def test_the_three_outcomes_partition_all_64_reading_combinations():
    """Proved by enumeration, not asserted in prose."""
    assert_exhaustive()
    seen = set()
    for triple in product(
        (EQUIVALENT, MATERIAL_POSITIVE, MATERIAL_NEGATIVE, UNRESOLVED), repeat=3
    ):
        decision = _decide(dict(zip(PRIMARY_CONTRASTS, triple)))
        assert decision.label in OUTCOMES
        seen.add(decision.label)
    assert seen == set(OUTCOMES), "every outcome must be reachable"


def test_the_rule_is_deterministic():
    primary = {"R_16": EQUIVALENT, "R_32": UNRESOLVED, "R_64": EQUIVALENT}
    first = _decide(primary)
    for _ in range(5):
        again = _decide(primary)
        assert again.label == first.label
        assert again.triggers == first.triggers
        assert again.reason == first.reason


# --------------------------------------------------------------------------- #
# What must NOT influence the outcome
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "r128", [EQUIVALENT, MATERIAL_POSITIVE, MATERIAL_NEGATIVE, UNRESOLVED]
)
def test_R_128_never_changes_the_outcome(r128):
    """R_128 is a secondary stress condition: measured, reported, excluded."""
    primary = {c: EQUIVALENT for c in PRIMARY_CONTRASTS}
    decision = _decide(primary, r128=r128)
    assert decision.label == REPLICATED, f"R_128={r128} changed the outcome"
    # ...and it is still REPORTED.
    assert decision.criteria["secondary_stress_reading"] == r128
    assert decision.criteria["secondary_stress_contrast"] == SECONDARY_STRESS_CONTRAST


@pytest.mark.parametrize(
    "b", [EQUIVALENT, MATERIAL_POSITIVE, MATERIAL_NEGATIVE, UNRESOLVED]
)
def test_B_g_never_changes_the_outcome(b):
    """B_g is secondary and confounded; it is reported alongside, never decisive."""
    primary = {c: EQUIVALENT for c in PRIMARY_CONTRASTS}
    decision = _decide(primary, b=b)
    assert decision.label == REPLICATED, f"B={b} changed the outcome"
    assert set(decision.criteria["b_readings"].values()) == {b}
    assert "never determines" in decision.criteria["b_note"]


def test_a_missing_primary_contrast_is_refused_not_silently_decided():
    """Deciding on a subset of the primary contrasts is not the frozen rule."""
    r_gaps = [_stat("R_16", EQUIVALENT), _stat("R_32", EQUIVALENT)]
    with pytest.raises(ReplicationRuleError, match="R_64"):
        classify_replication(
            ReplicationInputs(
                r_gaps=r_gaps, mean_clean_mae=MEAN_CLEAN_MAE, pilot_config=PILOT
            )
        )


# --------------------------------------------------------------------------- #
# What the record must say
# --------------------------------------------------------------------------- #

def test_the_sesoi_is_inherited_from_the_pilot_never_redefined():
    decision = _decide({c: EQUIVALENT for c in PRIMARY_CONTRASTS})
    assert decision.criteria["sesoi_delta"] == pytest.approx(DELTA)
    assert "inherited from the ETTh1 pilot" in decision.criteria["sesoi_source"]


def test_the_multiplicity_note_states_the_intersection_union_argument():
    criteria = _decide({c: EQUIVALENT for c in PRIMARY_CONTRASTS}).criteria
    note = criteria["multiplicity"]
    assert "no additional correction" in note.lower()
    assert "at least one component equivalence null is true" in note
    assert "regardless of dependence" in note


def test_authoritative_does_not_claim_interpretive_authority():
    decision = _decide({c: EQUIVALENT for c in PRIMARY_CONTRASTS})
    assert decision.authoritative is True
    assert "does NOT mean" in decision.status
    assert "NOT a claim" in decision.criteria["authoritative_meaning"]


def test_the_record_carries_the_hypothesis_provenance_and_the_etth1_result():
    criteria = _decide({c: EQUIVALENT for c in PRIMARY_CONTRASTS}).criteria
    assert "SELECTED after seeing ETTh1" in criteria["hypothesis_provenance"]
    assert criteria["etth1_result_being_replicated"] == (
        "INCONCLUSIVE / mixed_equivalent_and_unresolved"
    )
    assert "does not alter, relabel or reinterpret" in criteria["etth1_result_note"]


def test_even_REPLICATED_does_not_claim_a_mechanism():
    from stats.replication_decision import OUTCOME_INTERPRETATION

    text = OUTCOME_INTERPRETATION[REPLICATED]
    assert "does NOT establish a mechanism" in text
    assert "does not retrospectively upgrade" in text
    assert "exploratorily selected" in text
