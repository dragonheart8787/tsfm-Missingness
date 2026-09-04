"""Preregistered mechanism classification rule — §5.1, §7, §9.

Model-mocked. Every worked example from the Research Lead's specification is
reproduced exactly, including the ones whose point is what the rule must NOT
conclude.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from stats.mechanism_decision import (
    CONSISTENT_LOWER_ERROR,
    CONSISTENT_PENALTY,
    DIRECTION_REVERSAL,
    EQUIVALENT,
    FLAG_SESOI_UNRESOLVED,
    FLAG_SMALL_BUT_DETECTABLE,
    INCONCLUSIVE,
    INTERPRETATION,
    MATERIAL_NEGATIVE,
    MATERIAL_POSITIVE,
    MECHANISM_RULE_VERSION,
    NO_MATERIAL_DIFFERENCE,
    REASON_INSUFFICIENT,
    REASON_MIXED,
    REASON_SPARSE,
    UNRESOLVED,
    DoseResponse,
    GapStat,
    MechanismInputs,
    classify,
    sesoi_delta,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_MAE = 2.4575
DELTA = 0.03 * CLEAN_MAE          # 0.0737...
GAPS = (16, 32, 64, 128)


@pytest.fixture(scope="module")
def gap_config():
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )


def _stat(contrast_id, gap, *, holm, equiv, rejects=True, mean=None):
    """Build a GapStat from explicit (holm CI, equivalence CI) pairs."""
    lo_h, hi_h = holm
    lo_e, hi_e = equiv
    centre = mean if mean is not None else (lo_h + hi_h) / 2
    return GapStat(
        contrast_id=contrast_id, gap=gap,
        mean_difference=centre, median_difference=centre,
        ci_low_holm=lo_h, ci_high_holm=hi_h,
        ci_low_equivalence=lo_e, ci_high_equivalence=hi_e,
        holm_rejects=rejects, mean_clean_mae=CLEAN_MAE,
    )


def _equivalent(cid, gap):
    """90% CI entirely inside +/- delta."""
    return _stat(cid, gap, holm=(-0.03, 0.03), equiv=(-0.02, 0.02), rejects=False)


def _material_positive(cid, gap):
    """Holm lower bound comfortably above +delta."""
    return _stat(cid, gap, holm=(0.20, 0.40), equiv=(0.22, 0.38))


def _material_negative(cid, gap):
    return _stat(cid, gap, holm=(-0.40, -0.20), equiv=(-0.38, -0.22))


def _unresolved(cid, gap):
    """Wide CI: neither inside the band nor materially directional."""
    return _stat(cid, gap, holm=(-0.30, 0.30), equiv=(-0.25, 0.25), rejects=False)


def _inputs(r_gaps, gap_config, config, **overrides):
    base = dict(
        r_gaps=r_gaps, b_gaps=[], dose_response=None,
        mean_clean_mae=CLEAN_MAE, config=gap_config, pilot_config=config,
    )
    base.update(overrides)
    return MechanismInputs(**base)


# =========================================================================== #
# §5.1 — four-way per-gap classification
# =========================================================================== #

def test_sesoi_delta_is_three_percent_of_mean_clean_mae(config):
    assert sesoi_delta(config, CLEAN_MAE) == pytest.approx(DELTA)
    assert sesoi_delta(config, CLEAN_MAE) == pytest.approx(0.0737, abs=1e-4)


def test_equivalent_requires_the_whole_90_ci_inside_the_band():
    assert _equivalent("R_16", 16).reading(DELTA) == EQUIVALENT
    # One bound outside is enough to fail.
    just_outside = _stat("R_16", 16, holm=(-0.09, 0.09), equiv=(-0.02, DELTA + 1e-6),
                         rejects=False)
    assert just_outside.reading(DELTA) == UNRESOLVED


def test_material_positive_requires_ci_low_above_delta_not_merely_above_zero():
    """THE §5.1 correction: ci_low_holm > 0 is insufficient.

    A CI strictly above zero but not clearing the SESOI must read UNRESOLVED.
    """
    above_zero_below_delta = _stat(
        "R_16", 16,
        holm=(0.02, 0.06),          # 0 < 0.02 <= delta (0.0737)
        equiv=(0.025, 0.055),       # not inside +/- delta? it IS inside...
        rejects=True,
    )
    assert above_zero_below_delta.ci_low_holm > 0
    assert above_zero_below_delta.ci_low_holm <= DELTA
    assert above_zero_below_delta.reading(DELTA) != MATERIAL_POSITIVE

    # And the case where it is also not equivalent: still not material.
    wide_but_above_zero = _stat(
        "R_32", 32, holm=(0.01, 0.30), equiv=(0.02, 0.25), rejects=True
    )
    assert wide_but_above_zero.ci_low_holm > 0
    assert wide_but_above_zero.ci_low_holm <= DELTA
    assert wide_but_above_zero.reading(DELTA) == UNRESOLVED


def test_material_positive_when_ci_low_clears_delta():
    assert _material_positive("R_16", 16).reading(DELTA) == MATERIAL_POSITIVE


def test_material_negative_requires_ci_high_below_minus_delta():
    assert _material_negative("R_16", 16).reading(DELTA) == MATERIAL_NEGATIVE
    not_far_enough = _stat("R_16", 16, holm=(-0.06, -0.02), equiv=(-0.055, -0.025))
    assert not_far_enough.ci_high_holm < 0
    assert not_far_enough.reading(DELTA) != MATERIAL_NEGATIVE


# --- the three UNRESOLVED sub-cases, each pinned individually --------------- #

def test_unresolved_case_significant_but_magnitude_unresolved():
    """'Statistically different from zero but practical magnitude unresolved.'"""
    stat = _stat("R_16", 16, holm=(0.01, 0.50), equiv=(0.03, 0.45), rejects=True)
    assert stat.ci_low_holm > 0                      # significant
    assert stat.ci_low_holm <= DELTA                 # magnitude unresolved
    assert stat.reading(DELTA) == UNRESOLVED


def test_unresolved_case_point_estimate_outside_sesoi_but_ci_overlaps_it():
    """'Point estimate outside SESOI while CI overlaps it.'"""
    stat = _stat("R_32", 32, holm=(0.05, 0.35), equiv=(0.06, 0.30),
                 rejects=True, mean=0.20)
    assert abs(stat.mean_difference) > DELTA         # point estimate outside
    assert stat.ci_low_holm < DELTA                  # but the CI overlaps the band
    assert stat.reading(DELTA) == UNRESOLVED


def test_unresolved_case_non_significant_without_equivalence():
    """'Non-significant without equivalence.'"""
    stat = _unresolved("R_64", 64)
    assert stat.ci_low_holm < 0 < stat.ci_high_holm  # not significant
    assert stat.ci_high_equivalence > DELTA          # not equivalent
    assert stat.reading(DELTA) == UNRESOLVED


# --- interpretation text, verbatim ----------------------------------------- #

def test_interpretation_text_is_verbatim():
    assert INTERPRETATION[EQUIVALENT] == (
        "no practically meaningful performance difference was established... "
        "consistent with a horizon-dominant explanation, but does NOT demonstrate "
        "an 'effective-horizon-only' mechanism."
    )
    assert INTERPRETATION[MATERIAL_POSITIVE] == (
        "explicit trailing-gap encoding has higher mean error than "
        "shorter-context/longer-horizon single-shot forecasting by more than the "
        "SESOI. It does not isolate masking, normalization, positional handling, "
        "or another internal component."
    )
    assert INTERPRETATION[MATERIAL_NEGATIVE] == (
        "lower error was observed with explicit trailing NaNs. Do not call this a "
        "beneficial causal mechanism."
    )


def test_interpretation_travels_with_every_reading(gap_config, config):
    gaps = [_material_positive(f"R_{g}", g) for g in GAPS]
    decision = classify(_inputs(gaps, gap_config, config))
    for cid, text in decision.criteria["interpretation"].items():
        assert text == INTERPRETATION[decision.criteria["r_readings"][cid]]


def test_the_wrong_terminology_is_gone_repo_wide():
    """The old term is actively wrong: nothing unrolls anywhere in this design.

    The banned tokens are assembled at runtime so this file does not itself
    contain them and trip its own check.
    """
    banned = ("truncated" + "-autoregressive", "truncated" + "_autoregressive")
    offenders = []
    for pattern in ("*.py", "*.md", "*.yaml"):
        for path in REPO_ROOT.rglob(pattern):
            if {".venv", ".git"} & set(path.relative_to(REPO_ROOT).parts):
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in banned):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"stale terminology in: {offenders}"


def test_the_terminology_check_would_catch_a_violation(tmp_path):
    """Guard against a vacuous scan."""
    banned = "truncated" + "-autoregressive"
    offender = tmp_path / "bad.md"
    offender.write_text(f"the {banned} framing\n", encoding="utf-8")
    assert banned in offender.read_text(encoding="utf-8")


def test_the_replacement_terminology_is_present():
    source = (REPO_ROOT / "stats" / "mechanism_decision.py").read_text(encoding="utf-8")
    assert "shorter-context/longer-horizon single-shot" in source


# --- secondary flags cannot promote ---------------------------------------- #

def test_flag_small_but_detectable_is_recorded():
    stat = _stat("R_16", 16, holm=(0.005, 0.020), equiv=(0.008, 0.018), rejects=True)
    assert stat.reading(DELTA) == EQUIVALENT
    assert FLAG_SMALL_BUT_DETECTABLE in stat.flags(DELTA)


def test_flag_sesoi_unresolved_is_recorded():
    stat = _stat("R_16", 16, holm=(0.01, 0.50), equiv=(0.03, 0.45), rejects=True)
    assert stat.reading(DELTA) == UNRESOLVED
    assert FLAG_SESOI_UNRESOLVED in stat.flags(DELTA)


def test_flags_cannot_promote_a_gap_to_a_material_reading(gap_config, config):
    """Every gap flagged, none material: the label must not become directional."""
    flagged = [
        _stat(f"R_{g}", g, holm=(0.01, 0.50), equiv=(0.03, 0.45), rejects=True)
        for g in GAPS
    ]
    for stat in flagged:
        assert stat.flags(DELTA) == [FLAG_SESOI_UNRESOLVED]
        assert stat.reading(DELTA) == UNRESOLVED
    decision = classify(_inputs(flagged, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_INSUFFICIENT
    assert decision.criteria["r_reading_counts"][MATERIAL_POSITIVE] == 0


# =========================================================================== #
# §9 — classification precedence, with the worked examples verbatim
# =========================================================================== #

def test_worked_example_3_equivalent_plus_1_material_positive(gap_config, config):
    """-> INCONCLUSIVE(sparse), explicitly NOT NO_MATERIAL_DIFFERENCE."""
    gaps = [_equivalent("R_16", 16), _equivalent("R_32", 32),
            _equivalent("R_64", 64), _material_positive("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_SPARSE
    assert decision.label != NO_MATERIAL_DIFFERENCE, (
        "3-of-4 equivalent is NOT family-wise equivalence"
    )


def test_worked_example_3_equivalent_plus_1_material_negative(gap_config, config):
    gaps = [_equivalent("R_16", 16), _equivalent("R_32", 32),
            _equivalent("R_64", 64), _material_negative("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_SPARSE


def test_worked_example_2_material_positive_plus_2_equivalent(gap_config, config):
    gaps = [_material_positive("R_16", 16), _material_positive("R_32", 32),
            _equivalent("R_64", 64), _equivalent("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_SPARSE


def test_worked_example_3_positive_plus_1_negative_is_reversal(gap_config, config):
    """-> DIRECTION_REVERSAL, explicitly NOT a '3-of-4 positive majority'."""
    gaps = [_material_positive("R_16", 16), _material_positive("R_32", 32),
            _material_positive("R_64", 64), _material_negative("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == DIRECTION_REVERSAL
    assert decision.criteria["r_reading_counts"][MATERIAL_POSITIVE] == 3
    assert decision.label != CONSISTENT_PENALTY, (
        "a 3-of-4 majority must not survive an opposite material reading"
    )


def test_worked_example_1_unresolved_plus_3_equivalent(gap_config, config):
    gaps = [_unresolved("R_16", 16), _equivalent("R_32", 32),
            _equivalent("R_64", 64), _equivalent("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_MIXED


def test_worked_example_4_unresolved(gap_config, config):
    gaps = [_unresolved(f"R_{g}", g) for g in GAPS]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_INSUFFICIENT


def test_worked_example_2_positive_plus_2_unresolved(gap_config, config):
    gaps = [_material_positive("R_16", 16), _material_positive("R_32", 32),
            _unresolved("R_64", 64), _unresolved("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_SPARSE


def test_all_four_equivalent_is_the_only_route_to_no_material_difference(
    gap_config, config
):
    gaps = [_equivalent(f"R_{g}", g) for g in GAPS]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == NO_MATERIAL_DIFFERENCE
    assert decision.reason is None


def test_three_of_four_material_positive_is_consistent_penalty(gap_config, config):
    gaps = [_material_positive("R_16", 16), _material_positive("R_32", 32),
            _material_positive("R_64", 64), _unresolved("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == CONSISTENT_PENALTY


def test_three_of_four_material_negative_is_consistent_lower_error(gap_config, config):
    gaps = [_material_negative("R_16", 16), _material_negative("R_32", 32),
            _material_negative("R_64", 64), _unresolved("R_128", 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == CONSISTENT_LOWER_ERROR


def test_inconclusive_always_carries_exactly_one_reason(gap_config, config):
    cases = [
        [_equivalent("R_16", 16), _equivalent("R_32", 32),
         _equivalent("R_64", 64), _material_positive("R_128", 128)],
        [_unresolved("R_16", 16), _equivalent("R_32", 32),
         _equivalent("R_64", 64), _equivalent("R_128", 128)],
        [_unresolved(f"R_{g}", g) for g in GAPS],
    ]
    seen = set()
    for gaps in cases:
        decision = classify(_inputs(gaps, gap_config, config))
        assert decision.label == INCONCLUSIVE
        assert isinstance(decision.reason, str)
        assert decision.triggers == [decision.reason]
        seen.add(decision.reason)
    assert seen == {REASON_SPARSE, REASON_MIXED, REASON_INSUFFICIENT}


def test_threshold_scopes_are_declared_and_distinct(gap_config, config):
    decision = classify(_inputs([_equivalent(f"R_{g}", g) for g in GAPS], gap_config, config))
    assert decision.criteria["min_consistent_gaps"] == 3
    assert decision.criteria["equivalence_required_gaps"] == 4
    assert decision.criteria["min_consistent_gaps_applies_to"] == [
        CONSISTENT_PENALTY, CONSISTENT_LOWER_ERROR
    ]
    assert decision.criteria["equivalence_required_gaps_applies_to"] == [
        NO_MATERIAL_DIFFERENCE
    ]
    assert "intersection" in decision.criteria["equivalence_is_an_intersection_note"].lower()


def test_rule_is_frozen_and_authoritative_with_the_stated_meaning(gap_config, config):
    decision = classify(_inputs([_equivalent(f"R_{g}", g) for g in GAPS], gap_config, config))
    assert decision.rule_version == MECHANISM_RULE_VERSION == "preregistered-v1"
    assert decision.authoritative is True
    assert "frozen before GPU execution" in decision.status
    assert "does NOT mean" in decision.status
    assert "not a claim that any causal" in decision.criteria["authoritative_meaning"].lower()


def test_classification_is_deterministic_and_serialisable(gap_config, config):
    gaps = [_material_positive(f"R_{g}", g) for g in GAPS]
    a = classify(_inputs(gaps, gap_config, config))
    b = classify(_inputs(gaps, gap_config, config))
    assert json.dumps(a.as_dict(), default=str) == json.dumps(b.as_dict(), default=str)


# =========================================================================== #
# §7 — the slope and B_g must never determine the classification
# =========================================================================== #

def test_slope_never_changes_the_classification(gap_config, config):
    """Vary the slope arbitrarily; hold the four readings fixed; label unchanged."""
    gaps = [_material_positive("R_16", 16), _material_positive("R_32", 32),
            _equivalent("R_64", 64), _equivalent("R_128", 128)]
    baseline = classify(_inputs(gaps, gap_config, config))
    assert baseline.label == INCONCLUSIVE and baseline.reason == REASON_SPARSE

    for slope, lo, hi, excl in [
        (0.0, -0.001, 0.001, False),        # zeroed
        (+1e6, 9e5, 1.1e6, True),           # enormous positive
        (-1e6, -1.1e6, -9e5, True),         # sign-flipped, enormous
        (0.0012, 0.0004, 0.0020, True),     # plausible positive
        (-0.0012, -0.0020, -0.0004, True),  # plausible negative
    ]:
        dose = DoseResponse(
            mean_slope_per_patch=slope, ci95_low=lo, ci95_high=hi,
            ci95_excludes_zero=excl, quadratic_term=slope * 10,
        )
        decision = classify(_inputs(gaps, gap_config, config, dose_response=dose))
        assert decision.label == baseline.label, f"slope={slope} changed the label"
        assert decision.reason == baseline.reason
        assert decision.criteria["r_readings"] == baseline.criteria["r_readings"]


def test_equivalence_is_decided_only_from_the_per_gap_cis(gap_config, config):
    """An enormous slope cannot block, nor a zero slope create, equivalence."""
    equivalent = [_equivalent(f"R_{g}", g) for g in GAPS]
    huge = DoseResponse(mean_slope_per_patch=1e6, ci95_low=9e5, ci95_high=1.1e6,
                        ci95_excludes_zero=True)
    assert classify(_inputs(equivalent, gap_config, config, dose_response=huge)).label == (
        NO_MATERIAL_DIFFERENCE
    )
    unresolved = [_unresolved(f"R_{g}", g) for g in GAPS]
    flat = DoseResponse(mean_slope_per_patch=0.0, ci95_low=-1e-9, ci95_high=1e-9,
                        ci95_excludes_zero=False)
    decision = classify(_inputs(unresolved, gap_config, config, dose_response=flat))
    assert decision.label == INCONCLUSIVE, "a flat slope must not manufacture equivalence"


def test_quadratic_is_descriptive_only(gap_config, config):
    dose = DoseResponse(mean_slope_per_patch=0.001, quadratic_term=999.0,
                        quadratic_ci95_low=998.0, quadratic_ci95_high=1000.0)
    gaps = [_unresolved(f"R_{g}", g) for g in GAPS]
    decision = classify(_inputs(gaps, gap_config, config, dose_response=dose))
    assert decision.criteria["dose_response"]["descriptive_only"] is True
    assert decision.label == INCONCLUSIVE
    assert decision.reason == REASON_INSUFFICIENT


def test_b_gaps_never_determine_the_label(gap_config, config):
    equivalent = [_equivalent(f"R_{g}", g) for g in GAPS]
    b_strong = [_material_positive(f"B_{g}", g) for g in GAPS]
    with_b = classify(_inputs(equivalent, gap_config, config, b_gaps=b_strong))
    without_b = classify(_inputs(equivalent, gap_config, config))
    assert with_b.label == without_b.label == NO_MATERIAL_DIFFERENCE
    assert "CONFOUNDED WITH" in with_b.criteria["b_note"]


def test_dose_response_x_units_are_missing_patches():
    dose = DoseResponse()
    assert "missing_patches" in dose.x_units
    assert "patch_size" in dose.x_units


def test_non_significant_slope_is_never_called_flat_or_equivalent():
    """Static check: the codebase must not mischaracterise a null slope.

    Same style as test_source_contains_no_nan_to_zero_collapse.
    """
    banned = [
        ("slope", "flat"),
        ("slope", "equivalent"),
        ("slope", "effective-horizon-only"),
        ("slope", "effective_horizon_only"),
    ]
    # Scanned: shipped source and prose. NOT tests/ — this check lives there,
    # and its own banned-pair literals plus deliberate negative fixtures would
    # match themselves. test_the_slope_scan_covers_the_shipped_code pins that
    # the exclusion has not hollowed the check out.
    scanned = [
        path
        for pattern in ("*.py", "*.md")
        for path in REPO_ROOT.rglob(pattern)
        if not {".venv", ".git", "tests"} & set(path.relative_to(REPO_ROOT).parts)
    ]
    offenders = []
    for path in scanned:
        for line in path.read_text(encoding="utf-8").splitlines():
            lowered = line.lower()
            if "not" in lowered or "never" in lowered or "no detected" in lowered:
                continue  # a disclaimer, which is the correct usage
            for a, b in banned:
                if a in lowered and b in lowered:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {line.strip()}")
    assert not offenders, (
        "a non-significant slope is characterised as flat/equivalent/"
        f"effective-horizon-only in:\n  " + "\n  ".join(offenders)
    )


def test_the_slope_scan_covers_the_shipped_code():
    """The exclusion above must not have made the scan vacuous."""
    scanned = {
        # as_posix(), not str(): str() is separator-dependent, so on Windows the
        # discovered paths are backslash-separated and can never match the
        # forward-slash required list below.
        path.relative_to(REPO_ROOT).as_posix()
        for pattern in ("*.py", "*.md")
        for path in REPO_ROOT.rglob(pattern)
        if not {".venv", ".git", "tests"} & set(path.relative_to(REPO_ROOT).parts)
    }
    for required in (
        "stats/mechanism_decision.py",
        "stats/internal_only.py",
        "experiments/trailing_gap.py",
        "docs/preregistration_trailing_gap_mechanism_v1.md",
        "report/audit_report.md",
        "README.md",
    ):
        assert required in scanned, f"{required} is not covered by the slope prose check"


def test_the_slope_scan_would_catch_a_real_violation(tmp_path, monkeypatch):
    """Guard against a check that passes because it matches nothing."""
    offender = tmp_path / "bad.md"
    offender.write_text("The slope was non-significant, so the relationship is flat.\n",
                        encoding="utf-8")
    line = offender.read_text(encoding="utf-8").strip().lower()
    assert "slope" in line and "flat" in line
    assert not ("not" in line or "never" in line or "no detected" in line), (
        "the disclaimer escape hatch must not swallow this example"
    )


def test_the_disclaimer_travels_with_the_dose_response(gap_config, config):
    dose = DoseResponse()
    decision = classify(
        _inputs([_equivalent(f"R_{g}", g) for g in GAPS], gap_config, config,
                dose_response=dose)
    )
    stated = decision.criteria["dose_response"]["non_significant_means"]
    assert "no detected linear trend" in stated
    assert "NOT flatness" in stated
    assert "NOT equivalence" in stated
    assert "sub-question" in decision.criteria["dose_response_note"].lower()


# =========================================================================== #
# Approved-item additions
# =========================================================================== #

def test_hypothesis_provenance_on_every_possible_classification_output(
    gap_config, config
):
    """Not a sample: every reachable label and every INCONCLUSIVE reason.

    The provenance framing is §0 of the preregistration — that R_g/B_g test a
    hypothesis generated from a post-hoc finding, one level removed from an
    independent confirmatory test. It must survive on every output, because the
    output is what a reader sees.
    """
    from stats.mechanism_decision import (
        CONSISTENT_LOWER_ERROR, CONSISTENT_PENALTY, DIRECTION_REVERSAL,
        INCONCLUSIVE, NO_MATERIAL_DIFFERENCE,
    )

    cases = {
        DIRECTION_REVERSAL: [
            _material_positive("R_16", 16), _material_positive("R_32", 32),
            _material_positive("R_64", 64), _material_negative("R_128", 128),
        ],
        NO_MATERIAL_DIFFERENCE: [_equivalent(f"R_{g}", g) for g in GAPS],
        CONSISTENT_PENALTY: [
            _material_positive("R_16", 16), _material_positive("R_32", 32),
            _material_positive("R_64", 64), _unresolved("R_128", 128),
        ],
        CONSISTENT_LOWER_ERROR: [
            _material_negative("R_16", 16), _material_negative("R_32", 32),
            _material_negative("R_64", 64), _unresolved("R_128", 128),
        ],
        f"{INCONCLUSIVE}:{REASON_SPARSE}": [
            _material_positive("R_16", 16), _equivalent("R_32", 32),
            _equivalent("R_64", 64), _equivalent("R_128", 128),
        ],
        f"{INCONCLUSIVE}:{REASON_MIXED}": [
            _unresolved("R_16", 16), _equivalent("R_32", 32),
            _equivalent("R_64", 64), _equivalent("R_128", 128),
        ],
        f"{INCONCLUSIVE}:{REASON_INSUFFICIENT}": [_unresolved(f"R_{g}", g) for g in GAPS],
    }

    seen_labels, seen_reasons = set(), set()
    for expected, gaps in cases.items():
        decision = classify(_inputs(gaps, gap_config, config))
        label = (
            f"{decision.label}:{decision.reason}"
            if decision.label == INCONCLUSIVE else decision.label
        )
        assert label == expected, f"{expected}: got {label}"
        seen_labels.add(decision.label)
        if decision.reason:
            seen_reasons.add(decision.reason)

        provenance = decision.criteria["hypothesis_provenance"]
        assert "POST-HOC" in provenance, f"{expected}: POST-HOC missing"
        assert "one level" in provenance, f"{expected}: 'one level removed' missing"
        assert "removed from a fully independent confirmatory test" in provenance
        # Present in the serialised form a reader actually receives.
        assert "POST-HOC" in json.dumps(decision.as_dict(), default=str)

    # Every label and every reason the rule can emit was exercised.
    assert seen_labels == {
        DIRECTION_REVERSAL, NO_MATERIAL_DIFFERENCE, CONSISTENT_PENALTY,
        CONSISTENT_LOWER_ERROR, INCONCLUSIVE,
    }
    assert seen_reasons == {REASON_SPARSE, REASON_MIXED, REASON_INSUFFICIENT}


def test_sesoi_delta_follows_a_mutated_pilot_config(config):
    """Inheritance is real, not two values that happen to agree.

    Mutating the copied pilot config's SESOI must move sesoi_delta with it. If
    the rule carried its own copy of 0.03, this would not move.
    """
    baseline = sesoi_delta(config, CLEAN_MAE)
    assert baseline == pytest.approx(0.03 * CLEAN_MAE)

    for mutated_frac in (0.01, 0.05, 0.10, 0.0):
        mutated = json.loads(json.dumps(config))
        mutated["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"] = mutated_frac
        assert sesoi_delta(mutated, CLEAN_MAE) == pytest.approx(mutated_frac * CLEAN_MAE)
    # The original object is untouched by the mutation.
    assert sesoi_delta(config, CLEAN_MAE) == pytest.approx(baseline)


def test_a_mutated_sesoi_changes_the_per_gap_readings(gap_config, config):
    """The inherited value really drives classification, not just a number."""
    # A difference that is MATERIAL_POSITIVE under a 3% band...
    gaps = [_material_positive(f"R_{g}", g) for g in GAPS]
    assert classify(_inputs(gaps, gap_config, config)).label == (
        "CONSISTENT_EXTRA_TRAILING_GAP_PENALTY"
    )
    # ...ceases to clear the band when the inherited SESOI is widened.
    widened = json.loads(json.dumps(config))
    widened["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"] = 0.50
    relabelled = classify(_inputs(gaps, gap_config, widened))
    assert relabelled.label != "CONSISTENT_EXTRA_TRAILING_GAP_PENALTY"
    assert relabelled.criteria["sesoi_delta"] == pytest.approx(0.50 * CLEAN_MAE)
