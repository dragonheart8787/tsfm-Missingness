"""Gating tests for the trailing-gap mechanism experiment.

    *** THE EXPERIMENT IS NOT EXECUTED. *** These tests validate the DESIGN
    before any GPU time is requested. Model-mocked throughout.

The five mandatory checks from the round brief are tests 1-5 below; everything
after them is supporting coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.trailing_gap import (
    TrailingGapContractViolation,
    build_conditions,
    build_internal_block,
    build_trailing_nan,
    build_truncated_long,
    preflight,
    scored_timestamps,
)
from model.chronos2_runner import ModelContract, MockForecaster
from runner.windows import build_windows

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gap_config():
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def contract():
    """The mock contract, which mirrors the real pipeline's own arithmetic."""
    return MockForecaster().contract


def _contract(**overrides) -> ModelContract:
    base = dict(
        hf_model_id="amazon/chronos-2", revision="deadbeef",
        input_patch_size=16, input_patch_stride=16, output_patch_size=16,
        model_context_length=8192, max_output_patches=64,
        model_prediction_length=64 * 16,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        use_arcsinh=False, device="cpu", dtype="float32",
    )
    base.update(overrides)
    return ModelContract(**base)


@pytest.fixture(scope="module")
def window(config, synthetic_series):
    values, stamps = synthetic_series
    return build_windows(values=values, timestamps=stamps, config=config)[3]


# --------------------------------------------------------------------------- #
# MANDATORY 1 — patch alignment and the single-shot horizon limit
# --------------------------------------------------------------------------- #

def test_1_all_gaps_are_patch_aligned_and_fit_the_single_shot_horizon(
    gap_config, config, contract
):
    """L-g is a whole number of patches, and H+g fits the model's real limit.

    Both facts are read from the LOADED contract, never hardcoded.
    """
    L = int(config["design"]["context_length"])
    H = int(config["design"]["horizon"])
    gaps = [int(g) for g in gap_config["design"]["gap_lengths"]]
    patch = contract.input_patch_size
    single_shot = contract.max_output_patches * contract.output_patch_size

    assert gaps == [16, 32, 64, 128]
    for g in gaps:
        assert g % patch == 0, f"g={g} is not patch-aligned"
        assert (L - g) % patch == 0, f"L-g={L - g} is not a whole number of patches"
        assert L - g > 0
        assert H + g <= single_shot, (
            f"g={g}: H+g={H + g} exceeds the model's single-shot output {single_shot}"
        )

    assert preflight(
        contract=contract, context_length=L, horizon=H, gap_lengths=gaps,
        internal_block_distance=int(gap_config["design"]["internal_block_distance"]),
    ) == []


def test_1b_preflight_reads_the_limit_from_the_contract_not_a_constant():
    """Shrink the contract's output capacity and the check must start failing."""
    small = _contract(max_output_patches=8, model_prediction_length=8 * 16)  # 128
    violations = preflight(
        contract=small, context_length=320, horizon=96,
        gap_lengths=[16, 32, 64, 128], internal_block_distance=16,
    )
    # H+g = 112, 128, 160, 224 against a limit of 128: g=64 and g=128 fail.
    unrolling = [v for v in violations if "AUTOREGRESSIVE UNROLLING" in v]
    assert len(unrolling) == 2
    assert any("g=64" in v for v in unrolling)
    assert any("g=128" in v for v in unrolling)
    assert not any("g=16" in v for v in unrolling)


def test_1c_preflight_rejects_a_non_16_patch_grid():
    violations = preflight(
        contract=_contract(input_patch_size=24, output_patch_size=24),
        context_length=320, horizon=96, gap_lengths=[16, 32, 64, 128],
        internal_block_distance=16,
    )
    assert any("not a whole number of 24-step patches" in v for v in violations)


# --------------------------------------------------------------------------- #
# MANDATORY 2 — scored timestamps align with the ORIGINAL target
# --------------------------------------------------------------------------- #

def test_2_scored_timestamps_align_with_the_original_target(
    window, gap_config, config, contract
):
    """Not just equal lengths: the actual timestamps must coincide."""
    H = int(config["design"]["horizon"])
    context_timestamps = pd.date_range(
        window.origin_timestamp - pd.Timedelta(hours=len(window.context)),
        periods=len(window.context), freq="h",
    )
    conditions = build_conditions(
        context=window.context, horizon=H, config=gap_config, contract=contract
    )
    expected = pd.date_range(context_timestamps[-1] + pd.Timedelta(hours=1), periods=H, freq="h")

    for condition in conditions:
        actual = scored_timestamps(condition, context_timestamps=context_timestamps)
        assert len(actual) == H
        assert list(actual) == list(expected), (
            f"{condition.condition_id}: scored timestamps are misaligned with the "
            f"original target (starts {actual[0]}, expected {expected[0]})"
        )


def test_2b_truncated_long_scored_index_lands_on_the_original_target(
    window, gap_config, config, contract
):
    """The arithmetic, stated directly: (L-g) + g == L."""
    L, H = len(window.context), int(config["design"]["horizon"])
    for g in gap_config["design"]["gap_lengths"]:
        condition = build_truncated_long(
            context=window.context, horizon=H, gap=int(g),
            single_shot_limit=contract.max_output_patches * contract.output_patch_size,
        )
        assert condition.first_predicted_index == L - g
        assert condition.scored_first_index == L
        assert condition.scored_length == H
        assert condition.prediction_length == H + g


# --------------------------------------------------------------------------- #
# MANDATORY 3 — target byte-identity across every condition
# --------------------------------------------------------------------------- #

def test_3_target_sha256_is_identical_across_every_condition(
    window, gap_config, config, contract
):
    """Reuses the pilot's own target-integrity machinery."""
    from runner.run_pilot import assert_target_integrity

    H = int(config["design"]["horizon"])
    conditions = build_conditions(
        context=window.context, horizon=H, config=gap_config, contract=contract
    )
    reference = window.target_digest()
    digests = {}
    for condition in conditions:
        # No condition may touch the target; the digest must be unchanged after
        # each one is built.
        digests[condition.condition_id] = window.target_digest()
    assert set(digests.values()) == {reference}
    assert_target_integrity(window, digests)
    assert not window.target.flags.writeable


def test_3b_no_condition_mutates_the_source_context(window, gap_config, config, contract):
    original = np.array(window.context, copy=True)
    build_conditions(
        context=window.context, horizon=int(config["design"]["horizon"]),
        config=gap_config, contract=contract,
    )
    assert np.array_equal(window.context, original, equal_nan=True)


# --------------------------------------------------------------------------- #
# MANDATORY 4 — prove the original bug was real, and that the fix addresses it
# --------------------------------------------------------------------------- #

def test_4_naive_truncation_is_misaligned_by_exactly_g_steps(window, config, contract):
    """The bug this round exists to fix, demonstrated rather than asserted.

    NAIVE: context of L-g, ask for H steps -> scores [L-g, L-g+H).
    FIXED: context of L-g, ask for H+g steps, drop g -> scores [L, L+H).
    The two scored windows must differ by EXACTLY g steps.
    """
    L, H = len(window.context), int(config["design"]["horizon"])
    single_shot = contract.max_output_patches * contract.output_patch_size

    for g in (16, 32, 64, 128):
        # The naive design, reconstructed exactly as it would have been.
        naive_context_length = L - g
        naive_prediction_length = H
        naive_scored_first_index = naive_context_length  # forecasts from its own end
        naive_scored_last_index = naive_scored_first_index + naive_prediction_length

        fixed = build_truncated_long(
            context=window.context, horizon=H, gap=g, single_shot_limit=single_shot
        )

        # Same context; the fix does not change what the model is shown.
        assert fixed.context.shape[0] == naive_context_length

        # The misalignment is exactly g, at both ends.
        assert fixed.scored_first_index - naive_scored_first_index == g
        assert (fixed.scored_first_index + fixed.scored_length) - naive_scored_last_index == g

        # The naive arm would have scored a window that ENDS before the real
        # target ends, i.e. it scores partly inside the context period.
        assert naive_scored_first_index == L - g < L
        assert fixed.scored_first_index == L, "the fix must land on the original target"


def test_4b_the_two_designs_score_different_predicted_values(window, config, contract):
    """End-to-end with the mock: the misalignment changes the scored numbers."""
    L, H = len(window.context), int(config["design"]["horizon"])
    forecaster = MockForecaster()
    single_shot = contract.max_output_patches * contract.output_patch_size
    g = 64

    truncated_context = np.array(window.context[: L - g], dtype=np.float32)

    naive_prediction = forecaster.forecast_median(truncated_context, H)
    fixed = build_truncated_long(
        context=window.context, horizon=H, gap=g, single_shot_limit=single_shot
    )
    full_prediction = forecaster.forecast_median(fixed.context, fixed.prediction_length)
    fixed_scored = fixed.score(full_prediction)

    assert naive_prediction.shape == fixed_scored.shape == (H,)
    # The naive arm scores the FIRST H steps; the fix scores steps [g, g+H).
    assert np.array_equal(naive_prediction, full_prediction[:H])
    assert np.array_equal(fixed_scored, full_prediction[g : g + H])
    assert not np.array_equal(naive_prediction, fixed_scored), (
        "if these matched, the misalignment would be undetectable and the bug moot"
    )


# --------------------------------------------------------------------------- #
# MANDATORY 5 — autoregressive unrolling cannot happen silently
# --------------------------------------------------------------------------- #

def test_5_requesting_an_unroll_is_a_hard_stop_not_a_warning():
    """Chronos-2 only WARNS on an over-long horizon and then unrolls."""
    with pytest.raises(TrailingGapContractViolation, match="single-shot"):
        build_truncated_long(
            context=np.zeros(320, dtype=np.float32), horizon=96, gap=128,
            single_shot_limit=128,  # H+g = 224 > 128
        )


def test_5b_build_conditions_refuses_the_whole_matrix_on_an_unroll_risk(
    window, gap_config, config
):
    small = _contract(max_output_patches=8, model_prediction_length=8 * 16)
    with pytest.raises(TrailingGapContractViolation) as excinfo:
        build_conditions(
            context=window.context, horizon=int(config["design"]["horizon"]),
            config=gap_config, contract=small,
        )
    assert "AUTOREGRESSIVE UNROLLING" in str(excinfo.value)
    assert "HARD STOP" in str(excinfo.value)


def test_5c_config_asks_the_library_to_raise_rather_than_warn(gap_config):
    assert gap_config["truncated_long"]["limit_prediction_length"] is True
    assert gap_config["truncated_long"]["forbid_autoregressive_unroll"] is True


def test_5d_the_pipeline_default_would_have_warned_silently():
    """Documents WHY this guard exists, from the library's own source."""
    import chronos.chronos2.pipeline as pipeline_module
    import inspect

    source = inspect.getsource(pipeline_module.Chronos2Pipeline.predict)
    assert "limit_prediction_length: bool = False" in source, (
        "the default is what makes an over-long request warn instead of raise"
    )
    assert "warnings.warn(msg)" in source


# --------------------------------------------------------------------------- #
# Supporting coverage
# --------------------------------------------------------------------------- #

def test_truncated_context_contains_no_nan(window, config, contract):
    """It is genuinely shorter, not masked."""
    H = int(config["design"]["horizon"])
    single_shot = contract.max_output_patches * contract.output_patch_size
    for g in (16, 32, 64, 128):
        condition = build_truncated_long(
            context=window.context, horizon=H, gap=g, single_shot_limit=single_shot
        )
        assert not np.isnan(condition.context).any()
        assert condition.mask is None


def test_trailing_nan_generalises_the_pilots_d0_block(window, config, contract):
    H = int(config["design"]["horizon"])
    L = len(window.context)
    for g in (16, 32, 64, 128):
        condition = build_trailing_nan(
            context=window.context, horizon=H, gap=g, patch_size=16
        )
        assert condition.mask.distance_to_boundary == 0
        assert condition.mask.block_end_exclusive == L
        assert condition.mask.missing_count == g
        assert np.isnan(condition.context[-g:]).all()
        assert not np.isnan(condition.context[: L - g]).any()
        assert condition.prediction_length == H
        assert condition.scored_slice == slice(0, H)


def test_internal_block_is_fixed_at_d16_for_every_gap(window, config, contract, gap_config):
    """One full real patch always separates the block from the boundary."""
    H = int(config["design"]["horizon"])
    L = len(window.context)
    distance = int(gap_config["design"]["internal_block_distance"])
    assert distance == 16
    for g in (16, 32, 64, 128):
        condition = build_internal_block(
            context=window.context, horizon=H, gap=g, distance=distance, patch_size=16
        )
        assert condition.mask.distance_to_boundary == 16
        assert condition.mask.block_start == L - distance - g == 304 - g
        assert condition.mask.block_start >= 0
        assert condition.mask.block_start % 16 == 0
        assert condition.mask.block_end_exclusive == L - distance == 304
        # The final real patch is intact in every case.
        assert not np.isnan(condition.context[-distance:]).any()
    assert 304 - 128 == 176 >= 0


def test_matrix_is_thirteen_conditions_and_2314_forecasts(gap_config, config, contract, window):
    conditions = build_conditions(
        context=window.context, horizon=int(config["design"]["horizon"]),
        config=gap_config, contract=contract,
    )
    assert len(conditions) == 13 == gap_config["design"]["expected_conditions_per_origin"]
    kinds = [c.kind for c in conditions]
    assert kinds.count("clean") == 1
    for kind in ("trailing_nan", "truncated_long", "internal_block"):
        assert kinds.count(kind) == 4
    origins = int(config["design"]["expected_origins"])
    assert origins * 13 == 2314 == gap_config["design"]["expected_total_forecasts"]
    assert len({c.condition_id for c in conditions}) == 13


def test_score_rejects_a_wrong_length_prediction(window, config, contract):
    condition = build_truncated_long(
        context=window.context, horizon=int(config["design"]["horizon"]), gap=32,
        single_shot_limit=contract.max_output_patches * contract.output_patch_size,
    )
    with pytest.raises(ValueError, match="expected 128 predicted steps"):
        condition.score(np.zeros(96))


def test_config_does_not_redefine_shared_conventions(gap_config):
    """SESOI, bootstrap and design constants are inherited, never restated."""
    text = (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    assert "equivalence_band_frac_of_clean_mae" in gap_config["meta"]["inherited_keys"][-1]
    for forbidden in ("n_replicates:", "main_block_length:", "context_length:", "horizon:"):
        assert forbidden not in text, (
            f"{forbidden!r} is restated here instead of inherited from pilot_config.yaml"
        )


def test_pilot_config_design_constants_are_untouched(config):
    """This round must not alter the original preregistration."""
    assert config["design"]["context_length"] == 320
    assert config["design"]["horizon"] == 96
    assert config["design"]["stride"] == 96
    assert config["design"]["expected_origins"] == 178
    assert config["design"]["expected_conditions_per_origin"] == 17
    assert config["design"]["expected_total_forecasts"] == 3026
    assert config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"] == 0.03


# --------------------------------------------------------------------------- #
# DRAFT mechanism classification rule — unit-tested like stats/decision.py,
# but explicitly NOT authoritative.
# --------------------------------------------------------------------------- #

from stats.mechanism_decision import (  # noqa: E402
    GAP_EQUIVALENT,
    GAP_INDETERMINATE,
    GAP_NEGATIVE,
    GAP_POSITIVE,
    DoseResponse,
    GapStat,
    MechanismInputs,
    classify,
)


def _gap(contrast_id, gap, mean, *, rejects=True, span=0.01, clean_mae=2.4575):
    return GapStat(
        contrast_id=contrast_id, gap=gap,
        mean_difference=mean, median_difference=mean,
        ci_low_holm=mean - span, ci_high_holm=mean + span,
        ci_low_equivalence=mean - span * 0.7, ci_high_equivalence=mean + span * 0.7,
        holm_rejects=rejects, mean_clean_mae=clean_mae,
    )


def _inputs(r_gaps, gap_config, config, **overrides):
    base = dict(
        r_gaps=r_gaps, b_gaps=[], dose_response=None,
        mean_clean_mae=2.4575, config=gap_config, pilot_config=config,
    )
    base.update(overrides)
    return MechanismInputs(**base)


def test_rule_is_flagged_as_a_non_authoritative_draft(gap_config, config):
    decision = classify(_inputs([_gap("R_16", 16, 0.001, rejects=False)], gap_config, config))
    assert decision.authoritative is False
    assert "sign-off" in decision.status
    assert decision.rule_version == "draft-v1"
    assert decision.criteria["authoritative"] is False


def test_all_gaps_equivalent_reads_effective_horizon_only(gap_config, config):
    gaps = [_gap(f"R_{g}", g, 0.002, rejects=False) for g in (16, 32, 64, 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == "EFFECTIVE_HORIZON_ONLY"
    assert all(r == GAP_EQUIVALENT for r in decision.criteria["r_readings"].values())


def test_consistent_positive_reads_extra_trailing_nan_penalty(gap_config, config):
    gaps = [_gap(f"R_{g}", g, 0.30) for g in (16, 32, 64, 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == "EXTRA_TRAILING_NAN_PENALTY"
    assert all(r == GAP_POSITIVE for r in decision.criteria["r_readings"].values())


def test_consistent_negative_reads_nan_framing_outperforms(gap_config, config):
    gaps = [_gap(f"R_{g}", g, -0.30) for g in (16, 32, 64, 128)]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == "NAN_FRAMING_OUTPERFORMS"


def test_direction_disagreement_reads_inconsistent(gap_config, config):
    """A mechanism that reverses sign with gap length is not one mechanism."""
    gaps = [
        _gap("R_16", 16, +0.30), _gap("R_32", 32, +0.30),
        _gap("R_64", 64, +0.30), _gap("R_128", 128, -0.30),
    ]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == "INCONSISTENT_ACROSS_GAPS"
    assert decision.triggers == ["r_readings_disagree_in_direction"]


def test_inconsistency_outranks_a_majority_reading(gap_config, config):
    """3 positive + 1 negative must NOT be reported as a positive mechanism."""
    gaps = [
        _gap("R_16", 16, +0.30), _gap("R_32", 32, +0.30),
        _gap("R_64", 64, +0.30), _gap("R_128", 128, -0.30),
    ]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.criteria["r_reading_counts"][GAP_POSITIVE] == 3
    assert decision.label != "EXTRA_TRAILING_NAN_PENALTY"


def test_too_few_agreeing_gaps_reads_inconclusive(gap_config, config):
    gaps = [
        _gap("R_16", 16, +0.30), _gap("R_32", 32, +0.30),
        # Wide, zero-straddling intervals that also reach beyond the band:
        # neither equivalent nor significant, i.e. genuinely indeterminate.
        _gap("R_64", 64, 0.05, rejects=False, span=0.30),
        _gap("R_128", 128, 0.05, rejects=False, span=0.30),
    ]
    decision = classify(_inputs(gaps, gap_config, config))
    assert decision.label == "INCONCLUSIVE"
    assert decision.criteria["r_reading_counts"][GAP_INDETERMINATE] == 2
    assert decision.criteria["r_reading_counts"][GAP_POSITIVE] == 2


def test_equivalence_is_checked_before_significance(gap_config, config):
    """A significant but practically negligible effect reads as equivalent."""
    tiny = _gap("R_16", 16, 0.004, rejects=True, span=0.002)
    band = 0.03 * 2.4575
    assert abs(tiny.mean_difference) < band
    assert tiny.reading(band) == GAP_EQUIVALENT


def test_the_sesoi_is_inherited_not_redefined(gap_config, config):
    decision = classify(_inputs([_gap("R_16", 16, 0.3)], gap_config, config))
    assert decision.criteria["equivalence_band_abs"] == pytest.approx(0.03 * 2.4575)
    assert "inherited" in decision.criteria["equivalence_band_source"]
    from pathlib import Path

    source = Path("stats/mechanism_decision.py").read_text(encoding="utf-8")
    assert "0.03" not in source, "the SESOI literal was copied instead of inherited"


def test_b_gaps_never_determine_the_label(gap_config, config):
    """B_g is confounded with removed content and must not drive the mechanism."""
    r_equivalent = [_gap(f"R_{g}", g, 0.002, rejects=False) for g in (16, 32, 64, 128)]
    b_strong = [_gap(f"B_{g}", g, 0.9) for g in (16, 32, 64, 128)]
    with_b = classify(_inputs(r_equivalent, gap_config, config, b_gaps=b_strong))
    without_b = classify(_inputs(r_equivalent, gap_config, config))
    assert with_b.label == without_b.label == "EFFECTIVE_HORIZON_ONLY"
    assert "CONFOUNDED WITH" in with_b.criteria["b_interpretation_note"]


def test_hypothesis_provenance_is_recorded_in_every_decision(gap_config, config):
    decision = classify(_inputs([_gap("R_16", 16, 0.3)], gap_config, config))
    provenance = decision.criteria["hypothesis_provenance"]
    assert "POST-HOC" in provenance
    assert "one level" in provenance


def test_dose_response_is_carried_through(gap_config, config):
    dose = DoseResponse(
        mean_slope=0.0012, ci95_low=0.0004, ci95_high=0.0020, ci95_excludes_zero=True
    )
    decision = classify(
        _inputs([_gap("R_16", 16, 0.3)], gap_config, config, dose_response=dose)
    )
    assert decision.criteria["dose_response"]["mean_slope"] == pytest.approx(0.0012)
    assert decision.criteria["dose_response"]["ci95_excludes_zero"] is True


def test_classification_is_deterministic_and_serialisable(gap_config, config):
    import json

    gaps = [_gap(f"R_{g}", g, 0.30) for g in (16, 32, 64, 128)]
    a = classify(_inputs(gaps, gap_config, config))
    b = classify(_inputs(gaps, gap_config, config))
    assert a.label == b.label and a.triggers == b.triggers
    assert json.dumps(a.as_dict(), default=str) == json.dumps(b.as_dict(), default=str)
