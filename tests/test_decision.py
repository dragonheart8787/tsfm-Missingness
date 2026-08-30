"""The GO / PIVOT / NO-GO decision function must be deterministic and auditable.

Model-mocked: no weights required.
"""

from __future__ import annotations

import json

import pytest

from stats.decision import ContrastStat, DecisionInputs, decide


def _stat(contrast_id, family, rate, mean, median, *, rejects=True, clean_mae=1.0, **overrides):
    base = dict(
        contrast_id=contrast_id,
        family=family,
        kind="causal_contrast_within_pattern" if family == "location" else "pipeline_comparison",
        rate=rate,
        mean_difference=mean,
        median_difference=median,
        frac_positive=0.7 if mean > 0 else 0.3,
        holm_rejects=rejects,
        ci_low_holm=mean - 0.01,
        ci_high_holm=mean + 0.01,
        ci_low_equivalence=mean - 0.005,
        ci_high_equivalence=mean + 0.005,
        mean_clean_mae=clean_mae,
        block_length_estimates={4: mean, 8: mean, 12: mean},
        block_length_rejects={4: rejects, 8: rejects, 12: rejects},
        block_length_ci_low={4: mean - 0.01, 8: mean - 0.01, 12: mean - 0.01},
        block_length_ci_high={4: mean + 0.01, 8: mean + 0.01, 12: mean + 0.01},
        top_origin_share=0.1,
        rho_distance=0.6,
        rho_confounders={"removed_variance": 0.1, "n_patches_fully_missing": 0.1},
        rho_scaling=0.1,
        n_partially_missing_patches_max=0,
    )
    base.update(overrides)
    return ContrastStat(**base)


def _inputs(stats, config, **overrides):
    base = dict(
        contrasts=stats,
        mean_clean_mae=1.0,
        location_ordering_spearman={0.20: -0.9, 0.40: -0.9},
        seed_sensitivity_frac={0.20: 0.002, 0.40: 0.002},
        config=config,
    )
    base.update(overrides)
    return DecisionInputs(**base)


def test_clean_location_go(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "GO"
    assert decision.go_family == "location"
    assert decision.criteria["location_go"]["passed"] is True


def test_go_fails_when_the_magnitude_is_below_5_percent(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.02, 0.02),   # 2% of clean MAE
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label != "GO"
    assert decision.criteria["location_go"]["magnitude_ge_threshold_at_both_rates"] is False


def test_go_fails_on_a_sign_disagreement_between_rates(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, +0.10, +0.09),
        _stat("C2_loc_40", "location", 0.40, -0.12, -0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label != "GO"
    assert decision.criteria["location_go"]["same_sign_at_both_rates"] is False


def test_go_fails_when_the_median_disagrees_with_the_mean(config):
    """A mean driven by a tail of origins is not a GO."""
    stats = [
        _stat("C1_loc_20", "location", 0.20, +0.10, -0.02),
        _stat("C2_loc_40", "location", 0.40, +0.12, +0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label != "GO"
    assert decision.criteria["location_go"]["median_sign_matches_mean_at_both_rates"] is False


def test_go_fails_when_holm_does_not_reject(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09, rejects=False),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label != "GO"
    assert decision.criteria["location_go"]["holm_ci_excludes_zero_at_both_rates"] is False


def test_clean_no_go(config):
    """All four 90% CIs inside +/-3%, unstable ordering, small seed sensitivity."""
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.005, 0.004, rejects=False),
        _stat("C2_loc_40", "location", 0.40, -0.003, -0.002, rejects=False),
        _stat("C3_geom_20", "geometry", 0.20, 0.002, 0.002, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(
        _inputs(stats, config, location_ordering_spearman={0.20: 0.1, 0.40: -0.2})
    )
    assert decision.label == "NO-GO"
    assert decision.triggers == ["no_go_all_criteria_met"]
    assert decision.criteria["no_go"]["all_ci90_within_band"] is True
    assert decision.criteria["no_go"]["no_stable_location_ordering"] is True
    assert decision.criteria["no_go"]["seed_sensitivity_small"] is True


def test_no_go_blocked_by_a_stable_location_ordering(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.005, 0.004, rejects=False),
        _stat("C2_loc_40", "location", 0.40, 0.003, 0.002, rejects=False),
        _stat("C3_geom_20", "geometry", 0.20, 0.002, 0.002, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(
        _inputs(stats, config, location_ordering_spearman={0.20: -0.95, 0.40: -0.92})
    )
    assert decision.label != "NO-GO"
    assert decision.criteria["no_go"]["stable_location_ordering"] is True


def test_no_go_blocked_by_large_seed_sensitivity(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.005, 0.004, rejects=False),
        _stat("C2_loc_40", "location", 0.40, 0.003, 0.002, rejects=False),
        _stat("C3_geom_20", "geometry", 0.20, 0.002, 0.002, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(
        _inputs(
            stats, config,
            location_ordering_spearman={0.20: 0.1, 0.40: -0.2},
            seed_sensitivity_frac={0.20: 0.05, 0.40: 0.05},
        )
    )
    assert decision.label != "NO-GO"
    assert decision.criteria["no_go"]["seed_sensitivity_small"] is False


def test_pivot_when_the_effect_appears_only_at_40_percent(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.01, 0.01, rejects=False),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "PIVOT"
    assert any("effect_only_at_40pct" in t for t in decision.triggers)


def test_pivot_when_a_handful_of_origins_dominate(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09, top_origin_share=0.8),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "PIVOT"
    assert any("origin_dominance" in t for t in decision.triggers)


def test_pivot_when_a_confounder_out_tracks_distance(config):
    stats = [
        _stat(
            "C1_loc_20", "location", 0.20, 0.10, 0.09,
            rho_distance=0.2,
            rho_confounders={"removed_variance": 0.85, "n_patches_fully_missing": 0.1},
        ),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "PIVOT"
    assert any("confounder_out_tracks_distance" in t for t in decision.triggers)


def test_pivot_when_internal_scaling_explains_the_pattern(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09, rho_scaling=0.8),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "PIVOT"
    assert any("internal_scaling" in t for t in decision.triggers)


def test_pivot_when_a_block_arm_has_partially_missing_patches(config):
    """Patch-position confound: preregistered blocks must be fully patch-aligned."""
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09, n_partially_missing_patches_max=2),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "PIVOT"
    assert any("patch_position_confound" in t for t in decision.triggers)


def test_pivot_on_bootstrap_block_length_instability(config):
    """Sensitivity is judged on the intervals, which DO move with block length."""
    stats = [
        _stat(
            "C1_loc_20", "location", 0.20, 0.10, 0.09,
            block_length_rejects={4: True, 8: True, 12: False},
            block_length_ci_low={4: 0.09, 8: 0.05, 12: -0.02},
            block_length_ci_high={4: 0.11, 8: 0.15, 12: 0.22},
        ),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.label == "PIVOT"
    assert any("bootstrap_block_length_instability" in t for t in decision.triggers)


def test_pivot_outranks_go(config):
    """A real but confounded effect is a design problem, not a green light."""
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09, top_origin_share=0.9),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    decision = decide(_inputs(stats, config))
    assert decision.criteria["location_go"]["passed"] is True
    assert decision.label == "PIVOT"


def test_residual_case_is_labelled_pivot_inconclusive(config):
    """Neither a clean null nor a clean effect still yields one of three labels."""
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.04, 0.04),   # 4%: below the GO bar,
        _stat("C2_loc_40", "location", 0.40, 0.04, 0.04),   # above the NO-GO band
        _stat("C3_geom_20", "geometry", 0.20, 0.04, 0.04),
        _stat("C4_geom_40", "geometry", 0.40, 0.04, 0.04),
    ]
    decision = decide(
        _inputs(stats, config, location_ordering_spearman={0.20: 0.1, 0.40: -0.2})
    )
    assert decision.label == "PIVOT"
    assert decision.triggers == ["inconclusive_no_criteria_met"]


def test_decision_is_deterministic_and_json_serialisable(config):
    stats = [
        _stat("C1_loc_20", "location", 0.20, 0.10, 0.09),
        _stat("C2_loc_40", "location", 0.40, 0.12, 0.11),
        _stat("C3_geom_20", "geometry", 0.20, 0.001, 0.001, rejects=False),
        _stat("C4_geom_40", "geometry", 0.40, 0.001, 0.001, rejects=False),
    ]
    first = decide(_inputs(stats, config))
    second = decide(_inputs(stats, config))
    assert first.label == second.label
    assert first.triggers == second.triggers
    assert json.dumps(first.as_dict(), default=str) == json.dumps(second.as_dict(), default=str)


def test_exactly_four_preregistered_contrasts(config):
    assert len(config["contrasts"]) == 4
    ids = [c["id"] for c in config["contrasts"]]
    assert ids == ["C1_loc_20", "C2_loc_40", "C3_geom_20", "C4_geom_40"]
    kinds = {c["id"]: c["kind"] for c in config["contrasts"]}
    assert kinds["C3_geom_20"] == "pipeline_comparison"
    assert kinds["C4_geom_40"] == "pipeline_comparison"


def test_scope_exclusions_are_not_implemented(config):
    """Nothing out of scope crept into the design."""
    text = json.dumps(config).lower()
    for excluded in (
        "interpolat", "forward_fill", "ffill", "arima", "xgboost",
        "dlinear", "timesfm", "moment", "ts-icl", "proximity_weight",
    ):
        assert excluded not in text, f"out-of-scope item {excluded!r} present in the config"
    rates = {float(r["rate"]) for r in config["masks"]["rates"]}
    assert rates == {0.20, 0.40}, "only the two preregistered rates may exist"
