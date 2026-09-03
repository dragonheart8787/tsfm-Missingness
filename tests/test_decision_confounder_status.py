"""Distance-relative PIVOT checks must be honest about what they cannot ask.

The preregistered contrasts are fixed two-arm comparisons: every origin's
subtrahend arm sits at the SAME distance to the forecast boundary. A rank
correlation against a constant is mathematically undefined, not zero.

Decision rule v1 coerced that undefined value to 0.0 before comparing each
confounder's |rho| against it, which invented a baseline of "distance explains
nothing" — against which almost any confounder wins. These tests pin the v2
semantics: `not_evaluable` is structurally distinct from `false`, and neither
`not_evaluable` nor `invalid_input` may ever contribute a trigger.

Model-mocked: no weights required.
"""

from __future__ import annotations

import numpy as np
import pytest

from stats.decision import (
    DECISION_RULE_VERSION,
    INVALID_RHO_DISTANCE,
    NOT_EVALUABLE_CONSTANT_DISTANCE,
    ContrastStat,
    DecisionInputs,
    compare_confounders_to_distance,
    decide,
)

# Distance values as they actually are in the pilot: one fixed subtrahend
# condition, so the same value repeated across all 178 origins.
CONSTANT_DISTANCE = [128.0] * 178
VARYING_DISTANCE = [0.0, 64.0, 128.0, 192.0, 256.0] * 35 + [0.0, 64.0, 128.0]


def _stat(
    contrast_id="C1_loc_20",
    family="location",
    *,
    rate=0.20,
    mean=0.10,
    median=0.09,
    rho_distance=float("nan"),
    distance_values=None,
    rho_confounders=None,
    rho_scaling=0.1,
    n_partial=0,
    clean_mae=1.0,
):
    return ContrastStat(
        contrast_id=contrast_id,
        family=family,
        kind="causal_contrast_within_pattern" if family == "location" else "pipeline_comparison",
        rate=rate,
        mean_difference=mean,
        median_difference=median,
        frac_positive=0.7,
        holm_rejects=True,
        ci_low_holm=mean - 0.01,
        ci_high_holm=mean + 0.01,
        ci_low_equivalence=mean - 0.005,
        ci_high_equivalence=mean + 0.005,
        mean_clean_mae=clean_mae,
        block_length_estimates={4: mean, 8: mean, 12: mean},
        block_length_rejects={4: True, 8: True, 12: True},
        block_length_ci_low={4: mean - 0.01, 8: mean - 0.01, 12: mean - 0.01},
        block_length_ci_high={4: mean + 0.01, 8: mean + 0.01, 12: mean + 0.01},
        top_origin_share=0.1,
        rho_distance=rho_distance,
        distance_values=CONSTANT_DISTANCE if distance_values is None else distance_values,
        rho_confounders=(
            {"removed_variance": 0.30, "n_patches_fully_missing": 0.10}
            if rho_confounders is None
            else rho_confounders
        ),
        rho_scaling=rho_scaling,
        n_partially_missing_patches_max=n_partial,
    )


def _inputs(stats, config, **overrides):
    base = dict(
        contrasts=stats,
        mean_clean_mae=1.0,
        location_ordering_spearman={0.20: 0.1, 0.40: -0.2},
        seed_sensitivity_frac={0.20: 0.002, 0.40: 0.002},
        config=config,
    )
    base.update(overrides)
    return DecisionInputs(**base)


def _quiet_family(prefix="C9"):
    """Filler contrasts that trigger nothing, so one check can be isolated."""
    return [
        _stat(
            f"{prefix}_geom_20", "geometry", rate=0.20, mean=0.001, median=0.001,
            rho_confounders={"removed_variance": 0.01}, rho_scaling=0.01,
        ),
        _stat(
            f"{prefix}_geom_40", "geometry", rate=0.40, mean=0.001, median=0.001,
            rho_confounders={"removed_variance": 0.01}, rho_scaling=0.01,
        ),
    ]


# --------------------------------------------------------------------------- #
# 1. Distance constant -> not_evaluable
# --------------------------------------------------------------------------- #

def test_constant_distance_is_not_evaluable():
    """The real pilot situation: one fixed subtrahend arm, distance constant."""
    check = compare_confounders_to_distance(
        rho_distance=float("nan"),
        distance_values=CONSTANT_DISTANCE,
        candidates={"removed_variance": 0.85},
        margin=0.10,
    )
    assert check.status == "not_evaluable"
    assert check.reason == NOT_EVALUABLE_CONSTANT_DISTANCE
    assert check.n_unique_finite_distance == 1
    assert check.is_trigger is False
    assert check.exceeding == []


def test_empty_distance_values_are_not_evaluable():
    check = compare_confounders_to_distance(
        rho_distance=float("nan"), distance_values=[],
        candidates={"removed_variance": 0.85}, margin=0.10,
    )
    assert check.status == "not_evaluable"
    assert check.n_unique_finite_distance == 0


def test_constant_distance_does_not_reach_triggers(config):
    """Even with a huge confounder rho, a constant-distance contrast is silent."""
    stats = [
        _stat("C1_loc_20", "location", rate=0.20,
              rho_confounders={"removed_variance": 0.99, "n_patches_fully_missing": 0.99}),
        _stat("C2_loc_40", "location", rate=0.40,
              rho_confounders={"removed_variance": 0.99, "n_patches_fully_missing": 0.99}),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert not any("confounder_out_tracks_distance" in t for t in decision.triggers)
    assert not any("patch_position_confound" in t for t in decision.triggers)

    statuses = decision.criteria["pivot"]["confounder_out_tracks_distance_status"]
    for contrast_id in ("C1_loc_20", "C2_loc_40"):
        assert statuses[contrast_id]["status"] == "not_evaluable"
        assert statuses[contrast_id]["reason"] == NOT_EVALUABLE_CONSTANT_DISTANCE


# --------------------------------------------------------------------------- #
# 2. Distance varies, nothing beats it -> false
# --------------------------------------------------------------------------- #

def test_varying_distance_no_confounder_wins_is_false():
    check = compare_confounders_to_distance(
        rho_distance=0.60,
        distance_values=VARYING_DISTANCE,
        candidates={"removed_variance": 0.30, "removed_slope": 0.65},  # 0.65 < 0.60 + 0.10
        margin=0.10,
    )
    assert check.status == "false"
    assert check.reason == "no_confounder_exceeds_distance"
    assert check.exceeding == []
    assert check.is_trigger is False
    assert sorted(check.compared) == ["removed_slope", "removed_variance"]


def test_margin_is_respected_exactly_at_the_boundary():
    """A confounder exactly at rho_distance + margin does NOT win (strict >)."""
    check = compare_confounders_to_distance(
        rho_distance=0.50, distance_values=VARYING_DISTANCE,
        candidates={"x": 0.60}, margin=0.10,
    )
    assert check.status == "false"


def test_varying_distance_no_winner_does_not_reach_triggers(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_distance=0.60,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.30, "n_patches_fully_missing": 0.10}),
        _stat("C2_loc_40", "location", rate=0.40, rho_distance=0.60,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.30, "n_patches_fully_missing": 0.10}),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert not any("confounder_out_tracks_distance" in t for t in decision.triggers)
    statuses = decision.criteria["pivot"]["confounder_out_tracks_distance_status"]
    assert statuses["C1_loc_20"]["status"] == "false"


# --------------------------------------------------------------------------- #
# 3. Distance varies, a confounder beats it -> true
# --------------------------------------------------------------------------- #

def test_varying_distance_confounder_wins_is_true():
    check = compare_confounders_to_distance(
        rho_distance=0.20,
        distance_values=VARYING_DISTANCE,
        candidates={"removed_variance": 0.85, "n_patches_fully_missing": 0.10},
        margin=0.10,
    )
    assert check.status == "true"
    assert check.reason == "confounder_exceeds_distance"
    assert check.exceeding == ["removed_variance"]
    assert check.is_trigger is True


def test_varying_distance_confounder_wins_reaches_triggers(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_distance=0.20,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.85, "n_patches_fully_missing": 0.10}),
        _stat("C2_loc_40", "location", rate=0.40, rho_distance=0.60,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.30, "n_patches_fully_missing": 0.10}),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert decision.label == "PIVOT"
    assert any("confounder_out_tracks_distance:C1_loc_20" in t for t in decision.triggers)
    statuses = decision.criteria["pivot"]["confounder_out_tracks_distance_status"]
    assert statuses["C1_loc_20"]["status"] == "true"
    assert statuses["C2_loc_40"]["status"] == "false"


# --------------------------------------------------------------------------- #
# 4. Regression: not_evaluable and false must never be conflated
# --------------------------------------------------------------------------- #

def test_nan_to_zero_collapse_cannot_silently_reappear():
    """THE regression test for the v1 bug.

    NaN rho_distance with a small-but-nonzero confounder rho. Under v1's
    NaN-to-0.0 collapse this compared 0.30 > 0.0 + 0.10 and returned a trigger.
    It must now be not_evaluable.
    """
    check = compare_confounders_to_distance(
        rho_distance=float("nan"),
        distance_values=CONSTANT_DISTANCE,
        candidates={"removed_variance": 0.30},
        margin=0.10,
    )
    assert check.status == "not_evaluable", "the v1 NaN-to-0.0 collapse has reappeared"
    assert check.status != "true"
    assert check.is_trigger is False


def test_not_evaluable_is_distinguishable_from_false():
    """The two must not be represented by the same value."""
    not_evaluable = compare_confounders_to_distance(
        rho_distance=float("nan"), distance_values=CONSTANT_DISTANCE,
        candidates={"x": 0.30}, margin=0.10,
    )
    evaluated_false = compare_confounders_to_distance(
        rho_distance=0.60, distance_values=VARYING_DISTANCE,
        candidates={"x": 0.30}, margin=0.10,
    )
    assert not_evaluable.status != evaluated_false.status
    assert not_evaluable.reason != evaluated_false.reason
    # Both are non-triggers, but for structurally different reasons.
    assert not_evaluable.is_trigger is False and evaluated_false.is_trigger is False


def test_source_contains_no_nan_to_zero_collapse():
    """Static guard against the exact v1 idiom returning anywhere."""
    from pathlib import Path

    source = Path("stats/decision.py").read_text(encoding="utf-8")
    assert 'if np.isfinite(stat.rho_distance) else 0.0' not in source
    assert 'if np.isfinite(s.rho_distance) else 0.0' not in source


def test_varying_distance_with_nan_rho_is_invalid_input_not_a_null():
    """Distance varies so this SHOULD have been computable; flag, never zero."""
    check = compare_confounders_to_distance(
        rho_distance=float("nan"),
        distance_values=VARYING_DISTANCE,
        candidates={"removed_variance": 0.85},
        margin=0.10,
    )
    assert check.status == "invalid_input"
    assert check.reason == INVALID_RHO_DISTANCE
    assert check.is_trigger is False
    assert check.status != "not_evaluable", "must not share the constant-distance path"


def test_invalid_input_does_not_reach_triggers(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_distance=float("nan"),
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.99, "n_patches_fully_missing": 0.99}),
        _stat("C2_loc_40", "location", rate=0.40, rho_distance=0.60,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.30, "n_patches_fully_missing": 0.10}),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert not any("confounder_out_tracks_distance" in t for t in decision.triggers)
    statuses = decision.criteria["pivot"]["confounder_out_tracks_distance_status"]
    assert statuses["C1_loc_20"]["status"] == "invalid_input"


def test_infinite_distance_values_are_not_counted_as_finite():
    check = compare_confounders_to_distance(
        rho_distance=float("nan"),
        distance_values=[128.0, float("inf"), float("nan"), 128.0],
        candidates={"x": 0.9}, margin=0.10,
    )
    assert check.status == "not_evaluable"
    assert check.n_unique_finite_distance == 1


# --------------------------------------------------------------------------- #
# 5. The same semantics for the scaling and patch-occupancy triggers
# --------------------------------------------------------------------------- #

def test_scaling_absolute_branch_survives_the_fix(config):
    """The distance-free branch of (e) must remain fully live."""
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_scaling=0.80),  # >= 0.50 threshold
        _stat("C2_loc_40", "location", rate=0.40, rho_scaling=0.10),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert any("internal_scaling_explains_pattern:C1_loc_20" in t for t in decision.triggers)
    assert decision.criteria["pivot"]["scaling_flagged_by_absolute_threshold"] == {
        "C1_loc_20": 0.80
    }
    # ...but its distance-relative branch is not evaluable and claims nothing.
    statuses = decision.criteria["pivot"]["internal_scaling_distance_branch_status"]
    assert statuses["C1_loc_20"]["status"] == "not_evaluable"
    assert decision.criteria["pivot"]["scaling_flagged_by_distance_comparison"] == {}


def test_scaling_distance_branch_alone_cannot_fire_on_constant_distance(config):
    """rho_scaling below the absolute threshold but above a fabricated zero."""
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_scaling=0.30),  # < 0.50, > 0.0 + 0.10
        _stat("C2_loc_40", "location", rate=0.40, rho_scaling=0.30),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert not any("internal_scaling_explains_pattern" in t for t in decision.triggers), (
        "the scaling trigger fired on a fabricated zero distance baseline"
    )
    statuses = decision.criteria["pivot"]["internal_scaling_distance_branch_status"]
    assert statuses["C1_loc_20"]["status"] == "not_evaluable"


def test_scaling_distance_branch_fires_when_distance_genuinely_varies(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_scaling=0.30,
              rho_distance=0.05, distance_values=VARYING_DISTANCE),
        _stat("C2_loc_40", "location", rate=0.40, rho_scaling=0.01,
              rho_distance=0.60, distance_values=VARYING_DISTANCE),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert any("internal_scaling_explains_pattern:C1_loc_20" in t for t in decision.triggers)
    assert decision.criteria["pivot"]["scaling_flagged_by_distance_comparison"] == {
        "C1_loc_20": ["rho_scaling"]
    }


def test_patch_partial_branch_survives_the_fix(config):
    """The distance-free branch of (f) must remain fully live."""
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, n_partial=3),
        _stat("C2_loc_40", "location", rate=0.40),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert any("patch_position_confound:C1_loc_20" in t for t in decision.triggers)
    assert decision.criteria["pivot"]["block_arms_with_partial_patches"] == {"C1_loc_20": 3}
    statuses = decision.criteria["pivot"]["patch_occupancy_out_tracking_distance_status"]
    assert statuses["C1_loc_20"]["status"] == "not_evaluable"


def test_patch_distance_branch_alone_cannot_fire_on_constant_distance(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, n_partial=0,
              rho_confounders={"removed_variance": 0.01, "n_patches_fully_missing": 0.95}),
        _stat("C2_loc_40", "location", rate=0.40, n_partial=0,
              rho_confounders={"removed_variance": 0.01, "n_patches_fully_missing": 0.95}),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert not any("patch_position_confound" in t for t in decision.triggers), (
        "the patch trigger fired on a fabricated zero distance baseline"
    )
    assert decision.criteria["pivot"]["patch_occupancy_out_tracking_distance"] == {}


def test_patch_distance_branch_fires_when_distance_genuinely_varies(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20, rho_distance=0.10,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.01, "n_patches_fully_missing": 0.95}),
        _stat("C2_loc_40", "location", rate=0.40, rho_distance=0.60,
              distance_values=VARYING_DISTANCE,
              rho_confounders={"removed_variance": 0.01, "n_patches_fully_missing": 0.10}),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))

    assert any("patch_position_confound:C1_loc_20" in t for t in decision.triggers)
    assert decision.criteria["pivot"]["patch_occupancy_out_tracking_distance"] == {
        "C1_loc_20": 0.95
    }


# --------------------------------------------------------------------------- #
# 6. Rule versioning
# --------------------------------------------------------------------------- #

def test_decision_carries_the_rule_version(config):
    stats = [
        _stat("C1_loc_20", "location", rate=0.20),
        _stat("C2_loc_40", "location", rate=0.40),
    ] + _quiet_family()
    decision = decide(_inputs(stats, config))
    assert decision.decision_rule_version == DECISION_RULE_VERSION == "v2"
    assert decision.as_dict()["decision_rule_version"] == "v2"
    assert decision.criteria["decision_rule_version"] == "v2"


def test_all_three_distance_relative_checks_are_reported(config):
    """Every distance-relative check must publish a per-contrast status."""
    stats = [
        _stat("C1_loc_20", "location", rate=0.20),
        _stat("C2_loc_40", "location", rate=0.40),
    ] + _quiet_family()
    pivot = decide(_inputs(stats, config)).criteria["pivot"]

    for key in (
        "confounder_out_tracks_distance_status",
        "internal_scaling_distance_branch_status",
        "patch_occupancy_out_tracking_distance_status",
    ):
        assert key in pivot
        assert set(pivot[key]) == {s.contrast_id for s in stats}
        for entry in pivot[key].values():
            assert entry["status"] in {"not_evaluable", "invalid_input", "false", "true"}
            assert entry["reason"]
    # The raw rho dumps are KEPT alongside the statuses, not replaced by them.
    assert "rho_distance" in pivot
    assert "rho_confounders" in pivot
    assert "rho_scaling" in pivot


# --------------------------------------------------------------------------- #
# 7. The preregistered record must not be overwritten by a re-analysis
# --------------------------------------------------------------------------- #

def test_reanalysis_refuses_to_overwrite_a_different_rule_version(tmp_path, config):
    """results/pilot_v1/decision.json is the record of what v1 produced.

    Re-running the analysis under v2 must not silently rewrite it.
    """
    import json

    import pandas as pd
    import pytest as _pytest

    from stats.analyze import analyse

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "decision.json").write_text(
        json.dumps({"decision_rule_version": "v1", "label": "PIVOT"}), encoding="utf-8"
    )
    # Deliberately unreadable inputs: the guard must fire BEFORE any analysis.
    (run_dir / "window_results.csv").write_text("", encoding="utf-8")
    (run_dir / "mask_diagnostics.csv").write_text("", encoding="utf-8")

    with _pytest.raises(FileExistsError, match="produced by decision rule v1"):
        analyse(run_dir, config)

    preserved = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    assert preserved["decision_rule_version"] == "v1"
    assert preserved["label"] == "PIVOT"


def test_reanalysis_into_a_separate_out_dir_is_not_blocked(tmp_path, config):
    """The intended post-hoc path: read from run_dir, write somewhere else."""
    import json

    from stats.analyze import analyse

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "decision.json").write_text(json.dumps({"decision_rule_version": "v1"}), encoding="utf-8")
    (run_dir / "window_results.csv").write_text("", encoding="utf-8")
    (run_dir / "mask_diagnostics.csv").write_text("", encoding="utf-8")

    # The guard does not fire for a clean out_dir; analysis then fails on the
    # empty CSVs, which is a different error entirely.
    with pytest.raises(Exception) as excinfo:
        analyse(run_dir, config, out_dir=tmp_path / "post_hoc")
    assert not isinstance(excinfo.value, FileExistsError)
    assert json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))["decision_rule_version"] == "v1"
