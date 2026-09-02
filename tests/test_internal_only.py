"""Tests for the POST-HOC EXPLORATORY internal-only block-proximity analysis.

Model-mocked: no weights required. Nothing here validates a scientific claim;
these pin the mechanics — position selection, Holm family separation, the trend
slope, the confounder-difference status typing, and the rule that d=0 never
leaks into an internal-position aggregate.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from stats.internal_only import (
    ANALYSIS_KIND,
    CONFOUNDER_FAMILY,
    INVALID_NON_FINITE_DELTA,
    INVALID_NON_FINITE_RESPONSE,
    NOT_EVALUABLE_CONSTANT_DELTA,
    _delta_confounder,
    _holm_over_confounders,
    per_origin_trend_slopes,
    rate_positions,
)


# --------------------------------------------------------------------------- #
# 1. The four internal positions are correctly identified per rate
# --------------------------------------------------------------------------- #

def test_internal_positions_exclude_d0(config):
    positions = {p.rate_tag: p for p in rate_positions(config)}
    assert positions["20"].internal == (64, 128, 192, 256)
    assert positions["40"].internal == (48, 96, 144, 192)
    for pos in positions.values():
        assert 0 not in pos.internal, "d=0 leaked into the internal position set"
        assert len(pos.internal) == 4
        assert pos.trailing_boundary == 0


def test_primary_contrast_arms_match_the_specification(config):
    """IC1_20 is d=64 - d=256; IC2_40 is d=48 - d=192."""
    positions = {p.rate_tag: p for p in rate_positions(config)}
    assert (positions["20"].near, positions["20"].far) == (64, 256)
    assert (positions["40"].near, positions["40"].far) == (48, 192)
    assert positions["20"].column(64) == "block_r20_d64"
    assert positions["20"].column(256) == "block_r20_d256"
    assert positions["40"].column(48) == "block_r40_d48"
    assert positions["40"].column(192) == "block_r40_d192"


def test_internal_columns_never_include_the_d0_column(config):
    for pos in rate_positions(config):
        assert pos.trailing_boundary_column not in pos.internal_columns
        assert pos.trailing_boundary_column == f"block_r{pos.rate_tag}_d0"
        assert all(not c.endswith("_d0") for c in pos.internal_columns)


def test_positions_are_derived_from_config_not_hardcoded():
    """Change the config's positions and the internal set follows."""
    fake = {
        "masks": {
            "rates": [
                {"rate": 0.20, "missing_count": 64, "block_length": 64,
                 "block_distances": [0, 32, 96, 160]},
            ]
        }
    }
    pos = rate_positions(fake)[0]
    assert pos.internal == (32, 96, 160)
    assert (pos.near, pos.far) == (32, 160)


def test_missing_d0_or_too_few_internal_positions_is_rejected():
    with pytest.raises(ValueError, match="d=0"):
        rate_positions({"masks": {"rates": [
            {"rate": 0.2, "block_length": 64, "block_distances": [64, 128]}]}})
    with pytest.raises(ValueError, match=">= 2 internal"):
        rate_positions({"masks": {"rates": [
            {"rate": 0.2, "block_length": 64, "block_distances": [0, 64]}]}})


# --------------------------------------------------------------------------- #
# 2. Per-origin trend slope, against a hand-built toy example
# --------------------------------------------------------------------------- #

def test_trend_slope_matches_a_hand_calculated_example():
    """d = 64,128,192,256; y = 1,2,3,4  ->  slope = 1 per 64 steps = 1/64.

    By hand: x centred = (-96,-32,+32,+96), y centred = (-1.5,-0.5,+0.5,+1.5).
    sum(xy) = 144+16+16+144 = 320;  sum(x^2) = 9216+1024+1024+9216 = 20480.
    slope = 320/20480 = 0.015625 = 1/64.
    """
    distances = (64, 128, 192, 256)
    columns = [f"block_r20_d{d}" for d in distances]
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=columns, index=[0])
    slopes = per_origin_trend_slopes(
        paired_diff_matrix=frame, distances=distances, columns=columns
    )
    assert slopes.iloc[0] == pytest.approx(1.0 / 64.0)
    assert slopes.iloc[0] == pytest.approx(0.015625)


def test_trend_slope_signs_and_flat_case():
    distances = (64, 128, 192, 256)
    columns = [f"block_r20_d{d}" for d in distances]
    frame = pd.DataFrame(
        [
            [4.0, 3.0, 2.0, 1.0],      # decreasing -> negative slope
            [5.0, 5.0, 5.0, 5.0],      # flat -> zero
            [0.0, 0.0, 0.0, 6.4],      # only the far point raised
        ],
        columns=columns, index=[0, 1, 2],
    )
    slopes = per_origin_trend_slopes(
        paired_diff_matrix=frame, distances=distances, columns=columns
    )
    assert slopes.iloc[0] == pytest.approx(-1.0 / 64.0)
    assert slopes.iloc[1] == pytest.approx(0.0)
    assert slopes.iloc[2] > 0


def test_trend_slope_matches_numpy_polyfit_on_random_data():
    rng = np.random.default_rng(0)
    distances = (48, 96, 144, 192)
    columns = [f"block_r40_d{d}" for d in distances]
    values = rng.normal(size=(25, 4))
    frame = pd.DataFrame(values, columns=columns, index=range(25))
    slopes = per_origin_trend_slopes(
        paired_diff_matrix=frame, distances=distances, columns=columns
    )
    for i in range(25):
        expected = np.polyfit(np.asarray(distances, dtype=float), values[i], 1)[0]
        assert slopes.iloc[i] == pytest.approx(expected)


def test_trend_slope_is_invariant_to_the_clean_offset():
    """slope(MAE_d - MAE_clean) == slope(MAE_d): the clean term cancels."""
    distances = (64, 128, 192, 256)
    columns = [f"block_r20_d{d}" for d in distances]
    raw = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=columns, index=[0])
    shifted = raw - 7.25  # a per-origin clean offset
    a = per_origin_trend_slopes(paired_diff_matrix=raw, distances=distances, columns=columns)
    b = per_origin_trend_slopes(paired_diff_matrix=shifted, distances=distances, columns=columns)
    assert a.iloc[0] == pytest.approx(b.iloc[0])


def test_trend_slope_rejects_degenerate_distances():
    columns = ["a", "b"]
    frame = pd.DataFrame([[1.0, 2.0]], columns=columns, index=[0])
    with pytest.raises(ValueError, match="constant"):
        per_origin_trend_slopes(paired_diff_matrix=frame, distances=(64, 64), columns=columns)
    with pytest.raises(ValueError, match="at least two"):
        per_origin_trend_slopes(paired_diff_matrix=frame, distances=(64,), columns=["a"])


# --------------------------------------------------------------------------- #
# 3. Confounder-difference diagnostic: status typing, no NaN-to-0 regression
# --------------------------------------------------------------------------- #

def test_delta_confounder_evaluates_when_inputs_are_finite():
    rng = np.random.default_rng(1)
    delta_z = rng.normal(size=178)
    delta_mae = 0.8 * delta_z + rng.normal(scale=0.2, size=178)
    result = _delta_confounder(
        contrast_id="IC1_20", confounder="removed_variance",
        delta_mae=delta_mae, delta_z=delta_z,
    )
    assert result.status in {"false", "true"}
    assert np.isfinite(result.rho) and np.isfinite(result.p_value)
    assert result.n_unique_finite_delta == 178


def test_non_finite_delta_z_is_invalid_input_not_a_silent_value():
    """A NaN here means a DATA PROBLEM and must be surfaced, never suppressed."""
    delta_z = np.arange(178, dtype=float)
    delta_z[5] = np.nan
    result = _delta_confounder(
        contrast_id="IC1_20", confounder="removed_slope",
        delta_mae=np.arange(178, dtype=float), delta_z=delta_z,
    )
    assert result.status == "invalid_input"
    assert result.reason == INVALID_NON_FINITE_DELTA
    assert result.status != "not_evaluable", (
        "a non-finite delta must NOT be classed as the structurally-undefined case"
    )
    assert not np.isfinite(result.rho)
    assert result.holm_rejects is False


def test_non_finite_delta_mae_is_invalid_input():
    delta_mae = np.arange(178, dtype=float)
    delta_mae[3] = np.inf
    result = _delta_confounder(
        contrast_id="IC2_40", confounder="removed_variance",
        delta_mae=delta_mae, delta_z=np.arange(178, dtype=float),
    )
    assert result.status == "invalid_input"
    assert result.reason == INVALID_NON_FINITE_RESPONSE


def test_constant_delta_z_is_not_evaluable():
    """Distinct from invalid_input: the correlation is undefined, not broken."""
    result = _delta_confounder(
        contrast_id="IC1_20", confounder="removed_variance",
        delta_mae=np.random.default_rng(2).normal(size=178),
        delta_z=np.full(178, 3.5),
    )
    assert result.status == "not_evaluable"
    assert result.reason == NOT_EVALUABLE_CONSTANT_DELTA
    assert result.n_unique_finite_delta == 1


def test_invalid_and_not_evaluable_are_never_conflated():
    invalid = _delta_confounder(
        contrast_id="X", confounder="z",
        delta_mae=np.arange(10, dtype=float),
        delta_z=np.array([1.0] * 9 + [np.nan]),
    )
    undefined = _delta_confounder(
        contrast_id="X", confounder="z",
        delta_mae=np.arange(10, dtype=float), delta_z=np.full(10, 1.0),
    )
    assert invalid.status != undefined.status
    assert invalid.reason != undefined.reason
    assert invalid.holm_rejects is False and undefined.holm_rejects is False


def test_non_evaluable_members_are_excluded_from_the_holm_family():
    """Only evaluated members count toward m, and neither can become true."""
    rng = np.random.default_rng(3)
    strong_z = rng.normal(size=178)
    strong = _delta_confounder(
        contrast_id="IC1_20", confounder="removed_variance",
        delta_mae=0.9 * strong_z + rng.normal(scale=0.1, size=178), delta_z=strong_z,
    )
    broken = _delta_confounder(
        contrast_id="IC1_20", confounder="removed_slope",
        delta_mae=np.arange(178, dtype=float), delta_z=np.full(178, np.nan),
    )
    degenerate = _delta_confounder(
        contrast_id="IC1_20", confounder="removed_abs_diff_mean",
        delta_mae=rng.normal(size=178), delta_z=np.full(178, 1.0),
    )
    family = [strong, broken, degenerate]
    _holm_over_confounders(family, family_alpha=0.05)

    assert strong.status == "true"
    # m == 1 (only `strong` was evaluable), so its alpha is the full family alpha.
    assert strong.holm_alpha == pytest.approx(0.05)
    assert broken.status == "invalid_input" and broken.holm_rejects is False
    assert degenerate.status == "not_evaluable" and degenerate.holm_rejects is False


def test_holm_over_confounders_is_monotone_and_conservative():
    rng = np.random.default_rng(4)
    family = []
    for name, strength in zip(CONFOUNDER_FAMILY, (0.9, 0.5, 0.2, 0.0)):
        z = rng.normal(size=178)
        family.append(
            _delta_confounder(
                contrast_id="IC1_20", confounder=name,
                delta_mae=strength * z + rng.normal(scale=0.5, size=178), delta_z=z,
            )
        )
    _holm_over_confounders(family, family_alpha=0.05)
    evaluated = [r for r in family if r.status in {"false", "true"}]
    assert len(evaluated) == 4
    for r in evaluated:
        assert r.holm_adjusted_p >= r.p_value
        assert r.holm_alpha <= 0.05
    ordered = sorted(evaluated, key=lambda r: r.p_value)
    adjusted = [r.holm_adjusted_p for r in ordered]
    assert adjusted == sorted(adjusted)


def test_confounder_family_is_prespecified_and_exactly_four():
    assert CONFOUNDER_FAMILY == (
        "removed_variance",
        "removed_slope",
        "removed_abs_diff_mean",
        "retained_mean_change_frac_of_clean_std",
    )
    assert len(CONFOUNDER_FAMILY) == 4


# --------------------------------------------------------------------------- #
# 4. End-to-end: family independence, d=0 isolation, output shape
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory, config):
    """A synthetic 178x17 run directory, so the whole analysis can execute.

    The NUMBERS here are meaningless — this fixture exists to exercise the
    machinery, not to stand in for a Chronos-2 result.
    """
    run_dir = tmp_path_factory.mktemp("synthetic_run")
    rng = np.random.default_rng(11)
    n_origins = int(config["design"]["expected_origins"])

    conditions = ["clean"]
    for spec in config["masks"]["rates"]:
        tag = f"{int(round(float(spec['rate']) * 100))}"
        conditions += [f"random_r{tag}_s{s}" for s in config["masks"]["point_random_seeds"]]
    for spec in config["masks"]["rates"]:
        tag = f"{int(round(float(spec['rate']) * 100))}"
        conditions += [f"block_r{tag}_d{d}" for d in spec["block_distances"]]

    rows, diag_rows = [], []
    for origin in range(n_origins):
        clean_mae = 2.0 + rng.normal(scale=0.3)
        for condition in conditions:
            if condition == "clean":
                mae = clean_mae
                distance = np.nan
            else:
                distance = (
                    float(condition.rsplit("_d", 1)[1]) if "_d" in condition else np.nan
                )
                # A mild, arbitrary structure; not a claim about anything.
                bump = 0.0 if np.isnan(distance) else max(0.0, 0.4 - 0.001 * distance)
                mae = clean_mae + bump + rng.normal(scale=0.05)
            rows.append(
                {
                    "origin_id": origin, "condition_id": condition, "status": "ok",
                    "mae": mae, "mse": mae**2, "rmse": mae,
                    "pattern": "clean" if condition == "clean" else "contiguous_block",
                    "rate": 0.0, "seed": None, "distance_to_boundary": distance,
                }
            )
            diag_rows.append(
                {
                    "origin_id": origin, "condition_id": condition,
                    "removed_variance": abs(rng.normal(scale=2.0)),
                    "removed_slope": rng.normal(),
                    "removed_abs_diff_mean": abs(rng.normal()),
                    "retained_mean_change_frac_of_clean_std": rng.normal(scale=0.3),
                    "mean_missing_distance_to_boundary": distance,
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "window_results.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(run_dir / "mask_diagnostics.csv", index=False)
    return run_dir


@pytest.fixture(scope="module")
def fast_config(config):
    """Same design, far fewer bootstrap replicates so the suite stays quick."""
    import json

    cfg = json.loads(json.dumps(config))
    cfg["statistics"]["bootstrap"]["n_replicates"] = 200
    return cfg


@pytest.fixture(scope="module")
def payload(synthetic_run, fast_config, tmp_path_factory):
    from stats.internal_only import analyse_internal_only

    return analyse_internal_only(
        synthetic_run, fast_config, out_dir=tmp_path_factory.mktemp("internal_out")
    )


def test_analysis_is_labelled_post_hoc_everywhere(payload):
    assert payload["analysis_kind"] == ANALYSIS_KIND == "post_hoc_exploratory"
    assert payload["preregistered"] is False
    assert payload["feeds_decision_function"] is False
    for row in payload["primary_contrasts"] + payload["pairwise_contrasts"]:
        assert row["analysis_kind"] == "post_hoc_exploratory"
        assert "POST-HOC" in row["note"]
    for row in payload["trend_slopes"]:
        assert row["analysis_kind"] == "post_hoc_exploratory"


def test_primary_family_holds_exactly_the_two_internal_contrasts(payload):
    assert sorted(payload["holm_families"]["internal_primary"]) == ["IC1_20", "IC2_40"]
    assert {r["contrast_id"] for r in payload["primary_contrasts"]} == {"IC1_20", "IC2_40"}
    for row in payload["primary_contrasts"]:
        assert row["holm_family"] == "internal_primary{IC1_20,IC2_40}"


def test_original_preregistered_family_is_not_recorrected(payload, config):
    """The original {C1,C2,C3,C4} family must not be touched or merged."""
    original = payload["holm_families"]["original_preregistered_family_not_recorrected"]
    assert original == [c["id"] for c in config["contrasts"]]
    produced = {r["contrast_id"] for r in payload["primary_contrasts"]} | {
        r["contrast_id"] for r in payload["pairwise_contrasts"]
    }
    assert not produced & set(original), "an original contrast id leaked into this analysis"


def test_pairwise_families_are_six_per_rate_and_independent(payload, config):
    families = payload["holm_families"]["internal_pairwise_per_rate"]
    assert set(families) == {"20", "40"}
    for tag, members in families.items():
        assert len(members) == 6, f"rate {tag} must have C(4,2)=6 pairwise contrasts"
        assert len(set(members)) == 6
        assert all(m.startswith(f"IP_{tag}_") for m in members)
    # Two families of 6, never one family of 12.
    assert not set(families["20"]) & set(families["40"])
    assert len(payload["pairwise_contrasts"]) == 12


def test_holm_families_are_corrected_independently(payload):
    """Each family's Holm alphas reflect its OWN size, not a pooled one."""
    for row in payload["pairwise_contrasts"]:
        assert row["holm_family"].endswith("(6)")
    # A family of 6 corrected as a family of 12 would produce adjusted p-values
    # at most double; the cleanest structural check is that each rate's family
    # is corrected over exactly its own 6 members.
    by_rate = {}
    for row in payload["pairwise_contrasts"]:
        by_rate.setdefault(row["holm_family"], []).append(row)
    assert len(by_rate) == 2
    for members in by_rate.values():
        assert len(members) == 6
        adjusted = sorted(m["holm_adjusted_p"] for m in members)
        # Holm's smallest adjusted p is at most m * raw p for m = 6.
        smallest = min(members, key=lambda m: m["p_value_two_sided"])
        assert smallest["holm_adjusted_p"] <= min(1.0, 6 * smallest["p_value_two_sided"]) + 1e-12
        assert adjusted == sorted(adjusted)


def test_confounder_families_are_four_per_contrast_not_eight(payload):
    families = payload["holm_families"]["confounder_difference_per_contrast"]
    assert set(families) == {"IC1_20", "IC2_40"}
    for members in families.values():
        assert members == list(CONFOUNDER_FAMILY)
        assert len(members) == 4
    rows = payload["confounder_difference_correlations"]
    assert len(rows) == 8
    for contrast_id in ("IC1_20", "IC2_40"):
        subset = [r for r in rows if r["contrast_id"] == contrast_id]
        assert len(subset) == 4
        evaluated = [r for r in subset if r["status"] in {"false", "true"}]
        if evaluated:
            # Holm alpha of the least significant member is family_alpha / 1
            # for a family of 4 -> the largest alpha present is 0.05.
            assert max(r["holm_alpha"] for r in evaluated) <= 0.05 + 1e-12
            assert min(r["holm_alpha"] for r in evaluated) >= 0.05 / 4 - 1e-12


def test_d0_never_appears_in_any_internal_aggregate(payload, config):
    """The single most important invariant of this analysis."""
    internal_rows = payload["primary_contrasts"] + payload["pairwise_contrasts"]
    for row in internal_rows:
        assert row["position_kind"] == "internal_positions"
        assert not row["minuend"].endswith("_d0")
        assert not row["subtrahend"].endswith("_d0")
        assert row["near_distance"] > 0 and row["far_distance"] > 0
    for row in payload["trend_slopes"]:
        assert 0 not in row["internal_distances"]
        assert row["position_kind"] == "internal_positions"
    # ...and d=0 is present, but only under its own separate label.
    assert len(payload["trailing_boundary"]) == 2
    for row in payload["trailing_boundary"]:
        assert row["position_kind"] == "trailing_boundary_condition"
        assert row["distance_to_boundary"] == 0
        assert row["condition_id"].endswith("_d0")
        assert "NEVER pooled" in row["note"]


def test_rates_are_analysed_separately_never_pooled(payload):
    assert sorted(payload["rates_analysed_separately"]) == [0.20, 0.40]
    rates = {r["rate"] for r in payload["primary_contrasts"]}
    assert rates == {0.20, 0.40}
    assert len(payload["primary_contrasts"]) == 2, "one primary contrast per rate, not pooled"
    for row in payload["pairwise_contrasts"]:
        tag = f"{int(round(row['rate'] * 100))}"
        assert row["minuend"].startswith(f"block_r{tag}_")
        assert row["subtrahend"].startswith(f"block_r{tag}_")


def test_expected_output_files_are_written(payload, synthetic_run, fast_config, tmp_path):
    from stats.internal_only import analyse_internal_only

    out_dir = tmp_path / "out"
    analyse_internal_only(synthetic_run, fast_config, out_dir=out_dir)
    for name in (
        "internal_primary_contrasts.csv",
        "internal_pairwise_contrasts.csv",
        "internal_trend_slopes.csv",
        "per_origin_trend_slopes.csv",
        "trailing_boundary_d0.csv",
        "confounder_difference_correlations.csv",
        "internal_only_analysis.json",
    ):
        assert (out_dir / name).exists(), f"{name} was not written"
    slopes = pd.read_csv(out_dir / "per_origin_trend_slopes.csv", index_col=0)
    assert list(slopes.columns) == ["trend_slope_rate20", "trend_slope_rate40"]
    assert len(slopes) == 178


def test_does_not_write_into_the_run_directory(synthetic_run, fast_config, tmp_path):
    """Existing artifacts of the original run must be left alone."""
    from stats.internal_only import analyse_internal_only

    before = {p.name for p in synthetic_run.iterdir()}
    analyse_internal_only(synthetic_run, fast_config, out_dir=tmp_path / "elsewhere")
    assert {p.name for p in synthetic_run.iterdir()} == before


def test_bootstrap_settings_match_the_preregistered_machinery(payload, fast_config):
    assert payload["bootstrap"]["main_block_length"] == 8
    assert sorted(payload["bootstrap"]["block_lengths"]) == [4, 8, 12]
    for row in payload["primary_contrasts"]:
        for b in (4, 8, 12):
            assert f"estimate_block_len_{b}" in row
            assert f"holm_ci95_low_block_len_{b}" in row
