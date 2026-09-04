"""analysis_reporting_v2 — the B_g comparison-arm erratum.

Proves two things:
  1. the corrected text names B_g's real arm (internal_block), and
  2. NOTHING numeric moved — every metric, CI bound, p-value, per-gap reading,
     the SESOI, the dose-response result and the classification are identical
     between the original output and v2.

Adversarial tests perturb each of those in turn and confirm the identity check
fails loudly, so a passing check is evidence rather than decoration.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.reporting_v2 import (
    ADDED_COLUMNS,
    COMPARISON_ARM,
    ERRATUM_ID,
    INTERPRETATION_B,
    REPORTING_VERSION,
    REQUIRED_SOURCE_FILES,
    ReportingV2Error,
    build_reporting_v2,
    compare_reporting_versions,
)
from stats.mechanism_decision import INTERPRETATION

REPO_ROOT = Path(__file__).resolve().parents[1]
GAPS = (16, 32, 64, 128)

# The frozen ETTh1 readings, per family. THESE ARE DIFFERENT DISTRIBUTIONS and
# must never be interchanged.
#
# A prior round asserted that reporting_v2 changes "6, not 8" interpretation
# cells. That was wrong. It came from reusing an R-shaped reading distribution
# — which contains an UNRESOLVED at g=128, whose text is arm-neutral and so
# does not change — as a stand-in for B. On the real ETTh1 data ALL FOUR B
# readings are MATERIAL_POSITIVE, every one of which is arm-specific, so the
# real change count is 8: four in B_contrasts.csv and four in
# trailing_gap_analysis.json.
#
# Separate dicts, and a fixture builder that REQUIRES the family's own
# readings, so the substitution that caused that error cannot recur.
R_READINGS = {16: "EQUIVALENT", 32: "EQUIVALENT", 64: "EQUIVALENT", 128: "UNRESOLVED"}
B_READINGS = {
    16: "MATERIAL_POSITIVE", 32: "MATERIAL_POSITIVE",
    64: "MATERIAL_POSITIVE", 128: "MATERIAL_POSITIVE",
}
READINGS_BY_FAMILY = {"R": R_READINGS, "B": B_READINGS}


def _contrast_rows(prefix: str) -> list[dict]:
    """Rows for ONE family, using that family's own reading distribution.

    The family's readings are looked up here rather than passed in, so a caller
    cannot hand R's distribution to B.
    """
    if prefix not in READINGS_BY_FAMILY:
        raise AssertionError(f"unknown family {prefix!r}")
    readings = READINGS_BY_FAMILY[prefix]
    rows = []
    for i, g in enumerate(GAPS):
        reading = readings[g]
        rows.append({
            "contrast_id": f"{prefix}_{g}", "family": prefix, "gap": g,
            "reading": reading, "flags": "",
            "interpretation": INTERPRETATION[reading],
            "raw_mae_difference": f"{0.001234 + i * 0.0005:.17g}",
            "pct_of_mean_clean_mae": f"{0.05 + i * 0.02:.17g}",
            "median_paired_difference": f"{0.00111 + i * 0.0004:.17g}",
            "holm_ci95_low": f"{-0.0401 - i * 0.001:.17g}",
            "holm_ci95_high": f"{0.0422 + i * 0.001:.17g}",
            "holm_adjusted_p": f"{0.4123 - i * 0.01:.17g}",
            "holm_rejects": "False",
            "ci90_low": f"{-0.0301:.17g}", "ci90_high": f"{0.0333:.17g}",
            "frac_origins_positive": f"{0.5112 + i * 0.001:.17g}",
            "sesoi_delta": f"{0.0737250:.17g}",
            "estimate_block_len_4": f"{0.001231:.17g}",
            "estimate_block_len_8": f"{0.001234:.17g}",
            "estimate_block_len_12": f"{0.001237:.17g}",
        })
    return rows


@pytest.fixture
def analysis_dir(tmp_path) -> Path:
    """A realistic, completed analysis/ directory."""
    d = tmp_path / "analysis"
    d.mkdir()
    r_rows, b_rows = _contrast_rows("R"), _contrast_rows("B")
    pd.DataFrame(r_rows).to_csv(d / "R_contrasts.csv", index=False)
    pd.DataFrame(b_rows).to_csv(d / "B_contrasts.csv", index=False)
    (d / "dose_response.json").write_text(json.dumps({
        "x_units": "missing_patches (g / patch_size)",
        "mean_slope_per_patch": -0.00084271, "ci95_low": -0.00166,
        "ci95_high": 0.00033, "ci95_excludes_zero": False,
        "quadratic_term": 0.000112, "descriptive_only": True,
    }, indent=2), encoding="utf-8")
    (d / "classification.json").write_text(json.dumps({
        "label": "INCONCLUSIVE", "reason": "mixed_equivalent_and_unresolved",
        "rule_version": "preregistered-v1", "authoritative": True,
        "criteria": {"sesoi_delta": 0.073725,
                     "r_readings": {f"R_{g}": R_READINGS[g] for g in GAPS}},
    }, indent=2), encoding="utf-8")
    (d / "trailing_gap_analysis.json").write_text(json.dumps({
        "mean_clean_mae": 2.4575, "sesoi_delta": 0.073725, "n_origins": 178,
        "r_contrasts": r_rows, "b_contrasts": b_rows,
        "dose_response": {"mean_slope_per_patch": -0.00084271},
        "classification": {"label": "INCONCLUSIVE",
                           "reason": "mixed_equivalent_and_unresolved"},
    }, indent=2), encoding="utf-8")
    (d / "per_origin_slopes.csv").write_text("origin_id,slope\n0,0.001\n", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# The erratum is real, and confined to prose
# --------------------------------------------------------------------------- #

def test_the_computation_path_uses_internal_block_not_truncated_long():
    """The premise of the erratum: B_g was always computed correctly."""
    source = (REPO_ROOT / "experiments" / "analyze_trailing_gap.py").read_text(
        encoding="utf-8"
    )
    assert 'f"B_{g}": (matrix[f"trailing_nan_g{g}"] - matrix[f"internal_block_g{g}"])' in source
    assert 'f"R_{g}": (matrix[f"trailing_nan_g{g}"] - matrix[f"truncated_long_g{g}"])' in source


def test_the_original_r_texts_name_rs_arm_and_were_applied_to_b():
    """Documents exactly what was wrong, from the shipped strings."""
    assert "shorter-context/longer-horizon single-shot" in INTERPRETATION["MATERIAL_POSITIVE"]
    assert "horizon-dominant" in INTERPRETATION["EQUIVALENT"]
    source = (REPO_ROOT / "experiments" / "analyze_trailing_gap.py").read_text(
        encoding="utf-8"
    )
    # One shared dict, applied to both families -> B rows got R's arm.
    assert '"interpretation": INTERPRETATION[s.reading(delta)]' in source
    assert 'rows(b_stats, b_blocks, "B")' in source


def test_corrected_b_text_names_internal_block_and_never_truncation():
    for reading, text in INTERPRETATION_B.items():
        assert "shorter-context" not in text, reading
        assert "truncated" not in text.lower(), reading
        assert "horizon-dominant" not in text, reading
    assert "internal block" in INTERPRETATION_B["EQUIVALENT"]
    assert "internal block" in INTERPRETATION_B["MATERIAL_POSITIVE"]
    assert "internal block" in INTERPRETATION_B["MATERIAL_NEGATIVE"]


def test_corrected_b_text_carries_the_confounding_caveat_forward():
    """The existing caveat language is carried forward, not re-derived."""
    assert "CONFOUNDED WITH DIFFERENCES IN REMOVED CONTENT" in (
        INTERPRETATION_B["MATERIAL_POSITIVE"]
    )
    assert "confounded with differences in\nremoved content".replace("\n", " ") in (
        INTERPRETATION_B["EQUIVALENT"].replace("\n", " ")
    )


def test_r_family_text_is_untouched_by_the_erratum():
    """Only B was mislabelled; R's text must not drift."""
    from experiments import reporting_v2

    source = (REPO_ROOT / "experiments" / "reporting_v2.py").read_text(encoding="utf-8")
    assert "INTERPRETATION_R" not in source, "R's text is not redefined here"
    assert reporting_v2.COMPARISON_ARM["R"].startswith("truncated_long(g)")
    assert reporting_v2.COMPARISON_ARM["B"].startswith("internal_block(g, d=16)")


# --------------------------------------------------------------------------- #
# The identity proof
# --------------------------------------------------------------------------- #

def test_v2_is_built_beside_the_original_which_is_untouched(analysis_dir):
    before = {p.name: p.read_bytes() for p in analysis_dir.iterdir()}
    out = build_reporting_v2(analysis_dir)
    assert out.name == REPORTING_VERSION
    assert out != analysis_dir
    after = {p.name: p.read_bytes() for p in analysis_dir.iterdir()}
    assert before == after, "the original analysis directory was modified"


def test_identity_check_passes_and_reports_what_it_compared(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    report = compare_reporting_versions(analysis_dir, out)

    assert report.identical, report.violations
    assert not report.violations
    # It really did compare the substance, not a handful of cells.
    assert report.values_compared > 100
    assert report.numeric_fields_compared > 50
    assert "R_contrasts.csv" in report.files_compared
    assert "B_contrasts.csv" in report.files_compared
    assert "trailing_gap_analysis.json" in report.files_compared
    assert "classification.json" in report.files_compared
    assert "dose_response.json" in report.files_compared
    # On the real ETTh1 data all four B readings are MATERIAL_POSITIVE, which is
    # arm-specific, so all four change: four in the CSV and four again in the
    # payload = 8. The interpretation cells are the ONLY value differences;
    # additions are a separate category and are reported per category.
    interpretation_changes = report.interpretation_changes
    assert len(interpretation_changes) == 8, interpretation_changes
    assert all("R_contrasts" not in w for w in interpretation_changes)
    for gap in ("B_16", "B_32", "B_64", "B_128"):
        assert any(gap in w for w in interpretation_changes), gap


def test_addition_categories_are_counted_separately_and_8_is_not_the_total(analysis_dir):
    """The '8' figure is JSON contrast-row arms ONLY, and must not read as a total.

    Four r_contrasts[i].comparison_arm plus four b_contrasts[i].comparison_arm.
    The CSV column additions and the JSON root additions are separate, and
    together they outnumber it.
    """
    out = build_reporting_v2(analysis_dir)
    report = compare_reporting_versions(analysis_dir, out)
    assert report.identical, report.violations

    assert len(report.json_contrast_arm_additions) == 8, report.json_contrast_arm_additions
    assert all(a.endswith(".comparison_arm") for a in report.json_contrast_arm_additions)
    assert sum("r_contrasts" in a for a in report.json_contrast_arm_additions) == 4
    assert sum("b_contrasts" in a for a in report.json_contrast_arm_additions) == 4

    # 3 root keys, in trailing_gap_analysis.json only.
    assert len(report.json_root_additions) == 3, report.json_root_additions
    assert all(a.startswith("trailing_gap_analysis.json:") for a in report.json_root_additions)

    # 3 columns x 2 files = 6 columns, 4 rows each = 24 cells.
    assert len(report.csv_column_additions) == 6, report.csv_column_additions
    assert report.csv_added_cells == 24

    # The headline number is NOT the total.
    assert report.total_additions() == 8 + 3 + 24 == 35
    assert report.total_additions() != len(report.json_contrast_arm_additions)

    payload = report.as_dict()["additions"]
    assert payload["n_json_contrast_arm_additions"] == 8
    assert payload["total_additions"] == 35
    # No key in the reported payload calls 8 a total.
    assert "n_whitelisted_additions" not in payload


def test_all_four_real_b_readings_are_arm_specific():
    """Why the real change count is 8 and not 6.

    Every reading in B_READINGS names an arm in R's text, so every B row is
    rewritten. UNRESOLVED — which appears in R's distribution but NOT in B's —
    is the arm-neutral one that would not have changed.
    """
    for gap, reading in B_READINGS.items():
        assert INTERPRETATION[reading] != INTERPRETATION_B[reading], (
            f"B_{gap} reads {reading}, which must be rewritten"
        )
    assert "UNRESOLVED" not in B_READINGS.values()
    # ...and the arm-neutral reading really is identical across families.
    assert INTERPRETATION_B["UNRESOLVED"] == INTERPRETATION["UNRESOLVED"]
    for arm_word in ("shorter-context", "truncated", "horizon-dominant"):
        assert arm_word not in INTERPRETATION["UNRESOLVED"]


def test_r_and_b_reading_fixtures_are_structurally_separate():
    """The root cause of the '6 not 8' error, prevented structurally."""
    assert R_READINGS != B_READINGS
    assert set(R_READINGS.values()) != set(B_READINGS.values())
    assert READINGS_BY_FAMILY["R"] is R_READINGS
    assert READINGS_BY_FAMILY["B"] is B_READINGS
    # _contrast_rows looks the distribution up by family; it cannot be passed
    # the wrong one.
    import inspect

    assert list(inspect.signature(_contrast_rows).parameters) == ["prefix"]
    with pytest.raises(AssertionError, match="unknown family"):
        _contrast_rows("X")
    assert {r["reading"] for r in _contrast_rows("B")} == {"MATERIAL_POSITIVE"}
    assert "UNRESOLVED" in {r["reading"] for r in _contrast_rows("R")}


def test_every_named_field_class_is_actually_identical(analysis_dir):
    """Explicitly check the field classes the erratum promises are unchanged."""
    out = build_reporting_v2(analysis_dir)
    a = pd.read_csv(analysis_dir / "B_contrasts.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)

    for column in (
        "raw_mae_difference", "pct_of_mean_clean_mae", "median_paired_difference",
        "holm_ci95_low", "holm_ci95_high", "holm_adjusted_p", "holm_rejects",
        "ci90_low", "ci90_high", "frac_origins_positive", "sesoi_delta",
        "estimate_block_len_4", "estimate_block_len_8", "estimate_block_len_12",
        "reading", "contrast_id", "gap",
    ):
        assert list(a[column]) == list(b[column]), f"{column} changed"

    v1 = json.loads((analysis_dir / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    v2 = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    assert v1["sesoi_delta"] == v2["sesoi_delta"]
    assert v1["mean_clean_mae"] == v2["mean_clean_mae"]
    assert v1["dose_response"] == v2["dose_response"]
    assert v1["classification"] == v2["classification"]
    assert json.loads((analysis_dir / "classification.json").read_text(encoding="utf-8")) == (
        json.loads((out / "classification.json").read_text(encoding="utf-8"))
    )


def test_the_frozen_etth1_classification_is_not_altered(analysis_dir):
    """The erratum must not relabel or reinterpret the frozen result."""
    out = build_reporting_v2(analysis_dir)
    v2 = json.loads((out / "classification.json").read_text(encoding="utf-8"))
    assert v2["label"] == "INCONCLUSIVE"
    assert v2["reason"] == "mixed_equivalent_and_unresolved"
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    assert payload["classification"]["label"] == "INCONCLUSIVE"
    assert payload["classification"]["reason"] == "mixed_equivalent_and_unresolved"


def test_v2_adds_an_explicit_comparison_arm_to_both_families(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    for name, family in (("R_contrasts.csv", "R"), ("B_contrasts.csv", "B")):
        frame = pd.read_csv(out / name, dtype=str, keep_default_na=False)
        assert set(frame["comparison_arm"]) == {COMPARISON_ARM[family]}
        assert set(frame["reporting_version"]) == {REPORTING_VERSION}
        assert set(frame["erratum_id"]) == {ERRATUM_ID}
    assert set(ADDED_COLUMNS) == {"comparison_arm", "reporting_version", "erratum_id"}


# --------------------------------------------------------------------------- #
# Adversarial: the check must FAIL LOUDLY on any numeric drift
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "column",
    ["raw_mae_difference", "holm_ci95_low", "holm_ci95_high", "holm_adjusted_p",
     "ci90_low", "ci90_high", "sesoi_delta", "frac_origins_positive",
     "median_paired_difference", "estimate_block_len_8", "reading"],
)
def test_identity_check_fails_when_a_value_is_perturbed(analysis_dir, column):
    """One cell moved -> violation. Covers metrics, CIs, p-values, SESOI, readings."""
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
    original = frame.loc[0, column]
    frame.loc[0, column] = "EQUIVALENT_TAMPERED" if column == "reading" else "9.99999"
    frame.to_csv(out / "B_contrasts.csv", index=False)

    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, f"perturbing {column} was not detected"
    assert any(column in v for v in report.violations), report.violations
    assert str(original) in " ".join(report.violations)


def test_identity_check_fails_when_the_dose_response_changes(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "dose_response.json").read_text(encoding="utf-8"))
    payload["mean_slope_per_patch"] = 0.5
    (out / "dose_response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("mean_slope_per_patch" in v for v in report.violations)


def test_identity_check_fails_when_the_classification_changes(analysis_dir):
    """The frozen label must be impossible to alter silently."""
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "classification.json").read_text(encoding="utf-8"))
    payload["label"] = "NO_MATERIAL_DIFFERENCE_VS_TRUNCATION"
    (out / "classification.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("label" in v for v in report.violations)


def test_identity_check_fails_when_r_interpretation_is_touched(analysis_dir):
    """Only B's description may differ. R's must not."""
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "R_contrasts.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "interpretation"] = "something else"
    frame.to_csv(out / "R_contrasts.csv", index=False)
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("R_contrasts.csv" in v and "interpretation" in v for v in report.violations)


def test_identity_check_fails_on_a_dropped_row_or_column(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame.drop(columns=["holm_adjusted_p"]).to_csv(out / "B_contrasts.csv", index=False)
    assert any("dropped column" in v for v in
               compare_reporting_versions(analysis_dir, out).violations)

    out2 = build_reporting_v2(analysis_dir, analysis_dir.parent / "v2b")
    frame = pd.read_csv(out2 / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame.iloc[:-1].to_csv(out2 / "B_contrasts.csv", index=False)
    assert any("row count" in v for v in
               compare_reporting_versions(analysis_dir, out2).violations)


def test_build_refuses_an_unknown_reading(analysis_dir):
    frame = pd.read_csv(analysis_dir / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "reading"] = "NOT_A_READING"
    frame.to_csv(analysis_dir / "B_contrasts.csv", index=False)
    with pytest.raises(ReportingV2Error, match="unknown reading"):
        build_reporting_v2(analysis_dir)


def test_build_refuses_a_missing_analysis_directory(tmp_path):
    with pytest.raises(ReportingV2Error):
        build_reporting_v2(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# Hardening: the permitted TRANSFORMATION must be validated, not just the
# preservation of the originals. The first three below previously slipped
# through undetected — a v2 could keep every number and still assert the wrong
# arm, plant a rogue field, or corrupt a pass-through file.
# --------------------------------------------------------------------------- #

def test_wrong_comparison_arm_is_rejected(analysis_dir):
    """GAP 1 (previously undetected): every number preserved, arm still wrong."""
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame["comparison_arm"] = COMPARISON_ARM["R"]      # B row claiming R's arm
    frame.to_csv(out / "B_contrasts.csv", index=False)

    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, "a B row asserting R's comparison arm was accepted"
    assert any("comparison_arm" in v for v in report.violations), report.violations


def test_wrong_comparison_arm_on_the_r_family_is_also_rejected(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "R_contrasts.csv", dtype=str, keep_default_na=False)
    frame["comparison_arm"] = COMPARISON_ARM["B"]
    frame.to_csv(out / "R_contrasts.csv", index=False)
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("comparison_arm" in v for v in report.violations)


def test_rogue_extra_numeric_field_in_the_json_is_rejected(analysis_dir):
    """GAP 2 (previously undetected): a planted field nothing compared against."""
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    payload["smuggled_effect_size"] = 0.4213           # root-level rogue
    payload["b_contrasts"][0]["extra_p_value"] = 0.001  # row-level rogue
    (out / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, "a planted numeric field was accepted"
    assert any("smuggled_effect_size" in v for v in report.violations), report.violations
    assert any("extra_p_value" in v for v in report.violations), report.violations


def test_modified_passthrough_file_is_rejected(analysis_dir):
    """GAP 3 (previously undetected): a copied-through file quietly edited."""
    out = build_reporting_v2(analysis_dir)
    target = out / "per_origin_slopes.csv"
    assert target.exists(), "fixture must include a pass-through file"
    target.write_text("origin_id,slope\n0,999.0\n", encoding="utf-8")

    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, "a modified pass-through file was accepted"
    assert any("pass-through file modified" in v for v in report.violations), report.violations


def test_missing_passthrough_file_is_rejected(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    (out / "per_origin_slopes.csv").unlink()
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("pass-through file missing" in v for v in report.violations)


def test_unexpected_extra_file_is_rejected(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    (out / "surprise.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("unexpected file in v2" in v for v in report.violations)


def test_incorrect_b_interpretation_is_rejected(analysis_dir):
    """The text must EQUAL INTERPRETATION_B[reading], not merely differ from R's."""
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "interpretation"] = "some other plausible-sounding sentence"
    frame.to_csv(out / "B_contrasts.csv", index=False)
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("is not INTERPRETATION_B" in v for v in report.violations)


def test_b_interpretation_from_the_wrong_reading_is_rejected(analysis_dir):
    """Right dictionary, wrong key — still wrong."""
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "interpretation"] = INTERPRETATION_B["MATERIAL_NEGATIVE"]
    frame.to_csv(out / "B_contrasts.csv", index=False)
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("is not INTERPRETATION_B" in v for v in report.violations)


def test_rewriting_r_interpretation_is_rejected(analysis_dir):
    """Only B may be rewritten; R's text must carry across untouched."""
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "R_contrasts.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "interpretation"] = INTERPRETATION_B["MATERIAL_POSITIVE"]
    frame.to_csv(out / "R_contrasts.csv", index=False)
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("R_contrasts.csv" in v for v in report.violations)


def test_wrong_reporting_version_or_erratum_id_is_rejected(analysis_dir):
    for column, bad in (("reporting_version", "v3"), ("erratum_id", "ERRATUM-OTHER")):
        out = build_reporting_v2(analysis_dir, analysis_dir.parent / f"v2_{column}")
        frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
        frame[column] = bad
        frame.to_csv(out / "B_contrasts.csv", index=False)
        report = compare_reporting_versions(analysis_dir, out)
        assert not report.identical, column
        assert any(column in v for v in report.violations), (column, report.violations)


def test_unexpected_csv_column_is_rejected(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    frame = pd.read_csv(out / "B_contrasts.csv", dtype=str, keep_default_na=False)
    frame["editorial_note"] = "looks fine to me"
    frame.to_csv(out / "B_contrasts.csv", index=False)
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("not on the whitelist" in v for v in report.violations)


def test_unexpected_root_json_key_is_rejected(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    payload["editorial_summary"] = "the effect replicated"
    (out / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("not on the whitelist" in v for v in report.violations)


def test_whitelist_is_exactly_the_documented_set():
    from experiments.reporting_v2 import ADDED_ROOT_KEYS, ADDED_ROW_KEYS

    assert set(ADDED_COLUMNS) == {"comparison_arm", "reporting_version", "erratum_id"}
    assert set(ADDED_ROOT_KEYS) == {"reporting_version", "erratum_id", "erratum_note"}
    assert set(ADDED_ROW_KEYS) == {"comparison_arm"}


def test_build_refuses_a_preexisting_nonempty_output_directory(analysis_dir):
    """Same discipline as the runbook's initialization guard."""
    out = build_reporting_v2(analysis_dir)
    assert out.exists() and any(out.iterdir())
    with pytest.raises(ReportingV2Error, match="already exists and is not empty"):
        build_reporting_v2(analysis_dir)
    # An empty directory is fine to write into.
    empty = analysis_dir.parent / "empty_target"
    empty.mkdir()
    assert build_reporting_v2(analysis_dir, empty) == empty


def test_passthrough_files_are_actually_compared(analysis_dir):
    """The pass-through check must not be vacuous."""
    out = build_reporting_v2(analysis_dir)
    report = compare_reporting_versions(analysis_dir, out)
    assert report.identical
    assert "per_origin_slopes.csv" in report.passthrough_files_compared
    assert "classification.json" in report.passthrough_files_compared
    assert "dose_response.json" in report.passthrough_files_compared


# --------------------------------------------------------------------------- #
# Hardening round 2: the whitelist is a set of PATHS, not a set of key names;
# the erratum note is checked by content; and a missing required source
# artifact is a hard failure rather than a silently skipped comparison.
# --------------------------------------------------------------------------- #

def test_comparison_arm_outside_a_contrast_row_is_rejected(analysis_dir):
    """A key that merely SHARES THE WHITELISTED NAME, at a path that is not a
    contrast row, must be rejected. A name-based whitelist waves this through."""
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    payload["classification"]["comparison_arm"] = "NOT A CONTRAST ROW"
    (out / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, "comparison_arm planted in `classification` was accepted"
    assert any("classification.comparison_arm" in v for v in report.violations), (
        report.violations
    )
    # And it was NOT miscounted as a legitimate contrast-row addition.
    assert len(report.json_contrast_arm_additions) == 8
    assert all(
        "classification" not in a for a in report.json_contrast_arm_additions
    ), report.json_contrast_arm_additions


@pytest.mark.parametrize(
    "filename", ["classification.json", "dose_response.json"]
)
def test_comparison_arm_in_another_analysis_file_is_rejected(analysis_dir, filename):
    """Contrast-row additions are permitted in the payload file ONLY."""
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / filename).read_text(encoding="utf-8"))
    payload["comparison_arm"] = COMPARISON_ARM["B"]
    (out / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, f"comparison_arm accepted in {filename}"
    assert any("comparison_arm" in v for v in report.violations), report.violations


def test_comparison_arm_nested_deeper_than_a_contrast_row_is_rejected(analysis_dir):
    """`r_contrasts[i]` is a permitted path; anything below it is not."""
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    payload["dose_response"]["comparison_arm"] = COMPARISON_ARM["R"]
    (out / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical
    assert any("dose_response.comparison_arm" in v for v in report.violations)


def test_altered_erratum_note_is_rejected(analysis_dir):
    """Presence is not enough; the note must equal its single defined value."""
    from experiments.reporting_v2 import ERRATUM_NOTE

    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    assert payload["erratum_note"] == ERRATUM_NOTE
    payload["erratum_note"] = ERRATUM_NOTE.replace(
        "No number, interval", "Some numbers, intervals"
    )
    (out / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, "a rewritten erratum note was accepted"
    assert any("erratum_note" in v and "does not match" in v for v in report.violations), (
        report.violations
    )


def test_missing_erratum_note_is_rejected(analysis_dir):
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    del payload["erratum_note"]
    (out / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, "a v2 with no erratum note was accepted"
    assert any(
        "erratum_note" in v and "missing" in v for v in report.violations
    ), report.violations


@pytest.mark.parametrize(
    "where", ["wrong_file", "inside_a_contrast_row", "nested_object"]
)
def test_structurally_misplaced_erratum_note_is_rejected(analysis_dir, where):
    """The right text in the wrong place is still wrong — the same
    path-specificity that item 1 applies to `comparison_arm`."""
    from experiments.reporting_v2 import ERRATUM_NOTE

    out = build_reporting_v2(analysis_dir)
    if where == "wrong_file":
        target = out / "classification.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["erratum_note"] = ERRATUM_NOTE
    else:
        target = out / "trailing_gap_analysis.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        if where == "inside_a_contrast_row":
            payload["b_contrasts"][0]["erratum_note"] = ERRATUM_NOTE
        else:
            payload["classification"]["erratum_note"] = ERRATUM_NOTE
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = compare_reporting_versions(analysis_dir, out)
    assert not report.identical, f"a misplaced erratum note ({where}) was accepted"
    assert any("erratum_note" in v for v in report.violations), report.violations


def test_the_erratum_note_is_defined_exactly_once(analysis_dir):
    """One constant, used by both the builder and the verifier."""
    from experiments.reporting_v2 import ERRATUM_NOTE, EXPECTED_ROOT_VALUES

    assert EXPECTED_ROOT_VALUES["erratum_note"] is ERRATUM_NOTE
    source = (REPO_ROOT / "experiments" / "reporting_v2.py").read_text(encoding="utf-8")
    # The literal text appears once: in the constant's own definition.
    assert source.count("Reporting erratum only:") == 1
    out = build_reporting_v2(analysis_dir)
    payload = json.loads((out / "trailing_gap_analysis.json").read_text(encoding="utf-8"))
    assert payload["erratum_note"] == ERRATUM_NOTE


@pytest.mark.parametrize("filename", list(REQUIRED_SOURCE_FILES))
def test_build_hard_fails_on_each_individually_missing_source_file(analysis_dir, filename):
    """Non-vacuity: EACH of the five, removed on its own, must fail the build.

    Parametrised one file at a time. A single combined test could pass because
    some other file's absence tripped the check, proving nothing about this one.
    """
    (analysis_dir / filename).unlink()
    with pytest.raises(ReportingV2Error, match="required source artifact"):
        build_reporting_v2(analysis_dir)


@pytest.mark.parametrize("filename", list(REQUIRED_SOURCE_FILES))
def test_compare_hard_fails_on_each_individually_missing_source_file(
    analysis_dir, filename, tmp_path
):
    """The same five, one at a time, on the comparison path."""
    out = build_reporting_v2(analysis_dir)
    (analysis_dir / filename).unlink()
    with pytest.raises(ReportingV2Error, match="required source artifact"):
        compare_reporting_versions(analysis_dir, out)


@pytest.mark.parametrize("filename", list(REQUIRED_SOURCE_FILES))
def test_compare_hard_fails_on_each_individually_missing_v2_file(analysis_dir, filename):
    """A required artifact absent from v2 must fail too, not be skipped."""
    out = build_reporting_v2(analysis_dir)
    (out / filename).unlink()
    with pytest.raises(ReportingV2Error, match="required v2 artifact"):
        compare_reporting_versions(analysis_dir, out)


def test_the_required_file_list_is_exactly_the_five_named_artifacts():
    assert set(REQUIRED_SOURCE_FILES) == {
        "R_contrasts.csv", "B_contrasts.csv", "trailing_gap_analysis.json",
        "classification.json", "dose_response.json",
    }
    assert len(REQUIRED_SOURCE_FILES) == 5
