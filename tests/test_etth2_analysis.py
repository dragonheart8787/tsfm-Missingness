"""The ETTh2 analysis layer: reuse the shared statistics, add only the decision.

The point of these tests is that ``analyze_etth2`` introduces no new statistical
machinery. It calls the ETTh1 ``analyse`` and then applies the frozen
replication rule to the rows that were WRITTEN TO DISK, so the decision is
computed from exactly the numbers delivered to the reviewer.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.analyze_etth2 import REPLICATION_OUTPUT, gap_stats_from_rows
from stats.mechanism_decision import EQUIVALENT, MATERIAL_POSITIVE, UNRESOLVED
from stats.replication_decision import (
    CONTRADICTED,
    INCONCLUSIVE,
    PRIMARY_CONTRASTS,
    REPLICATED,
    ReplicationInputs,
    classify_replication,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = {"decision": {"no_go": {"equivalence_band_frac_of_clean_mae": 0.03}}}
MEAN_CLEAN_MAE = 2.0
DELTA = 0.06


def _row(contrast_id: str, *, lo90, hi90, lo95, hi95) -> dict:
    """A row shaped exactly like analyse()'s emitted contrast rows."""
    return {
        "contrast_id": contrast_id, "family": contrast_id[0],
        "gap": int(contrast_id.split("_")[1]),
        "raw_mae_difference": 0.5 * (lo95 + hi95),
        "median_paired_difference": 0.5 * (lo95 + hi95),
        "holm_ci95_low": lo95, "holm_ci95_high": hi95,
        "ci90_low": lo90, "ci90_high": hi90,
        "holm_rejects": bool(lo95 > 0 or hi95 < 0),
    }


EQUIV = dict(lo90=-0.5 * DELTA, hi90=0.5 * DELTA, lo95=-0.5 * DELTA, hi95=0.5 * DELTA)
MATPOS = dict(lo90=2 * DELTA, hi90=4 * DELTA, lo95=2 * DELTA, hi95=4 * DELTA)
UNRES = dict(lo90=-3 * DELTA, hi90=3 * DELTA, lo95=-3 * DELTA, hi95=3 * DELTA)


def test_gap_stats_are_rebuilt_faithfully_from_the_emitted_rows():
    """The decision must read the delivered numbers, not a parallel copy."""
    rows = [_row("R_16", **EQUIV), _row("R_32", **MATPOS), _row("R_64", **UNRES)]
    stats = gap_stats_from_rows(rows, mean_clean_mae=MEAN_CLEAN_MAE)
    assert [s.contrast_id for s in stats] == ["R_16", "R_32", "R_64"]
    assert [s.gap for s in stats] == [16, 32, 64]
    assert [s.reading(DELTA) for s in stats] == [EQUIVALENT, MATERIAL_POSITIVE, UNRESOLVED]
    for stat, row in zip(stats, rows):
        assert stat.ci_low_equivalence == row["ci90_low"]
        assert stat.ci_high_equivalence == row["ci90_high"]
        assert stat.ci_low_holm == row["holm_ci95_low"]
        assert stat.ci_high_holm == row["holm_ci95_high"]
        assert stat.mean_clean_mae == MEAN_CLEAN_MAE


@pytest.mark.parametrize(
    "readings,expected",
    [
        ((EQUIV, EQUIV, EQUIV), REPLICATED),
        ((EQUIV, MATPOS, EQUIV), CONTRADICTED),
        ((EQUIV, EQUIV, UNRES), INCONCLUSIVE),
        ((MATPOS, UNRES, EQUIV), CONTRADICTED),
    ],
)
def test_the_end_to_end_row_to_outcome_path(readings, expected):
    rows = [_row(cid, **spec) for cid, spec in zip(PRIMARY_CONTRASTS, readings)]
    rows.append(_row("R_128", **UNRES))
    stats = gap_stats_from_rows(rows, mean_clean_mae=MEAN_CLEAN_MAE)
    decision = classify_replication(
        ReplicationInputs(
            r_gaps=stats, mean_clean_mae=MEAN_CLEAN_MAE, pilot_config=PILOT
        )
    )
    assert decision.label == expected


def test_the_equivalence_branch_reads_the_uncorrected_90_percent_interval():
    """The resolved multiplicity treatment, checked on the numbers.

    A gap whose NOMINAL 90% CI lies inside the band is EQUIVALENT even when its
    Holm-corrected 95% interval does not — that is exactly what "no additional
    correction on the intersection" means operationally.
    """
    row = _row("R_16", lo90=-0.5 * DELTA, hi90=0.5 * DELTA,
               lo95=-3 * DELTA, hi95=3 * DELTA)
    stat = gap_stats_from_rows([row], mean_clean_mae=MEAN_CLEAN_MAE)[0]
    assert stat.reading(DELTA) == EQUIVALENT
    # Widening only the Holm interval must not change the reading...
    wider = _row("R_16", lo90=-0.5 * DELTA, hi90=0.5 * DELTA,
                 lo95=-9 * DELTA, hi95=9 * DELTA)
    assert gap_stats_from_rows([wider], mean_clean_mae=MEAN_CLEAN_MAE)[0].reading(
        DELTA
    ) == EQUIVALENT
    # ...but widening the 90% interval past the band must.
    narrow_holm = _row("R_16", lo90=-2 * DELTA, hi90=2 * DELTA,
                       lo95=-0.1 * DELTA, hi95=0.1 * DELTA)
    assert gap_stats_from_rows([narrow_holm], mean_clean_mae=MEAN_CLEAN_MAE)[0].reading(
        DELTA
    ) == UNRESOLVED


def test_the_material_branch_reads_the_holm_corrected_interval():
    """Holm is RETAINED for material directional readings, as on ETTh1."""
    # 90% CI clears the band, but the Holm interval does not: not material.
    row = _row("R_16", lo90=2 * DELTA, hi90=4 * DELTA,
               lo95=-0.5 * DELTA, hi95=6 * DELTA)
    assert gap_stats_from_rows([row], mean_clean_mae=MEAN_CLEAN_MAE)[0].reading(
        DELTA
    ) == UNRESOLVED


def test_the_analysis_module_adds_no_statistical_machinery():
    """It orchestrates; the statistics come from the shared ETTh1 analysis."""
    source = (REPO_ROOT / "experiments" / "analyze_etth2.py").read_text(encoding="utf-8")
    for token in ("bootstrap_paired_" + "difference", "holm_" + "correct",
                  "np.percentile", "aggregate_" + "mae"):
        assert token not in source, f"{token} is reimplemented in analyze_etth2"
    from experiments.analyze_trailing_gap import analyse
    import experiments.analyze_etth2 as mod

    assert mod.analyse is analyse
    assert REPLICATION_OUTPUT == "classification_etth2.json"


# --------------------------------------------------------------------------- #
# Family-correct reporting, built in — and mock refusal
# --------------------------------------------------------------------------- #

import copy
import json
import subprocess
import sys

import pandas as pd
import yaml

from experiments.analyze_etth2 import (
    COMPARISON_ARM,
    FORBIDDEN_ERRATUM_ID,
    MOCK_LABEL_PREFIX,
    MOCK_STAMP,
    Etth2ReportingError,
    MockAnalysisRefused,
    analyse_etth2,
    apply_family_correct_reporting,
    numeric_snapshot_of,
    assess_provenance,
    verify_reporting,
)
from experiments.reporting_v2 import INTERPRETATION_B
from stats.mechanism_decision import INTERPRETATION

B_READINGS = ("MATERIAL_POSITIVE",) * 4
R_READINGS = ("EQUIVALENT", "EQUIVALENT", "EQUIVALENT", "UNRESOLVED")
GAPS = (16, 32, 64, 128)


def _payload(mock: bool = False) -> dict:
    """A payload shaped exactly like analyse()'s, carrying v1's defect.

    v1's defect is INTERPRETATION[reading] on BOTH families — R's text, naming
    R's arm, applied to B rows. That is the input this pipeline must correct
    before anything is promoted.
    """
    def rows(prefix, readings):
        return [
            {
                "contrast_id": f"{prefix}_{g}", "family": prefix, "gap": g,
                "reading": r, "interpretation": INTERPRETATION[r],
                "raw_mae_difference": 0.001 * i, "pct_of_mean_clean_mae": 0.05 * i,
                "median_paired_difference": 0.002 * i,
                "holm_ci95_low": -0.04, "holm_ci95_high": 0.04,
                "holm_adjusted_p": 0.4, "holm_rejects": False,
                "ci90_low": -0.03, "ci90_high": 0.03,
                "frac_origins_positive": 0.51, "sesoi_delta": 0.0737,
            }
            for i, (g, r) in enumerate(zip(GAPS, readings))
        ]

    payload = {
        "mean_clean_mae": 2.4575,
        "r_contrasts": rows("R", R_READINGS),
        "b_contrasts": rows("B", B_READINGS),
    }
    if mock:
        payload.update(MOCK_STAMP)
    return payload


def test_b_rows_get_family_correct_text_and_r_rows_are_untouched():
    payload = _payload()
    before_r = [row["interpretation"] for row in payload["r_contrasts"]]
    apply_family_correct_reporting(payload, mock=False)
    for row in payload["b_contrasts"]:
        assert row["interpretation"] == INTERPRETATION_B[row["reading"]]
        assert "shorter-context" not in row["interpretation"]
        assert "truncated" not in row["interpretation"].lower()
        assert row["comparison_arm"] == COMPARISON_ARM["B"]
        assert row["comparison_arm"].startswith("internal_block(g, d=16)")
    assert [row["interpretation"] for row in payload["r_contrasts"]] == before_r
    for row in payload["r_contrasts"]:
        assert row["comparison_arm"] == COMPARISON_ARM["R"]
        assert row["comparison_arm"].startswith("truncated_long(g)")


def test_every_numeric_field_survives_the_correction_exactly():
    payload = _payload()
    snapshot = numeric_snapshot_of(payload)
    apply_family_correct_reporting(payload, mock=False)
    verify_reporting(payload, snapshot, mock=False)      # must not raise
    # And explicitly, field by field.
    for family in ("r_contrasts", "b_contrasts"):
        for column in ("raw_mae_difference", "holm_ci95_low", "holm_ci95_high",
                       "ci90_low", "ci90_high", "sesoi_delta", "holm_adjusted_p"):
            assert [r[column] for r in payload[family]] == snapshot[f"{family}|{column}"]


def test_verification_rejects_a_moved_number_and_refuses_promotion():
    payload = _payload()
    snapshot = numeric_snapshot_of(payload)
    apply_family_correct_reporting(payload, mock=False)
    payload["b_contrasts"][0]["holm_ci95_low"] = 9.99
    with pytest.raises(Etth2ReportingError, match="changed during reporting correction"):
        verify_reporting(payload, snapshot, mock=False)


def test_verification_rejects_a_missing_comparison_arm():
    payload = _payload()
    snapshot = numeric_snapshot_of(payload)
    apply_family_correct_reporting(payload, mock=False)
    del payload["r_contrasts"][0]["comparison_arm"]
    with pytest.raises(Etth2ReportingError, match="comparison_arm is wrong"):
        verify_reporting(payload, snapshot, mock=False)


def test_verification_rejects_uncorrected_b_text():
    """The exact ETTh1 defect, caught before promotion rather than after."""
    payload = _payload()
    snapshot = numeric_snapshot_of(payload)
    apply_family_correct_reporting(payload, mock=False)
    payload["b_contrasts"][0]["interpretation"] = INTERPRETATION["MATERIAL_POSITIVE"]
    with pytest.raises(Etth2ReportingError, match="not INTERPRETATION_B"):
        verify_reporting(payload, snapshot, mock=False)


def test_the_etth1_erratum_id_is_refused_anywhere_in_an_etth2_artifact():
    """ETTh2 never had that defect; claiming a correction would misdescribe it."""
    payload = _payload()
    snapshot = numeric_snapshot_of(payload)
    apply_family_correct_reporting(payload, mock=False)
    payload["erratum_id"] = FORBIDDEN_ERRATUM_ID
    with pytest.raises(Etth2ReportingError, match="never had that defect"):
        verify_reporting(payload, snapshot, mock=False)


def test_correct_output_carries_no_erratum_id_at_all():
    payload = _payload()
    apply_family_correct_reporting(payload, mock=False)
    blob = json.dumps(payload)
    assert FORBIDDEN_ERRATUM_ID not in blob
    # No erratum IDENTIFIER anywhere — the note may say the word while
    # explaining why there is none, which is the opposite of claiming one.
    assert "erratum_id" not in payload
    for family in ("r_contrasts", "b_contrasts"):
        for row in payload[family]:
            assert "erratum_id" not in row
    assert payload["reporting_provenance"] == "family_correct_from_the_start"


# --------------------------------------------------------------------------- #
# Mock detection and the stamp
# --------------------------------------------------------------------------- #

def test_the_mock_stamp_is_complete():
    assert MOCK_STAMP["mock_model"] is True
    assert MOCK_STAMP["scientifically_valid"] is False
    assert MOCK_STAMP["authoritative_result"] is False
    assert "not an etth2 finding" in MOCK_STAMP["mock_model_note"].lower()
    assert "pipeline exercise" in MOCK_STAMP["mock_model_note"].lower()


def test_verification_requires_the_full_stamp_on_a_mock_payload():
    payload = _payload(mock=True)
    snapshot = numeric_snapshot_of(payload)
    apply_family_correct_reporting(payload, mock=True)
    verify_reporting(payload, snapshot, mock=True)      # complete: must not raise
    for key in ("mock_model", "scientifically_valid", "authoritative_result",
                "mock_model_note"):
        broken = copy.deepcopy(payload)
        del broken[key]
        with pytest.raises(Etth2ReportingError, match="missing or misstates"):
            verify_reporting(broken, snapshot, mock=True)


def test_a_mock_label_cannot_be_mistaken_for_a_real_outcome():
    """Structural, not prose: the label itself is prefixed."""
    from stats.replication_decision import OUTCOMES

    labelled = f"{MOCK_LABEL_PREFIX}REPLICATED"
    assert labelled not in OUTCOMES
    for outcome in OUTCOMES:
        assert f"{MOCK_LABEL_PREFIX}{outcome}" not in OUTCOMES


# --------------------------------------------------------------------------- #
# The whole-directory sweep and the staging -> verify -> promote pattern
# --------------------------------------------------------------------------- #

from experiments.analyze_etth2 import (
    STAGING_SUFFIX,
    stamp_every_json,
    verify_staging_directory,
)


def _staged(tmp_path: Path, *, b_text: str, arm: str, mock: bool = False) -> Path:
    staging = tmp_path / "analysis.staging"
    staging.mkdir()
    rows = [
        {"contrast_id": f"B_{g}", "family": "B", "gap": g, "reading": "MATERIAL_POSITIVE",
         "interpretation": b_text, "comparison_arm": arm}
        for g in GAPS
    ]
    pd.DataFrame(rows).to_csv(staging / "B_contrasts.csv", index=False)
    payload = {"b_contrasts": rows}
    if mock:
        payload.update(MOCK_STAMP)
    (staging / "etth2_analysis.json").write_text(json.dumps(payload), encoding="utf-8")
    return staging


def test_the_sweep_passes_a_correct_directory(tmp_path):
    staging = _staged(
        tmp_path, b_text=INTERPRETATION_B["MATERIAL_POSITIVE"], arm=COMPARISON_ARM["B"]
    )
    verify_staging_directory(staging, mock=False)      # must not raise


def test_the_sweep_catches_r_arm_text_on_a_b_row_in_any_file(tmp_path):
    """The gap this closes: a correct file shipping beside a wrong one."""
    staging = _staged(
        tmp_path, b_text=INTERPRETATION["MATERIAL_POSITIVE"], arm=COMPARISON_ARM["B"]
    )
    with pytest.raises(Etth2ReportingError, match="R-arm text"):
        verify_staging_directory(staging, mock=False)


def test_the_sweep_catches_a_missing_comparison_arm_in_any_file(tmp_path):
    staging = _staged(
        tmp_path, b_text=INTERPRETATION_B["MATERIAL_POSITIVE"], arm=COMPARISON_ARM["R"]
    )
    with pytest.raises(Etth2ReportingError, match="comparison_arm"):
        verify_staging_directory(staging, mock=False)


def test_the_sweep_catches_the_erratum_id_in_any_file(tmp_path):
    staging = _staged(
        tmp_path, b_text=INTERPRETATION_B["MATERIAL_POSITIVE"], arm=COMPARISON_ARM["B"]
    )
    (staging / "stray.json").write_text(json.dumps({"note": FORBIDDEN_ERRATUM_ID}), encoding="utf-8")
    with pytest.raises(Etth2ReportingError, match="erratum id"):
        verify_staging_directory(staging, mock=False)


def test_the_sweep_catches_an_unstamped_json_on_a_mock_run(tmp_path):
    """The regression that was found by this very sweep: dose_response.json."""
    staging = _staged(
        tmp_path, b_text=INTERPRETATION_B["MATERIAL_POSITIVE"],
        arm=COMPARISON_ARM["B"], mock=True,
    )
    (staging / "dose_response.json").write_text(json.dumps({"mean_slope": 0.1}), encoding="utf-8")
    with pytest.raises(Etth2ReportingError, match="missing 'mock_model'"):
        verify_staging_directory(staging, mock=True)
    # Stamping every json fixes it, and the sweep then passes.
    stamped = stamp_every_json(staging)
    assert "dose_response.json" in stamped
    verify_staging_directory(staging, mock=True)


def test_stamping_sweeps_the_directory_rather_than_a_filename_list(tmp_path):
    """An artifact added later by the shared analysis must be stamped too."""
    staging = tmp_path / "s"
    staging.mkdir()
    for name in ("a.json", "b.json", "surprise_new_output.json"):
        (staging / name).write_text(json.dumps({"x": 1}), encoding="utf-8")
    (staging / "not_json.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    stamped = stamp_every_json(staging)
    assert set(stamped) == {"a.json", "b.json", "surprise_new_output.json"}
    for name in stamped:
        payload = json.loads((staging / name).read_text(encoding="utf-8"))
        assert payload["mock_model"] is True
        assert payload["scientifically_valid"] is False
        assert payload["authoritative_result"] is False
        assert payload["x"] == 1, "the stamp must not destroy existing content"


def test_promotion_refuses_to_overwrite_an_existing_official_directory(tmp_path):
    from experiments.analyze_etth2 import _promote

    staging = tmp_path / "staging"
    staging.mkdir()
    official = tmp_path / "analysis"
    official.mkdir()
    with pytest.raises(Etth2ReportingError, match="Refusing to overwrite"):
        _promote(staging, official)


def test_a_failed_verification_leaves_no_official_output(tmp_path):
    """The point of staging: a crash cannot leave a half-correct official result."""
    from experiments.analyze_etth2 import _promote

    staging = _staged(
        tmp_path, b_text=INTERPRETATION["MATERIAL_POSITIVE"], arm=COMPARISON_ARM["B"]
    )
    official = tmp_path / "analysis"
    with pytest.raises(Etth2ReportingError):
        verify_staging_directory(staging, mock=False)
    assert not official.exists(), "an official output was created despite a failure"
    # Promotion is a separate, later step; it was never reached.
    assert staging.exists()
    assert staging.name.endswith(STAGING_SUFFIX)


def test_the_official_path_is_never_the_staging_path():
    assert STAGING_SUFFIX and STAGING_SUFFIX.startswith(".")
    assert not STAGING_SUFFIX.endswith("/")


# --------------------------------------------------------------------------- #
# Positive provenance: REAL only on explicit, consistent evidence
# --------------------------------------------------------------------------- #

from experiments.analyze_etth2 import (
    EXPECTED_ENTRYPOINT,
    EXPECTED_EXPERIMENT,
    EXPECTED_PHASE,
    PROVENANCE_INVALID,
    PROVENANCE_MOCK,
    PROVENANCE_REAL,
    InvalidProvenance,
)

# The pinned revision, read from its ONE source of truth. Never a literal here:
# a second copy in the tests could drift from the config and the tests would
# then be asserting against the wrong value while passing.
PINNED_REVISION = yaml.safe_load(
    (REPO_ROOT / "configs" / "pilot_config.yaml").read_text(encoding="utf-8")
)["model"]["revision"]

REAL_MANIFEST = {
    "mock_model": False,
    "experiment": EXPECTED_EXPERIMENT,
    "phase": EXPECTED_PHASE,
    "execution_entrypoint": EXPECTED_ENTRYPOINT,
    "model_contract": {"revision": PINNED_REVISION},
}
REAL_SUMMARY = {k: REAL_MANIFEST[k] for k in
                ("mock_model", "experiment", "phase", "execution_entrypoint")}
MOCK_MANIFEST = dict(REAL_MANIFEST, mock_model=True,
                     model_contract={"revision": "mock-revision"})
MOCK_SUMMARY = dict(REAL_SUMMARY, mock_model=True)


def _run(tmp_path: Path, manifest, summary, name="run") -> Path:
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
    if summary is not None:
        (run / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run


def test_a_legitimate_real_run_is_REAL(tmp_path):
    verdict = assess_provenance(_run(tmp_path, REAL_MANIFEST, REAL_SUMMARY),
                              expected_revision=PINNED_REVISION)
    assert verdict.kind == PROVENANCE_REAL
    assert verdict.is_real and not verdict.is_mock
    assert verdict.reasons == []


def test_a_legitimate_mock_run_is_MOCK(tmp_path):
    verdict = assess_provenance(_run(tmp_path, MOCK_MANIFEST, MOCK_SUMMARY),
                              expected_revision=PINNED_REVISION)
    assert verdict.kind == PROVENANCE_MOCK
    assert verdict.is_mock and not verdict.is_real


@pytest.mark.parametrize(
    "label,manifest,summary,needle",
    [
        ("empty manifest", {}, REAL_SUMMARY, "does not carry"),
        ("only mock_model false", {"mock_model": False}, REAL_SUMMARY, "does not carry"),
        ("no summary at all", REAL_MANIFEST, None, "run_summary.json is absent"),
        ("no manifest at all", None, REAL_SUMMARY, "run_manifest.json is absent"),
        ("contradictory mock flags", REAL_MANIFEST,
         dict(REAL_SUMMARY, mock_model=True), "disagree on 'mock_model'"),
        ("wrong experiment",
         dict(REAL_MANIFEST, experiment="trailing-gap-mechanism-v1"),
         dict(REAL_SUMMARY, experiment="trailing-gap-mechanism-v1"), "experiment is"),
        ("wrong phase", dict(REAL_MANIFEST, phase="clean-reference"),
         dict(REAL_SUMMARY, phase="clean-reference"), "phase is"),
        ("wrong entrypoint",
         dict(REAL_MANIFEST, execution_entrypoint="experiments/run_trailing_gap.py"),
         dict(REAL_SUMMARY, execution_entrypoint="experiments/run_trailing_gap.py"),
         "execution_entrypoint is"),
        ("malformed mock_model", dict(REAL_MANIFEST, mock_model="false"),
         dict(REAL_SUMMARY, mock_model="false"), "not a boolean"),
        ("no revision", {k: v for k, v in REAL_MANIFEST.items()
                         if k != "model_contract"}, REAL_SUMMARY,
         "no model_contract.revision"),
        ("real flag, placeholder revision",
         dict(REAL_MANIFEST, model_contract={"revision": "mock-revision"}),
         REAL_SUMMARY, "is a placeholder, not a checkpoint"),
        ("mock flag, pinned revision",
         dict(MOCK_MANIFEST, model_contract={"revision": PINNED_REVISION}),
         MOCK_SUMMARY, "is not a placeholder"),
    ],
)
def test_every_broken_provenance_case_is_INVALID(tmp_path, label, manifest, summary, needle):
    verdict = assess_provenance(
        _run(tmp_path, manifest, summary, name=label[:20]),
        expected_revision=PINNED_REVISION,
    )
    assert verdict.kind == PROVENANCE_INVALID, label
    assert any(needle in reason for reason in verdict.reasons), (label, verdict.reasons)


def test_an_empty_directory_is_INVALID_not_real(tmp_path):
    run = tmp_path / "nothing"
    run.mkdir()
    assert assess_provenance(
        run, expected_revision=PINNED_REVISION
    ).kind == PROVENANCE_INVALID


def test_unreadable_json_is_INVALID(tmp_path):
    run = tmp_path / "broken"
    run.mkdir()
    (run / "run_manifest.json").write_text("{not json", encoding="utf-8")
    (run / "run_summary.json").write_text(json.dumps(REAL_SUMMARY), encoding="utf-8")
    verdict = assess_provenance(run, expected_revision=PINNED_REVISION)
    assert verdict.kind == PROVENANCE_INVALID
    assert any("unreadable" in r for r in verdict.reasons)


def test_the_three_verdicts_are_distinct(tmp_path):
    """REAL, MOCK and INVALID must be three outcomes, not two plus a fallback."""
    kinds = {
        assess_provenance(_run(tmp_path, REAL_MANIFEST, REAL_SUMMARY, "r"),
                          expected_revision=PINNED_REVISION).kind,
        assess_provenance(_run(tmp_path, MOCK_MANIFEST, MOCK_SUMMARY, "m"),
                          expected_revision=PINNED_REVISION).kind,
        assess_provenance(_run(tmp_path, {}, REAL_SUMMARY, "i"),
                          expected_revision=PINNED_REVISION).kind,
    }
    assert kinds == {PROVENANCE_REAL, PROVENANCE_MOCK, PROVENANCE_INVALID}


def test_allow_mock_analysis_does_NOT_bypass_invalid_provenance(tmp_path, config):
    """The core requirement: broken provenance is a DIFFERENT failure class."""
    etth2 = yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8"))
    gap = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8"))
    run = _run(tmp_path, {}, REAL_SUMMARY, "invalid")

    for allow in (False, True):
        with pytest.raises(InvalidProvenance) as excinfo:
            analyse_etth2(
                run, etth2_config=etth2, pilot_config=config, gap_config=gap,
                out_dir=tmp_path / f"out{int(allow)}", allow_mock_analysis=allow,
            )
        assert PROVENANCE_INVALID in str(excinfo.value)
        assert "does NOT bypass this" in str(excinfo.value)
    assert not (tmp_path / "out0").exists()
    assert not (tmp_path / "out1").exists()


def test_invalid_provenance_is_not_a_mock_refusal(tmp_path, config):
    """Distinct exception types, so a caller cannot conflate them."""
    etth2 = yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8"))
    gap = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8"))
    run = _run(tmp_path, {}, REAL_SUMMARY, "invalid2")
    with pytest.raises(InvalidProvenance):
        analyse_etth2(run, etth2_config=etth2, pilot_config=config, gap_config=gap,
                      out_dir=tmp_path / "o")
    assert not issubclass(InvalidProvenance, MockAnalysisRefused)
    assert not issubclass(MockAnalysisRefused, InvalidProvenance)


# --------------------------------------------------------------------------- #
# E4: REAL provenance requires the EXACT pinned revision
#
# E3 asked only whether a revision looked non-placeholder, so `deadbeef` — and
# any well-formed but wrong 40-character hash — established REAL provenance for
# a run produced against unknown weights. These tests pin the comparison to
# equality and prove it is not a length check, not a format check, not
# case-insensitive, and not whitespace-tolerant.
# --------------------------------------------------------------------------- #

def _with_revision(tmp_path: Path, revision, *, mock: bool = False, name="rev") -> Path:
    manifest = dict(
        MOCK_MANIFEST if mock else REAL_MANIFEST,
        model_contract={"revision": revision},
    )
    summary = MOCK_SUMMARY if mock else REAL_SUMMARY
    return _run(tmp_path, manifest, summary, name=name)


def _kind(tmp_path: Path, revision, *, mock: bool = False, name="rev") -> str:
    return assess_provenance(
        _with_revision(tmp_path, revision, mock=mock, name=name),
        expected_revision=PINNED_REVISION,
    ).kind


def test_the_pinned_revision_has_exactly_one_source_of_truth():
    """No second copy of the hash may live in analyze_etth2.py."""
    source = (REPO_ROOT / "experiments" / "analyze_etth2.py").read_text(encoding="utf-8")
    assert PINNED_REVISION not in source, (
        "the pinned revision is duplicated in analyze_etth2.py; it must be read "
        "from pilot_config['model']['revision'] at call time"
    )
    # ...and the config really is where it comes from.
    assert len(PINNED_REVISION) == 40
    assert PINNED_REVISION == "29ec3766d36d6f73f0696f85560a422f50e8498c"


def test_the_exact_pinned_revision_is_REAL(tmp_path):
    """Positive control. Without this every rejection below is vacuous."""
    assert _kind(tmp_path, PINNED_REVISION, name="exact") == PROVENANCE_REAL


def test_deadbeef_is_INVALID_not_REAL(tmp_path):
    """The reproduction case: E3 returned REAL for this."""
    verdict = assess_provenance(
        _with_revision(tmp_path, "deadbeef", name="deadbeef"),
        expected_revision=PINNED_REVISION,
    )
    assert verdict.kind == PROVENANCE_INVALID
    assert any("not the pinned revision" in r for r in verdict.reasons), verdict.reasons
    assert any("unknown weights" in r for r in verdict.reasons)


@pytest.mark.parametrize(
    "revision,why",
    [
        ("0000000000000000000000000000000000000000", "a different well-formed 40-char SHA"),
        ("ffffffffffffffffffffffffffffffffffffffff", "another well-formed 40-char SHA"),
        ("a1b2c3d4e5f60718293a4b5c6d7e8f9012345678", "a plausible-looking wrong SHA"),
    ],
)
def test_a_plausible_but_wrong_40_char_sha_is_INVALID(tmp_path, revision, why):
    """Proves this is not merely a length or format check.

    Each of these is exactly 40 lowercase hex characters — indistinguishable
    from the real thing by shape alone, and every one of them is a different
    checkpoint.
    """
    assert len(revision) == len(PINNED_REVISION) == 40
    assert revision != PINNED_REVISION
    assert _kind(tmp_path, revision, name=revision[:12]) == PROVENANCE_INVALID, why


@pytest.mark.parametrize(
    "mutate,label",
    [
        (lambda r: "a" + r, "one extra leading character"),
        (lambda r: r + "a", "one extra trailing character"),
        (lambda r: r[:-1] + ("d" if r[-1] != "d" else "e"), "last character changed"),
        (lambda r: ("d" if r[0] != "d" else "e") + r[1:], "first character changed"),
        (lambda r: r[:-1], "one character truncated"),
        (lambda r: r[:20], "truncated to a 20-char prefix"),
        (lambda r: r[20:], "only the 20-char suffix"),
    ],
)
def test_a_near_miss_is_INVALID(tmp_path, mutate, label):
    """Near misses are rejected, not just wildly different strings.

    A prefix match would accept the truncated cases; a substring match would
    accept the extended ones. Equality accepts none of them.
    """
    revision = mutate(PINNED_REVISION)
    assert revision != PINNED_REVISION
    assert _kind(tmp_path, revision, name=label[:18]) == PROVENANCE_INVALID, label


@pytest.mark.parametrize(
    "mutate,label",
    [
        (lambda r: r + " ", "trailing space"),
        (lambda r: " " + r, "leading space"),
        (lambda r: r + "\n", "trailing newline"),
        (lambda r: r + "\t", "trailing tab"),
        (lambda r: r.upper(), "uppercase"),
        (lambda r: r[:8].upper() + r[8:], "first eight characters uppercased"),
        (lambda r: r.replace("a", "A"), "some characters uppercased"),
    ],
)
def test_whitespace_and_case_variants_are_INVALID(tmp_path, mutate, label):
    """The comparison is exact: not stripped, not case-folded, not normalised.

    A run whose manifest recorded a differently-cased or space-padded hash did
    not record the pinned string, and this must not be papered over — the same
    zero-tolerance discipline the clean audit applies to its identity fields.
    """
    revision = mutate(PINNED_REVISION)
    assert revision != PINNED_REVISION
    assert revision.strip().lower() == PINNED_REVISION.lower() or label.startswith("some")
    assert _kind(tmp_path, revision, name=label[:18]) == PROVENANCE_INVALID, label


def test_a_mock_run_is_unaffected_by_the_revision_pin(tmp_path):
    """The pin governs REAL only; a declared mock still resolves to MOCK."""
    assert _kind(tmp_path, "mock-revision", mock=True, name="m1") == PROVENANCE_MOCK
    # ...but a mock claiming the pinned checkpoint is contradictory.
    assert _kind(tmp_path, PINNED_REVISION, mock=True, name="m2") == PROVENANCE_INVALID


def test_omitting_the_expected_revision_cannot_yield_REAL(tmp_path):
    """A caller that forgets to pass the pin must not get a free pass."""
    verdict = assess_provenance(_with_revision(tmp_path, PINNED_REVISION, name="nopin"))
    assert verdict.kind == PROVENANCE_INVALID
    assert any("no expected model revision was supplied" in r for r in verdict.reasons)


def test_the_caller_reads_the_pin_from_the_effective_config(tmp_path, config):
    """analyse_etth2 must source the pin from pilot_config, not a local copy."""
    import copy

    etth2 = yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8"))
    gap = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8"))
    run = _with_revision(tmp_path, PINNED_REVISION, name="cfg")

    # Move the pin in the config; the same run must stop being REAL.
    moved = copy.deepcopy(config)
    moved["model"]["revision"] = "0" * 40
    with pytest.raises(InvalidProvenance) as excinfo:
        analyse_etth2(run, etth2_config=etth2, pilot_config=moved, gap_config=gap,
                      out_dir=tmp_path / "out_moved")
    assert "not the pinned revision" in str(excinfo.value)
    assert not (tmp_path / "out_moved").exists()


def test_allow_mock_analysis_does_not_bypass_a_wrong_revision(tmp_path, config):
    """A wrong revision is INVALID_PROVENANCE, and that flag never reaches it."""
    etth2 = yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8"))
    gap = yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8"))
    run = _with_revision(tmp_path, "deadbeef", name="bypass")
    for allow in (False, True):
        with pytest.raises(InvalidProvenance) as excinfo:
            analyse_etth2(
                run, etth2_config=etth2, pilot_config=config, gap_config=gap,
                out_dir=tmp_path / f"o{int(allow)}", allow_mock_analysis=allow,
            )
        assert "does NOT bypass this" in str(excinfo.value)
        assert not (tmp_path / f"o{int(allow)}").exists()
