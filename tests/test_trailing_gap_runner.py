"""Runner and analysis workflow for the trailing-gap experiment (item 5/6).

Model-mocked. Mirrors tests/test_resume.py's conventions for the main pilot.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.run_trailing_gap import (
    completed_cells,
    completed_origins,
    run_trailing_gap,
)
from model.chronos2_runner import MockForecaster
from runner.run_pilot import RunPaths

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gap_config():
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture
def local_config(config, tmp_path, synthetic_series, monkeypatch):
    """Point the loader at a synthetic ETTh1 of the exact canonical length."""
    from pathlib import Path

    import data.fetch_etth1 as fetch
    from data.fetch_etth1 import sha256_file

    values, stamps = synthetic_series
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame({"date": stamps, "OT": values}).to_csv(csv, index=False)
    cfg = json.loads(json.dumps(config))
    cfg["dataset"]["local_path"] = str(csv)
    cfg["dataset"]["expected_sha256"] = sha256_file(csv)
    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))
    return cfg


def _run(cfg, gap_config, run_dir, limit, **kwargs):
    return run_trailing_gap(
        config=cfg, gap_config=gap_config, forecaster=MockForecaster(),
        run_dir=run_dir, limit_origins=limit, **kwargs,
    )


def test_thirteen_conditions_per_origin_are_written(local_config, gap_config, tmp_path):
    _run(local_config, gap_config, tmp_path / "r", 3)
    frame = pd.read_csv(tmp_path / "r" / "window_results.csv")
    assert len(frame) == 3 * 13
    assert sorted(frame.groupby("origin_id").size().unique()) == [13]
    kinds = frame[frame.origin_id == 0]["kind"].value_counts().to_dict()
    assert kinds == {"trailing_nan": 4, "truncated_long": 4, "internal_block": 4, "clean": 1}


def test_truncated_long_rows_record_the_realignment(local_config, gap_config, tmp_path):
    _run(local_config, gap_config, tmp_path / "r2", 1)
    frame = pd.read_csv(tmp_path / "r2" / "window_results.csv")
    tl = frame[frame.kind == "truncated_long"].sort_values("gap")
    assert tl["context_length_used"].tolist() == [304, 288, 256, 192]
    assert tl["prediction_length_requested"].tolist() == [112, 128, 160, 224]
    assert tl["dropped_leading_steps"].tolist() == [16, 32, 64, 128]
    # Every arm scores from the ORIGINAL target's first index.
    assert tl["scored_first_index"].unique().tolist() == [320]


def test_scored_first_timestamp_is_identical_across_all_conditions(
    local_config, gap_config, tmp_path
):
    _run(local_config, gap_config, tmp_path / "r3", 2)
    frame = pd.read_csv(tmp_path / "r3" / "window_results.csv")
    assert frame.groupby("origin_id")["scored_first_timestamp"].nunique().max() == 1


def test_target_digest_is_identical_across_all_conditions(local_config, gap_config, tmp_path):
    _run(local_config, gap_config, tmp_path / "r4", 4)
    frame = pd.read_csv(tmp_path / "r4" / "window_results.csv")
    assert frame.groupby("origin_id")["target_sha256"].nunique().max() == 1


def test_resume_does_not_duplicate_or_overwrite(local_config, gap_config, tmp_path):
    run_dir = tmp_path / "resume"
    first = _run(local_config, gap_config, run_dir, 3)
    assert first["origins_fully_completed"] == 3
    before = pd.read_csv(run_dir / "window_results.csv")
    assert len(before) == 3 * 13
    assert completed_origins(RunPaths(run_dir), conditions_per_origin=13) == {0, 1, 2}
    assert len(completed_cells(RunPaths(run_dir))) == 3 * 13

    second = _run(local_config, gap_config, run_dir, 6)
    after = pd.read_csv(run_dir / "window_results.csv")
    assert second["origins_fully_completed"] == 6
    assert len(after) == 6 * 13
    assert not after.duplicated(subset=["origin_id", "condition_id"]).any()
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True),
        after[after.origin_id < 3].reset_index(drop=True),
        check_exact=False,
    )
    third = _run(local_config, gap_config, run_dir, 6)
    assert third["forecasts_ok_this_process"] == 0


def test_a_lying_checkpoint_cannot_cause_a_skip(local_config, gap_config, tmp_path):
    """The checkpoint is advisory; the result rows are the only resume truth."""
    run_dir = tmp_path / "torn"
    _run(local_config, gap_config, run_dir, 3)
    paths = RunPaths(run_dir)
    state = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    state["completed_origins"] = [0, 1, 2, 3, 4]   # claims two never written
    state["n_completed_cells"] = 9999
    paths.checkpoint.write_text(json.dumps(state), encoding="utf-8")

    assert completed_origins(paths, conditions_per_origin=13) == {0, 1, 2}
    assert len(completed_cells(paths)) == 3 * 13

    # A resume must still execute origins 3 and 4 in full.
    _run(local_config, gap_config, run_dir, 5)
    frame = pd.read_csv(run_dir / "window_results.csv")
    assert len(frame) == 5 * 13
    assert not frame.duplicated(subset=["origin_id", "condition_id"]).any()


def test_failures_are_recorded_not_swallowed(local_config, gap_config, tmp_path):
    class Flaky(MockForecaster):
        def forecast_median(self, context, horizon):
            if len(self.calls) == 5:
                self.calls.append(context)
                raise RuntimeError("simulated CUDA OOM")
            return super().forecast_median(context, horizon)

    run_trailing_gap(
        config=local_config, gap_config=gap_config, forecaster=Flaky(),
        run_dir=tmp_path / "flaky", limit_origins=1,
    )
    frame = pd.read_csv(tmp_path / "flaky" / "window_results.csv")
    assert len(frame) == 13, "a failed forecast still gets its row"
    failed = frame[frame.status == "failed"]
    assert len(failed) == 1
    assert "simulated CUDA OOM" in failed.iloc[0]["error_message"]
    assert pd.isna(failed.iloc[0]["mae"])


def test_only_conditions_supports_the_clean_first_audit(local_config, gap_config, tmp_path):
    """Runbook step 3 runs the 178 clean forecasts and nothing else."""
    summary = _run(local_config, gap_config, tmp_path / "cleanonly", 5,
                   only_conditions=["clean"])
    frame = pd.read_csv(tmp_path / "cleanonly" / "window_results.csv")
    assert len(frame) == 5, "one clean forecast per origin, nothing else"
    assert frame["kind"].unique().tolist() == ["clean"]
    assert summary["only_conditions"] == ["clean"]


def test_runner_hard_stops_on_a_contract_violation(local_config, gap_config, tmp_path):
    """An unrollable gap can never reach the model."""
    from experiments.trailing_gap import TrailingGapContractViolation
    from model.chronos2_runner import ModelContract

    small = ModelContract(
        hf_model_id="mock", revision="x", input_patch_size=16, input_patch_stride=16,
        output_patch_size=16, model_context_length=8192, max_output_patches=8,
        model_prediction_length=8 * 16, quantiles=[0.5], use_arcsinh=False,
        device="cpu", dtype="float32",
    )
    forecaster = MockForecaster(contract=small)
    with pytest.raises(TrailingGapContractViolation, match="AUTOREGRESSIVE UNROLLING"):
        run_trailing_gap(
            config=local_config, gap_config=gap_config, forecaster=forecaster,
            run_dir=tmp_path / "stop", limit_origins=1,
        )


def test_analysis_produces_readings_and_a_classification(local_config, gap_config, tmp_path):
    """The analysis entry point runs end to end on a small matrix."""
    from experiments.analyze_trailing_gap import analyse

    run_dir = tmp_path / "analyse"
    _run(local_config, gap_config, run_dir, 40)
    fast = json.loads(json.dumps(local_config))
    fast["statistics"]["bootstrap"]["n_replicates"] = 100
    payload = analyse(run_dir, fast, gap_config, out_dir=tmp_path / "out")

    assert {r["contrast_id"] for r in payload["r_contrasts"]} == {
        "R_16", "R_32", "R_64", "R_128"
    }
    assert {r["contrast_id"] for r in payload["b_contrasts"]} == {
        "B_16", "B_32", "B_64", "B_128"
    }
    for row in payload["r_contrasts"]:
        assert row["reading"] in {
            "EQUIVALENT", "MATERIAL_POSITIVE", "MATERIAL_NEGATIVE", "UNRESOLVED"
        }
        assert row["interpretation"]
        assert row["sesoi_delta"] == pytest.approx(0.03 * payload["mean_clean_mae"])
    assert payload["classification"]["rule_version"] == "preregistered-v1"
    assert payload["classification"]["authoritative"] is True
    assert payload["dose_response"]["x_units"].startswith("missing_patches")
    for name in ("R_contrasts.csv", "B_contrasts.csv", "classification.json",
                 "dose_response.json", "per_origin_slopes.csv"):
        assert (tmp_path / "out" / name).exists()


# --------------------------------------------------------------------------- #
# Runbook step 4 — the clean-prediction checksum audit
# --------------------------------------------------------------------------- #

def _write_clean_predictions(run_dir, values, *, n_origins=3, horizon=4):
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for origin in range(n_origins):
        for step in range(horizon):
            rows.append({
                "run_id": "x", "origin_id": origin, "condition_id": "clean",
                "step_index": step, "forecast_timestamp": f"t{step}",
                "ground_truth": 1.0, "median_prediction": values[k],
            })
            k += 1
    pd.DataFrame(rows).to_csv(run_dir / "predictions_long.csv", index=False)


def test_clean_audit_hash_matches_for_identical_predictions(tmp_path):
    from scripts.audit_clean_predictions import canonical_clean_hash

    values = np.arange(12, dtype=np.float32) * 1.5
    _write_clean_predictions(tmp_path / "a", values)
    _write_clean_predictions(tmp_path / "b", values)
    assert canonical_clean_hash(tmp_path / "a")[0] == canonical_clean_hash(tmp_path / "b")[0]


def test_clean_audit_hash_differs_on_a_single_perturbed_value(tmp_path):
    """A float32-visible change anywhere must change the hash."""
    from scripts.audit_clean_predictions import canonical_clean_hash

    values = np.arange(12, dtype=np.float32) * 1.5
    perturbed = values.copy()
    perturbed[7] = np.float32(perturbed[7] + 1e-3)
    _write_clean_predictions(tmp_path / "a", values)
    _write_clean_predictions(tmp_path / "b", perturbed)
    assert canonical_clean_hash(tmp_path / "a")[0] != canonical_clean_hash(tmp_path / "b")[0]


def test_clean_audit_hash_is_row_order_independent(tmp_path):
    """Canonical sort: a shuffled file must hash the same."""
    from scripts.audit_clean_predictions import canonical_clean_hash

    values = np.arange(12, dtype=np.float32) * 1.5
    _write_clean_predictions(tmp_path / "a", values)
    frame = pd.read_csv(tmp_path / "a" / "predictions_long.csv")
    (tmp_path / "b").mkdir()
    frame.sample(frac=1.0, random_state=0).to_csv(
        tmp_path / "b" / "predictions_long.csv", index=False
    )
    assert canonical_clean_hash(tmp_path / "a")[0] == canonical_clean_hash(tmp_path / "b")[0]


def test_clean_audit_ignores_non_clean_conditions(tmp_path):
    from scripts.audit_clean_predictions import canonical_clean_hash

    values = np.arange(12, dtype=np.float32) * 1.5
    _write_clean_predictions(tmp_path / "a", values)
    frame = pd.read_csv(tmp_path / "a" / "predictions_long.csv")
    noise = frame.copy()
    noise["condition_id"] = "trailing_nan_g16"
    noise["median_prediction"] = 999.0
    (tmp_path / "b").mkdir()
    pd.concat([frame, noise]).to_csv(tmp_path / "b" / "predictions_long.csv", index=False)
    assert canonical_clean_hash(tmp_path / "a")[0] == canonical_clean_hash(tmp_path / "b")[0]


def test_clean_audit_offers_no_tolerance_flag():
    """Selecting a tolerance after seeing a discrepancy is what this prevents."""
    source = (REPO_ROOT / "scripts" / "audit_clean_predictions.py").read_text(
        encoding="utf-8"
    )
    assert "--tolerance" not in source
    assert "atol" not in source and "rtol" not in source
    assert "np.isclose" not in source and "allclose" not in source


# --------------------------------------------------------------------------- #
# BLOCKER 1 — clean-only -> full-matrix resume at cell granularity
# --------------------------------------------------------------------------- #

def test_clean_only_then_full_matrix_resume_executes_the_remaining_cells(
    local_config, gap_config, tmp_path
):
    """The exact scenario from the blocker report, at 5 origins.

    Before the fix: the clean-only pass marked all 5 origins complete, the full
    run skipped every one, and only 5 of the 65 cells ever existed.
    """
    run_dir = tmp_path / "b1"

    # 1. clean-only -> 5 rows, one per origin.
    first = run_trailing_gap(
        config=local_config, gap_config=gap_config, forecaster=MockForecaster(),
        run_dir=run_dir, limit_origins=5, only_conditions=["clean"],
    )
    after_clean = pd.read_csv(run_dir / "window_results.csv")
    assert len(after_clean) == 5
    assert after_clean["kind"].unique().tolist() == ["clean"]
    assert first["forecasts_ok_this_process"] == 5

    # 2. full matrix -> skips exactly the 5 clean cells, runs the other 60.
    full_forecaster = MockForecaster()
    second = run_trailing_gap(
        config=local_config, gap_config=gap_config, forecaster=full_forecaster,
        run_dir=run_dir, limit_origins=5,
    )
    assert second["forecasts_ok_this_process"] == 60, "must run 12 conditions x 5 origins"
    assert len(full_forecaster.calls) == 60, "exactly 60 model calls, no re-runs"

    # 3. 65 rows total, and the original clean rows byte-unchanged.
    after_full = pd.read_csv(run_dir / "window_results.csv")
    assert len(after_full) == 65
    preserved = after_full[after_full["kind"] == "clean"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        after_clean.reset_index(drop=True), preserved[after_clean.columns],
        check_exact=True,
    )

    # 4. a third full invocation -> zero model calls, zero duplicates.
    third_forecaster = MockForecaster()
    third = run_trailing_gap(
        config=local_config, gap_config=gap_config, forecaster=third_forecaster,
        run_dir=run_dir, limit_origins=5,
    )
    assert len(third_forecaster.calls) == 0, "a completed matrix must issue no calls"
    assert third["forecasts_ok_this_process"] == 0
    final = pd.read_csv(run_dir / "window_results.csv")
    assert len(final) == 65
    assert not final.duplicated(subset=["origin_id", "condition_id"]).any()
    assert len(completed_cells(RunPaths(run_dir))) == 65


def test_resume_after_clean_only_pairs_against_the_earlier_clean_row(
    local_config, gap_config, tmp_path
):
    """The clean row from an earlier pass must still supply clean_mae.

    Otherwise every paired difference written by the resuming pass is NaN.
    """
    run_dir = tmp_path / "b1pair"
    run_trailing_gap(
        config=local_config, gap_config=gap_config, forecaster=MockForecaster(),
        run_dir=run_dir, limit_origins=5, only_conditions=["clean"],
    )
    run_trailing_gap(
        config=local_config, gap_config=gap_config, forecaster=MockForecaster(),
        run_dir=run_dir, limit_origins=5,
    )
    frame = pd.read_csv(run_dir / "window_results.csv")
    assert frame["clean_mae"].notna().all(), "clean pairing lost across the resume"
    assert frame["paired_diff_mae_from_clean"].notna().all()
    # Every row of an origin pairs against that origin's single clean MAE.
    assert frame.groupby("origin_id")["clean_mae"].nunique().max() == 1


def test_partial_condition_subsets_resume_correctly(local_config, gap_config, tmp_path):
    """Generalisation: any subset, in any order, converges on the full matrix."""
    run_dir = tmp_path / "b1subset"
    for subset in (["clean"], ["internal_block"], ["trailing_nan"], None):
        run_trailing_gap(
            config=local_config, gap_config=gap_config, forecaster=MockForecaster(),
            run_dir=run_dir, limit_origins=3,
            only_conditions=subset,
        )
    frame = pd.read_csv(run_dir / "window_results.csv")
    assert len(frame) == 3 * 13
    assert not frame.duplicated(subset=["origin_id", "condition_id"]).any()
    assert sorted(frame["kind"].unique()) == [
        "clean", "internal_block", "trailing_nan", "truncated_long"
    ]


def test_full_scale_cell_expectations_are_declared(config, gap_config):
    """The 5-origin test is for iteration speed; full scale is asserted too."""
    origins = int(config["design"]["expected_origins"])
    per_origin = int(gap_config["design"]["expected_conditions_per_origin"])
    assert origins == 178 and per_origin == 13
    assert origins * per_origin == 2314 == int(
        gap_config["design"]["expected_total_forecasts"]
    )
    # The runner asserts DISTINCT cells at full scale, so a duplicated cell
    # cannot mask a missing one.
    source = (REPO_ROOT / "experiments" / "run_trailing_gap.py").read_text(
        encoding="utf-8"
    )
    assert "expected_distinct_cells" in source
    assert 'drop_duplicates(subset=["origin_id", "condition_id"])' in source
