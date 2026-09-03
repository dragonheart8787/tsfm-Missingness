"""Test 13 — a resumed run neither duplicates nor overwrites completed work.

Model-mocked: no weights required. This exercises the SAME entry point
(``run_pilot``) that the operator's resume command uses.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from model.chronos2_runner import MockForecaster
from runner.run_pilot import RunPaths, completed_origins, run_pilot


@pytest.fixture
def small_config(config, tmp_path, synthetic_series):
    """The real design, pointed at a written-out synthetic ETTh1 of exact length."""
    import numpy as np

    values, stamps = synthetic_series
    csv = tmp_path / "ETTh1.csv"
    pd.DataFrame({"date": stamps, "OT": values}).to_csv(csv, index=False)

    from data.fetch_etth1 import sha256_file

    cfg = json.loads(json.dumps(config))  # deep copy
    cfg["dataset"]["local_path"] = str(csv)
    cfg["dataset"]["expected_sha256"] = sha256_file(csv)
    return cfg


def _run(cfg, run_dir, limit):
    return run_pilot(
        config=cfg, forecaster=MockForecaster(), run_dir=run_dir, limit_origins=limit
    )


def test_13_resume_does_not_duplicate_or_overwrite(small_config, tmp_path, monkeypatch):
    import data.fetch_etth1 as fetch
    from pathlib import Path

    # load_series resolves paths relative to the repo root; the temp CSV is absolute.
    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))

    run_dir = tmp_path / "run"
    first = _run(small_config, run_dir, 4)
    assert first["origins_completed"] == 4

    paths = RunPaths(run_dir)
    frame_after_first = pd.read_csv(paths.window_results)
    assert len(frame_after_first) == 4 * 17
    assert completed_origins(paths) == {0, 1, 2, 3}
    checkpoint = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    assert checkpoint["completed_origins"] == [0, 1, 2, 3]

    # Interrupt and resume: extend to 7 origins with a FRESH forecaster.
    second = _run(small_config, run_dir, 7)
    assert second["origins_completed"] == 7

    frame_after_second = pd.read_csv(paths.window_results)
    # No duplication: exactly 7 x 17 rows, one per origin x condition.
    assert len(frame_after_second) == 7 * 17
    assert not frame_after_second.duplicated(subset=["origin_id", "condition_id"]).any()

    # No overwrite: the already-completed rows are byte-identical.
    reloaded = frame_after_second[frame_after_second["origin_id"] < 4].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        frame_after_first.reset_index(drop=True), reloaded, check_exact=False
    )

    # Re-running with nothing new to do is a genuine no-op.
    third = _run(small_config, run_dir, 7)
    assert third["origins_completed"] == 7
    assert third["forecasts_ok_this_process"] == 0
    assert len(pd.read_csv(paths.window_results)) == 7 * 17


def test_13b_predictions_file_also_resumes_without_duplication(small_config, tmp_path, monkeypatch):
    import data.fetch_etth1 as fetch
    from pathlib import Path

    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))
    run_dir = tmp_path / "run2"
    _run(small_config, run_dir, 2)
    _run(small_config, run_dir, 4)

    paths = RunPaths(run_dir)
    predictions = pd.read_csv(paths.predictions)
    assert len(predictions) == 4 * 17 * 96
    assert not predictions.duplicated(subset=["origin_id", "condition_id", "step_index"]).any()

    diagnostics = pd.read_csv(paths.diagnostics)
    assert len(diagnostics) == 4 * 17
    assert not diagnostics.duplicated(subset=["origin_id", "condition_id"]).any()


def test_13c_a_torn_checkpoint_does_not_strand_an_origin(small_config, tmp_path, monkeypatch):
    """A checkpoint claiming an origin whose rows never landed must redo it."""
    import data.fetch_etth1 as fetch
    from pathlib import Path

    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))
    run_dir = tmp_path / "run3"
    _run(small_config, run_dir, 3)

    paths = RunPaths(run_dir)
    state = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    state["completed_origins"] = [0, 1, 2, 3, 4]   # claims two origins never written
    paths.checkpoint.write_text(json.dumps(state), encoding="utf-8")

    assert completed_origins(paths) == {0, 1, 2}, (
        "resume trusted the checkpoint over the results file and would have skipped "
        "origins whose rows were never written"
    )

    _run(small_config, run_dir, 5)
    frame = pd.read_csv(paths.window_results)
    assert len(frame) == 5 * 17
    assert not frame.duplicated(subset=["origin_id", "condition_id"]).any()


def test_13d_failures_are_recorded_not_swallowed(small_config, tmp_path, monkeypatch):
    """A failing forecast lands in the results file with a status and a message."""
    import data.fetch_etth1 as fetch
    from pathlib import Path

    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))

    class FlakyForecaster(MockForecaster):
        def forecast_median(self, context, horizon):
            if len(self.calls) == 3:
                self.calls.append(context)
                raise RuntimeError("simulated CUDA OOM")
            return super().forecast_median(context, horizon)

    run_dir = tmp_path / "run4"
    run_pilot(config=small_config, forecaster=FlakyForecaster(), run_dir=run_dir, limit_origins=1)

    frame = pd.read_csv(run_dir / "window_results.csv")
    assert len(frame) == 17, "a failed forecast must still produce its result row"
    failed = frame[frame["status"] == "failed"]
    assert len(failed) == 1
    assert "simulated CUDA OOM" in failed.iloc[0]["error_message"]
    assert pd.isna(failed.iloc[0]["mae"])
