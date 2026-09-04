"""formal-full must FAIL THE PROCESS on any completion-invariant violation.

Before E3 a forecaster failure was recorded as a status value inside
window_results.csv while the invocation still exited 0 — invisible to any
wrapper script, and silently absorbed into a "resume later" path. These tests
drive the real CLI as a subprocess and assert on its actual exit code, because
an in-process exception proves nothing about what the process returns.

No cell is ever retried: the failure must stay visible.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.run_etth2 import (
    CompletionInvariantViolation,
    assert_completion_invariants,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINS = 178
CONDITIONS = 13
TOTAL = ORIGINS * CONDITIONS


def _complete_summary(**overrides) -> dict:
    summary = {
        "origins_fully_completed": ORIGINS,
        "result_rows": TOTAL,
        "distinct_cells": TOTAL,
        "rows_by_status": {"ok": TOTAL},
        "max_distinct_target_digests_per_origin": 1,
        "forecasts_failed_this_process": 0,
    }
    summary.update(overrides)
    return summary


def _assert(summary):
    assert_completion_invariants(
        summary, expected_origins=ORIGINS, conditions_per_origin=CONDITIONS
    )


# --------------------------------------------------------------------------- #
# The invariant check itself
# --------------------------------------------------------------------------- #

def test_a_complete_matrix_passes():
    """Non-vacuity: the check must not reject a good run."""
    _assert(_complete_summary())


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"origins_fully_completed": 177}, "origins_fully_completed is 177"),
        ({"result_rows": 2313}, "result_rows is 2313"),
        ({"distinct_cells": 2313}, "distinct_cells is 2313"),
        ({"rows_by_status": {"ok": 2313, "failed": 1}}, "rows_by_status is"),
        ({"max_distinct_target_digests_per_origin": 2},
         "max_distinct_target_digests_per_origin is 2"),
        ({"forecasts_failed_this_process": 1}, "1 forecast(s) failed"),
        ({"result_rows": None, "rows_by_status": None,
          "origins_fully_completed": 10}, "did not complete"),
    ],
)
def test_each_violated_invariant_raises(override, needle):
    with pytest.raises(CompletionInvariantViolation, match=r".*"):
        _assert(_complete_summary(**override))
    try:
        _assert(_complete_summary(**override))
    except CompletionInvariantViolation as exc:
        assert needle in str(exc), str(exc)


def test_the_message_says_no_cell_was_retried():
    try:
        _assert(_complete_summary(forecasts_failed_this_process=1))
    except CompletionInvariantViolation as exc:
        assert "No cell was retried" in str(exc)
        assert "absorbed into a resume path" in str(exc)


# --------------------------------------------------------------------------- #
# The real process, with a forecaster that fails one specific cell
# --------------------------------------------------------------------------- #

FAILING_DRIVER = textwrap.dedent('''
    """Drive run_etth2's CLI with a forecaster that fails one specific cell."""
    import sys
    sys.path.insert(0, {repo!r})
    from model.chronos2_runner import MockForecaster
    import experiments.run_etth2 as mod

    FAIL_ON_CALL = {fail_on}

    class FailsOneCell(MockForecaster):
        def __init__(self):
            super().__init__()
            self.n = 0
        def forecast_median(self, context, prediction_length):
            self.n += 1
            if self.n == FAIL_ON_CALL:
                raise RuntimeError("simulated forecaster failure on this cell")
            return super().forecast_median(context, prediction_length)

    mod._forecaster = lambda *a, **k: FailsOneCell()
    raise SystemExit(mod.main())
''')


def _run_cli(driver: Path, argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(driver), *argv],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200,
    )


@pytest.fixture
def prepared_run(tmp_path):
    """clean-reference + formal-clean + audit, ready for formal-full."""
    etth2 = REPO_ROOT / "configs" / "etth2_config.yaml"
    if not (REPO_ROOT / yaml.safe_load(etth2.read_text(encoding="utf-8"))
            ["dataset"]["local_path"]).exists():
        pytest.skip("ETTh2 not fetched; run data/fetch_etth2.py")
    reference, formal = tmp_path / "clean_reference", tmp_path / "formal"
    for phase in ("clean-reference", "formal-clean", "audit"):
        result = subprocess.run(
            [sys.executable, "experiments/run_etth2.py", "--phase", phase,
             "--reference-run-dir", str(reference), "--formal-run-dir", str(formal)]
            + (["--mock-model"] if phase != "audit" else []),
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200,
        )
        assert result.returncode == 0, (phase, result.stderr[-1500:])
    return reference, formal


def test_a_failed_cell_makes_the_PROCESS_exit_nonzero(tmp_path, prepared_run):
    """THE test: the CLI's own exit code, not a status inside a file."""
    reference, formal = prepared_run
    driver = tmp_path / "driver.py"
    driver.write_text(
        FAILING_DRIVER.format(repo=str(REPO_ROOT), fail_on=400), encoding="utf-8"
    )
    result = _run_cli(driver, [
        "--phase", "formal-full",
        "--reference-run-dir", str(reference), "--formal-run-dir", str(formal),
        "--mock-model",
    ])

    assert result.returncode != 0, (
        "the process exited 0 despite a failed cell — the failure is invisible "
        "to anything reading only the exit code"
    )
    assert result.returncode == 6, (result.returncode, result.stderr[-1500:])
    assert "completion invariant" in result.stderr
    assert "No cell was retried" in result.stderr

    # The failure is recorded too — the exit code is in ADDITION to the record,
    # not instead of it.
    frame = pd.read_csv(formal / "window_results.csv")
    assert (frame["status"] == "failed").sum() == 1
    summary = json.loads((formal / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["forecasts_failed_this_process"] == 1
    assert summary["rows_by_status"]["failed"] == 1


def test_the_failed_cell_was_not_silently_retried(tmp_path, prepared_run):
    """A retry would hide both the failure and its cause."""
    reference, formal = prepared_run
    driver = tmp_path / "driver.py"
    driver.write_text(
        FAILING_DRIVER.format(repo=str(REPO_ROOT), fail_on=400), encoding="utf-8"
    )
    _run_cli(driver, [
        "--phase", "formal-full",
        "--reference-run-dir", str(reference), "--formal-run-dir", str(formal),
        "--mock-model",
    ])
    frame = pd.read_csv(formal / "window_results.csv")
    failed = frame[frame["status"] == "failed"]
    assert len(failed) == 1
    key = (failed.iloc[0]["origin_id"], failed.iloc[0]["condition_id"])
    matching = frame[
        (frame["origin_id"] == key[0]) & (frame["condition_id"] == key[1])
    ]
    assert len(matching) == 1, "the failed cell was written more than once"
    assert matching.iloc[0]["status"] == "failed"


def test_a_clean_full_run_exits_zero(tmp_path, prepared_run):
    """Non-vacuity: the invariant check must not fail a good run."""
    reference, formal = prepared_run
    result = subprocess.run(
        [sys.executable, "experiments/run_etth2.py", "--phase", "formal-full",
         "--reference-run-dir", str(reference), "--formal-run-dir", str(formal),
         "--mock-model"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    summary = json.loads((formal / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_by_status"] == {"ok": TOTAL}
    assert summary["distinct_cells"] == TOTAL


def test_a_deliberately_limited_pass_is_not_a_violation(tmp_path, prepared_run):
    """--limit-origins is a partial run by instruction, not a broken matrix."""
    reference, formal = prepared_run
    result = subprocess.run(
        [sys.executable, "experiments/run_etth2.py", "--phase", "formal-full",
         "--reference-run-dir", str(reference), "--formal-run-dir", str(formal),
         "--mock-model", "--limit-origins", "5"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200,
    )
    assert result.returncode == 0, result.stderr[-1500:]
