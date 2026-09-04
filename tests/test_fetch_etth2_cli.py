"""fetch_etth2.py's EXIT CODE must be truthful — tested through the real CLI.

A function that raises correctly is not proof the CLI exits non-zero: the wrapper
could catch, print a note, and return 0, and a runbook checking `$?` would then
conclude the dataset was fine. So every case here launches the script as a
SUBPROCESS and asserts on its real exit status.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GOOD_SHA = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"


def _write_series(
    path: Path, *, n_rows: int = 17420, nan_at: list[int] | None = None,
    freq: str = "h", duplicate_a_timestamp: bool = False,
    descending: bool = False,
) -> Path:
    stamps = pd.date_range("2016-07-01", periods=n_rows, freq=freq)
    if duplicate_a_timestamp and n_rows > 1:
        stamps = stamps.delete(1).insert(1, stamps[0])
    if descending:
        stamps = stamps[::-1]
    ot = np.linspace(0.0, 50.0, n_rows)
    for i in nan_at or []:
        ot[i] = np.nan
    frame = pd.DataFrame({"date": [str(s) for s in stamps]})
    for column in ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"):
        frame[column] = 1.0
    frame["OT"] = ot
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _config_pointing_at(tmp_path: Path, csv: Path, **overrides) -> Path:
    """An ETTh2 config whose paths are inside tmp_path, so nothing real moves."""
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    # Relative to REPO_ROOT, which is how the script resolves them.
    config["dataset"]["local_path"] = str(csv.relative_to(REPO_ROOT))
    config["dataset"]["metadata_path"] = str(
        (csv.parent / "validation.json").relative_to(REPO_ROOT)
    )
    # A file:// URL so `download` never touches the network; it also short-
    # circuits because the file already exists.
    config["dataset"]["source_url"] = csv.as_uri()
    config["dataset"].update(overrides)
    path = csv.parent / "etth2_test_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def sandbox(tmp_path_factory) -> Path:
    """A scratch directory INSIDE the repo, so relative config paths resolve."""
    root = REPO_ROOT / ".pytest_etth2_cli"
    root.mkdir(exist_ok=True)
    yield root
    import shutil

    shutil.rmtree(root, ignore_errors=True)


def run_cli(config_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "data/fetch_etth2.py", "--config",
         str(config_path.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


# --------------------------------------------------------------------------- #
# The passing case must exit 0 — otherwise every test below is vacuous
# --------------------------------------------------------------------------- #

def test_the_real_dataset_exits_zero():
    result = run_cli(REPO_ROOT / "configs" / "etth2_config.yaml")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ZERO native missing values" in result.stdout
    assert "178 derived origins" in result.stdout


def test_a_synthetic_clean_series_exits_zero(sandbox):
    """Non-vacuity for the sandbox harness itself."""
    csv = _write_series(sandbox / "clean" / "ETTh2.csv")
    config = _config_pointing_at(
        sandbox / "clean" / "ETTh2.csv", csv, expected_sha256=None,
    )
    result = run_cli(config)
    assert result.returncode == 0, result.stderr[-2000:]


# --------------------------------------------------------------------------- #
# Every failure mode, through the real CLI
# --------------------------------------------------------------------------- #

def test_sha_mismatch_exits_nonzero(sandbox):
    csv = _write_series(sandbox / "sha" / "ETTh2.csv")
    config = _config_pointing_at(sandbox / "sha" / "ETTh2.csv", csv,
                                 expected_sha256=GOOD_SHA)
    result = run_cli(config)
    assert result.returncode != 0, result.stdout[-2000:]
    assert "sha256 mismatch" in result.stderr


def test_row_count_mismatch_exits_nonzero(sandbox):
    csv = _write_series(sandbox / "rows" / "ETTh2.csv", n_rows=17000)
    config = _config_pointing_at(sandbox / "rows" / "ETTh2.csv", csv,
                                 expected_sha256=None)
    result = run_cli(config)
    assert result.returncode != 0
    assert "row count 17000" in result.stderr


def test_grid_gap_exits_nonzero(sandbox):
    """A two-hourly grid is not the hourly grid the design assumes."""
    csv = _write_series(sandbox / "grid" / "ETTh2.csv", freq="2h")
    config = _config_pointing_at(sandbox / "grid" / "ETTh2.csv", csv,
                                 expected_sha256=None)
    result = run_cli(config)
    assert result.returncode != 0
    assert "gap(s) in the hourly grid" in result.stderr or (
        "frequency" in result.stderr
    )


def test_timestamp_range_mismatch_exits_nonzero(sandbox):
    csv = _write_series(sandbox / "range" / "ETTh2.csv")
    config = _config_pointing_at(
        sandbox / "range" / "ETTh2.csv", csv, expected_sha256=None,
        timestamp_last="1999-12-31 23:00:00",
    )
    result = run_cli(config)
    assert result.returncode != 0
    assert "timestamp_last" in result.stderr


def test_native_missingness_exits_nonzero_with_its_own_code(sandbox):
    csv = _write_series(sandbox / "nan" / "ETTh2.csv", nan_at=[1234])
    config = _config_pointing_at(sandbox / "nan" / "ETTh2.csv", csv,
                                 expected_sha256=None)
    result = run_cli(config)
    assert result.returncode == 2, (result.returncode, result.stderr[-2000:])
    assert "HARD STOP" in result.stderr
    assert "Do NOT impute" in result.stderr
    assert "Research Lead" in result.stderr


def test_derived_origin_count_mismatch_exits_nonzero(sandbox):
    """A length that yields 177 origins against a config recording 178."""
    csv = _write_series(sandbox / "origins" / "ETTh2.csv", n_rows=17407)
    config = _config_pointing_at(sandbox / "origins" / "ETTh2.csv", csv,
                                 expected_sha256=None, expected_rows=17407)
    result = run_cli(config)
    assert result.returncode != 0
    assert "derived origin count 177" in result.stderr


def test_non_monotonic_timestamps_exit_nonzero(sandbox):
    csv = _write_series(sandbox / "mono" / "ETTh2.csv", n_rows=500, descending=True)
    config = _config_pointing_at(sandbox / "mono" / "ETTh2.csv", csv,
                                 expected_sha256=None, expected_rows=500)
    result = run_cli(config)
    assert result.returncode != 0
    assert "monotonic" in result.stderr.lower()


def test_duplicate_timestamps_exit_nonzero(sandbox):
    csv = _write_series(sandbox / "dup" / "ETTh2.csv", n_rows=500,
                        duplicate_a_timestamp=True)
    config = _config_pointing_at(sandbox / "dup" / "ETTh2.csv", csv,
                                 expected_sha256=None, expected_rows=500)
    result = run_cli(config)
    assert result.returncode != 0


def test_an_absent_file_exits_nonzero(sandbox):
    """No file, no network: the CLI must fail rather than report success."""
    csv = sandbox / "absent" / "ETTh2.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    _write_series(csv)
    config = _config_pointing_at(csv, csv, expected_sha256=None)
    csv.unlink()
    result = run_cli(config)
    assert result.returncode != 0


# --------------------------------------------------------------------------- #
# The contract checker itself, as a unit
# --------------------------------------------------------------------------- #

def test_every_violation_is_reported_not_just_the_first(sandbox):
    """One run, every problem — so an operator fixes them together."""
    from data.fetch_etth2 import check_dataset_contract, validate

    csv = _write_series(sandbox / "many" / "ETTh2.csv", n_rows=17000)
    validation = validate(csv, source_url="x", target_column="OT", expected_freq="1h")
    problems = check_dataset_contract(
        validation,
        {"expected_rows": 17420, "expected_sha256": GOOD_SHA,
         "timestamp_last": "1999-01-01 00:00:00", "target_column": "OT"},
        {"context_length": 320, "horizon": 96, "stride": 96, "expected_origins": 178},
    )
    kinds = " | ".join(problems)
    assert "sha256 mismatch" in kinds
    assert "row count" in kinds
    assert "timestamp_last" in kinds
    assert "derived origin count" in kinds
    assert len(problems) >= 4
