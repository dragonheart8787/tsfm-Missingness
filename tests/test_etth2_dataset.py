"""ETTh2 dataset acquisition, validation, and the NATIVE-MISSINGNESS HARD STOP.

The hard stop is the point of this file. It is not enough that the check exists
and logs; it must be capable of actually halting the pipeline. Every test that
asserts it does so by constructing a dataset that violates the property and
confirming the code refuses to return, rather than by inspecting source text.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from data.fetch_etth2 import (
    DATASET_NAME,
    NativeMissingnessError,
    OriginCountError,
    assert_no_native_missingness,
    derive_origin_count,
    load_series,
    validate_etth2,
)
from runner.windows import build_windows, enumerate_origins

REPO_ROOT = Path(__file__).resolve().parents[1]

# MEASURED on 2026-09-04 from the canonical ETDataset release, and recorded in
# the preregistration's factual amendment A1. Asserted here so a swapped file
# fails the suite rather than silently changing the study.
ETTH2_SHA256 = "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"
ETTH2_ROWS = 17420
ETTH2_ORIGINS = 178
ETTH2_FIRST = "2016-07-01 00:00:00"
ETTH2_LAST = "2018-06-26 19:00:00"


@pytest.fixture(scope="module")
def etth2_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture
def effective_config(etth2_config, config) -> dict:
    from experiments.run_etth2 import build_effective_config

    return build_effective_config(config, etth2_config)


def _write_csv(path: Path, *, n_rows: int, nan_at: list[int] | None = None) -> Path:
    """A structurally valid ETT-shaped file, optionally with native NaNs in OT."""
    stamps = pd.date_range("2016-07-01", periods=n_rows, freq="h")
    ot = np.linspace(0.0, 50.0, n_rows)
    for i in nan_at or []:
        ot[i] = np.nan
    frame = pd.DataFrame({"date": stamps.astype(str)})
    for column in ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"):
        frame[column] = 1.0
    frame["OT"] = ot
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# The hard stop
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("nan_positions", [[0], [17419], [8000], [5, 6, 7], [0, 17419]])
def test_native_missingness_hard_stops_wherever_it_appears(tmp_path, etth2_config, nan_positions):
    """A native NaN anywhere in OT must HALT, not warn — first row, last, middle."""
    ds = copy.deepcopy(etth2_config["dataset"])
    csv = _write_csv(tmp_path / "ETTh2.csv", n_rows=ETTH2_ROWS, nan_at=nan_positions)
    with pytest.raises(NativeMissingnessError) as excinfo:
        validate_etth2(csv, ds)
    message = str(excinfo.value)
    assert "HARD STOP" in message
    assert str(len(nan_positions)) in message
    # It must tell the operator NOT to fix it themselves.
    assert "Do NOT impute" in message
    assert "Research Lead" in message


def test_the_hard_stop_halts_the_loader_not_just_the_validator(tmp_path, effective_config):
    """load_series must refuse too — the runner's entry point is what matters."""
    cfg = copy.deepcopy(effective_config)
    csv = _write_csv(tmp_path / "raw" / "ETTh2.csv", n_rows=ETTH2_ROWS, nan_at=[100])
    cfg["dataset"]["local_path"] = str(csv.relative_to(REPO_ROOT)) if csv.is_relative_to(
        REPO_ROOT
    ) else str(csv)
    # Point the loader at the temp file by absolute path.
    import data.fetch_etth2 as mod

    original = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = csv.parent.parent
        cfg["dataset"]["local_path"] = "raw/ETTh2.csv"
        with pytest.raises(NativeMissingnessError):
            load_series(cfg)
    finally:
        mod.REPO_ROOT = original


def test_the_hard_stop_precedes_the_row_count_and_checksum_assertions(tmp_path, effective_config):
    """A file that is BOTH short AND missing must report the missingness.

    Order matters: reporting a downstream symptom would send the operator after
    the wrong problem, and the missingness is the one that must reach the
    Research Lead.
    """
    cfg = copy.deepcopy(effective_config)
    csv = _write_csv(tmp_path / "raw" / "ETTh2.csv", n_rows=5000, nan_at=[42])
    import data.fetch_etth2 as mod

    original = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = csv.parent.parent
        cfg["dataset"]["local_path"] = "raw/ETTh2.csv"
        with pytest.raises(NativeMissingnessError):
            load_series(cfg)
    finally:
        mod.REPO_ROOT = original


def test_a_clean_series_passes_the_hard_stop(tmp_path, etth2_config):
    """Non-vacuity: the check must not reject everything."""
    ds = copy.deepcopy(etth2_config["dataset"])
    csv = _write_csv(tmp_path / "ETTh2.csv", n_rows=ETTH2_ROWS)
    validation = validate_etth2(csv, ds)
    assert validation.native_missing_in_target == 0
    assert_no_native_missingness(validation)   # must not raise


def test_the_config_declares_the_hard_stop_and_the_measured_zero(etth2_config):
    ds = etth2_config["dataset"]
    assert ds["native_missing_is_a_hard_stop"] is True
    assert ds["native_missing_in_target"] == 0
    assert etth2_config["preconditions_for_execution"]["native_missingness_hard_stop"] is True


# --------------------------------------------------------------------------- #
# The DERIVED origin count
# --------------------------------------------------------------------------- #

def test_origin_count_is_derived_from_row_count_not_inherited(config):
    """178 must fall out of ETTh2's own length, not be copied from ETTh1."""
    design = config["design"]
    derived = derive_origin_count(
        n_rows=ETTH2_ROWS,
        context_length=int(design["context_length"]),
        horizon=int(design["horizon"]),
        stride=int(design["stride"]),
    )
    assert derived == ETTH2_ORIGINS
    # The derivation is the runner's own enumeration, not a parallel formula.
    assert derived == len(
        enumerate_origins(
            n_rows=ETTH2_ROWS, context_length=320, horizon=96, stride=96
        )
    )


@pytest.mark.parametrize(
    "n_rows,expected",
    [(17420, 178), (17408, 178), (17407, 177), (17312, 177), (17311, 176),
     (416, 1), (415, 0)],
)
def test_the_derivation_is_sensitive_to_row_count(n_rows, expected):
    """Non-vacuity: the count must actually depend on the length it is given.

    A derivation that returned 178 regardless would pass the test above while
    proving nothing. The pairs straddle the exact boundaries: 17,408 =
    320 + 96 + 177*96 is the shortest series in which the 178th origin still
    fits, and 17,312 = 320 + 96 + 176*96 the shortest in which the 177th does.
    One row shorter than either and the count drops.
    """
    assert derive_origin_count(
        n_rows=n_rows, context_length=320, horizon=96, stride=96
    ) == expected


def test_load_series_rejects_a_derived_count_that_contradicts_the_config(
    tmp_path, effective_config
):
    """A truncated ETTh2 must fail, not silently run a shorter matrix."""
    cfg = copy.deepcopy(effective_config)
    short = 17420 - 96
    csv = _write_csv(tmp_path / "raw" / "ETTh2.csv", n_rows=short)
    cfg["dataset"]["expected_rows"] = short          # let it past the row check
    cfg["dataset"]["expected_sha256"] = None
    import data.fetch_etth2 as mod

    original = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = csv.parent.parent
        cfg["dataset"]["local_path"] = "raw/ETTh2.csv"
        with pytest.raises(OriginCountError, match="DERIVED, never assumed"):
            load_series(cfg)
    finally:
        mod.REPO_ROOT = original


def test_load_series_rejects_a_config_for_the_wrong_dataset(effective_config):
    cfg = copy.deepcopy(effective_config)
    cfg["dataset"]["name"] = "ETTh1"
    with pytest.raises(ValueError, match=f"this loader is for {DATASET_NAME}"):
        load_series(cfg)


# --------------------------------------------------------------------------- #
# The real file, when it has been fetched
# --------------------------------------------------------------------------- #

@pytest.fixture
def real_etth2(effective_config):
    csv = REPO_ROOT / effective_config["dataset"]["local_path"]
    if not csv.exists():
        pytest.skip("ETTh2 not fetched; run data/fetch_etth2.py")
    return load_series(effective_config)


def test_real_etth2_matches_the_recorded_dataset_contract(real_etth2):
    """The measured facts recorded in preregistration amendment A1."""
    _values, _stamps, validation = real_etth2
    assert validation.sha256 == ETTH2_SHA256
    assert validation.n_rows == ETTH2_ROWS
    assert validation.target_column == "OT"
    assert validation.timestamp_first == ETTH2_FIRST
    assert validation.timestamp_last == ETTH2_LAST
    assert validation.timestamps_monotonic_increasing is True
    assert validation.timestamps_unique is True
    assert validation.n_gaps_in_hourly_grid == 0
    assert validation.freq_matches_expected is True


def test_real_etth2_has_zero_native_missing_values_in_the_target(real_etth2):
    """CONFIRMED, not assumed. The precondition the whole design rests on."""
    _values, _stamps, validation = real_etth2
    assert validation.native_missing_in_target == 0
    assert validation.native_missing_positions_in_target == []
    assert validation.native_missing_per_column["OT"] == 0


def test_real_etth2_yields_178_origins(real_etth2, effective_config):
    """The ETTh2 equivalent of test_10g_real_etth1_yields_178_origins.

    Against the ACTUAL fetched dataset, not its declared length. That the number
    equals ETTh1's is a consequence of the two files having the same row count,
    not an assumption: the sensitivity test above proves the derivation moves
    when the length does.
    """
    values, stamps, validation = real_etth2
    assert validation.n_rows == ETTH2_ROWS
    windows = build_windows(values=values, timestamps=stamps, config=effective_config)
    assert len(windows) == ETTH2_ORIGINS
    assert windows[0].context_start == 0
    assert windows[-1].target_end_exclusive <= ETTH2_ROWS
    # One more stride would not fit.
    assert windows[-1].context_start + 96 + 320 + 96 > ETTH2_ROWS


def test_real_etth2_is_not_the_same_file_as_etth1(real_etth2, config):
    """A replication on a copy of ETTh1 would be worthless. Prove they differ."""
    _values, _stamps, validation = real_etth2
    assert validation.sha256 != config["dataset"]["expected_sha256"]
    etth1 = REPO_ROOT / config["dataset"]["local_path"]
    if etth1.exists():
        from data.fetch_etth1 import load_series as load_etth1

        v1, _s1, _val1 = load_etth1(config)
        assert not np.array_equal(v1, _values)
