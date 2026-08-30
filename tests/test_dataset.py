"""Dataset acquisition and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.fetch_etth1 import sha256_file, validate


def test_real_etth1_passes_every_validation(real_series, config):
    _, _, validation = real_series
    assert validation.n_rows == config["dataset"]["expected_rows"] == 17420
    assert validation.timestamps_monotonic_increasing
    assert validation.timestamps_unique
    assert validation.freq_matches_expected
    assert validation.n_gaps_in_hourly_grid == 0
    assert validation.target_column == "OT"
    assert validation.sha256 == config["dataset"]["expected_sha256"]


def test_no_native_missing_values_before_synthetic_missingness(real_series):
    """Recorded BEFORE any synthetic missingness is inserted anywhere."""
    _, _, validation = real_series
    assert validation.native_missing_in_target == 0
    assert validation.native_missing_positions_in_target == []
    assert all(v == 0 for v in validation.native_missing_per_column.values())


def test_loaded_series_is_finite_and_correctly_typed(real_series):
    values, stamps, validation = real_series
    assert values.dtype == np.float32
    assert np.all(np.isfinite(values))
    assert len(values) == len(stamps) == validation.n_rows


def test_validation_rejects_non_monotonic_timestamps(tmp_path):
    csv = tmp_path / "bad.csv"
    stamps = list(pd.date_range("2016-07-01", periods=10, freq="h"))
    stamps[3], stamps[4] = stamps[4], stamps[3]
    pd.DataFrame({"date": stamps, "OT": np.arange(10.0)}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="monotonic"):
        validate(csv, source_url="x", target_column="OT", expected_freq="1h")


def test_validation_rejects_duplicate_timestamps(tmp_path):
    csv = tmp_path / "dup.csv"
    stamps = list(pd.date_range("2016-07-01", periods=10, freq="h"))
    stamps[5] = stamps[4]
    pd.DataFrame({"date": stamps, "OT": np.arange(10.0)}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="monotonic|unique"):
        validate(csv, source_url="x", target_column="OT", expected_freq="1h")


def test_validation_counts_native_missing_values(tmp_path):
    csv = tmp_path / "holes.csv"
    values = np.arange(10.0)
    values[2] = np.nan
    values[7] = np.nan
    pd.DataFrame(
        {"date": pd.date_range("2016-07-01", periods=10, freq="h"), "OT": values}
    ).to_csv(csv, index=False)
    validation = validate(csv, source_url="x", target_column="OT", expected_freq="1h")
    assert validation.native_missing_in_target == 2
    assert validation.native_missing_positions_in_target == [2, 7]


def test_load_series_rejects_a_checksum_mismatch(tmp_path, config, monkeypatch):
    import json
    from pathlib import Path

    import data.fetch_etth1 as fetch

    csv = tmp_path / "ETTh1.csv"
    n = config["dataset"]["expected_rows"]
    pd.DataFrame(
        {"date": pd.date_range("2016-07-01", periods=n, freq="h"), "OT": np.arange(n, dtype=float)}
    ).to_csv(csv, index=False)

    cfg = json.loads(json.dumps(config))
    cfg["dataset"]["local_path"] = str(csv)
    cfg["dataset"]["expected_sha256"] = "0" * 64
    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fetch.load_series(cfg)


def test_load_series_rejects_a_wrong_row_count(tmp_path, config, monkeypatch):
    import json
    from pathlib import Path

    import data.fetch_etth1 as fetch

    csv = tmp_path / "short.csv"
    pd.DataFrame(
        {"date": pd.date_range("2016-07-01", periods=500, freq="h"), "OT": np.arange(500.0)}
    ).to_csv(csv, index=False)

    cfg = json.loads(json.dumps(config))
    cfg["dataset"]["local_path"] = str(csv)
    cfg["dataset"]["expected_sha256"] = sha256_file(csv)
    monkeypatch.setattr(fetch, "REPO_ROOT", Path("/"))
    with pytest.raises(ValueError, match="row count"):
        fetch.load_series(cfg)
