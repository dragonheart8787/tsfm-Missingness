"""Acquire and validate the canonical ETTh1 dataset.

Responsibilities (deliverable #2):
  * download the canonical ETTh1.csv from the ETDataset release,
  * record source URL, sha256, row count and timestamp range,
  * assert monotonic + unique hourly timestamps,
  * report NATIVE missing values in the raw data *before* any synthetic
    missingness is inserted anywhere in the pipeline,
  * persist all validation results as JSON next to the raw file.

This module never inserts missingness and never sees a forecast target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetValidation:
    """Everything we assert and persist about the raw series."""

    source_url: str
    local_path: str
    sha256: str
    n_rows: int
    n_columns: int
    columns: list[str]
    target_column: str
    timestamp_first: str
    timestamp_last: str
    timestamps_monotonic_increasing: bool
    timestamps_unique: bool
    inferred_freq: str | None
    expected_freq: str
    freq_matches_expected: bool
    n_gaps_in_hourly_grid: int
    native_missing_in_target: int
    native_missing_positions_in_target: list[int]
    native_missing_per_column: dict[str, int]
    target_min: float
    target_max: float
    target_mean: float
    target_std: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Fetch the canonical CSV unless it is already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    import requests  # imported lazily so offline validation needs no network stack

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def validate(csv_path: Path, *, source_url: str, target_column: str, expected_freq: str) -> DatasetValidation:
    """Validate the raw file and describe it. Raises on structural violations."""
    frame = pd.read_csv(csv_path)
    if "date" not in frame.columns:
        raise ValueError(f"ETTh1 must carry a 'date' column; found {list(frame.columns)}")
    if target_column not in frame.columns:
        raise ValueError(f"target column {target_column!r} absent; found {list(frame.columns)}")

    # Timestamps are preserved exactly as published: parsed for validation, but
    # never resampled, reindexed, or otherwise rewritten.
    stamps = pd.to_datetime(frame["date"])
    monotonic = bool(stamps.is_monotonic_increasing)
    unique = bool(stamps.is_unique)
    if not monotonic:
        raise ValueError("ETTh1 timestamps are not monotonically increasing")
    if not unique:
        raise ValueError("ETTh1 timestamps are not unique")

    deltas = stamps.diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    n_gaps = int((deltas != one_hour).sum())
    inferred = pd.infer_freq(pd.DatetimeIndex(stamps))
    freq_ok = inferred is not None and inferred.lower().rstrip("e") in {"h", "1h"}

    target = frame[target_column].to_numpy(dtype=np.float64)
    native_missing_mask = ~np.isfinite(target)
    native_missing_positions = np.flatnonzero(native_missing_mask).tolist()

    return DatasetValidation(
        source_url=source_url,
        local_path=str(csv_path.relative_to(REPO_ROOT)) if csv_path.is_relative_to(REPO_ROOT) else str(csv_path),
        sha256=sha256_file(csv_path),
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        columns=[str(c) for c in frame.columns],
        target_column=target_column,
        timestamp_first=str(stamps.iloc[0]),
        timestamp_last=str(stamps.iloc[-1]),
        timestamps_monotonic_increasing=monotonic,
        timestamps_unique=unique,
        inferred_freq=str(inferred) if inferred is not None else None,
        expected_freq=expected_freq,
        freq_matches_expected=bool(freq_ok),
        n_gaps_in_hourly_grid=n_gaps,
        native_missing_in_target=int(native_missing_mask.sum()),
        native_missing_positions_in_target=native_missing_positions[:1000],
        native_missing_per_column={
            str(col): int(frame[col].isna().sum()) for col in frame.columns
        },
        target_min=float(np.nanmin(target)),
        target_max=float(np.nanmax(target)),
        target_mean=float(np.nanmean(target)),
        target_std=float(np.nanstd(target)),
    )


def load_series(config: dict[str, Any]) -> tuple[np.ndarray, pd.DatetimeIndex, DatasetValidation]:
    """Load the validated univariate target series and its timestamps.

    Returns the target values as float32 (Chronos-2's own input dtype) together
    with the untouched hourly timestamps. Row count is asserted against the
    preregistered expectation; no row is ever dropped.
    """
    ds = config["dataset"]
    csv_path = REPO_ROOT / ds["local_path"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} is absent. Run: python data/fetch_etth1.py --config configs/pilot_config.yaml"
        )
    validation = validate(
        csv_path,
        source_url=ds["source_url"],
        target_column=ds["target_column"],
        expected_freq=ds["expected_freq"],
    )
    if validation.n_rows != ds["expected_rows"]:
        raise ValueError(
            f"ETTh1 row count {validation.n_rows} != preregistered {ds['expected_rows']}. "
            "The pilot design (178 origins) is derived from that length; refusing to proceed."
        )
    expected_sha = ds.get("expected_sha256")
    if expected_sha and expected_sha != validation.sha256:
        raise ValueError(
            f"ETTh1 sha256 mismatch: file is {validation.sha256}, config pins {expected_sha}."
        )
    frame = pd.read_csv(csv_path)
    values = frame[ds["target_column"]].to_numpy(dtype=np.float32)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    return values, stamps, validation


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description="Fetch and validate canonical ETTh1.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--write-checksum",
        action="store_true",
        help="Write the observed sha256 back into the config (first acquisition only).",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text())
    ds = config["dataset"]

    csv_path = REPO_ROOT / ds["local_path"]
    download(ds["source_url"], csv_path, force=args.force_download)
    validation = validate(
        csv_path,
        source_url=ds["source_url"],
        target_column=ds["target_column"],
        expected_freq=ds["expected_freq"],
    )

    metadata_path = REPO_ROOT / ds["metadata_path"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(validation.to_json())

    print(validation.to_json())

    if args.write_checksum:
        text = config_path.read_text()
        old = ds.get("expected_sha256") or "null"
        text = text.replace(f'expected_sha256: "{old}"', f'expected_sha256: "{validation.sha256}"')
        config_path.write_text(text)
        print(f"\nWrote expected_sha256={validation.sha256} into {args.config}")

    problems: list[str] = []
    if validation.n_rows != ds["expected_rows"]:
        problems.append(f"row count {validation.n_rows} != {ds['expected_rows']}")
    if validation.n_gaps_in_hourly_grid:
        problems.append(f"{validation.n_gaps_in_hourly_grid} gaps in the hourly grid")
    if validation.native_missing_in_target:
        problems.append(f"{validation.native_missing_in_target} native missing values in target")
    if problems:
        print("\nVALIDATION NOTES: " + "; ".join(problems))
    else:
        print("\nVALIDATION: clean — canonical length, no grid gaps, no native missing target values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
