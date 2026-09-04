"""Acquire and validate ETTh2 for the replication, with a NATIVE-MISSINGNESS HARD STOP.

Deliberately thin. Every structural check — timestamp monotonicity, uniqueness,
the hourly grid, the sha256, the per-column NaN census — is REUSED from
``data.fetch_etth1`` rather than reimplemented, so the two datasets cannot be
validated by subtly different rules. What this module adds is ETTh2-specific:

  1. a HARD STOP if ``OT`` carries any native missing value, and
  2. the DERIVED origin count, computed from ETTh2's own row count rather than
     inherited from ETTh1's 178.

Why the hard stop is a stop and not a warning
---------------------------------------------
ETTh1's zero-native-missingness property is load-bearing for the whole
target-integrity framework. Every forecast target is a contiguous slice of the
raw series, hashed and frozen read-only; every "missing" value anywhere in the
pipeline is one this project inserted deliberately, at a preregistered position
and count. A native NaN breaks that invariant in two places at once: a NaN
inside a target makes MAE undefined for that origin, and a NaN inside a context
is indistinguishable from an inserted mask, so the exact missing counts the
design preregisters would no longer be the counts the model actually sees.

ETTh2 has not been verified to share the property. If it does not, the correct
response is a Research Lead decision about the design, not an imputation, a
dropped row, or a quietly widened tolerance chosen by whoever happened to run
the fetch. So this raises, and the raise is what the runner depends on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))

from data.fetch_etth1 import (
    REPO_ROOT,
    DatasetValidation,
    download,
    sha256_file,
    validate,
)
from runner.windows import enumerate_origins

DATASET_NAME = "ETTh2"


class NativeMissingnessError(RuntimeError):
    """ETTh2 carries native missing values. A HARD STOP, never handled here.

    Raised, not logged. Imputing, dropping or interpolating is a design decision
    reserved to the Research Lead; this module's job is to make the condition
    impossible to proceed past silently.
    """


class OriginCountError(RuntimeError):
    """The derived origin count disagrees with the config's recorded value."""


def assert_no_native_missingness(validation: DatasetValidation) -> None:
    """HARD STOP on any native missing value in the target column.

    Checks the target column specifically — that is what the design's integrity
    guarantees are about — and reports the other columns for context without
    gating on them, since only ``OT`` is ever forecast or masked.
    """
    if validation.native_missing_in_target:
        positions = validation.native_missing_positions_in_target[:20]
        raise NativeMissingnessError(
            f"HARD STOP: {DATASET_NAME}'s target column "
            f"{validation.target_column!r} has "
            f"{validation.native_missing_in_target} native missing value(s), at "
            f"row index/indices {positions}"
            f"{' (first 20 shown)' if validation.native_missing_in_target > 20 else ''}.\n"
            f"\n"
            f"ETTh1 had zero, and that property is load-bearing for the entire "
            f"target-integrity framework: a native NaN in a target makes MAE "
            f"undefined for that origin, and a native NaN in a context is "
            f"indistinguishable from an inserted mask, so the preregistered exact "
            f"missing counts would no longer be the counts the model sees.\n"
            f"\n"
            f"Do NOT impute, drop, interpolate, or widen a tolerance. Stop and "
            f"report this to the Research Lead: how to handle native missingness "
            f"is a design decision, and it is theirs."
        )


def derive_origin_count(
    *, n_rows: int, context_length: int, horizon: int, stride: int
) -> int:
    """The origin count IMPLIED BY ETTh2's own length. Never inherited from ETTh1.

    Uses the same ``enumerate_origins`` the runner uses, so the number asserted
    here and the number produced at run time come from one implementation.
    """
    return len(
        enumerate_origins(
            n_rows=n_rows, context_length=context_length, horizon=horizon, stride=stride
        )
    )


def validate_etth2(csv_path: Path, dataset_config: dict[str, Any]) -> DatasetValidation:
    """Structural validation (shared) followed by the ETTh2 hard stop."""
    validation = validate(
        csv_path,
        source_url=dataset_config["source_url"],
        target_column=dataset_config["target_column"],
        expected_freq=dataset_config["expected_freq"],
    )
    assert_no_native_missingness(validation)
    return validation


def load_series(config: dict[str, Any]) -> tuple[np.ndarray, pd.DatetimeIndex, DatasetValidation]:
    """Load ETTh2's validated target series. Mirrors ``fetch_etth1.load_series``.

    Order matters: the native-missingness hard stop runs before the row-count and
    checksum assertions, so a corrupted file reports the missingness rather than
    a downstream symptom of it.
    """
    ds = config["dataset"]
    if ds.get("name") != DATASET_NAME:
        raise ValueError(
            f"this loader is for {DATASET_NAME}; config names {ds.get('name')!r}"
        )
    csv_path = REPO_ROOT / ds["local_path"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} is absent. Run: "
            f"python data/fetch_etth2.py --config configs/etth2_config.yaml"
        )

    validation = validate_etth2(csv_path, ds)

    if validation.n_rows != int(ds["expected_rows"]):
        raise ValueError(
            f"{DATASET_NAME} row count {validation.n_rows} != recorded "
            f"{ds['expected_rows']}. The derived origin count depends on that "
            f"length; refusing to proceed."
        )
    expected_sha = ds.get("expected_sha256")
    if expected_sha and expected_sha != validation.sha256:
        raise ValueError(
            f"{DATASET_NAME} sha256 mismatch: file is {validation.sha256}, "
            f"config pins {expected_sha}."
        )

    design = config["design"]
    derived = derive_origin_count(
        n_rows=validation.n_rows,
        context_length=int(design["context_length"]),
        horizon=int(design["horizon"]),
        stride=int(design["stride"]),
    )
    recorded = int(design["expected_origins"])
    if derived != recorded:
        raise OriginCountError(
            f"{DATASET_NAME} yields {derived} origins at L="
            f"{design['context_length']}, H={design['horizon']}, "
            f"stride={design['stride']}, but the config records {recorded}. "
            f"The origin count is DERIVED, never assumed; fix the config by "
            f"amendment rather than the derivation."
        )

    frame = pd.read_csv(csv_path)
    values = frame[ds["target_column"]].to_numpy(dtype=np.float32)
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    return values, stamps, validation


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(
        description=f"Fetch and validate canonical {DATASET_NAME}, with a "
                    f"native-missingness hard stop."
    )
    parser.add_argument("--config", default="configs/etth2_config.yaml")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--write-checksum", action="store_true",
        help="Write the observed sha256 back into the config (first acquisition only).",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ds, design = config["dataset"], config["design"]

    csv_path = REPO_ROOT / ds["local_path"]
    download(ds["source_url"], csv_path, force=args.force_download)

    # The hard stop is inside validate_etth2 and runs here, before anything else
    # reports success. A non-zero exit with the NativeMissingnessError message is
    # the intended behaviour, not a crash to be worked around.
    validation = validate_etth2(csv_path, ds)

    metadata_path = REPO_ROOT / ds["metadata_path"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(validation.to_json(), encoding="utf-8")
    print(validation.to_json())

    derived = derive_origin_count(
        n_rows=validation.n_rows,
        context_length=int(design["context_length"]),
        horizon=int(design["horizon"]),
        stride=int(design["stride"]),
    )
    print(
        f"\nDERIVED origin count at L={design['context_length']}, "
        f"H={design['horizon']}, stride={design['stride']}: {derived}"
    )

    if args.write_checksum:
        text = config_path.read_text(encoding="utf-8")
        old = ds.get("expected_sha256") or "null"
        text = text.replace(
            f'expected_sha256: "{old}"', f'expected_sha256: "{validation.sha256}"'
        )
        config_path.write_text(text, encoding="utf-8")
        print(f"Wrote expected_sha256={validation.sha256} into {args.config}")

    problems: list[str] = []
    if validation.n_rows != int(ds["expected_rows"]):
        problems.append(f"row count {validation.n_rows} != {ds['expected_rows']}")
    if validation.n_gaps_in_hourly_grid:
        problems.append(f"{validation.n_gaps_in_hourly_grid} gaps in the hourly grid")
    if derived != int(design["expected_origins"]):
        problems.append(f"derived origins {derived} != {design['expected_origins']}")
    if problems:
        print("\nVALIDATION NOTES: " + "; ".join(problems))
        return 1
    print(
        f"\nVALIDATION: clean — recorded length, no grid gaps, "
        f"ZERO native missing values in {validation.target_column}, "
        f"{derived} derived origins."
    )
    return 0


__all__ = [
    "DATASET_NAME",
    "NativeMissingnessError",
    "OriginCountError",
    "assert_no_native_missingness",
    "derive_origin_count",
    "load_series",
    "sha256_file",
    "validate_etth2",
]


if __name__ == "__main__":
    raise SystemExit(main())
