"""Shared fixtures. Everything here runs on CPU with NO model weights.

The real Chronos-2 call is mocked at the inference-module boundary
(``model.chronos2_runner.Forecaster``), so mask correctness, target integrity,
isolation and bootstrap pairing are all verifiable without a GPU. Tests that
genuinely need the real checkpoint are marked ``requires_model``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "pilot_config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def patch_size(config) -> int:
    """The preregistered grid. The runner re-verifies this against the real
    checkpoint and hard-fails on a mismatch; offline tests use the assumption."""
    return int(config["model"]["expected_patch_size"])


@pytest.fixture(scope="session")
def synthetic_series(config) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """A deterministic stand-in with ETTh1's exact length and hourly grid."""
    n = int(config["dataset"]["expected_rows"])
    t = np.arange(n, dtype=np.float64)
    values = (
        10.0
        + 5.0 * np.sin(2 * np.pi * t / 24.0)
        + 2.0 * np.sin(2 * np.pi * t / (24 * 7))
        + 0.0003 * t
        + np.random.default_rng(7).normal(scale=0.4, size=n)
    ).astype(np.float32)
    stamps = pd.date_range("2016-07-01", periods=n, freq="h")
    return values, stamps


@pytest.fixture(scope="session")
def real_series(config):
    """The actual validated ETTh1 target, when it has been fetched."""
    from data.fetch_etth1 import load_series

    csv = REPO_ROOT / config["dataset"]["local_path"]
    if not csv.exists():
        pytest.skip("ETTh1 not fetched; run data/fetch_etth1.py")
    values, stamps, validation = load_series(config)
    return values, stamps, validation
