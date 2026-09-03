"""Resumable pilot orchestrator.

Guarantees enforced here at RUNTIME (not only in the test suite):
  * origin count == 178, conditions per origin == 17, total forecasts == 3026;
  * the model's real patch grid matches the preregistered grid, else hard stop;
  * for every origin, every condition scores against BYTE-IDENTICAL ground
    truth (sha256 over the target bytes) and that target contains no NaN;
  * masking only ever touches the context: the target array is read-only and is
    never passed into masks/ or diagnostics/;
  * checkpointing after every completed origin, so an interrupted GPU run
    resumes without recomputing or duplicating finished work.

Failures are RECORDED (status + error message in the results file), never
silently skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.fetch_etth1 import load_series  # noqa: E402
from diagnostics.confounders import mask_diagnostics  # noqa: E402
from masks.base import apply_mask  # noqa: E402
from masks.plan import build_conditions  # noqa: E402
from metrics.pointwise import compute_metrics, seasonal_naive_denominator  # noqa: E402
from model.chronos2_runner import (  # noqa: E402
    Forecaster,
    MockForecaster,
    verify_contract,
)
from runner.windows import build_windows  # noqa: E402


class PatchGridMismatch(RuntimeError):
    """Raised when the loaded checkpoint's patch grid differs from the design's."""


# --------------------------------------------------------------------------- #
# Environment provenance
# --------------------------------------------------------------------------- #

def collect_environment(contract_dict: dict[str, Any] | None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_contract": contract_dict,
    }
    for module_name in ("torch", "transformers", "chronos", "numpy", "pandas", "scipy"):
        try:
            module = __import__(module_name)
            env[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - provenance is best-effort
            env[f"{module_name}_version"] = f"unavailable: {exc}"
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception as exc:  # pragma: no cover
        env["cuda_available"] = f"unavailable: {exc}"
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        env["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        )
    except Exception as exc:  # pragma: no cover
        env["git_commit"] = f"unavailable: {exc}"
    return env


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

@dataclass
class RunPaths:
    root: Path

    @property
    def window_results(self) -> Path:
        return self.root / "window_results.csv"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions_long.csv"

    @property
    def diagnostics(self) -> Path:
        return self.root / "mask_diagnostics.csv"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def log(self) -> Path:
        return self.root / "run.log"


def completed_origins(paths: RunPaths) -> set[int]:
    """Origins already fully written. The single source of resume truth.

    An origin counts as complete only if its checkpoint entry exists AND the
    results file actually carries its rows, so a crash between the two leaves
    the origin to be redone rather than silently dropped.
    """
    if not paths.checkpoint.exists():
        return set()
    state = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    claimed = {int(o) for o in state.get("completed_origins", [])}
    if not claimed or not paths.window_results.exists():
        return set()
    written = set(
        pd.read_csv(paths.window_results, usecols=["origin_id"])["origin_id"].astype(int).tolist()
    )
    return claimed & written


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    header = not path.exists()
    frame.to_csv(path, mode="a", header=header, index=False)


# --------------------------------------------------------------------------- #
# Runtime target-integrity guard
# --------------------------------------------------------------------------- #

def assert_target_integrity(window, digests: dict[str, str]) -> None:
    """Every condition of this origin scored against byte-identical truth.

    This is a RUNTIME guard, deliberately duplicated by the offline test suite.
    """
    reference = window.target_digest()
    mismatched = {c: d for c, d in digests.items() if d != reference}
    if mismatched:
        raise AssertionError(
            f"TARGET INTEGRITY VIOLATION at origin {window.origin_id}: conditions "
            f"{sorted(mismatched)} scored against a target differing from the clean "
            f"condition's (reference sha256 {reference})."
        )
    if not np.all(np.isfinite(np.asarray(window.target))):
        raise AssertionError(
            f"TARGET INTEGRITY VIOLATION at origin {window.origin_id}: the scored target "
            "contains a non-finite value."
        )
    if window.target.flags.writeable:
        raise AssertionError(
            f"TARGET INTEGRITY VIOLATION at origin {window.origin_id}: the target array is "
            "writeable; it must be read-only for the whole run."
        )


# --------------------------------------------------------------------------- #
# Main pilot loop
# --------------------------------------------------------------------------- #

def run_pilot(
    *,
    config: dict[str, Any],
    forecaster: Forecaster,
    run_dir: Path,
    limit_origins: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    design = config["design"]
    horizon = int(design["horizon"])
    context_length = int(design["context_length"])
    metrics_cfg = config["metrics"]
    diag_cfg = config["diagnostics"]

    # --- Model contract: hard stop on a patch-grid mismatch ----------------- #
    violations = verify_contract(forecaster.contract, config)
    if violations:
        raise PatchGridMismatch(
            "The loaded Chronos-2 checkpoint does not satisfy the preregistered model "
            "contract. Every block-alignment position depends on this grid, so the pilot "
            "STOPS rather than silently adjusting it:\n  - " + "\n  - ".join(violations)
        )
    patch_size = forecaster.contract.input_patch_size

    values, timestamps, validation = load_series(config)

    windows = build_windows(values=values, timestamps=timestamps, config=config)
    if len(windows) != int(design["expected_origins"]):
        raise AssertionError(f"origin count {len(windows)} != {design['expected_origins']}")

    paths = RunPaths(run_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    done = completed_origins(paths)

    run_id = run_id or (
        json.loads(paths.manifest.read_text(encoding="utf-8"))["run_id"]
        if paths.manifest.exists()
        else f"{config['run']['name']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )

    contract_dict = forecaster.contract.as_dict()
    manifest = {
        "run_id": run_id,
        "config": config,
        "dataset_validation": json.loads(validation.to_json()),
        "environment": collect_environment(contract_dict),
        "model_contract": contract_dict,
        "n_origins": len(windows),
        "conditions_per_origin": int(design["expected_conditions_per_origin"]),
        "expected_total_forecasts": int(design["expected_total_forecasts"]),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    selected = windows if limit_origins is None else windows[:limit_origins]
    n_ok = 0
    n_failed = 0

    for window in selected:
        if window.origin_id in done:
            continue
        started = time.time()
        conditions = build_conditions(
            config=config, origin_id=window.origin_id, patch_size=patch_size
        )
        context_timestamps = timestamps[window.context_start : window.context_end_exclusive]

        mase_denominator = (
            seasonal_naive_denominator(
                window.context,
                seasonal_period=int(metrics_cfg["mase_seasonal_period"]),
                min_denominator=float(metrics_cfg["mase_min_denominator"]),
            )
            if metrics_cfg.get("mase_enabled")
            else None
        )

        result_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        digests: dict[str, str] = {}
        clean_metrics = None

        for condition in conditions:
            # Masking receives ONLY the context. The target is not in scope here.
            masked_context = apply_mask(window.context, condition.mask)
            if masked_context.shape[0] != context_length:
                raise AssertionError("masking changed the context length")

            diag = mask_diagnostics(
                clean_context=window.context,
                mask=condition.mask,
                context_timestamps=context_timestamps,
                trailing_windows=[int(w) for w in diag_cfg["trailing_windows"]],
                patch_size=patch_size,
                seasonal_period=int(diag_cfg["seasonal_period"]),
            )
            diagnostic_rows.append(
                {
                    "run_id": run_id,
                    "origin_id": window.origin_id,
                    "condition_id": condition.condition_id,
                    "pattern": condition.pattern,
                    "rate": condition.rate,
                    "seed": condition.seed,
                    "distance_to_boundary": condition.distance,
                    "block_start": condition.mask.block_start,
                    "block_end_exclusive": condition.mask.block_end_exclusive,
                    **diag,
                }
            )

            status = "ok"
            error_message = ""
            metrics = None
            prediction = None
            call_started = time.time()
            try:
                prediction = forecaster.forecast_median(masked_context, horizon)
                metrics = compute_metrics(
                    truth=np.asarray(window.target),
                    prediction=prediction,
                    mase_denominator=mase_denominator,
                )
            except Exception as exc:  # recorded, never swallowed
                status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                n_failed += 1
            else:
                n_ok += 1
            runtime_s = time.time() - call_started

            # Integrity witness: recorded per condition, compared below.
            digests[condition.condition_id] = window.target_digest()

            if condition.condition_id == "clean" and metrics is not None:
                clean_metrics = metrics

            result_rows.append(
                {
                    "run_id": run_id,
                    "dataset_sha256": validation.sha256,
                    "model_revision": forecaster.contract.revision,
                    "model_id": forecaster.contract.hf_model_id,
                    "origin_id": window.origin_id,
                    "origin_timestamp": str(window.origin_timestamp),
                    "context_start": window.context_start,
                    "context_end_exclusive": window.context_end_exclusive,
                    "target_start": window.target_start,
                    "target_end_exclusive": window.target_end_exclusive,
                    "condition_id": condition.condition_id,
                    "pattern": condition.pattern,
                    "rate": condition.rate,
                    "seed": condition.seed,
                    "block_start": condition.mask.block_start,
                    "block_end_exclusive": condition.mask.block_end_exclusive,
                    "distance_to_boundary": condition.distance,
                    "missing_count": condition.mask.missing_count,
                    "mask_descriptor": condition.mask.descriptor(),
                    "mask_indices_sha256": _mask_digest(condition.mask),
                    "target_sha256": digests[condition.condition_id],
                    "mae": metrics.mae if metrics else float("nan"),
                    "mse": metrics.mse if metrics else float("nan"),
                    "rmse": metrics.rmse if metrics else float("nan"),
                    "mase": (metrics.mase if metrics else None),
                    "mase_denominator": mase_denominator,
                    "status": status,
                    "error_message": error_message,
                    "runtime_seconds": runtime_s,
                }
            )

            if prediction is not None:
                truth = np.asarray(window.target, dtype=np.float64)
                for step in range(horizon):
                    prediction_rows.append(
                        {
                            "run_id": run_id,
                            "origin_id": window.origin_id,
                            "condition_id": condition.condition_id,
                            "seed": condition.seed,
                            "step_index": step,
                            "forecast_timestamp": str(window.target_timestamps[step]),
                            "ground_truth": float(truth[step]),
                            "median_prediction": float(prediction[step]),
                        }
                    )

        # Runtime guard, before anything is persisted for this origin.
        assert_target_integrity(window, digests)

        # Paired clean metrics + paired difference, attached to every row.
        for row in result_rows:
            row["clean_mae"] = clean_metrics.mae if clean_metrics else float("nan")
            row["clean_mse"] = clean_metrics.mse if clean_metrics else float("nan")
            row["clean_rmse"] = clean_metrics.rmse if clean_metrics else float("nan")
            row["abs_paired_diff_mae_from_clean"] = (
                abs(row["mae"] - row["clean_mae"])
                if clean_metrics and np.isfinite(row["mae"])
                else float("nan")
            )
            row["paired_diff_mae_from_clean"] = (
                row["mae"] - row["clean_mae"]
                if clean_metrics and np.isfinite(row["mae"])
                else float("nan")
            )

        _append_rows(paths.window_results, result_rows)
        _append_rows(paths.predictions, prediction_rows)
        _append_rows(paths.diagnostics, diagnostic_rows)

        # Checkpoint AFTER the rows are on disk, so resume never duplicates.
        done.add(window.origin_id)
        _write_checkpoint(paths, run_id=run_id, done=done, n_ok=n_ok, n_failed=n_failed)

        elapsed = time.time() - started
        message = (
            f"origin {window.origin_id:4d}/{len(selected)} "
            f"({len(done)} complete) {elapsed:6.2f}s ok={n_ok} failed={n_failed}"
        )
        print(message, flush=True)
        with paths.log.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")

    summary = {
        "run_id": run_id,
        "run_dir": str(paths.root),
        "origins_completed": len(done),
        "origins_expected": len(selected),
        "forecasts_ok_this_process": n_ok,
        "forecasts_failed_this_process": n_failed,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Full-run assertions: only meaningful for a complete, unlimited run.
    if limit_origins is None and len(done) == int(design["expected_origins"]):
        frame = pd.read_csv(paths.window_results)
        n_rows = len(frame)
        expected_rows = int(design["expected_total_forecasts"])
        summary["result_rows"] = n_rows
        summary["expected_result_rows"] = expected_rows
        if n_rows != expected_rows:
            raise AssertionError(
                f"produced {n_rows} result rows, expected {expected_rows} "
                f"({design['expected_origins']} origins x "
                f"{design['expected_conditions_per_origin']} conditions)"
            )
        summary["rows_by_status"] = frame["status"].value_counts().to_dict()
        summary["distinct_target_digests_per_origin"] = int(
            frame.groupby("origin_id")["target_sha256"].nunique().max()
        )

    (paths.root / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _mask_digest(mask) -> str:
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(mask.missing_indices, dtype=np.int64).tobytes()
    ).hexdigest()


def _write_checkpoint(paths: RunPaths, *, run_id: str, done: set[int], n_ok: int, n_failed: int) -> None:
    payload = {
        "run_id": run_id,
        "completed_origins": sorted(done),
        "n_completed": len(done),
        "forecasts_ok_this_process": n_ok,
        "forecasts_failed_this_process": n_failed,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = paths.checkpoint.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, paths.checkpoint)  # atomic: a crash never leaves a torn checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Chronos-2 missingness geometry pilot.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--run-dir", default="results/pilot_v1")
    parser.add_argument("--limit-origins", type=int, default=None, help="Smoke-test subset.")
    parser.add_argument(
        "--mock-model",
        action="store_true",
        help="Use the deterministic mock forecaster (no weights, CPU-only). NOT a pilot result.",
    )
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    run_dir = REPO_ROOT / args.run_dir

    if args.mock_model:
        forecaster: Forecaster = MockForecaster()
        print("WARNING: running with the MOCK forecaster. Output is a pipeline exercise, "
              "not a Chronos-2 result.", flush=True)
    else:
        from model.chronos2_runner import Chronos2Forecaster

        forecaster = Chronos2Forecaster(config=config)

    summary = run_pilot(
        config=config,
        forecaster=forecaster,
        run_dir=run_dir,
        limit_origins=args.limit_origins,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
