"""Resumable runner for the trailing-gap mechanism experiment.

Mirrors ``runner/run_pilot.py``'s conventions exactly: preflight contract gate,
runtime count assertions, per-origin target-integrity guard, atomic checkpoint
written AFTER rows land, failures recorded rather than swallowed, and the same
CLI shape (``--config --run-dir --limit-origins --mock-model``).

    *** The experiment is gated on Research Lead sign-off. ***
    ``--mock-model`` is a pipeline exercise, NOT a Chronos-2 result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.fetch_etth1 import load_series  # noqa: E402
from experiments.trailing_gap import (  # noqa: E402
    TRUNCATED_LONG,
    build_conditions,
    scored_timestamps,
)
from metrics.pointwise import compute_metrics  # noqa: E402
from model.chronos2_runner import Forecaster, MockForecaster  # noqa: E402
from runner.run_pilot import RunPaths, assert_target_integrity, collect_environment  # noqa: E402
from runner.windows import build_windows  # noqa: E402


def completed_origins(paths: RunPaths) -> set[int]:
    """Resume truth: the checkpoint INTERSECTED with what actually landed."""
    if not paths.checkpoint.exists():
        return set()
    state = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
    claimed = {int(o) for o in state.get("completed_origins", [])}
    if not claimed or not paths.window_results.exists():
        return set()
    written = set(
        pd.read_csv(paths.window_results, usecols=["origin_id"])["origin_id"].astype(int)
    )
    return claimed & written


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_checkpoint(paths: RunPaths, *, run_id: str, done: set[int], ok: int, failed: int) -> None:
    payload = {
        "run_id": run_id,
        "completed_origins": sorted(done),
        "n_completed": len(done),
        "forecasts_ok_this_process": ok,
        "forecasts_failed_this_process": failed,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = paths.checkpoint.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, paths.checkpoint)  # atomic: never a torn checkpoint


def run_trailing_gap(
    *,
    config: dict[str, Any],
    gap_config: dict[str, Any],
    forecaster: Forecaster,
    run_dir: Path,
    limit_origins: int | None = None,
    only_conditions: list[str] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the matrix. ``only_conditions`` supports the clean-first audit step."""
    design = config["design"]
    horizon = int(design["horizon"])
    values, timestamps, validation = load_series(config)
    windows = build_windows(values=values, timestamps=timestamps, config=config)

    paths = RunPaths(run_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    done = completed_origins(paths)
    run_id = run_id or (
        json.loads(paths.manifest.read_text(encoding="utf-8"))["run_id"]
        if paths.manifest.exists()
        else f"{gap_config['meta']['name']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )

    contract_dict = forecaster.contract.as_dict()
    paths.manifest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment": gap_config["meta"]["name"],
                "preregistration": gap_config["meta"]["preregistration"],
                "pilot_config": config,
                "trailing_gap_config": gap_config,
                "dataset_validation": json.loads(validation.to_json()),
                "environment": collect_environment(contract_dict),
                "model_contract": contract_dict,
                "only_conditions": only_conditions,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    selected = windows if limit_origins is None else windows[:limit_origins]
    n_ok = n_failed = 0

    for window in selected:
        if window.origin_id in done:
            continue
        started = time.time()
        # Preflight runs inside build_conditions and hard-stops on a contract
        # violation, so an unrollable gap can never reach the model.
        conditions = build_conditions(
            context=window.context, horizon=horizon, config=gap_config,
            contract=forecaster.contract,
        )
        if only_conditions:
            conditions = [c for c in conditions if c.kind in only_conditions]

        context_timestamps = timestamps[window.context_start : window.context_end_exclusive]
        result_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        digests: dict[str, str] = {}
        clean_metrics = None

        for condition in conditions:
            expected_stamps = scored_timestamps(
                condition, context_timestamps=context_timestamps
            )
            # Runtime alignment guard: the scored window must land on the
            # ORIGINAL target's timestamps, whatever the condition did.
            if list(expected_stamps) != list(window.target_timestamps):
                raise AssertionError(
                    f"ALIGNMENT VIOLATION at origin {window.origin_id}, "
                    f"{condition.condition_id}: scored timestamps start "
                    f"{expected_stamps[0]}, target starts {window.target_timestamps[0]}"
                )

            status, error_message, metrics, scored = "ok", "", None, None
            call_started = time.time()
            try:
                prediction = forecaster.forecast_median(
                    condition.context, condition.prediction_length
                )
                scored = condition.score(prediction)
                metrics = compute_metrics(
                    truth=np.asarray(window.target), prediction=scored
                )
            except Exception as exc:  # recorded, never swallowed
                status, error_message = "failed", f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                n_failed += 1
            else:
                n_ok += 1
            runtime_s = time.time() - call_started

            digests[condition.condition_id] = window.target_digest()
            if condition.kind == "clean" and metrics is not None:
                clean_metrics = metrics

            result_rows.append(
                {
                    "run_id": run_id,
                    "dataset_sha256": validation.sha256,
                    "model_revision": forecaster.contract.revision,
                    "origin_id": window.origin_id,
                    "origin_timestamp": str(window.origin_timestamp),
                    "condition_id": condition.condition_id,
                    "kind": condition.kind,
                    "gap": condition.gap,
                    "context_length_used": int(condition.context.shape[0]),
                    "prediction_length_requested": condition.prediction_length,
                    "scored_first_index": condition.scored_first_index,
                    "dropped_leading_steps": (
                        condition.gap if condition.kind == TRUNCATED_LONG else 0
                    ),
                    "block_start": condition.mask.block_start if condition.mask else None,
                    "block_end_exclusive": (
                        condition.mask.block_end_exclusive if condition.mask else None
                    ),
                    "missing_count": condition.mask.missing_count if condition.mask else 0,
                    "target_sha256": digests[condition.condition_id],
                    "scored_first_timestamp": str(expected_stamps[0]),
                    "mae": metrics.mae if metrics else float("nan"),
                    "mse": metrics.mse if metrics else float("nan"),
                    "rmse": metrics.rmse if metrics else float("nan"),
                    "status": status,
                    "error_message": error_message,
                    "runtime_seconds": runtime_s,
                }
            )
            if scored is not None:
                truth = np.asarray(window.target, dtype=np.float64)
                for step in range(horizon):
                    prediction_rows.append(
                        {
                            "run_id": run_id,
                            "origin_id": window.origin_id,
                            "condition_id": condition.condition_id,
                            "step_index": step,
                            "forecast_timestamp": str(window.target_timestamps[step]),
                            "ground_truth": float(truth[step]),
                            "median_prediction": float(scored[step]),
                        }
                    )

        assert_target_integrity(window, digests)
        for row in result_rows:
            row["clean_mae"] = clean_metrics.mae if clean_metrics else float("nan")
            row["paired_diff_mae_from_clean"] = (
                row["mae"] - row["clean_mae"]
                if clean_metrics and np.isfinite(row["mae"]) else float("nan")
            )

        _append(paths.window_results, result_rows)
        _append(paths.predictions, prediction_rows)
        done.add(window.origin_id)
        _write_checkpoint(paths, run_id=run_id, done=done, ok=n_ok, failed=n_failed)

        message = (
            f"origin {window.origin_id:4d} ({len(done)}/{len(selected)}) "
            f"{time.time() - started:6.2f}s ok={n_ok} failed={n_failed}"
        )
        print(message, flush=True)
        with paths.log.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")

    summary = {
        "run_id": run_id,
        "run_dir": str(paths.root),
        "origins_completed": len(done),
        "forecasts_ok_this_process": n_ok,
        "forecasts_failed_this_process": n_failed,
        "only_conditions": only_conditions,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if limit_origins is None and only_conditions is None and len(done) == int(
        design["expected_origins"]
    ):
        frame = pd.read_csv(paths.window_results)
        expected = int(gap_config["design"]["expected_total_forecasts"])
        summary["result_rows"] = len(frame)
        summary["expected_result_rows"] = expected
        if len(frame) != expected:
            raise AssertionError(f"produced {len(frame)} rows, expected {expected}")
        per_origin = frame.groupby("origin_id").size().unique().tolist()
        if per_origin != [int(gap_config["design"]["expected_conditions_per_origin"])]:
            raise AssertionError(f"conditions per origin: {per_origin}")
        if frame.duplicated(subset=["origin_id", "condition_id"]).any():
            raise AssertionError("duplicate origin x condition cells")
        summary["rows_by_status"] = frame["status"].value_counts().to_dict()
        summary["max_distinct_target_digests_per_origin"] = int(
            frame.groupby("origin_id")["target_sha256"].nunique().max()
        )

    (paths.root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the trailing-gap mechanism experiment.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument("--gap-config", default="configs/trailing_gap_config.yaml")
    parser.add_argument("--run-dir", default="results/trailing_gap_v1")
    parser.add_argument("--limit-origins", type=int, default=None)
    parser.add_argument(
        "--only-conditions", default=None,
        help="Comma-separated kinds, e.g. 'clean' for the clean-first audit step.",
    )
    parser.add_argument("--mock-model", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    gap_config = yaml.safe_load((REPO_ROOT / args.gap_config).read_text(encoding="utf-8"))

    if args.mock_model:
        forecaster: Forecaster = MockForecaster()
        print("WARNING: MOCK forecaster. A pipeline exercise, not a Chronos-2 result.",
              flush=True)
    else:
        from model.chronos2_runner import Chronos2Forecaster

        forecaster = Chronos2Forecaster(config=config)

    summary = run_trailing_gap(
        config=config, gap_config=gap_config, forecaster=forecaster,
        run_dir=REPO_ROOT / args.run_dir, limit_origins=args.limit_origins,
        only_conditions=args.only_conditions.split(",") if args.only_conditions else None,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
