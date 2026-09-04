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

# The dataset this entrypoint exists for. Anything else is refused at the CLI.
THIS_EXPERIMENT_DATASET = "ETTh1"


def completed_cells(paths: RunPaths) -> set[tuple[int, str]]:
    """Resume truth, at ORIGIN x CONDITION granularity.

    The written result rows are the ONLY source of truth for what is done. A
    per-origin "seen it" flag is not sufficient: a ``--only-conditions clean``
    pass writes one row per origin, and an origin-level flag would then mark all
    178 origins complete, so the following full-matrix run would skip every one
    of them and silently never execute the other 2,136 forecasts. The count
    assertion would catch that only afterwards, with the run already wasted.

    The checkpoint is advisory metadata for humans; it never gates resumption.
    """
    if not paths.window_results.exists():
        return set()
    frame = pd.read_csv(paths.window_results, usecols=["origin_id", "condition_id"])
    return {
        (int(origin), str(condition))
        for origin, condition in zip(frame["origin_id"], frame["condition_id"])
    }


def completed_origins(paths: RunPaths, *, conditions_per_origin: int) -> set[int]:
    """Origins with EVERY condition written. Derived from the cell truth above."""
    counts: dict[int, int] = {}
    for origin, _condition in completed_cells(paths):
        counts[origin] = counts.get(origin, 0) + 1
    return {origin for origin, n in counts.items() if n >= conditions_per_origin}


def existing_clean_mae(paths: RunPaths) -> dict[int, float]:
    """Clean MAE per origin from earlier passes, for pairing across a resume.

    Without this, a full-matrix run resuming after a clean-only pass would find
    no clean row to pair against in ITS pass and write NaN paired differences.
    """
    if not paths.window_results.exists():
        return {}
    frame = pd.read_csv(paths.window_results)
    clean = frame[(frame["kind"] == "clean") & (frame["status"] == "ok")]
    return {int(r.origin_id): float(r.mae) for r in clean.itertuples()}


def existing_target_digests(paths: RunPaths) -> dict[int, set[str]]:
    """Target digests already recorded per origin, so integrity spans resumes."""
    if not paths.window_results.exists():
        return {}
    frame = pd.read_csv(paths.window_results, usecols=["origin_id", "target_sha256"])
    out: dict[int, set[str]] = {}
    for origin, digest in zip(frame["origin_id"], frame["target_sha256"]):
        out.setdefault(int(origin), set()).add(str(digest))
    return out


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_checkpoint(
    paths: RunPaths, *, run_id: str, cells: set[tuple[int, str]],
    fully_done: set[int], ok: int, failed: int,
) -> None:
    """Advisory progress metadata. NEVER consulted when deciding what to run."""
    payload = {
        "run_id": run_id,
        "note": (
            "Advisory only. Resumption is decided from window_results.csv at "
            "origin x condition granularity; this file never gates it."
        ),
        "n_completed_cells": len(cells),
        "completed_origins": sorted(fully_done),
        "n_completed": len(fully_done),
        "forecasts_ok_this_process": ok,
        "forecasts_failed_this_process": failed,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = paths.checkpoint.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, paths.checkpoint)  # atomic: never a torn checkpoint


class NativeMissingnessPolicyViolation(RuntimeError):
    """The dataset declares a native-missingness hard stop and violates it.

    Raised at the runner's lowest forecasting boundary, so it cannot be bypassed
    by calling this function directly instead of going through a dataset-specific
    wrapper.
    """


def enforce_native_missingness_policy(config: dict[str, Any], validation) -> None:
    """OPT-IN hard stop, keyed on ``dataset.native_missing_is_a_hard_stop``.

    ADDITIVE AND OPT-IN BY DESIGN. A dataset config that does not set the key —
    ETTh1's, which predates it — takes the same path it always did: this
    function inspects the flag, finds it absent, and returns without touching
    anything. ETTh1 behaviour is therefore byte-identical to before the key
    existed, which ``tests/test_native_missingness_policy.py`` proves against
    recorded output hashes rather than by inspection.

    For a dataset that DOES set it, the check runs here — after ``load_series``
    and before ``build_windows`` or any ``forecast_median`` call — because this
    is the last point common to every route into the matrix. A gate that lived
    only in a wrapper could be walked around by importing this module and
    calling ``run_trailing_gap`` directly, which is exactly the hole this closes.
    """
    dataset = config.get("dataset", {})
    if not dataset.get("native_missing_is_a_hard_stop"):
        return
    observed = int(getattr(validation, "native_missing_in_target", 0))
    if observed == 0:
        return
    positions = list(getattr(validation, "native_missing_positions_in_target", []))[:20]
    raise NativeMissingnessPolicyViolation(
        f"HARD STOP: {dataset.get('name', 'the dataset')}'s target column "
        f"{dataset.get('target_column', '?')!r} has {observed} native missing "
        f"value(s), at row index/indices {positions}"
        f"{' (first 20 shown)' if observed > 20 else ''}, and its config sets "
        f"dataset.native_missing_is_a_hard_stop.\n"
        f"\n"
        f"No window was built and no forecast was requested. Do NOT impute, drop, "
        f"interpolate, or widen a tolerance: how to handle native missingness is a "
        f"design decision, and it belongs to the Research Lead."
    )


def write_run_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    """Write run_summary.json atomically: temp file, then rename.

    A wrapper enriches this summary after the runner returns and writes it back.
    A torn summary — half the old content, half the new — would be worse than
    either, so the rename is what makes it visible, exactly as the checkpoint
    writer already does.
    """
    path = Path(run_dir) / "run_summary.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def run_trailing_gap(
    *,
    config: dict[str, Any],
    gap_config: dict[str, Any],
    forecaster: Forecaster,
    run_dir: Path,
    limit_origins: int | None = None,
    only_conditions: list[str] | None = None,
    run_id: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the matrix. ``only_conditions`` supports the clean-first audit step.

    ``run_metadata`` lets a dataset-specific wrapper OVERRIDE the manifest's
    provenance fields rather than inherit them. Without it the manifest defaults
    to ``gap_config['meta']``, which names the ETTh1 trailing-gap experiment —
    correct for that experiment and wrong for anything else, so a wrapper for a
    different dataset must pass its own identity explicitly.
    """
    design = config["design"]
    horizon = int(design["horizon"])
    values, timestamps, validation = load_series(config)
    # Lowest forecasting boundary: after validation, before any window is built
    # and before any call reaches the forecaster.
    enforce_native_missingness_policy(config, validation)
    windows = build_windows(values=values, timestamps=timestamps, config=config)

    conditions_per_origin = int(gap_config["design"]["expected_conditions_per_origin"])
    paths = RunPaths(run_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    done_cells = completed_cells(paths)
    prior_clean_mae = existing_clean_mae(paths)
    prior_digests = existing_target_digests(paths)
    run_metadata = dict(run_metadata or {})
    # Identity defaults to the trailing-gap experiment because that is what this
    # module was written for. A wrapper for a DIFFERENT dataset must override it:
    # inheriting ETTh1's experiment name into an ETTh2 manifest would misattribute
    # the run in the one file the reviewer uses to establish what was executed.
    experiment = run_metadata.pop("experiment", gap_config["meta"]["name"])
    preregistration = run_metadata.pop(
        "preregistration", gap_config["meta"]["preregistration"]
    )
    run_id = run_id or (
        json.loads(paths.manifest.read_text(encoding="utf-8"))["run_id"]
        if paths.manifest.exists()
        else f"{experiment}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )

    contract_dict = forecaster.contract.as_dict()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "experiment": experiment,
        "preregistration": preregistration,
        "pilot_config": config,
        "trailing_gap_config": gap_config,
        "dataset_validation": json.loads(validation.to_json()),
        "environment": collect_environment(contract_dict),
        "model_contract": contract_dict,
        "only_conditions": only_conditions,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Remaining wrapper metadata is additive. It cannot silently displace any key
    # above: a collision is a bug in the wrapper and is raised, not merged over.
    collisions = sorted(set(run_metadata) & set(manifest))
    if collisions:
        raise ValueError(
            f"run_metadata may not overwrite manifest key(s) {collisions}; only "
            f"'experiment' and 'preregistration' are overridable"
        )
    manifest.update(run_metadata)
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    selected = windows if limit_origins is None else windows[:limit_origins]
    n_ok = n_failed = 0

    for window in selected:
        started = time.time()
        # Preflight runs inside build_conditions and hard-stops on a contract
        # violation, so an unrollable gap can never reach the model.
        conditions = build_conditions(
            context=window.context, horizon=horizon, config=gap_config,
            contract=forecaster.contract,
        )
        if only_conditions:
            conditions = [c for c in conditions if c.kind in only_conditions]
        # Skip only the CELLS already written, never the whole origin.
        pending = [
            c for c in conditions
            if (window.origin_id, c.condition_id) not in done_cells
        ]
        if not pending:
            continue
        conditions = pending

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

        # Integrity spans resumes: digests written in EARLIER passes for this
        # origin must agree with the ones written now.
        for earlier in prior_digests.get(window.origin_id, set()):
            digests[f"__earlier_pass__{earlier[:8]}"] = earlier
        assert_target_integrity(window, digests)

        # Pair against this pass's clean row, or an earlier pass's if the clean
        # cell was written by a previous (e.g. clean-only) invocation.
        clean_mae = (
            clean_metrics.mae if clean_metrics
            else prior_clean_mae.get(window.origin_id, float("nan"))
        )
        for row in result_rows:
            row["clean_mae"] = clean_mae
            row["paired_diff_mae_from_clean"] = (
                row["mae"] - clean_mae
                if np.isfinite(clean_mae) and np.isfinite(row["mae"]) else float("nan")
            )
        if clean_metrics is not None:
            prior_clean_mae[window.origin_id] = clean_metrics.mae

        _append(paths.window_results, result_rows)
        _append(paths.predictions, prediction_rows)
        done_cells.update(
            (window.origin_id, row["condition_id"]) for row in result_rows
        )
        prior_digests.setdefault(window.origin_id, set()).update(digests.values())
        fully_done = completed_origins(paths, conditions_per_origin=conditions_per_origin)
        _write_checkpoint(
            paths, run_id=run_id, cells=done_cells, fully_done=fully_done,
            ok=n_ok, failed=n_failed,
        )

        message = (
            f"origin {window.origin_id:4d} (+{len(result_rows)} cells, "
            f"{len(done_cells)} total) {time.time() - started:6.2f}s "
            f"ok={n_ok} failed={n_failed}"
        )
        print(message, flush=True)
        with paths.log.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")

    fully_done = completed_origins(paths, conditions_per_origin=conditions_per_origin)
    summary = {
        "run_id": run_id,
        "run_dir": str(paths.root),
        "origins_fully_completed": len(fully_done),
        "cells_completed": len(done_cells),
        "forecasts_ok_this_process": n_ok,
        "forecasts_failed_this_process": n_failed,
        "only_conditions": only_conditions,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if limit_origins is None and only_conditions is None and len(fully_done) == int(
        design["expected_origins"]
    ):
        frame = pd.read_csv(paths.window_results)
        expected_rows = int(gap_config["design"]["expected_total_forecasts"])
        expected_cells = int(design["expected_origins"]) * conditions_per_origin
        summary["result_rows"] = len(frame)
        summary["expected_result_rows"] = expected_rows
        summary["distinct_cells"] = int(
            frame.drop_duplicates(subset=["origin_id", "condition_id"]).shape[0]
        )
        summary["expected_distinct_cells"] = expected_cells
        if len(frame) != expected_rows:
            raise AssertionError(f"produced {len(frame)} rows, expected {expected_rows}")
        # Cell-level assertion at full scale: 178 x 13 = 2314 DISTINCT cells,
        # not merely 2314 rows, so a duplicated cell cannot mask a missing one.
        if summary["distinct_cells"] != expected_cells:
            raise AssertionError(
                f"produced {summary['distinct_cells']} distinct origin x condition "
                f"cells, expected {expected_cells}"
            )
        per_origin = frame.groupby("origin_id").size().unique().tolist()
        if per_origin != [conditions_per_origin]:
            raise AssertionError(f"conditions per origin: {per_origin}")
        if frame.duplicated(subset=["origin_id", "condition_id"]).any():
            raise AssertionError("duplicate origin x condition cells")
        summary["rows_by_status"] = frame["status"].value_counts().to_dict()
        summary["max_distinct_target_digests_per_origin"] = int(
            frame.groupby("origin_id")["target_sha256"].nunique().max()
        )

    write_run_summary(paths.root, summary)
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

    # Refuse a non-ETTh1 dataset BEFORE constructing a forecaster. This CLI runs
    # the ETTh1 trailing-gap experiment; another dataset reaching it would get
    # ETTh1's experiment identity in its manifest, skip its own wrapper's
    # preconditions, and write into a directory the wrong runbook governs.
    dataset_name = str(config.get("dataset", {}).get("name", ""))
    if dataset_name and dataset_name != THIS_EXPERIMENT_DATASET:
        print(
            f"REFUSING TO RUN: --config names dataset {dataset_name!r}, but this "
            f"entrypoint runs the {THIS_EXPERIMENT_DATASET} trailing-gap experiment.\n"
            f"\n"
            f"No model was loaded and no forecast was requested. Use the dataset's "
            f"own entrypoint, which applies its own preconditions:\n"
            f"\n"
            f"    .venv/bin/python experiments/run_etth2.py --phase <phase>\n"
            f"\n"
            f"See docs/gpu_execution_runbook_etth2.md.",
            file=sys.stderr, flush=True,
        )
        return 2

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
