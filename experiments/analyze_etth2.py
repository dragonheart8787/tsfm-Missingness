"""ETTh2 replication analysis: R_g, B_g, and the three-way replication outcome.

    *** NOT YET EXECUTED. No ETTh2 forecast has been produced. ***

Reuses the ETTh1 analysis wholesale. ``experiments.analyze_trailing_gap.analyse``
computes the contrasts, the bootstrap CIs, the Holm correction and the per-gap
four-way readings; nothing statistical is reimplemented here, so the two datasets
cannot be analysed by subtly different rules. What this module adds is the part
that is genuinely new: the three-way REPLICATED / CONTRADICTED / INCONCLUSIVE
decision, applied to the three primary gaps under the resolved multiplicity
treatment.

Multiplicity, as resolved in the preregistration section 5.1
------------------------------------------------------------
* ``R_16``, ``R_32``, ``R_64`` are each checked against their own NOMINAL 90%
  equivalence CI. No additional Holm or Bonferroni correction is applied to the
  intersection claim — an intersection-union test is conservative under any
  dependence structure.
* Holm across the four ``R`` gaps is RETAINED, for MATERIAL directional readings
  only. That is already how ``GapStat.reading`` works: the equivalence branch
  reads the uncorrected 90% interval, the material branches read the Holm
  interval. The reused ``analyse`` therefore applies exactly this treatment
  without needing to be told.
* ``B`` remains a separately corrected family of four and never determines the
  outcome. It is not assumed statistically independent of ``R``; no sign is
  claimed for the dependence.

The trailing-gap mechanism classification is deliberately NOT recomputed as the
headline. ``analyse`` still produces it — it is part of the shared output — but
the ETTh2 question is the replication outcome, and that is what
``classification_etth2.json`` records.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_trailing_gap import analyse  # noqa: E402
from experiments.reporting_v2 import COMPARISON_ARM as _COMPARISON_ARM  # noqa: E402
from experiments.reporting_v2 import INTERPRETATION_B  # noqa: E402
from experiments.run_etth2 import (  # noqa: E402
    MOCK_NOT_A_FINDING,
    build_effective_config,
)
from stats.mechanism_decision import GapStat  # noqa: E402
from stats.replication_decision import (  # noqa: E402
    PRIMARY_CONTRASTS,
    SECONDARY_STRESS_CONTRAST,
    ReplicationInputs,
    classify_replication,
)

REPLICATION_OUTPUT = "classification_etth2.json"
STAGING_SUFFIX = ".staging"

# The comparison arm of each family, stated on every row rather than implied.
# Imported from the erratum module so ETTh2 and the corrected ETTh1 reporting
# cannot drift apart in wording — but see the note below on the erratum id.
COMPARISON_ARM = _COMPARISON_ARM

# ETTh2's pipeline writes family-correct B text FROM THE START. It never had the
# ETTh1 defect, so it carries NO erratum id: referencing
# ERRATUM-2026-09-04-bg-comparison-arm here would assert that this output was
# corrected after the fact, which is false and would misdescribe its provenance.
FORBIDDEN_ERRATUM_ID = "ERRATUM-2026-09-04-bg-comparison-arm"
REPORTING_PROVENANCE = "family_correct_from_the_start"


class Etth2ReportingError(RuntimeError):
    """The staged output failed verification and was not promoted."""


class MockAnalysisRefused(RuntimeError):
    """The run was produced by the mock forecaster. Refused by default."""


# Every field a mock artifact must carry. Applied to the payload AND to the
# classification object itself, so a mock result copied out of its directory is
# still structurally identifiable as mock — this is a requirement on the data
# shape, not on prose sitting next to it.
MOCK_STAMP: dict[str, Any] = {
    "mock_model": True,
    "scientifically_valid": False,
    "authoritative_result": False,
    "mock_model_note": MOCK_NOT_A_FINDING,
}
# Prefixed onto the outcome label. A consumer reading only `label` still cannot
# mistake it for REPLICATED / CONTRADICTED / INCONCLUSIVE.
MOCK_LABEL_PREFIX = "MOCK_NOT_A_FINDING__"


def run_is_mock(run_dir: Path) -> bool:
    """Was this run produced by the mock forecaster?

    Read from the run's own manifest and summary — the artifacts the runner
    wrote — rather than from a caller-supplied flag, so the answer cannot be
    lost by forgetting to pass an argument. A run whose provenance cannot be
    established is treated as mock: the safe direction is refusing a real run,
    not analysing a fake one.
    """
    run_dir = Path(run_dir)
    for name in ("run_manifest.json", "run_summary.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return True
        if payload.get("mock_model") is True:
            return True
        contract = payload.get("model_contract") or {}
        if str(contract.get("revision", "")).startswith("mock"):
            return True
    if not (run_dir / "run_manifest.json").exists() and not (
        run_dir / "run_summary.json"
    ).exists():
        return True
    return False


def gap_stats_from_rows(rows: list[dict[str, Any]], *, mean_clean_mae: float) -> list[GapStat]:
    """Rebuild GapStat objects from ``analyse``'s emitted contrast rows.

    Going through the emitted rows rather than an in-memory handoff means the
    decision is computed from exactly the numbers that were written to disk and
    delivered to the reviewer, not from a parallel copy that could differ.
    """
    stats: list[GapStat] = []
    for row in rows:
        stats.append(
            GapStat(
                contrast_id=str(row["contrast_id"]),
                gap=int(row["gap"]),
                mean_difference=float(row["raw_mae_difference"]),
                median_difference=float(row["median_paired_difference"]),
                ci_low_holm=float(row["holm_ci95_low"]),
                ci_high_holm=float(row["holm_ci95_high"]),
                ci_low_equivalence=float(row["ci90_low"]),
                ci_high_equivalence=float(row["ci90_high"]),
                holm_rejects=bool(row["holm_rejects"]),
                mean_clean_mae=float(mean_clean_mae),
            )
        )
    return stats


def apply_family_correct_reporting(
    payload: dict[str, Any], *, mock: bool
) -> dict[str, Any]:
    """Family-correct B text and an explicit comparison_arm, on every row.

    Built in, not patched on. ``analyse`` writes ``INTERPRETATION[reading]`` into
    both families' ``interpretation`` column — text authored for R, naming R's
    ``truncated_long`` arm. ETTh1 needed an erratum to fix that after the fact;
    ETTh2 fixes it here, before anything is promoted to an official output, so
    no ETTh2 artifact ever carries the wrong arm.

    Only descriptive fields are touched. Every numeric field is left exactly as
    ``analyse`` produced it, which ``verify_reporting`` proves against a snapshot
    taken before this function runs.
    """
    for row in payload.get("b_contrasts", []):
        row["interpretation"] = INTERPRETATION_B[row["reading"]]
        row["comparison_arm"] = COMPARISON_ARM["B"]
    for row in payload.get("r_contrasts", []):
        row["comparison_arm"] = COMPARISON_ARM["R"]
    payload["reporting_provenance"] = REPORTING_PROVENANCE
    payload["reporting_note"] = (
        "B_g rows carry family-correct interpretation text and an explicit "
        "comparison_arm from the start. This pipeline never produced the ETTh1 "
        "B_g arm defect, so it carries no erratum id."
    )
    if mock:
        payload.update(MOCK_STAMP)
    return payload


def verify_reporting(
    payload: dict[str, Any], numeric_snapshot: dict[str, list], *, mock: bool
) -> None:
    """Verify the staged payload BEFORE it is promoted. Raises, never warns.

    Checks, each a refusal:
      * every B row's text equals ``INTERPRETATION_B[reading]``;
      * every R and B row carries its family's ``comparison_arm``;
      * R's text is untouched;
      * every numeric field is bit-identical to the pre-correction snapshot;
      * the ETTh1 erratum id appears nowhere;
      * a mock payload carries the full mock stamp.
    """
    problems: list[str] = []

    for row in payload.get("b_contrasts", []):
        cid = row.get("contrast_id")
        expected = INTERPRETATION_B.get(row.get("reading"))
        if expected is None:
            problems.append(f"b_contrasts[{cid}]: unknown reading {row.get('reading')!r}")
        elif row.get("interpretation") != expected:
            problems.append(f"b_contrasts[{cid}].interpretation is not INTERPRETATION_B")
        if row.get("comparison_arm") != COMPARISON_ARM["B"]:
            problems.append(f"b_contrasts[{cid}].comparison_arm is wrong or absent")
    for row in payload.get("r_contrasts", []):
        if row.get("comparison_arm") != COMPARISON_ARM["R"]:
            problems.append(f"r_contrasts[{row.get('contrast_id')}].comparison_arm is wrong")

    for key, values in numeric_snapshot.items():
        family, column = key.split("|", 1)
        current = [row.get(column) for row in payload.get(family, [])]
        if current != values:
            problems.append(f"{family}[*].{column} changed during reporting correction")

    blob = json.dumps(payload, default=str)
    if FORBIDDEN_ERRATUM_ID in blob:
        problems.append(
            f"{FORBIDDEN_ERRATUM_ID} appears in an ETTh2 artifact; this pipeline "
            f"never had that defect and must not claim to have been corrected"
        )
    if mock:
        for key, value in MOCK_STAMP.items():
            if payload.get(key) != value:
                problems.append(f"mock payload is missing or misstates {key!r}")

    if problems:
        raise Etth2ReportingError(
            "staged ETTh2 reporting failed verification and was NOT promoted:\n  - "
            + "\n  - ".join(problems)
        )


def numeric_snapshot_of(payload: dict[str, Any]) -> dict[str, list]:
    """Every numeric contrast field, captured before any descriptive rewrite."""
    columns = (
        "raw_mae_difference", "pct_of_mean_clean_mae", "median_paired_difference",
        "holm_ci95_low", "holm_ci95_high", "holm_adjusted_p", "holm_rejects",
        "ci90_low", "ci90_high", "frac_origins_positive", "sesoi_delta",
        "gap", "reading", "contrast_id",
    )
    snapshot: dict[str, list] = {}
    for family in ("r_contrasts", "b_contrasts"):
        for column in columns:
            snapshot[f"{family}|{column}"] = [
                row.get(column) for row in payload.get(family, [])
            ]
    return snapshot


def stamp_every_json(staging: Path) -> list[str]:
    """Apply the mock stamp to every JSON artifact in the staged directory.

    Sweeping the directory rather than listing filenames means an artifact added
    later by the shared analysis is stamped automatically, instead of silently
    shipping unstamped until someone notices.
    """
    stamped: list[str] = []
    for path in sorted(staging.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        payload.update(MOCK_STAMP)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        stamped.append(path.name)
    return stamped


def verify_staging_directory(staging: Path, *, mock: bool) -> None:
    """Sweep EVERY staged artifact before promotion. Raises, never warns.

    File-by-file correctness is not enough: a directory is promoted as a whole,
    and a reviewer opening the wrong file in it would read the wrong thing. So
    every JSON and CSV is checked for R-arm text on a B row, for the ETTh1
    erratum id, and — on a mock run — for the mock stamp.
    """
    problems: list[str] = []
    r_arm_markers = ("shorter-context", "horizon-dominant")

    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if path.suffix == ".csv":
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            if "family" in frame.columns and "interpretation" in frame.columns:
                for _, row in frame.iterrows():
                    if row["family"] != "B":
                        continue
                    if any(m in row["interpretation"] for m in r_arm_markers):
                        problems.append(
                            f"{name}: a B row carries R-arm text "
                            f"({row.get('contrast_id')})"
                        )
                    if row.get("comparison_arm") != COMPARISON_ARM["B"]:
                        problems.append(f"{name}: B row without B's comparison_arm")
            if FORBIDDEN_ERRATUM_ID in path.read_text(encoding="utf-8"):
                problems.append(f"{name}: carries the ETTh1 erratum id")
            continue
        if path.suffix != ".json":
            continue

        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_ERRATUM_ID in text:
            problems.append(f"{name}: carries the ETTh1 erratum id")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            problems.append(f"{name}: is not valid JSON")
            continue
        for row in payload.get("b_contrasts", []) if isinstance(payload, dict) else []:
            if any(m in str(row.get("interpretation", "")) for m in r_arm_markers):
                problems.append(
                    f"{name}: b_contrasts[{row.get('contrast_id')}] carries R-arm text"
                )
            if row.get("comparison_arm") != COMPARISON_ARM["B"]:
                problems.append(f"{name}: a b_contrasts row lacks B's comparison_arm")
        if mock and isinstance(payload, dict):
            for key, value in MOCK_STAMP.items():
                if payload.get(key) != value:
                    problems.append(f"{name}: mock artifact is missing {key!r}")

    if problems:
        raise Etth2ReportingError(
            "the staged ETTh2 output directory failed verification and was NOT "
            "promoted:\n  - " + "\n  - ".join(sorted(set(problems)))
        )


def _promote(staging: Path, official: Path) -> None:
    """Atomically replace the official directory with the verified staging one.

    A crash before this point leaves the official location untouched or absent —
    never half-correct. A half-written directory carrying R's text on B rows and
    sitting where the reviewer looks for the final result is precisely the
    outcome this pattern exists to prevent.
    """
    if official.exists():
        raise Etth2ReportingError(
            f"{official} already exists. Refusing to overwrite an existing "
            f"official output; remove it deliberately or choose another path."
        )
    official.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, official)


def analyse_etth2(
    run_dir: Path,
    *,
    etth2_config: dict[str, Any],
    pilot_config: dict[str, Any],
    gap_config: dict[str, Any],
    out_dir: Path | None = None,
    allow_mock_analysis: bool = False,
) -> dict[str, Any]:
    """Shared analysis, family-correct reporting, then the replication decision.

    Refuses a mock run by default. Writes everything to a staging directory,
    verifies it, and only then promotes it atomically to the official path.
    """
    run_dir = Path(run_dir)
    effective = build_effective_config(pilot_config, etth2_config)
    official = Path(out_dir) if out_dir else run_dir / "analysis"

    mock = run_is_mock(run_dir)
    if mock and not allow_mock_analysis:
        raise MockAnalysisRefused(
            f"REFUSING TO ANALYSE: {run_dir} was produced by the MOCK forecaster.\n"
            f"\n"
            f"{MOCK_NOT_A_FINDING}\n"
            f"\n"
            f"Analysing it would produce a classification that looks exactly like "
            f"a real one. If you are exercising the pipeline deliberately, pass "
            f"--allow-mock-analysis; every artifact will then be stamped "
            f"mock_model=true, scientifically_valid=false and "
            f"authoritative_result=false, and must never be reported."
        )

    staging = official.with_name(official.name + STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    payload = analyse(run_dir, effective, gap_config, out_dir=staging)

    # Snapshot BEFORE the descriptive rewrite, so "no number moved" is proved
    # against what analyse produced rather than against itself.
    snapshot = numeric_snapshot_of(payload)
    payload = apply_family_correct_reporting(payload, mock=mock)

    mean_clean_mae = float(payload["mean_clean_mae"])
    r_stats = gap_stats_from_rows(payload["r_contrasts"], mean_clean_mae=mean_clean_mae)
    b_stats = gap_stats_from_rows(payload["b_contrasts"], mean_clean_mae=mean_clean_mae)
    decision = classify_replication(
        ReplicationInputs(
            r_gaps=r_stats, b_gaps=b_stats,
            mean_clean_mae=mean_clean_mae, pilot_config=effective,
        )
    )

    payload["dataset"] = effective["dataset"]["name"]
    payload["experiment"] = etth2_config["meta"]["name"]
    payload["preregistration"] = etth2_config["meta"]["preregistration"]
    payload["n_origins_expected"] = int(effective["design"]["expected_origins"])
    payload["replication_decision"] = decision.as_dict()
    payload["primary_contrasts"] = list(PRIMARY_CONTRASTS)
    payload["secondary_stress_contrast"] = SECONDARY_STRESS_CONTRAST

    classification = decision.as_dict()
    classification["dataset"] = effective["dataset"]["name"]
    classification["experiment"] = etth2_config["meta"]["name"]
    classification["preregistration"] = etth2_config["meta"]["preregistration"]
    if mock:
        # Stamped on the CLASSIFICATION itself, not only on the payload around
        # it. A mock classification file copied out of context must still be
        # structurally identifiable as mock.
        classification.update(MOCK_STAMP)
        classification["label"] = f"{MOCK_LABEL_PREFIX}{classification['label']}"
        payload["replication_decision"] = classification

    verify_reporting(payload, snapshot, mock=mock)

    # Rewrite the corrected contrast tables into staging, then the payload.
    pd.DataFrame(payload["r_contrasts"]).to_csv(staging / "R_contrasts.csv", index=False)
    pd.DataFrame(payload["b_contrasts"]).to_csv(staging / "B_contrasts.csv", index=False)
    (staging / REPLICATION_OUTPUT).write_text(
        json.dumps(classification, indent=2, default=str), encoding="utf-8"
    )
    (staging / "etth2_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # analyse() ALSO wrote its own trailing_gap_analysis.json and
    # classification.json into staging, before the correction ran. Left as they
    # were, the promoted directory would contain official ETTh2 artifacts
    # carrying R's interpretation text on B rows — the exact ETTh1 defect,
    # sitting one file away from the corrected ones. Rewrite them from the
    # corrected payload, and stamp the ETTh1-rule classification so it cannot be
    # mistaken for this replication's outcome.
    (staging / "trailing_gap_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    inherited = json.loads((staging / "classification.json").read_text(encoding="utf-8"))
    inherited["note"] = (
        "This is the ETTh1 trailing-gap MECHANISM classification, produced by the "
        "shared analysis. It is NOT the replication outcome. The replication "
        f"outcome is in {REPLICATION_OUTPUT}."
    )
    inherited["is_the_replication_outcome"] = False
    if mock:
        inherited.update(MOCK_STAMP)
        inherited["label"] = f"{MOCK_LABEL_PREFIX}{inherited['label']}"
    (staging / "classification.json").write_text(
        json.dumps(inherited, indent=2, default=str), encoding="utf-8"
    )

    if mock:
        (staging / "MOCK_NOT_A_FINDING.txt").write_text(
            MOCK_NOT_A_FINDING + "\n", encoding="utf-8"
        )
        # EVERY json artifact, not only the ones written above. A stamp that
        # covered the headline files but missed dose_response.json would leave a
        # mock artifact that reads as real the moment it is opened on its own.
        stamp_every_json(staging)

    # Whole-directory sweep before promotion. Nothing correct-adjacent may ship
    # next to something wrong, and no artifact may leave staging unstamped.
    verify_staging_directory(staging, mock=mock)
    _promote(staging, official)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse the ETTh2 replication.")
    parser.add_argument("--config", default="configs/etth2_config.yaml")
    parser.add_argument("--pilot-config", default="configs/pilot_config.yaml")
    parser.add_argument("--gap-config", default="configs/trailing_gap_config.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--allow-mock-analysis", action="store_true",
        help=("Analyse a MOCK run. Pipeline exercise only. Every artifact is "
              "stamped mock_model=true, scientifically_valid=false and "
              "authoritative_result=false, and must never be reported."),
    )
    args = parser.parse_args()

    etth2 = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    pilot = yaml.safe_load((REPO_ROOT / args.pilot_config).read_text(encoding="utf-8"))
    gap = yaml.safe_load((REPO_ROOT / args.gap_config).read_text(encoding="utf-8"))

    run_dir = Path(args.run_dir or etth2["clean_control"]["formal_run_dir"])
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    try:
        payload = analyse_etth2(
            run_dir, etth2_config=etth2, pilot_config=pilot, gap_config=gap,
            out_dir=out_dir, allow_mock_analysis=args.allow_mock_analysis,
        )
    except MockAnalysisRefused as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 3
    except Etth2ReportingError as exc:
        print(f"REPORTING VERIFICATION FAILED, nothing promoted:\n{exc}",
              file=sys.stderr, flush=True)
        return 4

    if payload.get("mock_model"):
        print("=" * 78)
        print(MOCK_NOT_A_FINDING)
        print("=" * 78)

    print(f"dataset: {payload['dataset']}   origins: {payload['n_origins']}")
    print(f"mean clean MAE: {payload['mean_clean_mae']:.6f}   "
          f"SESOI delta: +/-{payload['sesoi_delta']:.6f}")
    print("\n--- R_g ---")
    for row in payload["r_contrasts"]:
        marker = "PRIMARY  " if row["contrast_id"] in PRIMARY_CONTRASTS else "secondary"
        print(
            f"  {marker} {row['contrast_id']:<7} {row['raw_mae_difference']:+.6f} "
            f"({row['pct_of_mean_clean_mae']:+6.2f}%)  Holm CI "
            f"[{row['holm_ci95_low']:+.5f},{row['holm_ci95_high']:+.5f}]  "
            f"90% [{row['ci90_low']:+.5f},{row['ci90_high']:+.5f}]  -> {row['reading']}"
        )
    print("\n--- B_g (secondary, confounded, never decides) ---")
    for row in payload["b_contrasts"]:
        print(f"  {row['contrast_id']:<7} {row['raw_mae_difference']:+.6f} -> {row['reading']}")

    decision = payload["replication_decision"]
    print(f"\nREPLICATION OUTCOME: {decision['label']}"
          + (f" / {decision['reason']}" if decision["reason"] else ""))
    print(f"  triggers: {decision['triggers']}")
    print(f"  rule_version: {decision['rule_version']}")
    print(f"\n{decision['criteria']['hypothesis_provenance']}")
    if payload.get("mock_model"):
        print(f"\n{MOCK_NOT_A_FINDING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
