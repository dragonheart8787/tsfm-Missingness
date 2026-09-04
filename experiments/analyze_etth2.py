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
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_trailing_gap import analyse  # noqa: E402
from experiments.run_etth2 import build_effective_config  # noqa: E402
from stats.mechanism_decision import GapStat  # noqa: E402
from stats.replication_decision import (  # noqa: E402
    PRIMARY_CONTRASTS,
    SECONDARY_STRESS_CONTRAST,
    ReplicationInputs,
    classify_replication,
)

REPLICATION_OUTPUT = "classification_etth2.json"


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


def analyse_etth2(
    run_dir: Path,
    *,
    etth2_config: dict[str, Any],
    pilot_config: dict[str, Any],
    gap_config: dict[str, Any],
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Shared analysis, then the ETTh2-specific replication decision."""
    effective = build_effective_config(pilot_config, etth2_config)
    out_dir = Path(out_dir) if out_dir else Path(run_dir) / "analysis"

    payload = analyse(Path(run_dir), effective, gap_config, out_dir=out_dir)

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
    payload["n_origins_expected"] = int(effective["design"]["expected_origins"])
    payload["replication_decision"] = decision.as_dict()
    payload["primary_contrasts"] = list(PRIMARY_CONTRASTS)
    payload["secondary_stress_contrast"] = SECONDARY_STRESS_CONTRAST

    (out_dir / REPLICATION_OUTPUT).write_text(
        json.dumps(decision.as_dict(), indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "etth2_analysis.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse the ETTh2 replication.")
    parser.add_argument("--config", default="configs/etth2_config.yaml")
    parser.add_argument("--pilot-config", default="configs/pilot_config.yaml")
    parser.add_argument("--gap-config", default="configs/trailing_gap_config.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    etth2 = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    pilot = yaml.safe_load((REPO_ROOT / args.pilot_config).read_text(encoding="utf-8"))
    gap = yaml.safe_load((REPO_ROOT / args.gap_config).read_text(encoding="utf-8"))

    run_dir = Path(args.run_dir or etth2["clean_control"]["formal_run_dir"])
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    payload = analyse_etth2(
        run_dir, etth2_config=etth2, pilot_config=pilot, gap_config=gap, out_dir=out_dir
    )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
