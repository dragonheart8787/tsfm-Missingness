"""The ETTh2 analysis layer: reuse the shared statistics, add only the decision.

The point of these tests is that ``analyze_etth2`` introduces no new statistical
machinery. It calls the ETTh1 ``analyse`` and then applies the frozen
replication rule to the rows that were WRITTEN TO DISK, so the decision is
computed from exactly the numbers delivered to the reviewer.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.analyze_etth2 import REPLICATION_OUTPUT, gap_stats_from_rows
from stats.mechanism_decision import EQUIVALENT, MATERIAL_POSITIVE, UNRESOLVED
from stats.replication_decision import (
    CONTRADICTED,
    INCONCLUSIVE,
    PRIMARY_CONTRASTS,
    REPLICATED,
    ReplicationInputs,
    classify_replication,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = {"decision": {"no_go": {"equivalence_band_frac_of_clean_mae": 0.03}}}
MEAN_CLEAN_MAE = 2.0
DELTA = 0.06


def _row(contrast_id: str, *, lo90, hi90, lo95, hi95) -> dict:
    """A row shaped exactly like analyse()'s emitted contrast rows."""
    return {
        "contrast_id": contrast_id, "family": contrast_id[0],
        "gap": int(contrast_id.split("_")[1]),
        "raw_mae_difference": 0.5 * (lo95 + hi95),
        "median_paired_difference": 0.5 * (lo95 + hi95),
        "holm_ci95_low": lo95, "holm_ci95_high": hi95,
        "ci90_low": lo90, "ci90_high": hi90,
        "holm_rejects": bool(lo95 > 0 or hi95 < 0),
    }


EQUIV = dict(lo90=-0.5 * DELTA, hi90=0.5 * DELTA, lo95=-0.5 * DELTA, hi95=0.5 * DELTA)
MATPOS = dict(lo90=2 * DELTA, hi90=4 * DELTA, lo95=2 * DELTA, hi95=4 * DELTA)
UNRES = dict(lo90=-3 * DELTA, hi90=3 * DELTA, lo95=-3 * DELTA, hi95=3 * DELTA)


def test_gap_stats_are_rebuilt_faithfully_from_the_emitted_rows():
    """The decision must read the delivered numbers, not a parallel copy."""
    rows = [_row("R_16", **EQUIV), _row("R_32", **MATPOS), _row("R_64", **UNRES)]
    stats = gap_stats_from_rows(rows, mean_clean_mae=MEAN_CLEAN_MAE)
    assert [s.contrast_id for s in stats] == ["R_16", "R_32", "R_64"]
    assert [s.gap for s in stats] == [16, 32, 64]
    assert [s.reading(DELTA) for s in stats] == [EQUIVALENT, MATERIAL_POSITIVE, UNRESOLVED]
    for stat, row in zip(stats, rows):
        assert stat.ci_low_equivalence == row["ci90_low"]
        assert stat.ci_high_equivalence == row["ci90_high"]
        assert stat.ci_low_holm == row["holm_ci95_low"]
        assert stat.ci_high_holm == row["holm_ci95_high"]
        assert stat.mean_clean_mae == MEAN_CLEAN_MAE


@pytest.mark.parametrize(
    "readings,expected",
    [
        ((EQUIV, EQUIV, EQUIV), REPLICATED),
        ((EQUIV, MATPOS, EQUIV), CONTRADICTED),
        ((EQUIV, EQUIV, UNRES), INCONCLUSIVE),
        ((MATPOS, UNRES, EQUIV), CONTRADICTED),
    ],
)
def test_the_end_to_end_row_to_outcome_path(readings, expected):
    rows = [_row(cid, **spec) for cid, spec in zip(PRIMARY_CONTRASTS, readings)]
    rows.append(_row("R_128", **UNRES))
    stats = gap_stats_from_rows(rows, mean_clean_mae=MEAN_CLEAN_MAE)
    decision = classify_replication(
        ReplicationInputs(
            r_gaps=stats, mean_clean_mae=MEAN_CLEAN_MAE, pilot_config=PILOT
        )
    )
    assert decision.label == expected


def test_the_equivalence_branch_reads_the_uncorrected_90_percent_interval():
    """The resolved multiplicity treatment, checked on the numbers.

    A gap whose NOMINAL 90% CI lies inside the band is EQUIVALENT even when its
    Holm-corrected 95% interval does not — that is exactly what "no additional
    correction on the intersection" means operationally.
    """
    row = _row("R_16", lo90=-0.5 * DELTA, hi90=0.5 * DELTA,
               lo95=-3 * DELTA, hi95=3 * DELTA)
    stat = gap_stats_from_rows([row], mean_clean_mae=MEAN_CLEAN_MAE)[0]
    assert stat.reading(DELTA) == EQUIVALENT
    # Widening only the Holm interval must not change the reading...
    wider = _row("R_16", lo90=-0.5 * DELTA, hi90=0.5 * DELTA,
                 lo95=-9 * DELTA, hi95=9 * DELTA)
    assert gap_stats_from_rows([wider], mean_clean_mae=MEAN_CLEAN_MAE)[0].reading(
        DELTA
    ) == EQUIVALENT
    # ...but widening the 90% interval past the band must.
    narrow_holm = _row("R_16", lo90=-2 * DELTA, hi90=2 * DELTA,
                       lo95=-0.1 * DELTA, hi95=0.1 * DELTA)
    assert gap_stats_from_rows([narrow_holm], mean_clean_mae=MEAN_CLEAN_MAE)[0].reading(
        DELTA
    ) == UNRESOLVED


def test_the_material_branch_reads_the_holm_corrected_interval():
    """Holm is RETAINED for material directional readings, as on ETTh1."""
    # 90% CI clears the band, but the Holm interval does not: not material.
    row = _row("R_16", lo90=2 * DELTA, hi90=4 * DELTA,
               lo95=-0.5 * DELTA, hi95=6 * DELTA)
    assert gap_stats_from_rows([row], mean_clean_mae=MEAN_CLEAN_MAE)[0].reading(
        DELTA
    ) == UNRESOLVED


def test_the_analysis_module_adds_no_statistical_machinery():
    """It orchestrates; the statistics come from the shared ETTh1 analysis."""
    source = (REPO_ROOT / "experiments" / "analyze_etth2.py").read_text(encoding="utf-8")
    for token in ("bootstrap_paired_" + "difference", "holm_" + "correct",
                  "np.percentile", "aggregate_" + "mae"):
        assert token not in source, f"{token} is reimplemented in analyze_etth2"
    from experiments.analyze_trailing_gap import analyse
    import experiments.analyze_etth2 as mod

    assert mod.analyse is analyse
    assert REPLICATION_OUTPUT == "classification_etth2.json"
