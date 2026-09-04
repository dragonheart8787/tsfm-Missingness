"""The full ETTh2 phase sequence, INCLUDING an interrupted and resumed formal-full.

This is the sequence the gate broke before E3: an interrupted formal-full found
its own prior legitimate progress reported as a gate violation, because the gate
was bound to whole-file bytes and formal-full appends to those files by design.

The test runs the real sequence end to end on the mock forecaster:

    clean-reference -> formal-clean -> audit -> PARTIAL formal-full -> RESUME

and asserts the gate reports valid at every step, including immediately after the
resume, and that the finished matrix is exactly 2,314 distinct cells with no
duplicates.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.clean_reference_audit import (
    CleanReferenceAuditFailure,
    clean_content_digest,
    verify_gate_still_binds,
)
from experiments.run_etth2 import (
    GATE_FILENAME,
    PHASE_AUDIT,
    PHASE_CLEAN_REFERENCE,
    PHASE_FORMAL_CLEAN,
    PHASE_FORMAL_FULL,
    run_phase,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINS = 178
CONDITIONS = 13
TOTAL_CELLS = ORIGINS * CONDITIONS          # 2314


@pytest.fixture(scope="module")
def etth2_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def gap_config() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / "configs" / "trailing_gap_config.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(autouse=True)
def _needs_etth2(etth2_config):
    if not (REPO_ROOT / etth2_config["dataset"]["local_path"]).exists():
        pytest.skip("ETTh2 not fetched; run data/fetch_etth2.py")


def _gate_digests(formal: Path) -> dict:
    return json.loads((formal / GATE_FILENAME).read_text(encoding="utf-8"))[
        "clean_content_digests"
    ]


def test_the_full_sequence_survives_an_interruption_and_resume(
    tmp_path, etth2_config, config, gap_config
):
    reference, formal = tmp_path / "clean_reference", tmp_path / "formal"
    run = lambda phase, **kw: run_phase(
        phase=phase, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, reference_dir=reference, formal_dir=formal, **kw
    )

    # 1-2. The two independent clean passes.
    assert run(PHASE_CLEAN_REFERENCE, mock=True)["cells_completed"] == ORIGINS
    assert run(PHASE_FORMAL_CLEAN, mock=True)["cells_completed"] == ORIGINS

    # 3. Audit -> gate.
    audit = run(PHASE_AUDIT)
    assert audit["passed"] is True
    digests = _gate_digests(formal)
    verify_gate_still_binds(formal, reference, digests)

    # 4. PARTIAL formal-full: interrupted after 10 origins.
    partial = run(PHASE_FORMAL_FULL, mock=True, limit_origins=10)
    # 178 clean already written + 10 origins x 12 remaining conditions.
    assert partial["cells_completed"] == ORIGINS + 10 * (CONDITIONS - 1)
    assert partial["cells_completed"] < TOTAL_CELLS

    # THE REGRESSION: the gate must still bind over the run's own progress.
    verify_gate_still_binds(formal, reference, digests)

    # 5. RESUME. Under the whole-file digest this raised CleanReferenceAuditFailure.
    resumed = run(PHASE_FORMAL_FULL, mock=True)

    # ...and the gate is still valid immediately after the resume.
    verify_gate_still_binds(formal, reference, digests)

    assert resumed["result_rows"] == TOTAL_CELLS
    assert resumed["distinct_cells"] == TOTAL_CELLS
    assert resumed["origins_fully_completed"] == ORIGINS
    assert resumed["rows_by_status"] == {"ok": TOTAL_CELLS}
    assert resumed["max_distinct_target_digests_per_origin"] == 1

    frame = pd.read_csv(formal / "window_results.csv")
    assert len(frame) == TOTAL_CELLS
    assert not frame.duplicated(subset=["origin_id", "condition_id"]).any()
    assert frame["origin_id"].nunique() == ORIGINS
    assert sorted(frame.groupby("origin_id").size().unique().tolist()) == [CONDITIONS]

    # The clean projection is byte-stable across the whole sequence.
    assert clean_content_digest(formal) == digests["formal"]
    assert clean_content_digest(reference) == digests["reference"]


def test_the_resumed_run_did_not_re_forecast_the_clean_cells(
    tmp_path, etth2_config, config, gap_config
):
    """Resumption is at origin x condition granularity, so clean stays put."""
    reference, formal = tmp_path / "clean_reference", tmp_path / "formal"
    run = lambda phase, **kw: run_phase(
        phase=phase, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, reference_dir=reference, formal_dir=formal, **kw
    )
    run(PHASE_CLEAN_REFERENCE, mock=True)
    run(PHASE_FORMAL_CLEAN, mock=True)
    run(PHASE_AUDIT)
    before = clean_content_digest(formal)

    run(PHASE_FORMAL_FULL, mock=True, limit_origins=5)
    run(PHASE_FORMAL_FULL, mock=True)

    assert clean_content_digest(formal) == before
    frame = pd.read_csv(formal / "window_results.csv")
    assert (frame["condition_id"] == "clean").sum() == ORIGINS, "clean rows duplicated"


def test_a_gate_from_before_the_projection_is_refused_not_silently_accepted(
    tmp_path, etth2_config, config, gap_config
):
    """An E2-format gate records whole-file digests and cannot be verified here."""
    from experiments.run_etth2 import ReplicationGateError

    reference, formal = tmp_path / "clean_reference", tmp_path / "formal"
    run = lambda phase, **kw: run_phase(
        phase=phase, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, reference_dir=reference, formal_dir=formal, **kw
    )
    run(PHASE_CLEAN_REFERENCE, mock=True, limit_origins=3)
    run(PHASE_FORMAL_CLEAN, mock=True, limit_origins=3)
    gate_path = formal / GATE_FILENAME
    gate_path.write_text(json.dumps({
        "passed": True, "n_failures": 0,
        "dataset_sha256": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        "model_revision": "mock-revision",
        "clean_artifact_digests": {"formal": {}, "reference": {}},   # E2 format
    }), encoding="utf-8")

    with pytest.raises(ReplicationGateError, match="predates the clean-content"):
        run(PHASE_FORMAL_FULL, mock=True)


def test_clean_row_tampering_still_fails_across_the_resume_boundary(
    tmp_path, etth2_config, config, gap_config
):
    """The fix must not have over-corrected into permissiveness."""
    reference, formal = tmp_path / "clean_reference", tmp_path / "formal"
    run = lambda phase, **kw: run_phase(
        phase=phase, etth2_config=etth2_config, pilot_config=config,
        gap_config=gap_config, reference_dir=reference, formal_dir=formal, **kw
    )
    run(PHASE_CLEAN_REFERENCE, mock=True, limit_origins=4)
    run(PHASE_FORMAL_CLEAN, mock=True, limit_origins=4)
    from experiments.clean_reference_audit import audit_clean_reference
    from experiments.run_etth2 import write_gate

    report = audit_clean_reference(
        formal, reference, expected_origins=4, expected_horizon=96
    )
    assert report.passed, report.failures
    write_gate(
        formal, report,
        dataset_sha256="a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        model_revision="mock-revision",
    )
    digests = _gate_digests(formal)

    # Partial progress is fine...
    run(PHASE_FORMAL_FULL, mock=True, limit_origins=2)
    verify_gate_still_binds(formal, reference, digests)

    # ...but touching one clean prediction is not.
    path = formal / "predictions_long.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    clean = frame.index[frame["condition_id"] == "clean"]
    frame.loc[clean[0], "median_prediction"] = "42.0"
    frame.to_csv(path, index=False)
    with pytest.raises(CleanReferenceAuditFailure, match="CLEAN ROWS.*CHANGED"):
        verify_gate_still_binds(formal, reference, digests)
