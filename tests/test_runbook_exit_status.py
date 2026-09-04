"""Runbook pipelines must propagate the PYTHON command's exit status, not tee's.

`cmd | tee log` makes `$?` the status of **tee**, which is 0 even when cmd
failed. A runbook whose hard stops are checked with `$?` would then sail past a
failure. This is the same class of bug fixed in the trailing-gap runbook's §3.0
guard (D2 -> D3), checked here across every pipeline in the ETTh2 document.

A static scan for the string "PIPESTATUS" would prove nothing: it could appear
in a comment, be misspelled, or sit in a block whose status is still masked by a
trailing echo. So these tests EXTRACT each bash block from the document, run it
verbatim against a deliberately failing command, and assert on the real exit
code observed.

No inference, no GPU.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = REPO_ROOT / "docs" / "gpu_execution_runbook_etth2.md"

BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def bash_blocks() -> list[str]:
    return BASH_BLOCK.findall(RUNBOOK.read_text(encoding="utf-8"))


def pipeline_blocks() -> list[str]:
    """Blocks that pipe a python command into tee — the ones at risk."""
    return [b for b in bash_blocks() if "| tee" in b]


def run_bash(script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script], cwd=cwd, capture_output=True, text=True, timeout=120
    )


def test_the_runbook_actually_contains_pipelines():
    """Non-vacuity: if the extraction found nothing, every test below is empty."""
    blocks = pipeline_blocks()
    assert len(blocks) == 7, [b.splitlines()[0] for b in blocks]


@pytest.mark.parametrize("index", range(7))
def test_each_pipeline_propagates_a_failure(tmp_path, index):
    """THE behavioral test: substitute a failing command, observe the real status.

    The block is run verbatim except that the `.venv/bin/python ...` invocation
    is replaced by a command that exits 7. If the status were tee's, this would
    come back 0.
    """
    block = pipeline_blocks()[index]
    (tmp_path / "handoff").mkdir(exist_ok=True)
    # Replace the python invocation (possibly line-continued) with `exit 7`.
    failing = re.sub(
        r"\.venv/bin/python[^\n]*(?:\\\n[^\n]*)*(?= 2>&1 \| tee| \| tee)",
        "bash -c 'exit 7'",
        block,
    )
    assert ".venv/bin/python" not in failing, failing
    assert "| tee" in failing, failing

    result = run_bash(failing, tmp_path)
    assert result.returncode == 7, (
        f"pipeline {index} returned {result.returncode}, not the command's 7 — "
        f"the status is being masked.\nBLOCK:\n{failing}\nSTDOUT:{result.stdout}"
    )
    assert "exit status: 7" in result.stdout


@pytest.mark.parametrize("index", range(7))
def test_each_pipeline_still_reports_success_when_the_command_succeeds(tmp_path, index):
    """Non-vacuity the other way: it must not report failure unconditionally."""
    block = pipeline_blocks()[index]
    (tmp_path / "handoff").mkdir(exist_ok=True)
    passing = re.sub(
        r"\.venv/bin/python[^\n]*(?:\\\n[^\n]*)*(?= 2>&1 \| tee| \| tee)",
        "bash -c 'echo fine; exit 0'",
        block,
    )
    result = run_bash(passing, tmp_path)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "exit status: 0" in result.stdout


@pytest.mark.parametrize("index", range(7))
def test_each_pipeline_still_writes_its_log_on_failure(tmp_path, index):
    """The transcript must survive a failure — it is the evidence of what broke."""
    block = pipeline_blocks()[index]
    (tmp_path / "handoff").mkdir(exist_ok=True)
    failing = re.sub(
        r"\.venv/bin/python[^\n]*(?:\\\n[^\n]*)*(?= 2>&1 \| tee| \| tee)",
        "bash -c 'echo boom >&2; exit 7'",
        block,
    )
    run_bash(failing, tmp_path)
    logs = list((tmp_path / "handoff").glob("*.txt"))
    assert logs, "the tee log was not written"


def test_a_naive_pipeline_would_have_masked_the_failure(tmp_path):
    """Demonstrates the bug being prevented, so the tests above mean something."""
    naive = "bash -c 'exit 7' 2>&1 | tee handoff/x.txt"
    (tmp_path / "handoff").mkdir(exist_ok=True)
    result = run_bash(naive, tmp_path)
    assert result.returncode == 0, (
        "the naive form did NOT mask the failure; the premise of this fix is wrong"
    )


def test_no_pipeline_block_ends_with_a_status_masking_command():
    """A trailing `echo` after the subshell would replace the status with its 0."""
    for block in pipeline_blocks():
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        assert lines[-1].strip() == ")", (
            f"block ends with {lines[-1]!r}, which would mask the subshell's status"
        )


def test_every_tee_pipeline_in_the_document_is_covered():
    """No pipeline may exist outside the hardened blocks."""
    def tee_command_lines(text: str) -> list[str]:
        """Lines that actually pipe into tee, excluding comments about it."""
        return [
            line for line in text.splitlines()
            if "| tee" in line and not line.lstrip().startswith("#")
        ]

    text = RUNBOOK.read_text(encoding="utf-8")
    total = tee_command_lines(text)
    covered = [line for b in pipeline_blocks() for line in tee_command_lines(b)]
    assert len(total) == len(covered) == 7, (len(total), len(covered), total)
    for block in pipeline_blocks():
        assert "PIPESTATUS[0]" in block
