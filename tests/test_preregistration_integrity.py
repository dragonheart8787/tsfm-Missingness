"""The preregistration and its config must not carry obsolete status language.

Contradictory historical text left live alongside a frozen rule is an
operational hazard: a reader following the document top-to-bottom would act on
a superseded proposal. Frozen text lives in the numbered sections; superseded
text lives ONLY in the clearly-labelled appendix and is never operative.

Static scan, in the established style. Model-mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG = REPO_ROOT / "docs" / "preregistration_trailing_gap_mechanism_v1.md"
GAP_CONFIG = REPO_ROOT / "configs" / "trailing_gap_config.yaml"

BANNED_STATUS = ("AWAITING SIGN-OFF", "DRAFT", "PROPOSAL", "NOT AUTHORITATIVE")

# Everything from this heading onward is the quarantined history.
APPENDIX_HEADING = "## 14. Appendix A — Superseded draft history"


def _operative_text(path: Path) -> str:
    """The document with the superseded-history appendix removed."""
    text = path.read_text(encoding="utf-8")
    if APPENDIX_HEADING in text:
        text = text[: text.index(APPENDIX_HEADING)]
    return text


def test_no_banned_status_string_in_the_operative_preregistration():
    text = _operative_text(PREREG)
    offenders = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for banned in BANNED_STATUS:
            if banned in line:
                offenders.append(f"{PREREG.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "obsolete status language outside the superseded-history appendix:\n  "
        + "\n  ".join(offenders)
    )


def test_no_banned_status_string_anywhere_in_the_gap_config():
    """The config is embedded in the run manifest; it must read frozen."""
    text = GAP_CONFIG.read_text(encoding="utf-8")
    offenders = [
        f"{GAP_CONFIG.name}:{n}: {line.strip()}"
        for n, line in enumerate(text.splitlines(), start=1)
        for banned in BANNED_STATUS
        if banned in line
    ]
    assert not offenders, "obsolete status language in the config:\n  " + "\n  ".join(offenders)


def test_config_status_fields_read_frozen():
    config = yaml.safe_load(GAP_CONFIG.read_text(encoding="utf-8"))
    assert "frozen" in config["meta"]["status"].lower()
    assert "frozen" in config["mechanism_rule"]["status"].lower()


def test_the_superseded_appendix_exists_and_is_marked_non_operative():
    text = PREREG.read_text(encoding="utf-8")
    assert APPENDIX_HEADING in text
    appendix = text[text.index(APPENDIX_HEADING):]
    assert "NOT OPERATIVE" in appendix
    assert "the body governs" in appendix
    # It must actually hold the superseded content, not be an empty stub.
    assert "three-way" in appendix
    assert "Replaced" in appendix


def test_the_scan_would_catch_a_reintroduced_draft_marker(tmp_path):
    """Guard against a vacuous scan."""
    sample = tmp_path / "sample.md"
    sample.write_text(
        "## 5.1 Readings\n\nStatus: AWAITING SIGN-OFF\n\n"
        f"{APPENDIX_HEADING}\n\nold text with DRAFT in it\n",
        encoding="utf-8",
    )
    operative = _operative_text(sample)
    assert "AWAITING SIGN-OFF" in operative, "the scan must see the operative body"
    assert "DRAFT" not in operative, "the appendix must be excluded from the scan"


def test_no_correction_note_is_appended_after_the_operative_sections():
    """A correction appended at the end is the pattern this replaced.

    Section 13 must be an amendment RECORD, not a restatement of rules that
    contradicts the sections above it.
    """
    text = _operative_text(PREREG)
    assert "## 13. Amendment record" in text
    assert "## 13. Frozen for execution" not in text
    section_13 = text[text.index("## 13. Amendment record"):]
    # The record points at the sections; it does not redefine them.
    assert "Amendment" in section_13
    assert "min_consistent_gaps = 3" not in section_13


def test_header_declares_the_document_frozen():
    text = PREREG.read_text(encoding="utf-8")
    header = text[: text.index("## 0.")] if "## 0." in text else text[:3000]
    assert "FROZEN AND PREREGISTERED" in header
    assert "Appendix A" in header


def test_module_and_rule_status_are_frozen():
    from stats.mechanism_decision import MECHANISM_RULE_VERSION, RULE_STATUS

    assert MECHANISM_RULE_VERSION == "preregistered-v1"
    assert "frozen before GPU execution" in RULE_STATUS
    module = (REPO_ROOT / "experiments" / "trailing_gap.py").read_text(encoding="utf-8")
    assert "FROZEN AND PREREGISTERED" in module
    for banned in BANNED_STATUS:
        assert banned not in module, f"{banned} still present in experiments/trailing_gap.py"


def test_section_6_no_longer_claims_a_forbidden_literal_test():
    """§6's old claim became false once the docstring mentioned the formula."""
    text = _operative_text(PREREG)
    assert "tests assert the `0.03`\nliteral appears in neither" not in text
    assert "test_sesoi_delta_follows_a_mutated_pilot_config" in text


def test_section_10_describes_an_exact_audit_with_no_tolerance():
    text = _operative_text(PREREG)
    section = text[text.index("## 10. Preconditions for execution"):]
    assert "EXACT match, no" in section
    assert "identity before values" in section.lower()
    assert "no tolerance mechanism" in section
    assert "checksum/tolerance-audited" not in section
