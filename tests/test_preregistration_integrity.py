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


# =========================================================================== #
# Final release-gate static regressions
#
# Each scan is paired with a proof that it would actually catch a
# reintroduction, so none of them can pass vacuously.
# =========================================================================== #

STALE_CONFIG_KEY = "clean_audit_tolerance_note"
STALE_AUDIT_PHRASE = "checksum/tolerance-audited"
# Assembled at runtime so this file does not itself contain the literal and
# trip its own repo-wide scan below.
STALE_LABEL = "INCONSISTENT" + "_ACROSS_GAPS"


# --- 1. the renamed config key must not come back ------------------------- #

def test_stale_clean_audit_tolerance_key_is_absent_from_the_config():
    """The key was renamed to clean_audit_exact_match_note.

    The old name implies a tolerance the audit does not have and never had.
    """
    text = GAP_CONFIG.read_text(encoding="utf-8")
    offenders = [
        f"{GAP_CONFIG.name}:{n}: {line.strip()}"
        for n, line in enumerate(text.splitlines(), start=1)
        if STALE_CONFIG_KEY in line
    ]
    assert not offenders, "the stale tolerance key reappeared:\n  " + "\n  ".join(offenders)


def test_the_renamed_key_is_present_and_states_exact_matching():
    config = yaml.safe_load(GAP_CONFIG.read_text(encoding="utf-8"))
    note = config["preconditions_for_execution"]["clean_audit_exact_match_note"]
    assert "EXACT" in note
    assert "no numerical tolerance" in note.lower()
    assert STALE_CONFIG_KEY not in config["preconditions_for_execution"]


def test_the_stale_key_scan_would_catch_a_reintroduction(tmp_path):
    """Non-vacuity: the scan's own predicate fires on a planted violation."""
    planted = tmp_path / "cfg.yaml"
    planted.write_text(
        f"preconditions_for_execution:\n  {STALE_CONFIG_KEY}: >-\n    something\n",
        encoding="utf-8",
    )
    offenders = [
        line for line in planted.read_text(encoding="utf-8").splitlines()
        if STALE_CONFIG_KEY in line
    ]
    assert offenders, "the scan predicate failed to detect a planted stale key"


# --- 2. the tolerance-audit phrase must not appear in operative text ------- #

def _operative_files_for_audit_phrase() -> list[tuple[str, str]]:
    """(label, operative text) for every file the phrase must stay out of.

    The preregistration contributes only its operative body: Appendix A
    legitimately records that the phrase WAS used and was replaced, and that
    historical note must survive.
    """
    return [
        (GAP_CONFIG.name, GAP_CONFIG.read_text(encoding="utf-8")),
        (PREREG.name, _operative_text(PREREG)),
        (
            "gpu_execution_runbook_trailing_gap.md",
            (REPO_ROOT / "docs" / "gpu_execution_runbook_trailing_gap.md").read_text(
                encoding="utf-8"
            ),
        ),
    ]


def test_checksum_tolerance_phrase_is_absent_from_operative_config_and_docs():
    offenders = [
        f"{label}: {line.strip()}"
        for label, text in _operative_files_for_audit_phrase()
        for line in text.splitlines()
        if STALE_AUDIT_PHRASE in line
    ]
    assert not offenders, (
        "the tolerance-audit phrasing reappeared in operative text:\n  "
        + "\n  ".join(offenders)
    )


def test_the_historical_mention_survives_in_the_appendix():
    """The scan must not be satisfied by deleting the audit trail."""
    text = PREREG.read_text(encoding="utf-8")
    appendix = text[text.index(APPENDIX_HEADING):]
    assert "checksum/tolerance" in appendix, (
        "Appendix A must keep the record of what amendment 3 replaced"
    )


def test_the_audit_phrase_scan_would_catch_a_reintroduction(tmp_path):
    """Non-vacuity, including that the appendix exemption is not a blanket one."""
    planted = tmp_path / "doc.md"
    planted.write_text(
        f"## 10. Preconditions\n\nPredictions are {STALE_AUDIT_PHRASE} against the "
        f"original run.\n\n{APPENDIX_HEADING}\n\nhistorical {STALE_AUDIT_PHRASE} note\n",
        encoding="utf-8",
    )
    operative = _operative_text(planted)
    assert STALE_AUDIT_PHRASE in operative, "a violation in the body must be visible"
    appendix_only = planted.read_text(encoding="utf-8")[
        planted.read_text(encoding="utf-8").index(APPENDIX_HEADING):
    ]
    assert STALE_AUDIT_PHRASE in appendix_only, "the appendix copy is correctly exempt"


# --- 3. the superseded classification label ------------------------------- #

def test_stale_classification_label_is_absent_from_the_operative_body():
    """The frozen label is DIRECTION_REVERSAL_ACROSS_GAPS."""
    text = _operative_text(PREREG)
    offenders = [
        f"{PREREG.name}:{n}: {line.strip()}"
        for n, line in enumerate(text.splitlines(), start=1)
        if STALE_LABEL in line
    ]
    assert not offenders, (
        "the superseded label reappeared in the operative body:\n  "
        + "\n  ".join(offenders)
    )


def test_the_frozen_label_is_the_one_the_document_uses():
    text = _operative_text(PREREG)
    assert "DIRECTION_REVERSAL_ACROSS_GAPS" in text
    # ...and it is the label the implementation actually emits.
    from stats.mechanism_decision import DIRECTION_REVERSAL

    assert DIRECTION_REVERSAL == "DIRECTION_REVERSAL_ACROSS_GAPS"
    assert DIRECTION_REVERSAL in text


def test_the_stale_label_scan_would_catch_a_reintroduction(tmp_path):
    """Non-vacuity: a body occurrence is caught, an appendix one is exempt."""
    planted = tmp_path / "doc.md"
    planted.write_text(
        f"## 9. Rule\n\nreadings disagree -> `{STALE_LABEL}`\n\n"
        f"{APPENDIX_HEADING}\n\nformerly `{STALE_LABEL}`\n",
        encoding="utf-8",
    )
    operative = _operative_text(planted)
    assert STALE_LABEL in operative, "a violation in the body must be visible"
    assert operative.count(STALE_LABEL) == 1, "only the body occurrence is in scope"


def test_no_stale_label_anywhere_in_the_implementation():
    """The code must not carry the interim name either."""
    offenders = []
    for pattern in ("*.py", "*.yaml"):
        for path in REPO_ROOT.rglob(pattern):
            if {".venv", ".git"} & set(path.relative_to(REPO_ROOT).parts):
                continue
            if STALE_LABEL in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"the superseded label persists in: {offenders}"
