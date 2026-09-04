"""analysis_reporting_v2 — corrected B_g comparison-arm description.

    *** REPORTING ERRATUM ONLY. NO NUMBER CHANGES. ***

What was wrong
--------------
``experiments/analyze_trailing_gap.py`` writes ``INTERPRETATION[reading]`` into
the ``interpretation`` column of BOTH ``R_contrasts.csv`` and
``B_contrasts.csv``. Those texts were authored for the R family, and two of the
four name or imply R's comparison arm:

  EQUIVALENT        "...consistent with a horizon-dominant explanation..."
  MATERIAL_POSITIVE "...higher mean error than shorter-context/longer-horizon
                     single-shot forecasting..."

That arm is ``truncated_long``. It is R_g's comparison, not B_g's. B_g's arm
has always been ``internal_block(g, d=16)``:

    R_g = MAE(trailing_nan(g)) - MAE(truncated_long(g))
    B_g = MAE(trailing_nan(g)) - MAE(internal_block(g, d=16))

The COMPUTATION was always correct — ``analyze_trailing_gap.py`` builds
``b_series`` from ``internal_block_g{g}`` — so no reported number, interval,
p-value, reading, SESOI, dose-response result or classification is affected.
The error is confined to the prose attached to B rows.

How this module fixes it
------------------------
Additively, and WITHOUT touching the code that produces the numbers. It reads a
completed ``analysis/`` directory and writes a parallel
``analysis_reporting_v2/`` directory with:

  * B rows carrying family-correct interpretation text;
  * an explicit ``comparison_arm`` column on every row, R and B alike, so the
    arm can never again be left implicit;
  * a ``reporting_version`` marker.

Every other value is carried across UNCHANGED. CSV cells are read as strings
(``dtype=str``), so non-description cells are copied byte-for-byte and cannot
be perturbed by a float round-trip. ``compare_reporting_versions`` proves it.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REPORTING_VERSION = "analysis_reporting_v2"
ERRATUM_ID = "ERRATUM-2026-09-04-bg-comparison-arm"
ERRATUM_DOC = "docs/erratum_2026-09-04_bg_comparison_arm.md"

# The comparison arm of each family, stated explicitly rather than implied.
COMPARISON_ARM: dict[str, str] = {
    "R": "truncated_long(g) — shorter context of L-g with a longer H+g single-shot request",
    "B": "internal_block(g, d=16) — an equal-length gap held one full patch off the boundary",
}

# Family-correct interpretation text. The R entries are IDENTICAL to
# stats.mechanism_decision.INTERPRETATION (verified by a test); only the B
# entries are new, and they name B's actual arm.
INTERPRETATION_B: dict[str, str] = {
    "EQUIVALENT": (
        "no practically meaningful performance difference was established between a "
        "trailing gap and an equal-length internal block held one full patch off the "
        "forecast boundary. This does NOT demonstrate that gap position is "
        "irrelevant, and the comparison remains confounded with differences in "
        "removed content between the two arms."
    ),
    "MATERIAL_POSITIVE": (
        "explicit trailing-gap encoding has higher mean error than an equal-length "
        "internal block held one full patch off the forecast boundary, by more than "
        "the SESOI. This is CONFOUNDED WITH DIFFERENCES IN REMOVED CONTENT between "
        "the two arms: it does not isolate boundary position, masking, "
        "normalization, positional handling, or any other internal component."
    ),
    "MATERIAL_NEGATIVE": (
        "lower error was observed with a trailing gap than with an equal-length "
        "internal block held off the boundary. Do not call this a beneficial causal "
        "mechanism; the arms differ in removed content as well as in position."
    ),
    "UNRESOLVED": (
        "neither equivalence nor a material directional difference was established "
        "at this gap length."
    ),
}

# The erratum note, defined ONCE. The builder writes this exact string and the
# verifier requires exact equality against it. A note that merely exists, or
# that paraphrases this, is rejected: an erratum whose own explanatory text can
# drift is not an auditable record.
ERRATUM_NOTE = (
    "Reporting erratum only: B_g's comparison arm was described using the "
    "R_g interpretation text, which names truncated_long. B_g's arm is and "
    "always was internal_block(g, d=16). No number, interval, p-value, "
    "reading, SESOI, dose-response result or classification is affected. "
    f"See {ERRATUM_DOC}."
)

# Columns a v2 row is permitted to differ in or to add. Everything else must
# carry across byte-for-byte.
DESCRIPTION_COLUMNS = ("interpretation",)
ADDED_COLUMNS = ("comparison_arm", "reporting_version", "erratum_id")
# Keys the v2 payload may add at the JSON root. Nothing else, anywhere.
ADDED_ROOT_KEYS = ("reporting_version", "erratum_id", "erratum_note")
# Keys the v2 payload may add inside a contrast row.
ADDED_ROW_KEYS = ("comparison_arm",)

# Exact expected values for every whitelisted root addition. Presence is not
# enough; each must match.
EXPECTED_ROOT_VALUES: dict[str, str] = {
    "reporting_version": REPORTING_VERSION,
    "erratum_id": ERRATUM_ID,
    "erratum_note": ERRATUM_NOTE,
}

# The ONLY file that may carry root-level or contrast-row additions. The other
# two analysis JSONs are pure pass-through.
PAYLOAD_FILE = "trailing_gap_analysis.json"

# The whitelist is PATH-SPECIFIC, not name-based. `comparison_arm` is permitted
# at exactly these locations inside trailing_gap_analysis.json:
#
#     r_contrasts[i]        b_contrasts[i]
#
# and nowhere else. A key called `comparison_arm` appearing in
# classification.json, in dose_response.json, or in any other nested object —
# `classification.comparison_arm`, say — is a VIOLATION, not a permitted
# addition that happened to share a name.
CONTRAST_ROW_PATH = re.compile(r"^[rb]_contrasts\[\d+\]$")
# Likewise for the one rewritten leaf: only B's interpretation, only inside a
# b_contrasts row.
B_INTERPRETATION_PATH = re.compile(r"^b_contrasts\[\d+\]\.interpretation$")

# Source artifacts that must all be present. A missing one is a hard failure,
# never a silently skipped comparison: an identity check that quietly compares
# four files instead of five proves nothing about the fifth.
REQUIRED_SOURCE_FILES = (
    "R_contrasts.csv",
    "B_contrasts.csv",
    "trailing_gap_analysis.json",
    "classification.json",
    "dose_response.json",
)


class ReportingV2Error(RuntimeError):
    """The source analysis directory is not in a state this can correct."""


@dataclass
class IdentityReport:
    """Field-by-field comparison of an original analysis against its v2."""

    identical: bool = False
    files_compared: list[str] = field(default_factory=list)
    values_compared: int = 0
    numeric_fields_compared: int = 0
    # A REWRITTEN value: B's interpretation text. This is the erratum itself.
    interpretation_changes: list[str] = field(default_factory=list)
    passthrough_files_compared: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    # --- ADDITIONS, counted per category ------------------------------------ #
    # Additions are a different category from a rewrite: nothing that existed
    # before was altered. They are reported per category and never as one
    # number, because no single count is the total. In particular
    # ``json_contrast_arm_additions`` is 8 on the real ETTh1 data — four
    # ``r_contrasts[i].comparison_arm`` and four ``b_contrasts[i].comparison_arm``
    # — and that 8 is NOT the total of everything added. The CSV column
    # additions and the JSON root additions are separate, and larger.
    json_contrast_arm_additions: list[str] = field(default_factory=list)
    json_root_additions: list[str] = field(default_factory=list)
    csv_column_additions: list[str] = field(default_factory=list)
    csv_added_cells: int = 0

    def total_additions(self) -> int:
        """Every added value, across every file and category."""
        return (
            len(self.json_contrast_arm_additions)
            + len(self.json_root_additions)
            + self.csv_added_cells
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "files_compared": self.files_compared,
            "values_compared": self.values_compared,
            "numeric_fields_compared": self.numeric_fields_compared,
            "passthrough_files_compared": self.passthrough_files_compared,
            "n_interpretation_changes": len(self.interpretation_changes),
            "interpretation_changes": self.interpretation_changes,
            "additions": {
                "note": (
                    "Counted per category. No single count below is the total; "
                    "total_additions is."
                ),
                "n_json_contrast_arm_additions": len(self.json_contrast_arm_additions),
                "json_contrast_arm_additions": self.json_contrast_arm_additions,
                "n_json_root_additions": len(self.json_root_additions),
                "json_root_additions": self.json_root_additions,
                "n_csv_column_additions": len(self.csv_column_additions),
                "csv_column_additions": self.csv_column_additions,
                "n_csv_added_cells": self.csv_added_cells,
                "total_additions": self.total_additions(),
            },
            "violations": self.violations,
        }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def _correct_contrast_csv(path: Path, out_path: Path, *, family: str) -> None:
    """Copy a contrasts CSV, correcting only B's interpretation text.

    Read as strings so every non-description cell is carried across verbatim;
    no float parses, so no round-trip can perturb a number.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "reading" not in frame.columns:
        raise ReportingV2Error(f"{path} has no 'reading' column")

    if family == "B":
        missing = sorted(set(frame["reading"]) - set(INTERPRETATION_B))
        if missing:
            raise ReportingV2Error(f"{path}: unknown reading(s) {missing}")
        frame["interpretation"] = frame["reading"].map(INTERPRETATION_B)

    frame["comparison_arm"] = COMPARISON_ARM[family]
    frame["reporting_version"] = REPORTING_VERSION
    frame["erratum_id"] = ERRATUM_ID
    frame.to_csv(out_path, index=False)


def require_source_files(analysis_dir: Path, *, what: str = "source") -> None:
    """Every one of REQUIRED_SOURCE_FILES must be present. Hard failure if not.

    Checked individually and reported by name, so the failure says which file is
    absent rather than that "something" is. Skipping a missing artifact would
    make both the build and the identity check vacuous for that artifact: a
    comparison that never opens ``dose_response.json`` proves nothing about it,
    yet would still report ``identical=True``.
    """
    missing = [n for n in REQUIRED_SOURCE_FILES if not (Path(analysis_dir) / n).is_file()]
    if missing:
        raise ReportingV2Error(
            f"required {what} artifact(s) missing from {analysis_dir}: "
            f"{', '.join(missing)}. All of {', '.join(REQUIRED_SOURCE_FILES)} must "
            f"be present; a partial analysis directory cannot be corrected or verified."
        )


def build_reporting_v2(analysis_dir: Path, out_dir: Path | None = None) -> Path:
    """Emit analysis_reporting_v2/ beside an existing analysis/ directory.

    The source directory is never modified.
    """
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir else analysis_dir.parent / REPORTING_VERSION
    if not analysis_dir.is_dir():
        raise ReportingV2Error(f"{analysis_dir} is not a directory")
    require_source_files(analysis_dir)
    # Same discipline as the runbook's initialization guard: never write into a
    # directory that could retain stale files from an earlier run, which would
    # silently mix two v2 builds and defeat the pass-through byte check.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ReportingV2Error(
            f"{out_dir} already exists and is not empty. Refusing to write into it: "
            f"stale files from a previous build would survive and be compared as if "
            f"they were this build's. Remove it deliberately, or choose another path."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, family in (("R_contrasts.csv", "R"), ("B_contrasts.csv", "B")):
        _correct_contrast_csv(analysis_dir / name, out_dir / name, family=family)

    # trailing_gap_analysis.json carries the same b_contrasts prose. It is a
    # required source, so this is unconditional: a build that silently omitted
    # it would emit a v2 whose payload still carried the wrong arm.
    payload_path = analysis_dir / PAYLOAD_FILE
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    for row in payload.get("b_contrasts", []):
        row["interpretation"] = INTERPRETATION_B[row["reading"]]
        row["comparison_arm"] = COMPARISON_ARM["B"]
    for row in payload.get("r_contrasts", []):
        row["comparison_arm"] = COMPARISON_ARM["R"]
    payload["reporting_version"] = REPORTING_VERSION
    payload["erratum_id"] = ERRATUM_ID
    payload["erratum_note"] = ERRATUM_NOTE
    (out_dir / PAYLOAD_FILE).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    # Everything else is copied through untouched.
    for source in sorted(analysis_dir.iterdir()):
        if source.is_file() and not (out_dir / source.name).exists():
            shutil.copy2(source, out_dir / source.name)
    return out_dir


# --------------------------------------------------------------------------- #
# Identity proof
# --------------------------------------------------------------------------- #

def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _compare_csv(v1: Path, v2: Path, *, family: str, report: IdentityReport) -> None:
    """Original values must carry across; added values must be exactly right.

    Validating only that the originals are unchanged is necessary but not
    sufficient: a v2 could preserve every number and still assert the wrong
    comparison arm, the wrong interpretation, or an unexpected extra column.
    All of that is checked here.
    """
    a = pd.read_csv(v1, dtype=str, keep_default_na=False)
    b = pd.read_csv(v2, dtype=str, keep_default_na=False)
    report.files_compared.append(v1.name)

    if len(a) != len(b):
        report.violations.append(f"{v1.name}: row count {len(a)} != {len(b)}")
        return
    missing_cols = [c for c in a.columns if c not in b.columns]
    if missing_cols:
        report.violations.append(f"{v1.name}: v2 dropped column(s) {missing_cols}")
        return
    added = [c for c in b.columns if c not in a.columns]
    unexpected = [c for c in added if c not in ADDED_COLUMNS]
    if unexpected:
        report.violations.append(
            f"{v1.name}: v2 added column(s) not on the whitelist: {unexpected}"
        )
    for column in added:
        if column in ADDED_COLUMNS:
            report.csv_column_additions.append(f"{v1.name}:{column}")
            report.csv_added_cells += len(b)

    # --- original values ---------------------------------------------------- #
    for column in a.columns:
        for i, (x, y) in enumerate(zip(a[column], b[column])):
            report.values_compared += 1
            if _is_numeric(x):
                report.numeric_fields_compared += 1
            if x == y:
                continue
            where = f"{v1.name}[row {i}, {a.iloc[i].get('contrast_id', '?')}].{column}"
            if column in DESCRIPTION_COLUMNS and family == "B":
                report.interpretation_changes.append(where)
            else:
                report.violations.append(f"{where}: {x!r} -> {y!r}")

    # --- the transformation itself ------------------------------------------ #
    for i in range(len(b)):
        row = b.iloc[i]
        cid = row.get("contrast_id", f"row {i}")

        if family == "B":
            expected = INTERPRETATION_B.get(row["reading"])
            if expected is None:
                report.violations.append(f"{v1.name}[{cid}]: unknown reading {row['reading']!r}")
            elif row["interpretation"] != expected:
                report.violations.append(
                    f"{v1.name}[{cid}].interpretation is not INTERPRETATION_B"
                    f"[{row['reading']}]"
                )
        else:
            # R's text must be carried across untouched, not rewritten.
            if row["interpretation"] != a.iloc[i]["interpretation"]:
                report.violations.append(
                    f"{v1.name}[{cid}].interpretation: R's text must not be rewritten"
                )

        for column, expected in (
            ("comparison_arm", COMPARISON_ARM[family]),
            ("reporting_version", REPORTING_VERSION),
            ("erratum_id", ERRATUM_ID),
        ):
            if column not in b.columns:
                report.violations.append(f"{v1.name}[{cid}]: missing {column}")
            elif row[column] != expected:
                report.violations.append(
                    f"{v1.name}[{cid}].{column}: {row[column]!r} != {expected!r}"
                )


def _compare_json(v1: Path, v2: Path, report: IdentityReport) -> None:
    """Compare one analysis JSON, whitelisting additions BY PATH, not by name.

    The whitelist is a set of locations, not a set of key names. ``comparison_arm``
    is permitted only at ``trailing_gap_analysis.json:r_contrasts[i]`` and
    ``:b_contrasts[i]``. The same key appearing anywhere else — inside
    ``classification``, inside ``dose_response``, at any depth, or in either of
    the other two JSON files — is a violation. A name-based whitelist would have
    waved it through purely because the string matched.
    """
    a = json.loads(v1.read_text(encoding="utf-8"))
    b = json.loads(v2.read_text(encoding="utf-8"))
    report.files_compared.append(v1.name)
    is_payload = v1.name == PAYLOAD_FILE

    def addition_is_permitted(path: str, key: str) -> bool:
        """A nested key v2 invented. Permitted at exactly two path shapes."""
        return (
            is_payload
            and key in ADDED_ROW_KEYS
            and CONTRAST_ROW_PATH.match(path) is not None
        )

    def rewrite_is_permitted(path: str) -> bool:
        """A pre-existing leaf v2 changed. Only B's interpretation text."""
        return is_payload and B_INTERPRETATION_PATH.match(path) is not None

    def walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict):
            if not isinstance(y, dict):
                report.violations.append(f"{v1.name}:{path}: type changed")
                return
            for key in x:
                if key not in y:
                    report.violations.append(f"{v1.name}:{path}.{key}: dropped in v2")
                    continue
                walk(x[key], y[key], f"{path}.{key}")
            for key in y:
                if key in x:
                    continue
                if addition_is_permitted(path, key):
                    report.json_contrast_arm_additions.append(f"{v1.name}:{path}.{key}")
                else:
                    report.violations.append(
                        f"{v1.name}:{path}.{key}: key not on the whitelist appeared "
                        f"in v2 (additions are permitted only at "
                        f"{PAYLOAD_FILE}:r_contrasts[i] / b_contrasts[i])"
                    )
            return
        if isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                report.violations.append(f"{v1.name}:{path}: list length changed")
                return
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]")
            return
        report.values_compared += 1
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            report.numeric_fields_compared += 1
        if x == y:
            return
        if rewrite_is_permitted(path):
            report.interpretation_changes.append(path)
        else:
            report.violations.append(f"{v1.name}:{path}: {x!r} -> {y!r}")

    for key in a:
        if key not in b:
            report.violations.append(f"{v1.name}:{key}: dropped in v2")
            continue
        walk(a[key], b[key], key)

    # --- root-level additions ------------------------------------------------ #
    # Permitted only in the payload file, only the documented keys, and each
    # must EQUAL its documented value. Presence alone is never sufficient.
    for key in b:
        if key in a:
            continue
        if not is_payload or key not in ADDED_ROOT_KEYS:
            report.violations.append(
                f"{v1.name}: root key {key!r} is not on the whitelist "
                f"(root additions are permitted only in {PAYLOAD_FILE})"
            )
            continue
        report.json_root_additions.append(f"{v1.name}:{key}")
        expected = EXPECTED_ROOT_VALUES[key]
        if b[key] != expected:
            report.violations.append(
                f"{v1.name}.{key} does not match its single defined value: "
                f"{b[key]!r} != {expected!r}"
            )

    if not is_payload:
        return

    # --- the payload's own required additions -------------------------------- #
    for key in ADDED_ROOT_KEYS:
        if key not in b:
            report.violations.append(f"{v1.name}: required root key {key!r} is missing")

    # Every B contrast row must carry the corrected text and the correct arm.
    for i, row in enumerate(b.get("b_contrasts", [])):
        expected = INTERPRETATION_B.get(row.get("reading"))
        if expected is None:
            report.violations.append(f"b_contrasts[{i}]: unknown reading")
        elif row.get("interpretation") != expected:
            report.violations.append(
                f"b_contrasts[{i}].interpretation is not INTERPRETATION_B[{row.get('reading')}]"
            )
        if row.get("comparison_arm") != COMPARISON_ARM["B"]:
            report.violations.append(f"b_contrasts[{i}].comparison_arm is wrong")
    for i, row in enumerate(b.get("r_contrasts", [])):
        if row.get("comparison_arm") != COMPARISON_ARM["R"]:
            report.violations.append(f"r_contrasts[{i}].comparison_arm is wrong")


def _compare_passthrough(analysis_dir: Path, v2_dir: Path, report: IdentityReport) -> None:
    """Files not rewritten by the erratum must be byte-identical, and complete."""
    rewritten = {
        "R_contrasts.csv", "B_contrasts.csv", "trailing_gap_analysis.json",
        "identity_check.json",
    }
    source = {p.name for p in analysis_dir.iterdir() if p.is_file()}
    produced = {p.name for p in v2_dir.iterdir() if p.is_file()}

    for name in sorted(source - produced):
        report.violations.append(f"pass-through file missing from v2: {name}")
    for name in sorted(produced - source - rewritten):
        report.violations.append(f"unexpected file in v2: {name}")

    for name in sorted(source & produced - rewritten):
        a, b = (analysis_dir / name).read_bytes(), (v2_dir / name).read_bytes()
        report.passthrough_files_compared.append(name)
        if a != b:
            report.violations.append(
                f"pass-through file modified (byte-identity required): {name}"
            )


def compare_reporting_versions(analysis_dir: Path, v2_dir: Path) -> IdentityReport:
    """Prove the erratum performed EXACTLY the permitted transformation.

    Validates three things, not one:
      1. every original value is unchanged (except B's interpretation);
      2. every added value is exactly right — B's text equals
         INTERPRETATION_B[reading], both families' comparison_arm is correct,
         and the version/erratum markers match;
      3. nothing else appeared — no unwhitelisted column, JSON key, or file,
         and every pass-through file is byte-identical.

    Raises ReportingV2Error if any of REQUIRED_SOURCE_FILES is absent on either
    side. That is deliberate: a missing artifact must fail, never be skipped.
    """
    analysis_dir, v2_dir = Path(analysis_dir), Path(v2_dir)
    # Hard failure before any comparison begins. Every required artifact must
    # exist on BOTH sides; otherwise a comparison would quietly skip it and
    # still report identical=True, which is worse than no check at all.
    require_source_files(analysis_dir, what="source")
    require_source_files(v2_dir, what="v2")
    report = IdentityReport()

    for name, family in (("R_contrasts.csv", "R"), ("B_contrasts.csv", "B")):
        _compare_csv(analysis_dir / name, v2_dir / name, family=family, report=report)

    for name in REQUIRED_SOURCE_FILES:
        if name.endswith(".json"):
            _compare_json(analysis_dir / name, v2_dir / name, report)

    if analysis_dir.is_dir() and v2_dir.is_dir():
        _compare_passthrough(analysis_dir, v2_dir, report)

    report.identical = not report.violations
    return report
