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

# Columns a v2 row is permitted to differ in or to add. Everything else must
# carry across byte-for-byte.
DESCRIPTION_COLUMNS = ("interpretation",)
ADDED_COLUMNS = ("comparison_arm", "reporting_version", "erratum_id")
# Keys the v2 payload may add at the JSON root. Nothing else, anywhere.
ADDED_ROOT_KEYS = ("reporting_version", "erratum_id", "erratum_note")
# Keys the v2 payload may add inside a contrast row.
ADDED_ROW_KEYS = ("comparison_arm",)


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
    # An ADDED field on the whitelist. A different category from a rewrite:
    # nothing that existed before was altered.
    whitelisted_additions: list[str] = field(default_factory=list)
    passthrough_files_compared: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "files_compared": self.files_compared,
            "values_compared": self.values_compared,
            "numeric_fields_compared": self.numeric_fields_compared,
            "passthrough_files_compared": self.passthrough_files_compared,
            "n_interpretation_changes": len(self.interpretation_changes),
            "interpretation_changes": self.interpretation_changes,
            "n_whitelisted_additions": len(self.whitelisted_additions),
            "whitelisted_additions": self.whitelisted_additions,
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


def build_reporting_v2(analysis_dir: Path, out_dir: Path | None = None) -> Path:
    """Emit analysis_reporting_v2/ beside an existing analysis/ directory.

    The source directory is never modified.
    """
    analysis_dir = Path(analysis_dir)
    out_dir = Path(out_dir) if out_dir else analysis_dir.parent / REPORTING_VERSION
    if not analysis_dir.is_dir():
        raise ReportingV2Error(f"{analysis_dir} is not a directory")
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
        source = analysis_dir / name
        if not source.exists():
            raise ReportingV2Error(f"{source} is missing")
        _correct_contrast_csv(source, out_dir / name, family=family)

    # trailing_gap_analysis.json carries the same b_contrasts prose.
    payload_path = analysis_dir / "trailing_gap_analysis.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        for row in payload.get("b_contrasts", []):
            row["interpretation"] = INTERPRETATION_B[row["reading"]]
            row["comparison_arm"] = COMPARISON_ARM["B"]
        for row in payload.get("r_contrasts", []):
            row["comparison_arm"] = COMPARISON_ARM["R"]
        payload["reporting_version"] = REPORTING_VERSION
        payload["erratum_id"] = ERRATUM_ID
        payload["erratum_note"] = (
            "Reporting erratum only: B_g's comparison arm was described using the "
            "R_g interpretation text, which names truncated_long. B_g's arm is and "
            "always was internal_block(g, d=16). No number, interval, p-value, "
            "reading, SESOI, dose-response result or classification is affected. "
            f"See {ERRATUM_DOC}."
        )
        (out_dir / "trailing_gap_analysis.json").write_text(
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
    unexpected = [c for c in b.columns if c not in a.columns and c not in ADDED_COLUMNS]
    if unexpected:
        report.violations.append(
            f"{v1.name}: v2 added column(s) not on the whitelist: {unexpected}"
        )

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
    a = json.loads(v1.read_text(encoding="utf-8"))
    b = json.loads(v2.read_text(encoding="utf-8"))
    report.files_compared.append(v1.name)

    def walk(x: Any, y: Any, path: str, *, in_b_contrast: bool) -> None:
        if isinstance(x, dict):
            if not isinstance(y, dict):
                report.violations.append(f"{path}: type changed")
                return
            for key in x:
                if key not in y:
                    report.violations.append(f"{path}.{key}: dropped in v2")
                    continue
                walk(x[key], y[key], f"{path}.{key}",
                     in_b_contrast=in_b_contrast or path.endswith("b_contrasts"))
            # Any key v2 invented inside a nested object is a violation unless
            # it is a whitelisted per-row addition.
            for key in y:
                if key in x:
                    continue
                if key in ADDED_ROW_KEYS:
                    report.whitelisted_additions.append(f"{path}.{key}")
                else:
                    report.violations.append(
                        f"{path}.{key}: key not on the whitelist appeared in v2"
                    )
            return
        if isinstance(x, list):
            if not isinstance(y, list) or len(x) != len(y):
                report.violations.append(f"{path}: list length changed")
                return
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{path}[{i}]", in_b_contrast=in_b_contrast)
            return
        report.values_compared += 1
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            report.numeric_fields_compared += 1
        if x == y:
            return
        leaf = path.rsplit(".", 1)[-1]
        if in_b_contrast and leaf in DESCRIPTION_COLUMNS:
            report.interpretation_changes.append(path)
        else:
            report.violations.append(f"{path}: {x!r} -> {y!r}")

    for key in a:
        if key not in b:
            report.violations.append(f"{key}: dropped in v2")
            continue
        walk(a[key], b[key], key, in_b_contrast=key == "b_contrasts")

    # Root-level additions must be exactly the documented ones, with the
    # documented values.
    for key in b:
        if key in a:
            continue
        if key not in ADDED_ROOT_KEYS:
            report.violations.append(
                f"{v1.name}: root key {key!r} is not on the whitelist"
            )
            continue
        expected = {"reporting_version": REPORTING_VERSION, "erratum_id": ERRATUM_ID}.get(key)
        if expected is not None and b[key] != expected:
            report.violations.append(f"{v1.name}.{key}: {b[key]!r} != {expected!r}")

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
    """
    analysis_dir, v2_dir = Path(analysis_dir), Path(v2_dir)
    report = IdentityReport()

    for name, family in (("R_contrasts.csv", "R"), ("B_contrasts.csv", "B")):
        v1, v2 = analysis_dir / name, v2_dir / name
        if not v1.exists() or not v2.exists():
            report.violations.append(f"{name}: missing on one side")
            continue
        _compare_csv(v1, v2, family=family, report=report)

    for name in ("trailing_gap_analysis.json", "classification.json", "dose_response.json"):
        v1, v2 = analysis_dir / name, v2_dir / name
        if v1.exists() and v2.exists():
            _compare_json(v1, v2, report)

    if analysis_dir.is_dir() and v2_dir.is_dir():
        _compare_passthrough(analysis_dir, v2_dir, report)

    report.identical = not report.violations
    return report
