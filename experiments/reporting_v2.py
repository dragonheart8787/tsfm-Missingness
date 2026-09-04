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


class ReportingV2Error(RuntimeError):
    """The source analysis directory is not in a state this can correct."""


@dataclass
class IdentityReport:
    """Field-by-field comparison of an original analysis against its v2."""

    identical: bool = False
    files_compared: list[str] = field(default_factory=list)
    values_compared: int = 0
    numeric_fields_compared: int = 0
    allowed_differences: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "files_compared": self.files_compared,
            "values_compared": self.values_compared,
            "numeric_fields_compared": self.numeric_fields_compared,
            "n_allowed_differences": len(self.allowed_differences),
            "allowed_differences": self.allowed_differences,
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
        report.violations.append(f"{v1.name}: v2 added unexpected column(s) {unexpected}")

    for column in a.columns:
        for i, (x, y) in enumerate(zip(a[column], b[column])):
            report.values_compared += 1
            if _is_numeric(x):
                report.numeric_fields_compared += 1
            if x == y:
                continue
            where = f"{v1.name}[row {i}, {a.iloc[i].get('contrast_id', '?')}].{column}"
            if column in DESCRIPTION_COLUMNS and family == "B":
                report.allowed_differences.append(where)
            else:
                report.violations.append(f"{where}: {x!r} -> {y!r}")


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
            report.allowed_differences.append(path)
        else:
            report.violations.append(f"{path}: {x!r} -> {y!r}")

    for key in a:
        if key not in b:
            report.violations.append(f"{key}: dropped in v2")
            continue
        walk(a[key], b[key], key, in_b_contrast=key == "b_contrasts")


def compare_reporting_versions(analysis_dir: Path, v2_dir: Path) -> IdentityReport:
    """Prove the erratum changed nothing but the B_g description.

    Compares every value in the contrast tables and the analysis payload —
    every metric, CI bound, p-value, per-gap reading, the SESOI, the
    dose-response result and the final classification. Any difference outside
    B's ``interpretation`` is a violation.
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

    report.identical = not report.violations
    return report
