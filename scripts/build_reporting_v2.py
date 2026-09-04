"""Emit analysis_reporting_v2/ and prove it changed no number.

Reporting erratum only. Reads a completed analysis directory, writes the
corrected parallel directory, then verifies field-by-field identity and exits
non-zero if anything numeric moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reporting_v2 import (  # noqa: E402
    ERRATUM_DOC,
    build_reporting_v2,
    compare_reporting_versions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify analysis_reporting_v2.")
    parser.add_argument(
        "--analysis-dir", default="results/trailing_gap_v1/analysis",
        help="The original, untouched analysis output directory.",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    analysis_dir = REPO_ROOT / args.analysis_dir
    out_dir = build_reporting_v2(
        analysis_dir, REPO_ROOT / args.out_dir if args.out_dir else None
    )
    print(f"original (untouched): {analysis_dir}")
    print(f"corrected           : {out_dir}")

    report = compare_reporting_versions(analysis_dir, out_dir)
    print("\n--- identity check ---")
    print(f"  files compared          : {', '.join(report.files_compared)}")
    print(f"  values compared         : {report.values_compared}")
    print(f"  numeric fields compared : {report.numeric_fields_compared}")
    print(f"  allowed differences     : {len(report.allowed_differences)} "
          f"(B_g interpretation only)")
    for where in report.allowed_differences:
        print(f"      {where}")
    (out_dir / "identity_check.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )

    if report.identical:
        print(f"\nPASS: every number, CI, p-value, reading, SESOI, dose-response result "
              f"and the classification are identical. Only the B_g comparison-arm "
              f"description differs. See {ERRATUM_DOC}.")
        return 0
    print(f"\nFAIL: {len(report.violations)} value(s) changed outside the allowed "
          f"description field:")
    for violation in report.violations:
        print(f"  - {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
