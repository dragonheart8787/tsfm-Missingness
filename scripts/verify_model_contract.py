"""STOP-AND-REPORT gate: verify the real Chronos-2 checkpoint before the pilot runs.

Run this FIRST on any host with Hugging Face access. It:
  1. resolves and prints the exact commit hash of `amazon/chronos-2`,
  2. loads the checkpoint and reads its REAL patch size / stride / quantiles,
  3. compares them against the preregistered assumptions in the config,
  4. exercises the NaN missing-value convention on a synthetic context,
  5. exits NON-ZERO if the preregistered patch grid does not hold.

The pilot's block positions (d values) are all computed on a 16-step grid. If
the real grid differs, every position and every alignment assertion must be
recomputed before the pilot may run — this script refuses rather than adjusting
the design silently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from model.chronos2_runner import Chronos2Forecaster, verify_contract  # noqa: E402


def resolve_revision(model_id: str) -> str:
    """Resolve the model repo's current HEAD commit hash (not a moving tag)."""
    from huggingface_hub import HfApi

    return HfApi().model_info(model_id).sha


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the Chronos-2 model contract.")
    parser.add_argument("--config", default="configs/pilot_config.yaml")
    parser.add_argument(
        "--write-revision",
        action="store_true",
        help="Resolve the current HEAD commit of the HF repo and pin it into the config.",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    config = yaml.safe_load(config_path.read_text())
    model_id = config["model"]["hf_model_id"]

    if args.write_revision or not config["model"].get("revision"):
        revision = resolve_revision(model_id)
        print(f"Resolved {model_id} HEAD commit: {revision}")
        if args.write_revision:
            text = config_path.read_text()
            current = config["model"].get("revision")
            needle = "revision: null" if not current else f'revision: "{current}"'
            if needle not in text:
                raise SystemExit(f"could not locate `{needle}` in {args.config} to pin the revision")
            config_path.write_text(text.replace(needle, f'revision: "{revision}"', 1))
            print(f"Pinned model.revision={revision} into {args.config}")
            config = yaml.safe_load(config_path.read_text())
        else:
            config["model"]["revision"] = revision

    forecaster = Chronos2Forecaster(config=config)
    contract = forecaster.contract
    print("\n=== LOADED MODEL CONTRACT ===")
    print(json.dumps(contract.as_dict(), indent=2))

    print("\n=== NaN MISSING-VALUE CONVENTION CHECK ===")
    context_length = int(config["design"]["context_length"])
    horizon = int(config["design"]["horizon"])
    base = np.sin(np.arange(context_length) / 12.0).astype(np.float32) * 5.0 + 20.0
    clean_forecast = forecaster.forecast_median(base, horizon)
    holed = base.copy()
    holed[100:164] = np.nan
    holed_forecast = forecaster.forecast_median(holed, horizon)
    print(f"clean forecast finite      : {bool(np.all(np.isfinite(clean_forecast)))}")
    print(f"NaN-context forecast finite: {bool(np.all(np.isfinite(holed_forecast)))}")
    print(f"forecasts differ           : {bool(not np.allclose(clean_forecast, holed_forecast))}")
    nan_ok = bool(np.all(np.isfinite(holed_forecast)))
    if not nan_ok:
        print("  -> NaN passthrough produced non-finite forecasts; the assumed missing-value "
              "convention does NOT hold. STOP.")

    print("\n=== PREREGISTERED CONTRACT VERIFICATION ===")
    violations = verify_contract(contract, config)
    if violations or not nan_ok:
        print("FAILED. The preregistered design does not match this checkpoint:")
        for violation in violations:
            print(f"  - {violation}")
        if not nan_ok:
            print("  - NaN passthrough did not yield finite forecasts.")
        print(
            "\nSTOP AND REPORT to the Research Lead. Do NOT adjust the block positions or the "
            "patch-grid assertions to fit; the design's d values are preregistered on the "
            f"{config['model']['expected_patch_size']}-step grid."
        )
        return 1

    print(
        f"PASSED. Real patch size = {contract.input_patch_size} "
        f"(stride {contract.input_patch_stride}), matching the preregistered grid. "
        f"q=0.5 present in the model's trained quantiles. Context length "
        f"{config['design']['context_length']} is a whole number of patches, so Chronos-2 "
        f"applies no left NaN padding and every block position sits on a real patch boundary."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
