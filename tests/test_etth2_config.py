"""The ETTh2 config must INHERIT the scientific design, never restate it.

A replication that redefined its own constants could drift from the study it
replicates without the diff showing anything. These tests make the inheritance
mechanical: every shared constant is asserted EQUAL to its source, and the
frozen source files are asserted UNMODIFIED.

Model-mocked. No inference, no GPU.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from experiments.run_etth2 import OVERRIDE_SECTIONS, build_effective_config

REPO_ROOT = Path(__file__).resolve().parents[1]


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


# --------------------------------------------------------------------------- #
# Non-duplication
# --------------------------------------------------------------------------- #

def test_the_etth2_config_declares_only_dataset_design_and_whats_new(etth2_config):
    """It may not carry a statistics, decision, model, masks or metrics block.

    Those are the blocks that hold the SESOI, the bootstrap settings and the
    pinned model revision. Their absence here is what makes drift impossible.
    """
    forbidden = {"statistics", "decision", "model", "masks", "metrics", "contrasts"}
    assert not (forbidden & set(etth2_config)), (
        f"etth2_config restates {sorted(forbidden & set(etth2_config))}; these must "
        f"be inherited from pilot_config.yaml / trailing_gap_config.yaml"
    )
    assert set(etth2_config) == {
        "meta", "dataset", "design", "clean_control", "replication_rule",
        "preconditions_for_execution",
    }


def test_no_gap_length_sesoi_or_bootstrap_value_is_restated(etth2_config):
    """The frozen scientific constants must not appear as ETTh2 config values."""
    text = (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    # Assembled at runtime so this test cannot match its own source text.
    for key in ("gap_" + "lengths", "equivalence_band_frac_of_clean_" + "mae",
                "n_" + "replicates", "sensitivity_block_" + "lengths",
                "main_block_" + "length", "family_" + "alpha", "revision"):
        assert f"\n  {key}:" not in text and f"\n    {key}:" not in text, (
            f"{key} is restated in etth2_config.yaml; it must be inherited"
        )
    assert "gap_lengths" not in etth2_config.get("design", {})


def test_the_geometry_it_does_restate_is_equal_to_the_pilots(etth2_config, config):
    """L, H and stride are repeated only because build_windows reads them.

    Repeated, therefore checked: if they ever differ from the pilot's, this
    fails rather than silently running a different geometry on ETTh2.
    """
    for key in ("context_length", "horizon", "stride"):
        assert etth2_config["design"][key] == config["design"][key], key


def test_conditions_per_origin_matches_the_trailing_gap_config(etth2_config, gap_config):
    assert (
        etth2_config["design"]["expected_conditions_per_origin"]
        == gap_config["design"]["expected_conditions_per_origin"]
        == 13
    )
    assert (
        etth2_config["design"]["expected_total_forecasts"]
        == etth2_config["design"]["expected_origins"]
        * etth2_config["design"]["expected_conditions_per_origin"]
        == 2314
    )


def test_the_frozen_source_configs_are_untouched_by_this_round():
    """The replication may not edit the study it replicates."""
    import subprocess

    for path in ("configs/pilot_config.yaml", "configs/trailing_gap_config.yaml"):
        diff = subprocess.run(
            ["git", "diff", "5b62949", "--", path],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert diff.stdout.strip() == "", f"{path} was modified: {diff.stdout[:400]}"


# --------------------------------------------------------------------------- #
# The effective config
# --------------------------------------------------------------------------- #

def test_effective_config_substitutes_only_dataset_and_design(etth2_config, config):
    effective = build_effective_config(config, etth2_config)
    assert set(OVERRIDE_SECTIONS) == {"dataset", "design"}
    assert effective["dataset"]["name"] == "ETTh2"
    assert effective["design"]["expected_origins"] == 178
    # Everything scientific is carried through byte-for-byte.
    for section in ("statistics", "decision", "model", "masks", "metrics", "contrasts"):
        assert effective[section] == config[section], section


def test_effective_config_carries_the_pilots_sesoi_and_bootstrap_exactly(
    etth2_config, config
):
    effective = build_effective_config(config, etth2_config)
    assert (
        effective["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
        == config["decision"]["no_go"]["equivalence_band_frac_of_clean_mae"]
        == 0.03
    )
    boot = effective["statistics"]["bootstrap"]
    assert boot["main_block_length"] == 8
    assert boot["sensitivity_block_lengths"] == [4, 12]
    assert boot["n_replicates"] == 5000
    assert effective["statistics"]["ci_level_equivalence"] == 0.90
    assert effective["model"]["revision"] == config["model"]["revision"]


def test_effective_config_does_not_mutate_its_inputs(etth2_config, config):
    """A merge that mutated the pilot config would corrupt every later reader."""
    before_pilot = copy.deepcopy(config)
    before_etth2 = copy.deepcopy(etth2_config)
    build_effective_config(config, etth2_config)
    assert config == before_pilot
    assert etth2_config == before_etth2


def test_effective_config_refuses_an_etth2_config_missing_a_section(etth2_config, config):
    from experiments.run_etth2 import ReplicationGateError

    broken = copy.deepcopy(etth2_config)
    del broken["design"]
    with pytest.raises(ReplicationGateError, match="no 'design' section"):
        build_effective_config(config, broken)


# --------------------------------------------------------------------------- #
# The declared clean control and replication rule
# --------------------------------------------------------------------------- #

def test_clean_control_declares_the_preregistered_discipline(etth2_config):
    control = etth2_config["clean_control"]
    assert control["design"] == "two_independent_runs"
    assert control["reference_is_qc_only"] is True
    assert control["reference_excluded_from_analysis"] is True
    assert control["separate_process_required"] is True
    assert control["separate_model_load_required"] is True
    assert control["tolerance"] == "none"
    assert "HARD STOP" in control["on_mismatch"]
    assert control["reference_run_dir"] != control["formal_run_dir"]
    # All eight identity fields the preregistration names.
    fields = " ".join(control["exact_match_fields"]).lower()
    for needle in ("cardinality", "key sets", "forecast_timestamp", "ground_truth",
                   "target_sha256", "dataset_sha256", "model_revision", "float32"):
        assert needle in fields, needle


def test_replication_rule_block_matches_the_frozen_rule_module(etth2_config):
    from stats.replication_decision import (
        PRIMARY_CONTRASTS,
        REPLICATION_RULE_VERSION,
        SECONDARY_STRESS_CONTRAST,
    )

    rule = etth2_config["replication_rule"]
    assert rule["rule_version"] == REPLICATION_RULE_VERSION
    assert tuple(rule["primary_contrasts"]) == PRIMARY_CONTRASTS
    assert rule["secondary_stress_contrast"] == SECONDARY_STRESS_CONTRAST
    assert rule["intersection_multiplicity_correction"] == "none"
    assert "intersection-union" in rule["intersection_multiplicity_justification"].lower()
    assert "regardless of dependence" in rule["intersection_multiplicity_justification"]


def test_the_config_never_claims_r_and_b_are_independent(etth2_config):
    text = (REPO_ROOT / "configs" / "etth2_config.yaml").read_text(encoding="utf-8")
    # YAML folds long scalars across lines, so compare on collapsed whitespace.
    lowered = " ".join(text.lower().split())
    assert "not assumed statistically independent" in lowered
    for forbidden in ("positively dependent", "negatively dependent"):
        assert forbidden not in lowered, forbidden
    # And the loaded value says it too, not just the file.
    note = " ".join(etth2_config["replication_rule"]["b_family_dependence_note"].lower().split())
    assert "not assumed statistically independent" in note
    assert "no sign is claimed" in note


# --------------------------------------------------------------------------- #
# The model contract is UNCHANGED by this round — confirmed, not assumed
# --------------------------------------------------------------------------- #

def test_the_pinned_model_contract_is_byte_identical_to_the_frozen_one(config):
    """The replication must run against the same weights and the same grid.

    Confirmed rather than assumed: the ETTh2 config inherits the model block, so
    a drift here would mean the pilot config itself changed. The real-checkpoint
    re-verification (tests/test_model_contract.py, marked requires_model) is the
    other half of this and must be run on the GPU host before execution.
    """
    contract = config["model"]["verified_contract"]
    assert config["model"]["revision"] == "29ec3766d36d6f73f0696f85560a422f50e8498c"
    assert config["model"]["hf_model_id"] == "amazon/chronos-2"
    assert contract["input_patch_size"] == 16
    assert contract["input_patch_stride"] == 16
    assert contract["output_patch_size"] == 16
    assert contract["max_output_patches"] == 64
    assert contract["single_shot_horizon"] == 1024
    assert (
        contract["max_output_patches"] * contract["output_patch_size"]
        == contract["single_shot_horizon"]
    )
    assert config["model"]["limit_prediction_length"] is True
    assert config["model"]["cross_learning"] is False
    assert config["model"]["quantile_level"] == 0.5


def test_every_etth2_gap_still_fits_the_single_shot_horizon(config, gap_config):
    """H + g must fit in ONE forward pass for every gap, on ETTh2 as on ETTh1.

    The dataset changed; the model did not. This re-derives the constraint from
    the pinned contract rather than trusting that it still holds.
    """
    contract = config["model"]["verified_contract"]
    single_shot = min(
        contract["single_shot_horizon"],
        contract["max_output_patches"] * contract["output_patch_size"],
    )
    horizon = int(config["design"]["horizon"])
    patch = int(contract["input_patch_size"])
    for gap in gap_config["design"]["gap_lengths"]:
        assert horizon + gap <= single_shot, f"g={gap} would unroll autoregressively"
        assert gap % patch == 0, f"g={gap} is not patch-aligned"
        assert (int(config["design"]["context_length"]) - gap) % patch == 0, gap
