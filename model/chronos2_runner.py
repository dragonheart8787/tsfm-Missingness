"""Chronos-2 inference wrapper: revision-pinned, q=0.5, isolation-guaranteed.

Contract verification
---------------------
The pilot's block positions are aligned to a preregistered 16-step patch grid.
Rather than trusting that, ``ModelContract`` is read from the LOADED checkpoint
(``Chronos2Model.chronos_config``) and compared against the preregistered
expectation. A mismatch is a hard stop with an explicit report, never a silent
adjustment of the grid.

Verified facts about Chronos-2 (chronos-forecasting 2.3.1 source):
  * Missing history is expressed as float NaN. ``Chronos2Model``'s
    ``_prepare_patched_context`` computes ``context_mask = ~isnan(context)``,
    NaN-aware-normalises, patches, zeroes unobserved positions and feeds the
    mask to the model as an explicit channel. NaN passthrough is therefore the
    supported convention, not a hack.
  * ``Chronos2Pipeline.predict(..., cross_learning=False)`` is the default and
    keeps tasks independent; ``cross_learning=True`` would share information
    across every task in the call. We force False AND submit one origin per
    call.
  * ``predict_quantiles`` returns a second value named ``mean`` that is in fact
    the q=0.5 column. We do NOT use it: we request ``quantile_levels=[0.5]``
    and additionally assert 0.5 is one of the model's own trained quantiles so
    no interpolation path is taken.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ModelContract:
    """Facts read off the loaded checkpoint, recorded in every output artifact."""

    hf_model_id: str
    revision: str
    input_patch_size: int
    input_patch_stride: int
    output_patch_size: int
    model_context_length: int
    max_output_patches: int
    model_prediction_length: int
    quantiles: list[float]
    use_arcsinh: bool
    device: str
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Forecaster(Protocol):
    """The seam at which the real model is mocked for CPU-only testing."""

    contract: ModelContract

    def forecast_median(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Return the q=0.5 forecast of shape (horizon,) for one context window."""
        ...


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class Chronos2Forecaster:
    """Thin, isolation-enforcing wrapper around ``Chronos2Pipeline``."""

    def __init__(self, *, config: dict[str, Any]) -> None:
        import torch
        from chronos import Chronos2Pipeline

        model_cfg = config["model"]
        revision = model_cfg.get("revision")
        if not revision:
            raise ValueError(
                "model.revision is not pinned in configs/pilot_config.yaml. The pilot refuses "
                "to run against a floating tag. Run `python scripts/verify_model_contract.py "
                "--write-revision` on a host with Hugging Face access first."
            )

        device = _resolve_device(str(model_cfg.get("device", "auto")))
        dtype_name = str(model_cfg.get("dtype", "float32"))
        torch_dtype = getattr(torch, dtype_name)

        self._pipeline = Chronos2Pipeline.from_pretrained(
            model_cfg["hf_model_id"],
            revision=revision,
            torch_dtype=torch_dtype,
        )
        self._pipeline.model.to(device)
        self._pipeline.model.eval()  # never fine-tuned; inference only

        self._torch = torch
        self._device = device
        self._cross_learning = bool(model_cfg.get("cross_learning", False))
        if self._cross_learning:
            raise ValueError(
                "model.cross_learning must be false: cross-learning shares information across "
                "every task in a call and would break per-origin isolation."
            )
        self._quantile_level = float(model_cfg.get("quantile_level", 0.5))

        chronos_config = self._pipeline.model.chronos_config
        actual_dtype = str(next(self._pipeline.model.parameters()).dtype).replace("torch.", "")
        self.contract = ModelContract(
            hf_model_id=str(model_cfg["hf_model_id"]),
            revision=str(revision),
            input_patch_size=int(chronos_config.input_patch_size),
            input_patch_stride=int(chronos_config.input_patch_stride),
            output_patch_size=int(chronos_config.output_patch_size),
            model_context_length=int(chronos_config.context_length),
            max_output_patches=int(chronos_config.max_output_patches),
            model_prediction_length=int(self._pipeline.model_prediction_length),
            quantiles=[float(q) for q in chronos_config.quantiles],
            use_arcsinh=bool(chronos_config.use_arcsinh),
            device=device,
            dtype=actual_dtype,
        )

        if self._quantile_level not in self.contract.quantiles:
            raise ValueError(
                f"q={self._quantile_level} is not one of the model's trained quantiles "
                f"{self.contract.quantiles}; requesting it would silently interpolate. "
                "Refusing to proceed."
            )
        if actual_dtype != dtype_name:
            warnings.warn(
                f"requested dtype {dtype_name} but parameters are {actual_dtype}; "
                "recorded in the model contract.",
                stacklevel=2,
            )

    def forecast_median(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast one context window in complete isolation.

        One origin per call and ``cross_learning=False`` together mean no other
        evaluation window is present in the computation graph at all.
        """
        torch = self._torch
        if context.ndim != 1:
            raise ValueError("context must be 1-dimensional")

        tensor = torch.as_tensor(np.asarray(context, dtype=np.float32))
        with torch.inference_mode():
            quantiles, _mean = self._pipeline.predict_quantiles(
                [tensor],                      # exactly one task per call
                prediction_length=horizon,
                quantile_levels=[self._quantile_level],
                cross_learning=self._cross_learning,
                batch_size=1,
            )
        # shape: (n_variates=1, horizon, n_quantile_levels=1)
        prediction = quantiles[0]
        if prediction.shape[-1] != 1:
            raise AssertionError(f"expected a single quantile column, got {tuple(prediction.shape)}")
        median = prediction[..., 0].reshape(-1).to(torch.float32).cpu().numpy()
        if median.shape != (horizon,):
            raise AssertionError(f"forecast shape {median.shape} != ({horizon},)")
        return median


def verify_contract(contract: ModelContract, config: dict[str, Any]) -> list[str]:
    """Compare the loaded checkpoint against preregistered assumptions.

    Returns a list of violation strings; empty means the preregistered grid
    holds. Callers must treat a non-empty result as a STOP-AND-REPORT, because
    every block position in the design is computed on this grid.
    """
    model_cfg = config["model"]
    design = config["design"]
    violations: list[str] = []

    expected_patch = int(model_cfg["expected_patch_size"])
    if contract.input_patch_size != expected_patch:
        violations.append(
            f"PATCH GRID MISMATCH: checkpoint input_patch_size={contract.input_patch_size}, "
            f"preregistered expected_patch_size={expected_patch}. Every block-alignment "
            f"position (d values) and every alignment assertion must be recomputed on the "
            f"real grid before this pilot may run."
        )
    expected_stride = int(model_cfg["expected_patch_stride"])
    if contract.input_patch_stride != expected_stride:
        violations.append(
            f"PATCH STRIDE MISMATCH: checkpoint input_patch_stride={contract.input_patch_stride}, "
            f"preregistered {expected_stride}. Overlapping patches change what "
            f"'patch-aligned' means for the block positions."
        )
    quantile_level = float(model_cfg["quantile_level"])
    if quantile_level not in contract.quantiles:
        violations.append(
            f"q={quantile_level} absent from the model's trained quantiles {contract.quantiles}."
        )
    context_length = int(design["context_length"])
    if context_length > contract.model_context_length:
        violations.append(
            f"design context_length={context_length} exceeds the model's "
            f"{contract.model_context_length}; the context would be truncated."
        )
    if contract.input_patch_size and context_length % contract.input_patch_size != 0:
        violations.append(
            f"context_length={context_length} is not a whole number of patches "
            f"(patch_size={contract.input_patch_size}); Chronos-2 would LEFT-PAD the context "
            f"with NaN, shifting every block position relative to the patch grid."
        )
    # Drift check against the values measured on the GPU host for the pinned
    # revision. A pin alone does not prove the checkpoint still reports what it
    # reported: this makes a change visible immediately rather than at analysis
    # time, and is what lets the recorded capacity be relied on downstream.
    recorded = model_cfg.get("verified_contract") or {}
    for field in (
        "input_patch_size", "input_patch_stride", "output_patch_size", "max_output_patches"
    ):
        if field in recorded and getattr(contract, field) != int(recorded[field]):
            violations.append(
                f"CONTRACT DRIFT: checkpoint reports {field}={getattr(contract, field)}, but "
                f"model.verified_contract records {recorded[field]} for pinned revision "
                f"{contract.revision}. The recorded value is what downstream designs were "
                f"sized against; re-verify before running."
            )
    if "single_shot_horizon" in recorded:
        actual = contract.max_output_patches * contract.output_patch_size
        if actual != int(recorded["single_shot_horizon"]):
            violations.append(
                f"CONTRACT DRIFT: single-shot horizon is {actual} "
                f"({contract.max_output_patches} x {contract.output_patch_size}), but "
                f"model.verified_contract records {recorded['single_shot_horizon']}."
            )

    horizon = int(design["horizon"])
    if horizon > contract.model_prediction_length:
        violations.append(
            f"horizon={horizon} exceeds the model's single-pass prediction length "
            f"{contract.model_prediction_length}; predictions would be produced by "
            f"autoregressive unrolling, which is a different inference regime."
        )
    return violations


class MockForecaster:
    """Deterministic stand-in used by CPU-only tests.

    It is NOT a model: it produces a reproducible function of the observed
    context so that mask correctness, target integrity, isolation and bootstrap
    pairing can all be exercised without weights. Tests that need real model
    behaviour are marked ``requires_model``.
    """

    def __init__(self, *, contract: ModelContract | None = None, noise_scale: float = 0.0) -> None:
        self.contract = contract or ModelContract(
            hf_model_id="mock/chronos-2",
            revision="mock-revision",
            input_patch_size=16,
            input_patch_stride=16,
            output_patch_size=16,
            model_context_length=8192,
            # Self-consistent: model_prediction_length IS
            # max_output_patches * output_patch_size in the real pipeline, and a
            # mock that disagreed would mask the long-horizon contract check.
            max_output_patches=64,
            model_prediction_length=64 * 16,
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            use_arcsinh=False,
            device="cpu",
            dtype="float32",
        )
        self.noise_scale = float(noise_scale)
        self.calls: list[np.ndarray] = []

    def forecast_median(self, context: np.ndarray, horizon: int) -> np.ndarray:
        self.calls.append(np.array(context, copy=True))
        observed = np.asarray(context, dtype=np.float64)
        finite = observed[np.isfinite(observed)]
        if finite.size == 0:
            return np.zeros(horizon, dtype=np.float32)

        # A seasonal-naive-like echo of the observed context, so that *which*
        # observations were removed measurably changes the forecast.
        level = float(finite.mean())
        recent = finite[-min(24, finite.size):]
        season = np.resize(recent - recent.mean(), horizon)
        out = level + season
        if self.noise_scale:
            rng = np.random.default_rng(abs(int(level * 1e6)) % (2**32))
            out = out + rng.normal(scale=self.noise_scale, size=horizon)
        return out.astype(np.float32)


def forecaster_from_config(config: dict[str, Any], *, mock: bool = False) -> Forecaster:
    if mock:
        return MockForecaster()
    return Chronos2Forecaster(config=config)
