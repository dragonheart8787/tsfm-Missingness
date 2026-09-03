"""Trailing-gap mechanism experiment conditions.

    *** DRAFT. NOT EXECUTED. AWAITING RESEARCH LEAD SIGN-OFF. ***

No forecasts have been produced under this design. This module builds and
validates the conditions; it does not run them.

The corrected ``truncated_long(g)``
-----------------------------------
The naive version of this arm — hand the model the first ``L-g`` real
observations and ask for ``H`` steps — is BROKEN. The model forecasts the
``H`` steps immediately following its context, i.e. absolute indices
``[L-g, L-g+H)``, while the pilot scores against the original target at
``[L, L+H)``. That is a ``g``-step timestamp misalignment which silently
invalidates every comparison involving the arm.

The corrected design:

    context           = original_context[:L-g]   # real observations, no NaN
    prediction_length = H + g                    # NOT H
    scored_prediction = prediction[g : g+H]      # drop the first g steps
    ground truth      = the original, unchanged H-step target

so the scored window lands on absolute indices ``[(L-g)+g, (L-g)+g+H)`` =
``[L, L+H)`` — byte-identical ground truth to every other condition.

Long-horizon hazard
-------------------
``H+g`` must fit in the model's SINGLE-SHOT output. ``Chronos2Pipeline.predict``
only *warns* when ``prediction_length`` exceeds ``model_prediction_length`` and
then silently unrolls autoregressively — a different inference regime, and a
contract violation under the original pilot's rules. ``preflight`` treats it as
a hard stop, and the pipeline is additionally called with
``limit_prediction_length=True`` so the library raises rather than warns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from masks.base import Mask, apply_mask, clean_mask
from masks.contiguous_block import contiguous_block_mask

CLEAN = "clean"
TRAILING_NAN = "trailing_nan"
TRUNCATED_LONG = "truncated_long"
INTERNAL_BLOCK = "internal_block"

CONFOUNDED_NOTE = (
    "CONFOUNDED WITH REMOVED CONTENT: the two arms remove different observations, "
    "so this tests a boundary-specific penalty but is not a pure mechanism test."
)


class TrailingGapContractViolation(RuntimeError):
    """The design cannot be executed against the loaded checkpoint as specified."""


@dataclass(frozen=True)
class GapCondition:
    """One condition of the trailing-gap matrix, for one origin.

    ``scored_slice`` selects which of the returned predicted steps are scored.
    It is ``slice(0, H)`` everywhere except ``truncated_long``, where the first
    ``g`` predicted steps are dropped so the scored window realigns with the
    original target.
    """

    condition_id: str
    kind: str
    gap: int | None
    context: np.ndarray
    prediction_length: int
    scored_slice: slice
    mask: Mask | None = None
    # Absolute index in the source series of the FIRST predicted step, before
    # slicing. Used to prove timestamp alignment rather than assert it.
    first_predicted_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scored_first_index(self) -> int:
        return self.first_predicted_index + self.scored_slice.start

    @property
    def scored_length(self) -> int:
        return self.scored_slice.stop - self.scored_slice.start

    def score(self, prediction: np.ndarray) -> np.ndarray:
        """Select the scored window from a full-length prediction."""
        if prediction.shape[0] != self.prediction_length:
            raise ValueError(
                f"{self.condition_id}: expected {self.prediction_length} predicted steps, "
                f"got {prediction.shape[0]}"
            )
        scored = prediction[self.scored_slice]
        if scored.shape[0] != self.scored_length:
            raise AssertionError(f"{self.condition_id}: scored window has the wrong length")
        return scored


# --------------------------------------------------------------------------- #
# Preflight: refuse to build anything the checkpoint cannot honour
# --------------------------------------------------------------------------- #

def preflight(
    *, contract, context_length: int, horizon: int, gap_lengths: list[int],
    internal_block_distance: int,
) -> list[str]:
    """Validate the whole matrix against the LOADED model contract.

    Returns a list of violations; empty means the design is executable as
    specified. Nothing here is hardcoded from memory — the patch size, stride
    and single-shot horizon all come from the checkpoint the caller loaded.
    """
    violations: list[str] = []
    patch = int(contract.input_patch_size)

    # The usable single-shot horizon is the MINIMUM of the two fields the
    # contract reports. They currently agree (1024 and 64 x 16 = 1024), but the
    # formula must not hardcode that agreement: if a future checkpoint reports a
    # smaller model_prediction_length than its patch arithmetic implies, or vice
    # versa, the smaller number is the one that binds.
    patch_capacity = int(contract.max_output_patches) * int(contract.output_patch_size)
    single_shot = min(int(contract.model_prediction_length), patch_capacity)

    if patch_capacity != int(contract.model_prediction_length):
        violations.append(
            f"contract is internally inconsistent: model_prediction_length="
            f"{contract.model_prediction_length} but max_output_patches x "
            f"output_patch_size={patch_capacity}. Using the smaller "
            f"({single_shot}) as the binding limit."
        )

    if context_length % patch != 0:
        violations.append(
            f"context_length={context_length} is not a whole number of "
            f"{patch}-step patches."
        )

    for g in gap_lengths:
        if g % patch != 0:
            violations.append(
                f"g={g} is not a whole number of {patch}-step patches; the trailing "
                f"and internal blocks would straddle patch boundaries."
            )
        # truncated_long: the shortened context must stay patch-aligned...
        if (context_length - g) % patch != 0:
            violations.append(
                f"g={g}: truncated context length {context_length - g} is not a whole "
                f"number of {patch}-step patches; Chronos-2 would left-pad it with NaN."
            )
        if context_length - g <= 0:
            violations.append(f"g={g}: truncated context would be empty.")
        # ...and H+g must fit in a SINGLE forward pass.
        if horizon + g > single_shot:
            violations.append(
                f"g={g}: truncated_long requires prediction_length={horizon + g}, which "
                f"exceeds the model's binding single-shot limit of {single_shot} "
                f"= min(model_prediction_length={contract.model_prediction_length}, "
                f"{contract.max_output_patches} x {contract.output_patch_size}"
                f"={patch_capacity}). Chronos-2 would fall back to "
                f"AUTOREGRESSIVE UNROLLING — a different inference regime and a contract "
                f"violation. HARD STOP: this gap length is not executable as specified."
            )
        # internal_block(g) at a fixed distance must fit and stay aligned.
        block_start = context_length - internal_block_distance - g
        if block_start < 0:
            violations.append(
                f"g={g}: internal_block start {block_start} is negative at "
                f"d={internal_block_distance}."
            )
        elif block_start % patch != 0:
            violations.append(
                f"g={g}: internal_block start {block_start} is not patch-aligned."
            )
    if internal_block_distance % patch != 0:
        violations.append(
            f"internal_block_distance={internal_block_distance} is not a whole number "
            f"of {patch}-step patches, so the gap to the boundary is not a whole patch."
        )
    return violations


# --------------------------------------------------------------------------- #
# Condition builders
# --------------------------------------------------------------------------- #

def build_clean(*, context: np.ndarray, horizon: int) -> GapCondition:
    length = context.shape[0]
    return GapCondition(
        condition_id=CLEAN,
        kind=CLEAN,
        gap=None,
        context=apply_mask(context, clean_mask(length)),
        prediction_length=horizon,
        scored_slice=slice(0, horizon),
        mask=clean_mask(length),
        first_predicted_index=length,
    )


def build_trailing_nan(
    *, context: np.ndarray, horizon: int, gap: int, patch_size: int
) -> GapCondition:
    """Context length L, the final `g` steps set to NaN. Generalises d=0."""
    length = context.shape[0]
    # Reuses the pilot's contiguous-block masking: d=0 with block_length=g.
    mask = contiguous_block_mask(
        context_length=length, block_length=gap, distance=0,
        rate=gap / length, patch_size=patch_size,
    )
    return GapCondition(
        condition_id=f"{TRAILING_NAN}_g{gap}",
        kind=TRAILING_NAN,
        gap=gap,
        context=apply_mask(context, mask),
        prediction_length=horizon,
        scored_slice=slice(0, horizon),
        mask=mask,
        first_predicted_index=length,
    )


def build_truncated_long(
    *, context: np.ndarray, horizon: int, gap: int, single_shot_limit: int
) -> GapCondition:
    """The CORRECTED truncated arm: shorter context, longer request, sliced back.

    ``single_shot_limit`` is the model's real max_output_patches x
    output_patch_size. Exceeding it is a hard stop, not a warning.
    """
    length = context.shape[0]
    prediction_length = horizon + gap
    if prediction_length > single_shot_limit:
        raise TrailingGapContractViolation(
            f"truncated_long(g={gap}) needs prediction_length={prediction_length}, "
            f"exceeding the model's single-shot output of {single_shot_limit}. "
            f"Chronos-2 would silently unroll autoregressively. Refusing to build."
        )
    truncated = np.array(context[: length - gap], dtype=np.float32, copy=True)
    if np.isnan(truncated).any():
        raise AssertionError(
            f"truncated_long(g={gap}): the truncated context must contain no NaN — "
            "it is genuinely shorter, not masked."
        )
    return GapCondition(
        condition_id=f"{TRUNCATED_LONG}_g{gap}",
        kind=TRUNCATED_LONG,
        gap=gap,
        context=truncated,
        prediction_length=prediction_length,
        # Drop the first g predicted steps so the scored window realigns with
        # the ORIGINAL target.
        scored_slice=slice(gap, gap + horizon),
        mask=None,
        # The model forecasts from the end of its shortened context.
        first_predicted_index=length - gap,
        extra={
            "truncated_context_length": length - gap,
            "requested_prediction_length": prediction_length,
            "dropped_leading_steps": gap,
        },
    )


def build_internal_block(
    *, context: np.ndarray, horizon: int, gap: int, distance: int, patch_size: int
) -> GapCondition:
    """Context length L, `g` missing at a FIXED distance from the boundary."""
    length = context.shape[0]
    mask = contiguous_block_mask(
        context_length=length, block_length=gap, distance=distance,
        rate=gap / length, patch_size=patch_size,
    )
    return GapCondition(
        condition_id=f"{INTERNAL_BLOCK}_g{gap}",
        kind=INTERNAL_BLOCK,
        gap=gap,
        context=apply_mask(context, mask),
        prediction_length=horizon,
        scored_slice=slice(0, horizon),
        mask=mask,
        first_predicted_index=length,
        extra={"internal_block_distance": distance},
    )


def build_conditions(
    *, context: np.ndarray, horizon: int, config: dict[str, Any], contract
) -> list[GapCondition]:
    """The full 13-condition matrix for one origin, preflighted first."""
    design = config["design"]
    gaps = [int(g) for g in design["gap_lengths"]]
    distance = int(design["internal_block_distance"])
    length = context.shape[0]

    violations = preflight(
        contract=contract, context_length=length, horizon=horizon,
        gap_lengths=gaps, internal_block_distance=distance,
    )
    if violations:
        raise TrailingGapContractViolation(
            "the trailing-gap matrix is not executable against this checkpoint:\n  - "
            + "\n  - ".join(violations)
        )

    patch = int(contract.input_patch_size)
    single_shot = min(
        int(contract.model_prediction_length),
        int(contract.max_output_patches) * int(contract.output_patch_size),
    )

    conditions = [build_clean(context=context, horizon=horizon)]
    for gap in gaps:
        conditions.append(
            build_trailing_nan(context=context, horizon=horizon, gap=gap, patch_size=patch)
        )
        conditions.append(
            build_truncated_long(
                context=context, horizon=horizon, gap=gap, single_shot_limit=single_shot
            )
        )
        conditions.append(
            build_internal_block(
                context=context, horizon=horizon, gap=gap,
                distance=distance, patch_size=patch,
            )
        )

    expected = int(design["expected_conditions_per_origin"])
    if len(conditions) != expected:
        raise AssertionError(f"built {len(conditions)} conditions, expected {expected}")
    return conditions


def scored_timestamps(
    condition: GapCondition, *, context_timestamps: pd.DatetimeIndex, freq: str = "h"
) -> pd.DatetimeIndex:
    """Timestamps of the SCORED forecast steps, derived from the design itself.

    Computed from ``first_predicted_index`` and the scored slice rather than
    assumed, so a misaligned arm shows up as different timestamps rather than
    merely a different length.
    """
    origin = context_timestamps[0]
    start = origin + pd.tseries.frequencies.to_offset(freq) * condition.scored_first_index
    return pd.date_range(start=start, periods=condition.scored_length, freq=freq)
