"""The preregistered 17-condition grid, built per evaluation origin.

Per origin:
      1 clean
  +   6 point-random   (2 rates x 3 seeds)
  + 10 contiguous block (2 rates x 5 positions)
  = 17 forecasts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Mask, clean_mask
from .contiguous_block import contiguous_block_mask
from .point_random import point_random_mask


@dataclass(frozen=True)
class Condition:
    """One (origin-independent) cell of the design, bound to a concrete mask."""

    condition_id: str
    pattern: str            # clean | point_random | contiguous_block
    rate: float
    seed: int | None
    distance: int | None
    mask: Mask


def build_conditions(*, config: dict[str, Any], origin_id: int, patch_size: int) -> list[Condition]:
    """Build all 17 conditions for one origin. Never sees the series or target."""
    context_length = config["design"]["context_length"]
    mask_cfg = config["masks"]
    rng_cfg = mask_cfg["rng"]

    conditions: list[Condition] = [
        Condition(
            condition_id="clean",
            pattern="clean",
            rate=0.0,
            seed=None,
            distance=None,
            mask=clean_mask(context_length),
        )
    ]

    for spec in mask_cfg["rates"]:
        rate = float(spec["rate"])
        rate_tag = f"{int(round(rate * 100))}"
        for base_seed in mask_cfg["point_random_seeds"]:
            conditions.append(
                Condition(
                    condition_id=f"random_r{rate_tag}_s{base_seed}",
                    pattern="point_random",
                    rate=rate,
                    seed=int(base_seed),
                    distance=None,
                    mask=point_random_mask(
                        context_length=context_length,
                        missing_count=int(spec["missing_count"]),
                        rate=rate,
                        base_seed=int(base_seed),
                        origin_id=origin_id,
                        namespace=rng_cfg["namespace"],
                        digest_size=int(rng_cfg["digest_size_bytes"]),
                    ),
                )
            )

    for spec in mask_cfg["rates"]:
        rate = float(spec["rate"])
        rate_tag = f"{int(round(rate * 100))}"
        for distance in spec["block_distances"]:
            conditions.append(
                Condition(
                    condition_id=f"block_r{rate_tag}_d{distance}",
                    pattern="contiguous_block",
                    rate=rate,
                    seed=None,
                    distance=int(distance),
                    mask=contiguous_block_mask(
                        context_length=context_length,
                        block_length=int(spec["block_length"]),
                        distance=int(distance),
                        rate=rate,
                        patch_size=patch_size,
                    ),
                )
            )

    expected = config["design"]["expected_conditions_per_origin"]
    if len(conditions) != expected:
        raise AssertionError(f"built {len(conditions)} conditions, expected {expected}")
    if len({c.condition_id for c in conditions}) != expected:
        raise AssertionError("duplicate condition ids in the design grid")
    return conditions
