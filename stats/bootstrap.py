"""Moving-block bootstrap over chronologically ordered evaluation origins.

Pairing discipline
------------------
The bootstrap resamples ORIGIN INDICES, never per-condition values. Every
condition is therefore carried jointly by a resampled block, so the pairing
between conditions within an origin survives resampling by construction. This
is what ``tests/test_bootstrap_pairing.py`` verifies adversarially.

Independence discipline
-----------------------
  * the 3 point-random seeds are averaged WITHIN origin before any contrast;
  * the 96 forecast timestamps inside an origin are collapsed into that
    origin's MAE before resampling.
Neither is ever treated as an independent sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def moving_block_indices(
    *, n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """One moving-block resample of the origin index set, length n.

    Blocks of `block_length` consecutive origins are drawn with replacement from
    all n - block_length + 1 possible start positions, concatenated, then
    truncated to exactly n. Consecutive origins overlap in time (stride 96 with
    L=320), so blocking is what keeps the resample honest about that dependence.
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    block_length = min(block_length, n)
    n_starts = n - block_length + 1
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n_starts, size=n_blocks)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]).reshape(-1)
    return indices[:n]


@dataclass
class BootstrapResult:
    contrast_id: str
    block_length: int
    n_replicates: int
    n_origins: int
    observed_mean: float
    observed_median: float
    frac_positive: float
    ci_low_corrected: float
    ci_high_corrected: float
    ci_level_corrected: float
    ci_low_equivalence: float
    ci_high_equivalence: float
    ci_level_equivalence: float
    p_value_two_sided: float
    boot_mean: float
    boot_std: float
    holm_adjusted_p: float | None = None
    holm_alpha: float | None = None
    holm_rejects: bool | None = None
    ci_low_holm: float | None = None
    ci_high_holm: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_paired_difference(
    *,
    contrast_id: str,
    paired_differences: np.ndarray,
    block_length: int,
    n_replicates: int,
    seed: int,
    ci_level_corrected: float = 0.95,
    ci_level_equivalence: float = 0.90,
    joint_matrix: np.ndarray | None = None,
) -> tuple[BootstrapResult, np.ndarray]:
    """Bootstrap the mean paired difference for one contrast.

    ``paired_differences`` is one value per origin, in chronological order.
    ``joint_matrix``, when supplied, is the full origins x conditions matrix
    resampled with the SAME index draw, so callers can verify that pairing was
    preserved.
    """
    diffs = np.asarray(paired_differences, dtype=np.float64)
    if diffs.ndim != 1:
        raise ValueError("paired_differences must be 1-dimensional")
    n = diffs.size
    if n == 0:
        raise ValueError("no origins to bootstrap")

    rng = np.random.default_rng(seed)
    replicate_means = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        idx = moving_block_indices(n=n, block_length=block_length, rng=rng)
        replicate_means[r] = diffs[idx].mean()
        if joint_matrix is not None:
            # Same draw applied to every condition: pairing is structural.
            _ = joint_matrix[idx, :]

    def percentile_ci(level: float) -> tuple[float, float]:
        alpha = 1.0 - level
        low, high = np.percentile(replicate_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(low), float(high)

    ci_low_c, ci_high_c = percentile_ci(ci_level_corrected)
    ci_low_e, ci_high_e = percentile_ci(ci_level_equivalence)

    # Bootstrap percentile p-value, consistent BY CONSTRUCTION with whether the
    # percentile CI at the same level excludes zero.
    frac_le_zero = float(np.mean(replicate_means <= 0.0))
    frac_ge_zero = float(np.mean(replicate_means >= 0.0))
    p_value = min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero))
    p_value = max(p_value, 1.0 / (n_replicates + 1))

    result = BootstrapResult(
        contrast_id=contrast_id,
        block_length=int(block_length),
        n_replicates=int(n_replicates),
        n_origins=int(n),
        observed_mean=float(diffs.mean()),
        observed_median=float(np.median(diffs)),
        frac_positive=float(np.mean(diffs > 0)),
        ci_low_corrected=ci_low_c,
        ci_high_corrected=ci_high_c,
        ci_level_corrected=float(ci_level_corrected),
        ci_low_equivalence=ci_low_e,
        ci_high_equivalence=ci_high_e,
        ci_level_equivalence=float(ci_level_equivalence),
        p_value_two_sided=float(p_value),
        boot_mean=float(replicate_means.mean()),
        boot_std=float(replicate_means.std(ddof=1)),
    )
    return result, replicate_means


def holm_correct(
    results: list[BootstrapResult],
    replicate_means: dict[str, np.ndarray],
    *,
    family_alpha: float = 0.05,
) -> list[BootstrapResult]:
    """Apply Holm-Bonferroni across the family of contrasts, in place.

    Holm is a step-down procedure on p-values. We record, per contrast:
      * ``holm_adjusted_p`` — the standard monotone-enforced adjusted p-value;
      * ``holm_alpha``      — that contrast's step-specific alpha, alpha/(m-k+1);
      * ``holm_rejects``    — adjusted p < family_alpha;
      * ``ci_low_holm/ci_high_holm`` — the percentile interval at level
        1 - holm_alpha.

    The decision function's "Holm-corrected 95% CI excludes zero" criterion is
    operationalised as ``holm_rejects``. Because our p-value is the percentile
    p-value from the same bootstrap distribution, rejecting at a given alpha and
    the corresponding percentile interval excluding zero are the same event, so
    the two phrasings agree rather than merely approximating each other.
    """
    m = len(results)
    order = sorted(range(m), key=lambda i: results[i].p_value_two_sided)
    running_max = 0.0
    for step, i in enumerate(order, start=1):
        alpha_k = family_alpha / (m - step + 1)
        adjusted = min(1.0, (m - step + 1) * results[i].p_value_two_sided)
        adjusted = max(adjusted, running_max)  # enforce monotonicity
        running_max = adjusted

        results[i].holm_alpha = float(alpha_k)
        results[i].holm_adjusted_p = float(adjusted)
        results[i].holm_rejects = bool(adjusted < family_alpha)

        draws = replicate_means[results[i].contrast_id]
        low, high = np.percentile(draws, [100 * alpha_k / 2, 100 * (1 - alpha_k / 2)])
        results[i].ci_low_holm = float(low)
        results[i].ci_high_holm = float(high)
    return results
