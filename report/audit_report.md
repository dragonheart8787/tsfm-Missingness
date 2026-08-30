# Chronos-2 Missingness Geometry Pilot — Audit Report

**Status: implementation complete and verified on CPU; the pilot's forecasting
run has NOT been executed.** The build environment has no GPU and its network
policy blocks `huggingface.co`, so no Chronos-2 weights could be loaded here.
Every section below that would report empirical results is marked
`PENDING EXECUTION` rather than filled with a placeholder. No number in this
document is fabricated.

This report is written to be read alongside `results/<run>/decision.json`, which
is the machine-readable, deterministic decision output. Where this prose and
that file could ever disagree, the file is authoritative.

---

## 1. What this pilot tests

Whether, for Chronos-2 forecasting ETTh1's `OT` series with a 320-step context
and a 96-step horizon, forecast error depends on **where** missing history sits
(distance of a contiguous gap from the forecast boundary) and on the **shape**
of the missingness (scattered points vs. one contiguous block), at 20% and 40%
missing rates.

It is a pilot. Its output is a GO / PIVOT / NO-GO decision about whether a
larger study is warranted. A NO-GO is a complete, successful outcome.

### Novelty claims: none

This pilot claims no novelty for TSFM missingness benchmarking, for evaluating
Chronos-2 under missing history, for the observation that imputation can harm
forecasts, for forecast proximity as a causal mechanism, or for any
proximity-weighted severity score. It is a small, preregistered measurement.

---

## 2. Observed result

`PENDING EXECUTION.` Populated from `results/<run>/condition_summary.csv` and
`contrast_results.csv` after the GPU run.

To be reported here, without interpretation:

* mean clean MAE across the 178 origins;
* per-condition mean MAE, pooled RMSE and RED for all 17 conditions;
* the four preregistered contrasts as raw MAE differences and as percentages of
  mean clean MAE;
* the number of failed forecasts, if any, with their recorded error messages.

---

## 3. Statistical evidence

`PENDING EXECUTION.` Populated from `bootstrap_results.csv` and `analysis.json`.

The analysis that will run, fixed in advance:

* **Pairing.** Everything is paired by evaluation origin. The three point-random
  seeds are averaged *within* origin before any contrast is formed; they are
  never treated as three independent samples. The 96 forecast timestamps inside
  an origin are collapsed into that origin's MAE before any resampling; they are
  never treated as 96 independent samples.
* **Bootstrap.** Moving-block bootstrap over the 178 chronologically ordered
  origins. Main block length 8; sensitivity at 4 and 12. 5,000 replicates.
  Every condition is resampled **jointly** within a block — the resample draws
  origin indices and applies them to the whole origins-by-conditions matrix, so
  pairing survives by construction rather than by convention.
* **Multiplicity.** Holm–Bonferroni across the four contrasts at family
  alpha = 0.05. Reported per contrast: raw MAE difference, difference as a
  percentage of mean clean MAE, Holm-corrected 95% interval, Holm-adjusted
  p-value, median paired difference, fraction of origins with a positive paired
  difference, and the estimate and interval under each of the three bootstrap
  block lengths.
* **Equivalence.** Separately, uncorrected 90% intervals, used only for the
  NO-GO equivalence assessment.

**A note on "Holm-corrected CI excludes zero".** Holm is a procedure on
p-values, so the decision function operationalises this criterion as the Holm
rejection decision (adjusted p < 0.05). Because the p-value used is the
percentile p-value from the same bootstrap distribution that generates the
interval, "the interval at that level excludes zero" and "Holm rejects" are the
same event, not an approximation of one another. The Holm-adjusted interval
bounds are reported alongside so an auditor can check this directly.

---

## 4. Plausible explanation

`PENDING EXECUTION.` This section is reserved for explanations that are
consistent with the observed numbers but **not** tested by this pilot. It will
be written only after sections 2 and 3 are populated, and each entry will be
explicitly labelled as untested here.

Nothing in this report will describe any association found in this pilot as a
causal mechanism.

---

## 5. Tested mechanism

**This pilot tests no mechanism.** It measures associations between mask
geometry and forecast error under a fixed model and dataset. The design has no
intervention that isolates a mechanism, and no mechanism is estimated.

The one thing the design does control cleanly: for the location contrasts (C1,
C2) the two arms have the **same missing count**, the **same block length**, the
**same pattern**, the **same series**, and the **same scored target** — they
differ only in where the block sits. That makes the location contrast a clean
comparison *of block position*. It does **not** make it an estimate of "recency
importance", because block position varies several things at once (see §7).

### Contrasts 3 and 4 are pipeline comparisons

C3 and C4 compare a point-random arm against a centred-block arm. These are
labelled `pipeline_comparison` in the config, in `contrast_results.csv`, in
`analysis.json`, in the code, and in the figure titles. They are **not** pure
causal estimates of contiguity: the two arms differ in the position of the
removed observations, in patch occupancy, and in which specific observations
are removed, all at once. Any difference between them is a difference between
two whole pipelines, not an isolated contiguity effect.

---

## 6. Untested speculation

`PENDING EXECUTION.` Anything in this section is explicitly speculation, is
labelled as such, and feeds no part of the decision rule.

---

## 7. Live alternative explanations, and what this pilot can and cannot separate

If a boundary block (d=0) underperforms a far block (d=256 or d=192), **do not
conclude that recency is intrinsically more important.** At least the following
alternatives are live, and they are not mutually exclusive:

| # | Alternative explanation | Can this pilot's diagnostics distinguish it? |
|---|---|---|
| 1 | **Effective forecast distance.** With the last 64 (or 128) observations removed, the model's nearest real observation is 64 (128) steps further from the forecast, so it is effectively forecasting a longer horizon. | **No.** This is inseparable from block position by construction — moving the block toward the boundary *is* increasing the effective forecast distance. The diagnostics record `nearest_missing_to_boundary` and `trailing_run_length`, which quantify it, but the design cannot break the confound. This is the single most important limitation of the location contrast. |
| 2 | **Removed local volatility / level / slope.** The removed segment near the boundary may simply be more informative — higher variance, a stronger trend, a level shift — than a segment 256 steps back. | **Partially.** `removed_variance`, `removed_slope`, `removed_abs_diff_mean`, `removed_mean`, `removed_min/max` are recorded per mask, and the decision function computes their rank correlation with the paired difference and compares it against distance's. It can detect that a confounder *out-tracks* distance; it cannot cleanly decompose the two. |
| 3 | **Chronos-2's internal scaling.** Chronos-2 applies a NaN-aware instance normalisation whose location and scale are computed from the *retained* observations only. Removing a boundary block therefore changes the normalisation itself, not just the information content. | **Partially.** `retained_context_mean/std`, `retained_mean_change_frac_of_clean_std` and `retained_std_ratio` are recorded per mask, and `rho_scaling` is a PIVOT trigger. This detects association with the scaling shift; it does not separate "the model was renormalised" from "the model lost information". |
| 4 | **Patch occupancy / patch position.** Chronos-2 patches the context; a mask that wipes whole patches removes whole tokens, while one that partially fills patches degrades them. | **Largely yes, for the block contrasts.** All ten block positions are asserted to be exactly patch-aligned on the model's real, verified patch grid, so every block arm removes whole patches and `n_patches_partially_missing == 0`. Any nonzero value there is a PIVOT trigger. For the random arms partial-patch occupancy is unavoidable and large — which is precisely part of why C3/C4 are pipeline comparisons. |
| 5 | **Seasonal phase of the removed observations.** A 64-step block spans exactly 2.67 days; hour-of-day composition varies with position. | **Partially.** `removed_hour_entropy_normalised` and `removed_hour_max_share` are recorded. For blocks whose length is a multiple of 24 the composition is near-uniform by construction, which limits how much this can explain — but 64 and 128 are not multiples of 24, so it is not eliminated. |
| 6 | **A handful of origins.** ETTh1 has regime shifts; a few origins could carry the whole effect. | **Yes.** The top-5% share of the summed paired difference is computed per contrast and is a PIVOT trigger, and the fraction of origins with a positive paired difference plus the median paired difference are reported for every contrast. |
| 7 | **Serial dependence between origins.** Consecutive origins overlap (stride 96 < context 320), so they are not independent. | **Partially.** The moving-block bootstrap is specifically there to respect this, and results are reported at three block lengths. Instability across block lengths is a PIVOT trigger. |

**Bottom line for the Research Lead:** even a large, clean, statistically strong
location effect in this pilot would be consistent with alternatives 1, 2 and 3,
and this design cannot rank them. Alternative 1 in particular is structural: it
cannot be removed by more origins, more seeds or a bigger dataset. Separating it
requires a design change, not more data.

---

## 8. Decision

`PENDING EXECUTION.` The label comes from `stats/decision.py`, a pure
deterministic function over the computed statistics. It is not a narrative
judgement and cannot be overridden by this document.

Precedence, fixed in advance and unit-tested:

1. **NO-GO** — all four 90% intervals lie entirely within ±3% of mean clean MAE,
   **and** there is no stable location ordering, **and** random-seed sensitivity
   is small. If this fires, the pilot **stops**: no datasets, conditions,
   models or rates are added to search for significance.
2. **PIVOT** — any PIVOT trigger fires. PIVOT deliberately **outranks** GO: a
   real but confounded effect is a design problem, not a green light.
3. **GO** — location GO or geometry GO criteria met, and no PIVOT trigger.
4. **PIVOT (residual)** — neither a clean null nor a clean effect; recorded with
   trigger `inconclusive_no_criteria_met`.

**Location GO** requires, at *both* 20% and 40%: the boundary-vs-far contrast is
the same sign at both rates; its magnitude is ≥ 5% of mean clean MAE; the
Holm-corrected 95% interval excludes zero; and the median paired difference has
the same sign as the mean. **Geometry GO** is the analogous rule on C3/C4 —
with the standing caveat that those are pipeline comparisons.

**PIVOT triggers**, each implemented and unit-tested: effect only at 40%;
material shift across bootstrap block lengths; a handful of origins dominating;
a removed-segment confounder out-tracking distance; internal scaling explaining
the pattern; patch position as a plausible confound.

---

## 9. Verified facts about Chronos-2 used by this design

Read from `chronos-forecasting` 2.3.1 source on the build host, and re-verified
against the loaded checkpoint at run time by `scripts/verify_model_contract.py`
and again inside the runner:

* **Missing history is expressed as float NaN.** `Chronos2Model._prepare_patched_context`
  computes `context_mask = ~torch.isnan(context)`, normalises NaN-aware, patches
  both, zeroes unobserved positions, and feeds the mask to the model as an
  explicit channel. NaN passthrough is the supported convention. No timestamp is
  ever dropped.
* **Normalisation is NaN-aware and context-wide.** `InstanceNorm` uses
  `nanmean`, so removing observations changes the location and scale actually
  applied. This is alternative explanation #3 above, and the reason the
  retained-context diagnostics exist.
* **`cross_learning` defaults to `False`** and, when `True`, shares information
  across every task in a call. The config forces it `False` and the runner
  additionally submits exactly one origin per call.
* **`predict_quantiles` returns a second value named `mean` that is in fact the
  q=0.5 column.** The pilot does not use it. It requests
  `quantile_levels=[0.5]` explicitly and asserts that 0.5 is one of the model's
  own trained quantiles, so no interpolation path is taken.
* **The context is left-padded with NaN if its length is not a whole number of
  patches.** L = 320 is 20 patches on a 16-step grid, so no padding occurs and
  every block position sits on a real patch boundary. The contract check fails
  loudly if this ever stops being true.

### Patch grid: verified at run time, not assumed

The preregistered design assumes a **16-step patch grid**, and all ten block
positions are aligned to it:

| Rate | Block length | d values | block_start values | All multiples of 16 |
|---|---|---|---|---|
| 20% | 64 | 0, 64, 128, 192, 256 | 256, 192, 128, 64, 0 | yes |
| 40% | 128 | 0, 48, 96, 144, 192 | 192, 144, 96, 48, 0 | yes |

The real patch size lives in the checkpoint's `chronos_config`, which the build
host could not reach. It is therefore **never hardcoded**: the runner reads
`input_patch_size` from the loaded model and raises `PatchGridMismatch`,
refusing to run, if it is not 16. That is a stop-and-report, not a silent
adjustment — every d value would need recomputing on a different grid.

---

## 10. Integrity guarantees, and how each is enforced

| Guarantee | Enforcement |
|---|---|
| The scored target is byte-identical across all 17 conditions of an origin | sha256 of the target bytes recorded per condition and compared **at run time** in `assert_target_integrity`, before anything is persisted; plus offline tests |
| Masking can never reach the target | Structural: mask generators accept only `context_length: int`, never a series or a target array. `apply_mask` receives only the context. There is no code path to an index ≥ the context boundary. Asserted by signature inspection in the test suite |
| The target cannot be mutated | The target is a read-only numpy view; a write raises `ValueError` |
| No missing value in the scored target | `compute_metrics` refuses a non-finite target; the runtime guard re-checks |
| Rows and timestamps are never changed | Missingness is written in place as NaN; length is asserted after every mask application |
| Exact missing counts | Sampling without replacement, never Bernoulli; asserted per mask |
| Block position, length, contiguity and patch alignment | Asserted per block against the model's verified patch size |
| Deterministic, process-stable masks | BLAKE2b digest over a canonical payload; builtin `hash()` never used; verified by recomputing in a fresh interpreter |
| Per-origin isolation | One origin per call, `cross_learning=False`, plus an adversarial leakage probe |
| Resume never duplicates or overwrites | Atomic checkpoint written **after** rows land on disk; resume intersects the checkpoint with what the results file actually contains |
| Failures are recorded, not swallowed | Every failure produces its result row with `status="failed"` and the exception message |

---

## 11. Scope

Implemented exactly as specified. Deliberately **not** implemented, and not to
be added without a Research Lead decision: 10% missingness; periodic
missingness; linear interpolation; forward-fill; ARIMA; XGBoost; DLinear;
TimesFM; MOMENT; TS-ICL; additional datasets; hyperparameter searches; any
fitted forecast-proximity weighting parameter.

No severity metric is fitted from the confounder diagnostics in this pilot.
