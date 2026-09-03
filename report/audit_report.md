# Chronos-2 Missingness Geometry Pilot — Audit Report

**Status: implementation complete and verified on CPU; the pilot's forecasting
run has NOT been executed.** The build environment has no GPU and its network
policy blocks `huggingface.co`, so no Chronos-2 weights could be loaded here.
Every section below that would report empirical results is marked
`PENDING EXECUTION` rather than filled with a placeholder. No number in this
document is fabricated.

**Update.** The pilot has since been executed and `results/pilot_v1/` exists on
the operator's host. Its outputs are not present in the environment where this
report is maintained, because `results/` is gitignored and was never committed.
The `PENDING EXECUTION` markers below therefore still stand, and still mean
exactly what they said: the numbers are not available here, and none have been
invented to fill the gap. §9.10 lists the commands that produce them.

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

`PENDING EXECUTION` — **and still pending for a reason that is not the GPU run.**
The pilot has been executed and `results/pilot_v1/` exists on the operator's
host, but that directory is not present in the environment where this report is
maintained (`results/` is gitignored and the outputs were never committed). The
numbers are therefore not available to paste in here, and none have been
invented. Filling this section requires running, on the host that holds the run:

```bash
.venv/bin/python scripts/analyze_run.py --run-dir results/pilot_v1 \
  --out-dir results/pilot_v1/post_hoc_reanalysis_v2
.venv/bin/python scripts/analyze_internal_only.py --run-dir results/pilot_v1
```

and pasting the outputs. Sources once available: `condition_summary.csv` and
`contrast_results.csv`.

To be reported here, without interpretation:

* mean clean MAE across the 178 origins;
* per-condition mean MAE, pooled RMSE and RED for all 17 conditions;
* the four preregistered contrasts as raw MAE differences and as percentages of
  mean clean MAE;
* the number of failed forecasts, if any, with their recorded error messages.

---

## 3. Statistical evidence

`PENDING EXECUTION` — for the same reason as §2: the run exists, but its outputs
are not present in this environment. Sources once available:
`bootstrap_results.csv` and `analysis.json`, plus the v2 decision re-derivation
in `post_hoc_reanalysis_v2/`.

The analysis that runs, fixed in advance:

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

`PENDING EXECUTION` — for the same reason as §2. Anything eventually written
here is explicitly speculation, is labelled as such, and feeds no part of the
decision rule.

**Language discipline for this section when filled.** The trailing-gap and
effective-horizon readings of the boundary jump (§9.4) belong here, as
speculation, until the experiment drafted in
`docs/next_round_trailing_gap_mechanism.md` has been run. Specifically: this
pilot does not distinguish loss of recent observations from Chronos-2-specific
handling of trailing missingness, and no wording here may imply that it does.

---

## 7. Live alternative explanations, and what this pilot can and cannot separate

If a boundary block (d=0) underperforms a far block (d=256 or d=192), **do not
conclude that recency is intrinsically more important.** At least the following
alternatives are live, and they are not mutually exclusive:

| # | Alternative explanation | Can this pilot's diagnostics distinguish it? |
|---|---|---|
| 1 | **Effective forecast distance.** With the last 64 (or 128) observations removed, the model's nearest real observation is 64 (128) steps further from the forecast, so it is effectively forecasting a longer horizon. | **No.** This is inseparable from block position by construction — moving the block toward the boundary *is* increasing the effective forecast distance. The diagnostics record `nearest_missing_to_boundary` and `trailing_run_length`, which quantify it, but the design cannot break the confound. This is the single most important limitation of the location contrast. **§9 partially addresses this post-hoc** by restricting a re-analysis to the internal positions, where effective forecast distance is constant — but that is an exploratory restriction of the comparison, not a fix to the preregistered contrast. |
| 2 | **Removed local volatility / level / slope.** The removed segment near the boundary may simply be more informative — higher variance, a stronger trend, a level shift — than a segment 256 steps back. | **Diagnostics recorded, but the automated comparison is NOT EVALUABLE here.** `removed_variance`, `removed_slope`, `removed_abs_diff_mean`, `removed_mean`, `removed_min/max` are recorded per mask and remain available for inspection. The automated "does this confounder out-track distance?" check, however, cannot be run on the preregistered contrasts: each is a fixed two-arm comparison in which distance is constant across all 178 origins, so distance's rank correlation is undefined. See §8.1. Distinguishing these confounders from distance needs a design that varies distance within the comparison. |
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

`PENDING EXECUTION` — for the same reason as §2. The label comes from
`stats/decision.py`, a pure deterministic function over the computed statistics.
It is not a narrative judgement and cannot be overridden by this document.

Two files will carry it, and they must not be conflated (see §8.1):
`results/pilot_v1/decision.json` (rule **v1**, the preregistered record of what
the original run produced) and
`results/pilot_v1/post_hoc_reanalysis_v2/decision.json` (rule **v2**, the
post-hoc correction). Quote the label from both, with their versions.

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
the pattern; patch position as a plausible confound. Three of these compare a
confounder's rank correlation against distance's; on the preregistered contrasts
those comparisons are **not evaluable** and carry no weight — see §8.1.

### 8.1 Post-run correction: `confounder_out_tracks_distance` is not evaluable

PIVOT was originally triggered by three automated checks. Post-run audit found
that `confounder_out_tracks_distance` was not evaluable for the preregistered
two-arm contrasts because distance was constant and its undefined correlation
had been coerced to zero. This trigger therefore receives no evidentiary weight.
The PIVOT decision remains unchanged based on bootstrap instability,
scaling-related diagnostics, distributional inconsistency, and the unresolved
effective-forecast-distance confound.

**Why the correlation is undefined.** Each preregistered contrast compares two
fixed arms — for C1, the d=0 block against the d=256 block. Every origin's
subtrahend arm therefore sits at the *same* distance to the forecast boundary,
so `mean_missing_distance_to_boundary` is constant across all 178 origins and a
Spearman correlation against it has no defined value. Decision rule v1 replaced
that undefined value with `0.0` before comparing each confounder's |rho| against
it. That manufactures a baseline of "distance explains nothing", against which
almost any confounder wins — so the check fired largely independently of the
data. Raising the margin would not repair this: it would only make an invalid
comparison more conservative, not valid.

**What changed.** Decision rule **v2** reports each distance-relative check as
one of four explicit states — `not_evaluable`, `invalid_input`, `false`, `true`
— and only `true` may contribute a trigger. `not_evaluable` is structurally
distinct from `false`: it means the question could not be asked, not that the
answer was no. The same NaN-to-zero collapse appeared in two further places, and
both are fixed identically:

| Check | Distance-free branch | Distance-relative branch |
|---|---|---|
| `confounder_out_tracks_distance` | none — wholly distance-relative | `not_evaluable` |
| `internal_scaling_explains_pattern` | absolute \|rho_scaling\| ≥ threshold — **still live** | `not_evaluable` |
| `patch_position_confound` | partially-missing patches in a block arm — **still live** | `not_evaluable` |

The distance-free branches are unaffected and continue to carry evidentiary
weight normally.

**Where to find each version's output.**

| Version | File | Status |
|---|---|---|
| v1 | `results/pilot_v1/decision.json`, `results/pilot_v1/analysis.json` | **Preregistered record of the original run. Not modified.** |
| v2 | `results/pilot_v1/post_hoc_reanalysis_v2/decision.json`, `.../analysis.json` | **Post-hoc sensitivity analysis**, re-derived from the same `window_results.csv` and `mask_diagnostics.csv` with no new model inference. Not a replacement for the preregistered output. |

Every decision output carries a `decision_rule_version` field, and the analysis
step refuses to overwrite a `decision.json` written under a different rule
version unless explicitly told to.

**This is a correction to the decision logic, not to the measurements.** No
forecast, metric, bootstrap interval or contrast estimate changes between v1 and
v2 — only which automated checks are permitted to claim they found something.

---

## 9. POST-HOC EXPLORATORY: internal-only block-proximity re-analysis

> **This entire section is post-hoc and exploratory.** It was not part of the
> original preregistration, it feeds no GO/PIVOT/NO-GO decision function, and it
> carries no confirmatory weight. It re-analyses data already collected in the
> completed run — no new forecasts, no new data.

### 9.1 Rationale

The preregistered C1/C2 contrasts compare `d=0` against a far position. That
conflates two things that move together: block position within the context, and
**effective forecast distance** — how far back the model's nearest *real*
observation sits from the forecast start.

| Position | Nearest real observation | Effective forecast distance |
|---|---|---|
| `d=0` (block abuts the boundary) | `block_length + 1` steps back — 65 at 20%, 129 at 40% | inflated |
| `d=64, 128, 192, 256` (20%) | exactly 1 step back | unchanged |
| `d=48, 96, 144, 192` (40%) | exactly 1 step back | unchanged |

Restricting the analysis to the **internal** positions therefore holds effective
forecast distance constant, isolating genuine block-proximity variation from
that confound — the confound §7 row 1 identified as structurally inseparable in
the preregistered contrasts. It is separable here only because the comparison is
restricted, not because more data was gathered.

`d=0` is **not discarded**. It is reported separately in §9.5 as a distinct
**trailing-boundary condition**, and is never pooled into any internal-position
statistic. An automated invariant enforces this.

Rates are analysed separately and never pooled. Everything is paired by
evaluation origin and uses the same moving-block bootstrap already implemented
(`stats/bootstrap.py`): main block length 8, sensitivity at 4 and 12, 5,000
replicates, every condition resampled jointly. Nothing was reimplemented.

### 9.2 Multiplicity families

Four **independent** Holm families. The original preregistered family is not
merged into any of them and is **not re-corrected**.

| Family | Members | Note |
|---|---|---|
| `{C1, C2, C3, C4}` | 4 | Original preregistered family — untouched |
| `{IC1_20, IC2_40}` | 2 | §9.3 internal-position primary |
| `{BJ1_20, BJ2_40}` | 2 | §9.4 boundary jump, independent of every other family |
| Pairwise, 20% | 6 | §9.4, corrected within rate |
| Pairwise, 40% | 6 | §9.4, corrected within rate — two families of 6, never one of 12 |
| Confounder Δz, per contrast | 4 each | §9.6 — two families of 4, never one of 8 |

### 9.3 Primary: internal-near vs. internal-far

`IC1_20` = 20%, `d=64` − `d=256`. `IC2_40` = 40%, `d=48` − `d=192`.

`PENDING EXECUTION` — populated from
`results/pilot_v1/post_hoc_internal_only_analysis/internal_primary_contrasts.csv`.
Reported in the same format as C1–C4: raw MAE difference, % of mean clean MAE,
Holm-corrected 95% CI, Holm-adjusted p, 90% CI, median paired difference,
fraction of origins > 0, and sensitivity across bootstrap block lengths 4/8/12.

### 9.4 Boundary-jump contrasts (`BJ1_20`, `BJ2_40`)

* `BJ1_20` = 20%: `d=0` − `d=64`
* `BJ2_40` = 40%: `d=0` − `d=48`

**Computed directly, not by subtraction.** An earlier reading obtained the
boundary jump by subtracting two already-bootstrapped contrasts
(`C1_loc_20 − IC1_20`). That arithmetic is valid for the **point estimate** —
the operation is linear, and a test asserts the two agree — but it is **not a
valid confidence interval**. Both contrasts are measured on the same 178
origins, so their difference's sampling distribution carries a covariance
structure that differencing two separately drawn intervals discards. A test
demonstrates that the naive subtraction yields a materially wider, and therefore
wrong, interval. `BJ1_20` and `BJ2_40` are bootstrapped directly from the
per-origin paired MAE values at the two conditions.

Own Holm family `{BJ1_20, BJ2_40}`, independent of `{C1..C4}`, of
`{IC1_20, IC2_40}`, of the two 6-item pairwise families, and of the
confounder-difference families.

`PENDING EXECUTION` — populated from
`results/pilot_v1/post_hoc_internal_only_analysis/boundary_jump_contrasts.csv`.
Reported in the same format as C1–C4 and IC1/IC2, plus the top-5% origin share
(computed by the shared implementation in `stats/decision.py`).

**Language discipline for this subsection when filled.** Report magnitude, CI
and significance factually. Do **not** write that the jump is "localised
entirely at the boundary", or upgrade it to a causal or definitive locational
claim: the contrast measures a difference between two conditions, and §9.9 and
the next-round design (`docs/next_round_trailing_gap_mechanism.md`) exist
precisely because its mechanism is not established here.

### 9.5 Equivalence check on the internal contrasts — non-significance is not equivalence

`IC1_20` and `IC2_40` not reaching significance does **not** establish that
there is no internal-position effect. The pilot already has a convention for
when equivalence may be claimed, and it is applied here rather than eyeballed:
the preregistered NO-GO rule's band, `decision.no_go.equivalence_band_frac_of_clean_mae`
(±3% of mean clean MAE), with the requirement that the **90% CI lie entirely
inside it**. The threshold is imported from that single definition; a test
asserts the literal is not copied.

Each contrast reports an explicit boolean `equivalence_established`, in
`internal_equivalence_check.csv`, alongside the band's absolute value.

`PENDING EXECUTION` — the boolean result per contrast.

**Mandated wording for the internal-only finding**, replacing any formulation
that says the location effect "vanishes" or that there is "no effect":

> In a post-hoc internal-only analysis, we found no consistent evidence of a
> monotonic block-location effect when observations immediately preceding the
> forecast boundary remained available. The large endpoint contrast was
> concentrated in the boundary-touching condition. This pattern is consistent
> with trailing-gap or effective-horizon fragility, but does not distinguish the
> loss of recent observations from Chronos-2-specific handling of trailing
> missingness.

### 9.6 Secondary: repeated-measures across the four internal positions

**Categorical (diagnostic only, feeds nothing).** All 6 pairwise paired
contrasts among the internal positions, per rate, Holm-corrected within rate.
`PENDING EXECUTION` — `internal_pairwise_contrasts.csv`.

**Ordered trend.** Per origin, the OLS slope of paired-difference-from-clean
against `d` across that origin's four internal positions; then the across-origin
mean slope bootstrapped with the same machinery. Reported per rate with its 95%
CI and whether that CI excludes zero. No multiplicity correction is applied —
this is one descriptive slope per rate, not a test family, and the output says
so. `PENDING EXECUTION` — `internal_trend_slopes.csv`, with the per-origin
slopes in `per_origin_trend_slopes.csv`.

*Note on the slope's response variable:* the per-origin clean MAE is a constant
offset across the four positions, so `slope(MAE_d − MAE_clean)` is identically
`slope(MAE_d)`. The paired difference is used to keep the response on the same
scale as everything else reported here; a test pins the equivalence.

### 9.7 The trailing-boundary condition (`d=0`), reported separately

`PENDING EXECUTION` — `trailing_boundary_d0.csv`. Reported per rate with the
same summary statistics as the internal contrasts, under
`position_kind: trailing_boundary_condition`, and never pooled with them. The
underlying numbers are from the original run; only their presentation is new.

### 9.8 Revised confounder diagnostic: two-arm difference correlation

**Correction from the previous round.** This diagnostic was previously computed
against `IC1_20` / `IC2_40`, i.e. against the small internal-position variation.
That tests what might explain the *internal* variation and says nothing about
what drives the boundary jump. It is now computed against the **boundary-jump
contrasts**, where Δz = z(`d=0`) − z(nearest internal position), correlated
against the corresponding `BJ` paired MAE difference, per origin.

The earlier `IC1_20` / `IC2_40` results are **retained, not deleted**, in
`confounder_difference_internal_contrasts.csv`, relabelled to state exactly what
they show: association (or its absence) with the small internal-position
variation, and **not** a test of what drives the boundary jump.

This is **well-posed**, and does not suffer the NaN problem that made the
distance-based check not evaluable (§8.1): Δz genuinely varies across origins
because the removed content differs by origin, even though the two positions
being compared are fixed.

The confounder family is **pre-specified**, fixed before any result was
inspected — the same discipline the rest of the pilot is built on:

* `removed_variance`
* `removed_slope`
* `removed_abs_diff_mean`
* `retained_mean_change_frac_of_clean_std` (the scaling-shift proxy)

Each result carries the same `not_evaluable` / `invalid_input` / `false` /
`true` status typing introduced in the v1→v2 patch, so this diagnostic cannot
regress into the NaN-to-zero collapse that was fixed there. Because Δz is a real
difference of real per-origin values, a non-finite Δz here would indicate an
actual **data problem**: it is classified `invalid_input` and surfaced, never
`not_evaluable` and never silently zeroed. `not_evaluable` and `invalid_input`
results are excluded from the Holm family entirely and can never become `true`.

**Both a parametric p-value and a bootstrap CI on ρ are reported, and neither
stands alone.** The parametric Spearman p-value assumes independent
observations; evaluation origins are not independent (stride 96 < context 320).
ρ is therefore additionally bootstrapped with the same moving-block procedure
(block lengths 4/8/12, 5,000 replicates, the same resampled origin index applied
to both series so the pairing that defines ρ is preserved). The two can and do
disagree — a Holm-adjusted `false` alongside a bootstrap CI excluding zero is a
meaningful signal, not a contradiction to be resolved by picking one.

**Language discipline for this subsection when filled.** Replace any formulation
claiming "no alternative driver detected" or that a "confound audit has ruled
out" an explanation with:

> No statistically detectable association was found among the four measured
> internal-contrast diagnostics; this analysis does not evaluate drivers of the
> boundary jump.

— and report the `BJ`-based results (above) as the actually-relevant diagnostic
for the boundary jump specifically. Four measured diagnostics returning null is
not an audit that rules anything out.

`PENDING EXECUTION` — `confounder_difference_correlations.csv` (boundary jump,
primary) and `confounder_difference_internal_contrasts.csv` (internal variation,
retained and relabelled).

### 9.9 Rejected alternative: the sliding-forecast-origin design

A previously considered design would have slid the forecast origin forward to
vary effective forecast distance directly. **It is rejected and no code for it
exists.**

The reason: sliding the forecast origin changes the target window, the clean
context, the market/seasonal regime, and the intrinsic difficulty of the target
*all at once*. Any difference it produced would be attributable to those changes
as readily as to effective forecast distance. It swaps one confound for another
rather than isolating anything — and unlike the present confound, the
substitution is not even visible in the diagnostics.

**Sketch of what a corrected version would require** — documentation only, for a
future preregistration round, not implemented here:

* a **crossed panel** of multiple fixed anchor blocks × multiple shifted origins;
* **each shifted-origin cell carrying its own clean control**, so target
  difficulty is differenced out rather than assumed constant;
* response `ΔMAE_{block,origin} = MAE_masked − MAE_clean`, i.e. always measured
  against that cell's own control;
* **block fixed effects**, so the shifted-origin comparison is made within
  anchor block rather than across blocks.

That design is out of scope for this pilot and is recorded here only so the
reasoning is not lost.

**A different next-round design — the trailing-gap mechanism experiment — is
drafted in `docs/next_round_trailing_gap_mechanism.md`.** It would separate loss
of recent observations from effective-horizon extension from Chronos-2-specific
trailing-missingness handling, via `trailing_nan(g)` / `truncated(g)` /
`internal_block(g)` conditions at patch-aligned gap lengths. It requires new
forecasts, **no code for it exists**, and it is deliberately sequenced to follow
confirmation that the boundary jump is robust under §9.4 — not to run
concurrently with it.

### 9.10 Outputs

All in `results/pilot_v1/post_hoc_internal_only_analysis/`. Nothing under
`results/pilot_v1/` that existed before is read-modified or overwritten —
`decision.json`, `analysis.json` and `post_hoc_reanalysis_v2/` are untouched,
and a test enforces that the analysis writes nothing into the run directory.

| File | Contents |
|---|---|
| `internal_primary_contrasts.csv` | IC1_20, IC2_40 — §9.3 |
| `boundary_jump_contrasts.csv` | BJ1_20, BJ2_40, directly bootstrapped — §9.4 |
| `internal_equivalence_check.csv` | `equivalence_established` per internal contrast — §9.5 |
| `internal_pairwise_contrasts.csv` | 12 rows, 6 per rate — §9.4 |
| `internal_trend_slopes.csv` | Mean trend slope and CIs, per rate — §9.4 |
| `per_origin_trend_slopes.csv` | 178 per-origin slopes per rate |
| `trailing_boundary_d0.csv` | `d=0`, separately labelled — §9.5 |
| `confounder_difference_correlations.csv` | ρ(ΔMAE, Δz) vs. the **boundary jump**, with status typing and bootstrap CI — §9.8 |
| `confounder_difference_internal_contrasts.csv` | The same against the internal variation — retained and relabelled — §9.8 |
| `internal_only_analysis.json` | All of the above plus the family declarations |

### 9.11 Interpretation

Deliberately not offered here. The three-branch reading of these numbers — does
an internal-only effect hold at both rates; is `d=0` the only condition that
degrades; is any apparent effect driven by a handful of origins or by
removed-content differences — is a judgement for the Research Lead, not
something this pilot encodes as a second automated classifier.

---

## 10. Verified facts about Chronos-2 used by this design

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

## 11. Integrity guarantees, and how each is enforced

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

**Test suite size.** 261 non-model tests + 3 real-model tests (require GPU/model access) = 264 total. Stated this way rather than as a single number,
because a bare count reads differently depending on whether the environment can
load the model: on a CPU-only host 3 tests are deselected, so "175 tests" and
"172 passed" are both true and neither is the whole picture.

---

## 12. Scope

Implemented exactly as specified. Deliberately **not** implemented, and not to
be added without a Research Lead decision: 10% missingness; periodic
missingness; linear interpolation; forward-fill; ARIMA; XGBoost; DLinear;
TimesFM; MOMENT; TS-ICL; additional datasets; hyperparameter searches; any
fitted forecast-proximity weighting parameter.

No severity metric is fitted from the confounder diagnostics in this pilot.
