# Preregistration v1 — ETTh2/OT trailing-gap replication

> **STATUS: DRAFT SUBMITTED FOR RESEARCH LEAD SIGN-OFF. NOT IMPLEMENTED, NOT RUN.**
>
> No config change, no data-fetch script, no implementation, no execution exists
> for this document. ETTh2 has not been downloaded. Nothing here is frozen until
> signed off; on sign-off, every item below becomes frozen and changeable only
> by numbered amendment, as in
> `docs/preregistration_trailing_gap_mechanism_v1.md`.

---

## 0. Provenance — read this first

**This is a confirmatory hypothesis derived from an exploratory finding. It is
not an independent a priori claim, and this document does not soften that.**

The chain is three deep, and each link matters:

1. The **original ETTh1 pilot** was preregistered. Its own §0 already recorded
   that the *boundary jump* was a **post-hoc** finding, identified after
   inspecting preregistered results.
2. The **trailing-gap experiment** (`R_g` / `B_g`) was designed in response to
   that post-hoc boundary jump — one level removed from an independent
   confirmatory test, as its §0 states.
3. **This replication's primary hypothesis is generated from that experiment's
   observed per-gap readings.** `R_16`, `R_32` and `R_64` are named as the
   primary claim *because those are the three gaps that read `EQUIVALENT` on
   ETTh1*. The hypothesis was selected after seeing the ETTh1 result.

So a confirmatory pass on ETTh2 would establish that an **exploratorily
selected** pattern replicates on a second dataset. That is a real and useful
result. It is **not** the same as a hypothesis specified before any data were
seen, and no report of this work may present it as such.

The ETTh1 result being replicated is
**`INCONCLUSIVE / mixed_equivalent_and_unresolved`**. This document does not
alter, relabel or reinterpret it. A replication does not retrospectively
upgrade the ETTh1 classification, whatever it finds.

---

## 1. Design — identical to F3, dataset changed only

**The only intended difference from the frozen trailing-gap design is the
dataset.** Everything else is inherited, not restated, so it cannot drift.

| Element | Value | Source |
|---|---|---|
| Dataset | **ETTh2**, `OT` column | **the only change** |
| Context length `L` | 320 | inherited |
| Horizon `H` | 96 | inherited |
| Stride | 96 | inherited |
| Gap lengths `g` | {16, 32, 64, 128} | inherited |
| Conditions per gap | `trailing_nan`, `truncated_long`, `internal_block(d=16)` | inherited |
| Model revision | `29ec3766d36d6f73f0696f85560a422f50e8498c` | inherited — **pinned, identical to ETTh1** |
| Patch grid | 16 / stride 16, single-shot capacity 1,024 | inherited; re-verified at load |
| Quantile | explicit q=0.5 | inherited |
| Isolation | one origin per call, `cross_learning=False`, `limit_prediction_length=True` | inherited |
| `R_g` / `B_g` definitions | unchanged (§3) | inherited |
| Bootstrap | moving-block, lengths 4/8/12, 5,000 replicates, joint resampling | inherited |
| SESOI | ±3% of **ETTh2's own** mean clean MAE | inherited convention (§4) |
| Classification rule | `preregistered-v1`, unchanged precedence and thresholds | inherited |

**Origin count is NOT inherited.** It is a function of ETTh2's row count and
must be derived from the fetched data, asserted at runtime, and recorded here by
amendment before execution. Do not assume 178 — that number is ETTh1's. If
ETTh2's canonical length differs, the origin count differs, and the design must
say so explicitly rather than silently producing a different denominator.

---

## 2. Dataset

* **ETTh2**, target column `OT`, univariate.
* Same acquisition and validation discipline as ETTh1: canonical source URL,
  sha256 recorded and asserted on every load, row count asserted, monotonic and
  unique hourly timestamps asserted, **native missing values counted and
  reported before any synthetic missingness is introduced**.
* Timestamps preserved as published. No row is ever deleted; missingness is
  represented in place as NaN.
* The forecast target is never touched by any masking or preprocessing step, and
  target integrity is asserted at runtime across all conditions of an origin, as
  in the ETTh1 runner.

**Not yet done:** ETTh2 has not been fetched, and its checksum, row count and
native-missing count are therefore unknown at the time of writing. They must be
recorded by amendment before execution.

---

## 3. Contrasts — unchanged

```
R_g = MAE(trailing_nan(g)) − MAE(truncated_long(g))
B_g = MAE(trailing_nan(g)) − MAE(internal_block(g, d=16))
```

Two independent Holm families of 4, `{R_16..R_128}` and `{B_16..B_128}`,
independent of each other and of every family in the ETTh1 work.

### 3.1 `B_g` remains secondary and content-confounded

Carried forward verbatim from the frozen preregistration §5.2:

> **This tests a boundary-specific penalty but remains CONFOUNDED WITH
> DIFFERENCES IN REMOVED CONTENT between the two arms. It is not a pure
> mechanism test.**

`B_g` is reported alongside and **never determines** the classification.

**Note the erratum.** `docs/erratum_2026-09-04_bg_comparison_arm.md` records
that `B_g`'s comparison arm was mislabelled in generated ETTh1 output prose
(the R-family interpretation text was applied to B rows). The computation was
always correct. Any ETTh2 run must emit `analysis_reporting_v2`-style output
with an explicit `comparison_arm` column from the outset, so the same error
cannot recur.

---

## 4. SESOI

±3% of mean clean MAE, via the single definition at
`configs/pilot_config.yaml: decision.no_go.equivalence_band_frac_of_clean_mae`,
read through `stats.decision.equivalence_band()`.

**Computed from ETTh2's own mean clean MAE**, not ETTh1's. The *convention* is
inherited; the *value* is dataset-specific and will differ. Reusing ETTh1's
absolute band would be an error.

Equivalence is established **only** by the preregistered convention — the 90%
CI lying entirely inside ±delta. Non-significance is never equivalence.

---

## 5. Primary replication hypothesis

> **`R_16`, `R_32`, and `R_64` must ALL establish equivalence on ETTh2.**

* **Intersection claim.** All three must read `EQUIVALENT` under the frozen
  four-way per-gap classification. Two of three is a failure to replicate, not a
  partial success, and must be reported as such.
* **Provenance label, to appear wherever this hypothesis is reported:** *this
  is an intersection claim generated from the ETTh1 result — `R_16`, `R_32` and
  `R_64` are named because those were the three gaps that read `EQUIVALENT` on
  ETTh1. It is a confirmatory hypothesis derived from an exploratory finding,
  not an independent a priori claim.*
* Per-gap readings use the frozen four-way rule unchanged: `EQUIVALENT` requires
  the whole 90% CI inside ±delta; `MATERIAL_POSITIVE` requires the Holm lower
  bound above +delta; `MATERIAL_NEGATIVE` the Holm upper bound below −delta;
  everything else is `UNRESOLVED`.

**Open for sign-off:** whether the three-gap intersection needs a multiplicity
adjustment of its own, and if so which. An intersection of three equivalence
claims is not the same testing problem as the four-member Holm family the gaps
already sit in. This document does **not** decide it; a decision is required
before execution.

---

## 6. Secondary conditions

* **`R_128` is a mandatory secondary stress condition.** It read `UNRESOLVED` on
  ETTh1. It **must** be measured and reported on ETTh2 with the same
  completeness as the primary gaps, and it is **not** part of the primary
  confirmatory claim. It may not be dropped, and its outcome — whatever it is —
  may not be used to strengthen or weaken the primary result.
* **`B_16`…`B_128`** are secondary and content-confounded, per §3.1.
* **Dose-response** across the four gaps, slope against `x = g / patch_size`,
  primary for that sub-question only, never primary evidence, never an input to
  the classification. A non-significant slope means only "no detected linear
  trend".

---

## 7. Explicit exclusions

* **No pooled ETTh1+ETTh2 primary analysis is permitted** unless separately
  preregistered in a future round. The two datasets are analysed separately.
  Pooling after seeing both results would be a different, unregistered test.
* **No new model, imputation method, or baseline this round.** The pinned
  revision is the same; no imputation is introduced; no comparison model is
  added.
* No new gap lengths, contrasts, thresholds, or bootstrap settings.
* No change to the frozen ETTh1 result or to the F3 execution tree.

---

## 8. Preconditions for execution

1. **Sign-off on this document**, including the §5 multiplicity question.
2. **Amendment recording ETTh2's checksum, row count, native-missing count and
   derived origin count**, before any forecast.
3. **Model contract re-verification** — same revision, same patch grid, same
   single-shot capacity — as in the ETTh1 runbook step 2.
4. **A clean-condition audit.** ETTh1's audit compared against a prior ETTh1
   run; there is no prior ETTh2 run to compare against, so the equivalent
   control must be specified at sign-off. **Open for sign-off:** the natural
   substitute is a within-run determinism check (the same clean origin forecast
   twice, byte-identical), but this document does not decide it.
5. **Initialization guard** — refuse to start into an existing ETTh2 run
   directory, as in runbook §3.0.

## 8.1 Results delivery

Per `docs/results_delivery_policy.md`: executed config, run manifest, the audit
result, summary tables, bootstrap sensitivity across block lengths 4/8/12, the
per-gap `R_g`/`B_g` readings, the dose-response output and the final
classification must reach the reviewer. Raw per-forecast files may stay
uncommitted. Summaries may not.

---

## 9. What a result here would and would not establish

**Would:** that the ETTh1 per-gap equivalence pattern at `g ∈ {16, 32, 64}`
does, or does not, reproduce on a second ETT series under an identical design
and an identically pinned model.

**Would not:** establish a mechanism. The `R_g` arms differ in whether the model
is shown a trailing NaN run or given a shorter context with a longer single-shot
request; neither reading isolates masking, normalization, positional handling or
any other internal component. Nor would it convert the exploratorily selected
hypothesis of §0 into an a priori one.

---

## 10. Summary of what needs a decision before execution

| Item | Status |
|---|---|
| Design inherited from F3, dataset changed only (§1) | **Proposed** |
| ETTh2 origin count | **Unknown — must be derived and recorded by amendment** |
| Primary hypothesis: `R_16 ∧ R_32 ∧ R_64` equivalent (§5) | **Proposed** |
| Provenance label as an exploratorily generated hypothesis (§0, §5) | **Proposed, and not negotiable in substance** |
| Multiplicity treatment of the three-gap intersection (§5) | **OPEN — needs sign-off** |
| Clean-condition control in the absence of a prior ETTh2 run (§8.4) | **OPEN — needs sign-off** |
| `R_128` mandatory secondary (§6) | **Proposed** |
| `B_g` secondary and content-confounded (§3.1) | **Carried forward verbatim** |
| No pooled analysis; no new model/imputation/baseline (§7) | **Proposed** |
