# Preregistration v1 — Trailing-gap mechanism experiment

> **STATUS: FROZEN AND PREREGISTERED. NOT YET EXECUTED.**
>
> Every rule in this document is frozen. No forecasts have been produced under
> it and no GPU time has been spent; execution is gated on the Research Lead's
> release audit, not on any further design decision. Nothing below may change
> without a numbered amendment.
>
> Superseded earlier wording is quarantined in Appendix A (§14) and is **not
> operative**. Nothing outside that appendix describes a proposal.

Supersedes the sketch in `docs/next_round_trailing_gap_mechanism.md`, whose
`truncated(g)` arm contained a fatal timestamp-misalignment bug (§3.1).

---

## 0. Provenance of the hypothesis — read this first

**FROZEN.** `R_g` and `B_g` test a **new hypothesis**, not a confirmatory
replication of the original pilot's C1–C4 or of the boundary-jump finding.

They were designed **in response to the boundary jump**, which is itself a
**post-hoc finding** — identified after inspecting the original preregistered
results. So even a clean `R_g` / `B_g` result here tests a hypothesis generated
from prior data, and is **one level removed from a fully independent
confirmatory test**. Every report of these results must say so. The
classification rule embeds this statement in its own output
(`criteria.hypothesis_provenance`) so it cannot be dropped in transcription.

---

## 1. Question

Does Chronos-2 treat a trailing run of missing values **equivalently** to
forecasting the same target from the last observed point with an explicitly
extended prediction horizon?

The boundary jump (audit report §9.4) established that the boundary-touching
condition costs more than internal positions. It did not establish **why**. At
least three explanations remain live, and the original design cannot separate
them: loss of recent observations; effective-horizon extension; and
Chronos-2-specific handling of trailing missingness.

---

## 2. Conditions

**FROZEN.** Per origin, at each gap length `g ∈ {16, 32, 64, 128}`:

| Condition | Context | `prediction_length` | Scored steps | Isolates |
|---|---|---|---|---|
| `trailing_nan(g)` | Length `L`=320, final `g` steps NaN | `H`=96 | `[0, 96)` | Full effect: recency loss + horizon extension + NaN handling |
| `truncated_long(g)` | First `L−g` real observations, **no NaN** | `H+g` | `[g, g+96)` | Recency loss + horizon extension, with **no** trailing-missingness handling |
| `internal_block(g)` | Length `L`, `g` missing at fixed `d=16` | `H` | `[0, 96)` | Equal-sized loss with the boundary **intact** |

Plus one `clean` condition per origin (full 320-step context, unchanged).

`trailing_nan(g)` reuses the pilot's existing contiguous-block masking with
`block_length=g, d=0`; it is the pilot's `d=0` block generalised to variable
length.

### 2.1 Why `internal_block(g)` is fixed at `d=16` for every `g`

**FROZEN.** Holding the distance constant means **exactly one full real patch
always separates the block from the forecast boundary**, whatever the gap
length. If `d` instead scaled with `g`, the comparison across gap lengths would
confound "how much was removed" with "how far from the boundary" — the same
entanglement the original pilot could not escape.

`block_start = 320 − 16 − g`, giving 288, 272, 240, **176** at `g` = 16, 32, 64,
128. All non-negative and all multiples of 16, so both the block and the
16-step gap to the boundary stay patch-aligned. Asserted, not assumed.

---

## 3. The corrected `truncated_long(g)`

**FROZEN.**

```
context           = original_context[:L-g]   # real observations, no NaN
prediction_length = H + g                    # NOT H
scored_prediction = prediction[g : g+H]      # drop the first g predicted steps
ground truth      = the original, unchanged H-step target
```

The scored window lands on absolute indices `[(L−g)+g, (L−g)+g+H)` = `[L, L+H)`
— byte-identical ground truth to every other condition.

### 3.1 The bug this corrects

The earlier sketch supplied the first `L−g` real observations and asked for `H`
steps. The model forecasts the `H` steps immediately following **its own**
context, i.e. `[L−g, L−g+H)`, while the pilot scores against `[L, L+H)`. That is
a **`g`-step timestamp misalignment**, and it would have silently invalidated
every comparison involving that arm — including the entire `R_g` family, which
is the experiment's primary output.

It is not caught by a length check: both designs return 96 scored values. Only a
timestamp check catches it. `tests/test_trailing_gap.py::test_4_naive_truncation_is_misaligned_by_exactly_g_steps`
reconstructs the naive design alongside the corrected one and asserts their
scored windows differ by exactly `g` at both ends — demonstrating the bug is
real rather than asserting the fix looks right.

### 3.2 The long-horizon hazard — RESOLVED, hard stop retained

**FROZEN as a hard stop. Capacity RESOLVED: the matrix fits single-shot.**

`H+g` must fit in the model's **single-shot** output,
`max_output_patches × output_patch_size`. `Chronos2Pipeline.predict` only
**warns** when `prediction_length` exceeds that (its `limit_prediction_length`
defaults to `False`) and then silently falls back to
`_autoregressive_unroll_for_long_horizon` — a different inference regime, and a
contract violation under the original pilot's rules.

At `g=128` this asks for **224 steps**.

**Resolved.** The original pilot's real-weights verification
(`scripts/verify_model_contract.py`, run on the GPU host against revision
`29ec3766d36d6f73f0696f85560a422f50e8498c`) reported
`max_output_patches: 64`, `output_patch_size: 16`, giving a single-shot capacity
of **1,024 steps**. The largest request in this matrix is `H+g = 224` at
`g=128`, comfortably inside it.

Confirmed against the library's control flow, not the arithmetic alone.
`Chronos2Pipeline._predict_batch` computes
`get_num_output_patches(remaining) = min(ceil(remaining / output_patch_size), max_output_patches)`.
For 224 steps that is `min(ceil(224/16), 64) = min(14, 64) = 14` patches = 224
steps, all produced by the **first** `_predict_step`, leaving `remaining = 0`.
The `if remaining > 0` branch that would enter
`_prepare_inputs_for_long_horizon_unrolling` is therefore never taken. **No
autoregressive unrolling occurs at any gap length in this matrix.**

| `g` | `H+g` | Output patches | Single-shot capacity | Unrolls? |
|---|---|---|---|---|
| 16 | 112 | 7 | 1,024 (64 × 16) | no |
| 32 | 128 | 8 | 1,024 | no |
| 64 | 160 | 10 | 1,024 | no |
| 128 | **224** | **14** | 1,024 | **no** |

**The hard stop is retained regardless.** The capacity is a property of the
checkpoint, and revisions can drift. `experiments/trailing_gap.preflight()`
still raises `TrailingGapContractViolation` naming the offending gap if the
loaded contract cannot serve the request, and the pipeline is still called with
`limit_prediction_length=True` so the library raises instead of warning. A cheap
re-verification on the GPU host remains a precondition (§10) — it costs seconds
and the value above was measured on a different day.

---

## 4. Matrix and counts

**FROZEN.**

```
178 origins × (1 clean + 4 gaps × 3 conditions) = 178 × 13 = 2,314 forecasts
```

Asserted at run time (`expected_conditions_per_origin: 13`,
`expected_total_forecasts: 2314` in `configs/trailing_gap_config.yaml`), in the
same style as the original pilot's 178 × 17 = 3,026 assertion. **Not asserted
now** — nothing has been run.

---

## 5. Contrasts

### 5.1 Primary: the `R_g` family

**FROZEN.** Own Holm family of 4, independent of every prior family
(`{C1..C4}`, `{IC1_20, IC2_40}`, `{BJ1_20, BJ2_40}`, the pairwise families, and
the confounder-difference families).

```
R_g = MAE(trailing_nan(g)) − MAE(truncated_long(g))
```

Both arms lose the same `g` most-recent observations and are scored against the
same target over the same timestamps. They differ **only** in whether the model
is shown a trailing run of NaNs or handed a shorter context and a longer
horizon request.

**FROZEN.** Per-gap classification is **four-way** against
`delta = 0.03 x mean_clean_mae`:

| Reading | Criterion | Interpretation (verbatim, used wherever reported) |
|---|---|---|
| `EQUIVALENT` | the complete **90%** equivalence CI lies inside `[-delta, +delta]` | "no practically meaningful performance difference was established... consistent with a horizon-dominant explanation, but does NOT demonstrate an 'effective-horizon-only' mechanism." |
| `MATERIAL_POSITIVE` | Holm-adjusted CI **lower** bound `> +delta` | "explicit trailing-gap encoding has higher mean error than shorter-context/longer-horizon single-shot forecasting by more than the SESOI. It does not isolate masking, normalization, positional handling, or another internal component." |
| `MATERIAL_NEGATIVE` | Holm-adjusted CI **upper** bound `< -delta` | "lower error was observed with explicit trailing NaNs. Do not call this a beneficial causal mechanism." |
| `UNRESOLVED` | everything else | neither equivalence nor a material directional difference was established at this gap length |

`ci_low_holm > 0` is **not** sufficient for `MATERIAL_POSITIVE`; the bound must
clear `+delta`. `UNRESOLVED` explicitly covers: statistically different from
zero but practical magnitude unresolved; point estimate outside the SESOI while
the CI overlaps it; and non-significant without equivalence.

Two secondary flags — `statistically_detectable_but_practically_small` and
`statistically_detectable_but_sesoi_unresolved` — are recorded for diagnostic
visibility and can never promote a gap to a material directional reading.

### 5.2 Secondary: the `B_g` family

**FROZEN.** Own Holm family of 4, independent of `R` and of every prior family.

```
B_g = MAE(trailing_nan(g)) − MAE(internal_block(g, d=16))
```

> **This tests a boundary-specific penalty but remains CONFOUNDED WITH
> DIFFERENCES IN REMOVED CONTENT between the two arms. It is not a pure
> mechanism test.**

That sentence travels with `B_g` into the code, the output schema, and every
report. `B_g` **never determines** the mechanism label; a unit test enforces
that arbitrarily strong `B_g` values cannot change the classification.

---

## 6. Equivalence (SESOI)

**FROZEN, and inherited rather than redefined.** ±3% of mean clean MAE — the
same convention as the original pilot's NO-GO rule and the internal-only
analysis. The single definition lives at
`configs/pilot_config.yaml: decision.no_go.equivalence_band_frac_of_clean_mae`,
read via `stats.decision.equivalence_band()`.

`configs/trailing_gap_config.yaml` does not restate it. Inheritance is proved
behaviourally rather than by forbidding a literal: `stats/mechanism_decision.py`
mentions `0.03` only in a docstring stating the formula, and
`test_sesoi_delta_follows_a_mutated_pilot_config` mutates the pilot config's
value and asserts `sesoi_delta` moves with it — so the two cannot be independent
values that merely happen to agree.

**Equivalence is checked before significance.** A result both statistically
significant and practically negligible reads as *equivalent*, not as an effect.
Non-significance alone never establishes equivalence.

---

## 7. Dose-response across the four gap lengths

**FROZEN.**

Per origin, the OLS slope of `R_g` against `g` across that origin's four gap
values; then the across-origin mean slope bootstrapped with the same
moving-block procedure used everywhere else. This mirrors the internal-only
trend analysis exactly (`stats.internal_only.per_origin_trend_slopes`), so the
two are directly comparable.

The slope is computed against **`x = g / patch_size`** — missing patches, not
raw `g`. Equivalent linearity, interpretable per-patch units.

The **linear slope is primary for the dose-response SUB-QUESTION ONLY**. It is
**not** primary evidence for the experiment: the four categorical per-gap `R_g`
readings of §5.1 are. Nonlinearity is assessed as a **secondary descriptive
check** only (a quadratic term plus the per-gap reading pattern). Neither the
slope nor the curvature feeds the classification of §9, and equivalence is
decided **only** from the four per-gap CIs.

**A non-significant slope means only "no detected linear trend."** It does not
mean the relationship is flat, does not establish equivalence, and does not
support an effective-horizon-only explanation. A static test enforces that no
part of this codebase characterises it otherwise.

*Reasoning recorded for the audit trail.* An
effective-horizon-only story predicts `R_g ≈ 0` at every `g` — a flat line, and
the linear test is the sharpest instrument against it. A representation-specific
penalty most plausibly grows with the number of unobserved trailing patches,
which is again monotonic in `g`. Four points is too few to fit a shape with any
confidence: a quadratic on four points has 1 residual degree of freedom, so a
nonlinearity test here would be badly underpowered and prone to
over-interpretation. The per-gap readings in §5.1 already surface a non-monotone
pattern qualitatively — and the classification rule treats direction
disagreement across gaps as `DIRECTION_REVERSAL_ACROSS_GAPS` rather than averaging it
away.

**The known limitation, accepted with this choice:** if the effect is a
threshold — nothing until the trailing gap exceeds some number of patches, then
a jump — a linear slope could dilute it toward zero. That is precisely why the
slope is confined to the sub-question and cannot establish equivalence: only the
four per-gap CIs can, and a threshold effect would surface there as a
`MATERIAL_*` reading at the large gaps with `EQUIVALENT` or `UNRESOLVED` at the
small ones, which §9 classifies as `INCONCLUSIVE`, never as equivalence.

---

## 8. Bootstrap and multiplicity

**FROZEN.** Moving-block bootstrap over chronologically ordered origins: main
block length 8, sensitivity at 4 and 12, 5,000 replicates, every condition
resampled jointly to preserve pairing. Reuses
`stats.bootstrap.bootstrap_paired_difference` and `holm_correct` — not
reimplemented. Inherited from `configs/pilot_config.yaml: statistics.bootstrap`,
not restated in the new config.

Paired by evaluation origin throughout. Holm applied **within** `{R_16, R_32,
R_64, R_128}` and **within** `{B_16, B_32, B_64, B_128}` separately — two
families of 4, never one of 8, and never merged with any prior family.

---

## 9. Mechanism classification rule

**FROZEN. `rule_version: preregistered-v1`, `authoritative: True`.**

`authoritative: True` means **the decision rule itself was preregistered** — its
thresholds and precedence were fixed before any forecast, so the classification
cannot have been tuned to the data. **It does NOT mean any resulting causal
interpretation is authoritative.** No label isolates masking, normalization,
positional handling, or any other internal component.

Implemented as a deterministic function, `stats/mechanism_decision.py::classify`,
unit-tested in the style of `stats/decision.py`.

**Precedence, in order:**

1. At least one `MATERIAL_POSITIVE` **and** at least one `MATERIAL_NEGATIVE`
   → `DIRECTION_REVERSAL_ACROSS_GAPS`.
2. All 4 gaps `EQUIVALENT` → `NO_MATERIAL_DIFFERENCE_VS_TRUNCATION`.
3. ≥3 of 4 `MATERIAL_POSITIVE` and zero `MATERIAL_NEGATIVE`
   → `CONSISTENT_EXTRA_TRAILING_GAP_PENALTY`.
4. ≥3 of 4 `MATERIAL_NEGATIVE` and zero `MATERIAL_POSITIVE`
   → `CONSISTENT_LOWER_ERROR_WITH_TRAILING_NAN`.
5. Otherwise → `INCONCLUSIVE`, with exactly one machine-readable reason:
   * `sparse_or_gap_dependent_directional_evidence` — ≥1 material directional
     reading, fewer than three, and no opposite material direction;
   * `mixed_equivalent_and_unresolved` — no material direction, but a mix of
     equivalent and unresolved gaps;
   * `insufficient_precision` — all four unresolved.

**Threshold scopes are distinct.** `min_consistent_gaps = 3` governs **only**
rules 3–4. `equivalence_required_gaps = 4` governs **only** rule 2: family-wise
equivalence is an **intersection** across all four gaps. A 3-of-4 equivalent
subset is **not** family-wise equivalence without a separately adjusted
procedure, and falls through to `INCONCLUSIVE`.

Rule 1 deliberately **outranks** a 3-of-4 majority: 3 positive + 1 negative is a
direction reversal, not a positive result.

**Worked examples, each reproduced as a test:**

| Readings | Label |
|---|---|
| 3 `EQUIVALENT` + 1 `MATERIAL_POSITIVE` (or `_NEGATIVE`) | `INCONCLUSIVE` (`sparse_or_gap_dependent_directional_evidence`) — **not** an equivalence result |
| 2 `MATERIAL_POSITIVE` + 2 `EQUIVALENT` | `INCONCLUSIVE` (`sparse_or_gap_dependent_directional_evidence`) |
| 3 `MATERIAL_POSITIVE` + 1 `MATERIAL_NEGATIVE` | `DIRECTION_REVERSAL_ACROSS_GAPS` — **not** a 3-of-4 majority |
| 1 `UNRESOLVED` + 3 `EQUIVALENT` | `INCONCLUSIVE` (`mixed_equivalent_and_unresolved`) |
| 4 `UNRESOLVED` | `INCONCLUSIVE` (`insufficient_precision`) |
| 2 `MATERIAL_POSITIVE` + 2 `UNRESOLVED` | `INCONCLUSIVE` (`sparse_or_gap_dependent_directional_evidence`) |

`B_g` and the dose-response slope are computed and reported alongside the
classification but **never determine it**; tests prove that arbitrarily strong
values of either cannot change the label.

---

## 10. Preconditions for execution

**FROZEN as requirements. None performed yet.**

1. **Re-verify the model contract.** `python scripts/verify_model_contract.py`,
   confirming the pinned revision, the patch grid, and the single-shot horizon
   limit against §3.2. Expected, from the original pilot's verification:
   `revision: 29ec3766d36d6f73f0696f85560a422f50e8498c`, `input_patch_size: 16`, `input_patch_stride: 16`,
   `output_patch_size: 16`, `max_output_patches: 64` → capacity 1,024. Any
   departure means §3.2 must be re-derived before the run, not during it.
2. **Re-run `clean` and audit it against the original pilot — EXACT match, no
   tolerance.** Before any new result is trusted, the clean condition must be
   re-run and audited by `scripts/audit_clean_predictions.py`, which verifies
   **identity before values**: matching `(origin_id, step_index)` key sets, a
   row count of exactly 178 x 96, exact forecast timestamps, exact
   `ground_truth`, per-origin `target_sha256` agreeing with `window_results.csv`
   on both sides, matching dataset checksum and model revision, no duplicate or
   missing cells — and only then the canonical origin/step-sorted float32
   SHA256 of the predictions.

   **A mismatch in any of those fields is a HARD STOP.** There is no tolerance
   mechanism for any of them, and selecting one after seeing a discrepancy is
   exactly what this audit exists to prevent. A `clean` that has moved
   invalidates cross-run comparison, and the audit must run **before** any `R_g`
   or `B_g` is read.
3. **Confirm the origin set and target integrity** are byte-identical to the
   original run, via the existing `target_sha256` machinery.

### 10.1 Process requirement — results must reach the reviewer

**FROZEN.** The Research Lead **cannot recompute values that live only in an
uncommitted `results/` directory.** This has already bitten this project: three
consecutive analysis rounds were implemented and tested but could not be
populated with real numbers, because `results/pilot_v1/` exists only on the
execution host and `results/` is gitignored.

Therefore, for the execution round, the following must be **committed or
otherwise delivered to the reviewer**, not merely produced:

- the executed config (`configs/trailing_gap_config.yaml` as run, with the
  resolved model revision);
- the run manifest (environment, dependency versions, GPU, git commit);
- summary statistics — the contrast tables, bootstrap intervals, dose-response
  output and classification result;
- the `clean` re-run audit from §10.2.

Raw per-forecast files may stay uncommitted. **Summaries may not.**

This requirement has been generalised repo-wide as
`docs/results_delivery_policy.md`, so it applies to every execution round and
not only to this experiment.

---

## 11. Referenced gating tests

All in `tests/test_trailing_gap.py`, all passing, all model-mocked:

| Test | Gates |
|---|---|
| `test_1_all_gaps_are_patch_aligned_and_fit_the_single_shot_horizon` | `L−g` patch-aligned; `H+g` within the contract's real limit |
| `test_1b_preflight_reads_the_limit_from_the_contract_not_a_constant` | The limit is queried, not hardcoded |
| `test_2_scored_timestamps_align_with_the_original_target` | Timestamps coincide, not merely lengths |
| `test_2b_truncated_long_scored_index_lands_on_the_original_target` | `(L−g)+g == L` |
| `test_3_target_sha256_is_identical_across_every_condition` | Target byte-identity, via the pilot's own machinery |
| `test_4_naive_truncation_is_misaligned_by_exactly_g_steps` | The bug is real and the fix addresses it |
| `test_4b_the_two_designs_score_different_predicted_values` | The misalignment changes the scored numbers |
| `test_5_requesting_an_unroll_is_a_hard_stop_not_a_warning` | Unrolling cannot happen silently |
| `test_5d_the_pipeline_default_would_have_warned_silently` | Documents why the guard is needed, from the library source |

---

## 12. Summary — every item is frozen

| Item | Status |
|---|---|
| Hypothesis provenance statement (§0) | **FROZEN** |
| Conditions and `internal_block` at `d=16` (§2) | **FROZEN** |
| Corrected `truncated_long(g)` (§3) | **FROZEN** |
| Unrolling as a hard stop (§3.2) | **FROZEN** |
| `g=128` executability (§3.2) | **RESOLVED** — capacity 1,024 vs. a 224-step maximum; cheap re-verification retained as a precondition |
| Matrix, 2,314 forecasts (§4) | **FROZEN** |
| `R_g` / `B_g` definitions and Holm families (§5) | **FROZEN** |
| `R_g` interpretation readings (§5.1) | **FROZEN** — four-way against the SESOI; interpretation text verbatim |
| `B_g` confounding label (§5.2) | **FROZEN** |
| SESOI ±3%, inherited (§6) | **FROZEN** |
| Dose-response method (§7) | **FROZEN** |
| Dose-response: linear primary, x = missing patches (§7) | **FROZEN** — sub-question only; never feeds the classification |
| Bootstrap and multiplicity (§8) | **FROZEN** |
| Classification rule structure and precedence (§9) | **FROZEN** — `preregistered-v1` |
| Classification thresholds (§9) | **FROZEN** — `min_consistent_gaps=3` (rules 3-4), `equivalence_required_gaps=4` (rule 2) |
| Execution preconditions (§10) | **FROZEN** |
| Results-delivery requirement (§10.1) | **FROZEN** |


---

## 13. Amendment record

| # | Date | Amendment |
|---|---|---|
| 1 | 2026-09-03 | `g=128` executability resolved: the pinned checkpoint's single-shot capacity is 1,024 steps against a 224-step maximum request. The hard stop and the cheap re-verification precondition were both retained. |
| 2 | 2026-09-03 | §5.1 replaced by the four-way SESOI classification; §7 confined to the dose-response sub-question with `x = missing patches`; §9 replaced by the frozen precedence. Superseded text moved to Appendix A. |
| 3 | 2026-09-03 | §10 audit tightened to identity-before-values with no tolerance mechanism, after review found the audit hashed prediction values only. |

No further amendment has been made. Any future change requires a numbered entry
here.

---

## 14. Appendix A — Superseded draft history

> **NOT OPERATIVE. NOTHING IN THIS APPENDIX IS A RULE.**
>
> Retained only so the audit trail shows what changed and when. Every statement
> here was replaced by the frozen text in the numbered sections above. Where
> this appendix and the body disagree, **the body governs** — this appendix is
> never the tiebreaker.

**Superseded 2026-09-03 (amendment 2).** Before the Research Lead's
conditional-GO corrections, this document carried three items marked as awaiting
sign-off, and their earlier content is recorded here for provenance:

* **§5.1** previously proposed a **three-way** reading — inside the equivalence
  band, above it, below it — with `ci_low_holm > 0` sufficient for a positive
  directional reading. **Replaced** by the four-way classification in which a
  material reading requires the Holm bound to clear `±delta`, and everything
  else is `UNRESOLVED`.
* **§7** previously left the linear-versus-nonlinear choice open, and computed
  the slope against raw `g`. **Replaced** by: linear primary for the
  dose-response sub-question only, `x = g / patch_size`, never feeding the
  classification, curvature descriptive only.
* **§9** previously described a draft rule whose outputs carried
  `authoritative: false`, keyed on a single `min_consistent_gaps` threshold.
  **Replaced** by the frozen five-step precedence with distinct threshold
  scopes, family-wise equivalence as an intersection, and three machine-readable
  inconclusive reasons.

**Superseded 2026-09-03 (amendment 3).** §10 previously described a
"checksum/tolerance" audit of the clean re-run. **Replaced** by an exact audit
verifying identity before values, with no tolerance mechanism for any field.

The earlier sketch that preceded this document,
`docs/next_round_trailing_gap_mechanism.md`, is separately marked superseded and
carries its own description of the `truncated(g)` timestamp-misalignment bug
that this preregistration's §3.1 corrects.
