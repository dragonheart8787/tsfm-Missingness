# Preregistration v1 — Trailing-gap mechanism experiment

> **STATUS: DRAFT SUBMITTED FOR RESEARCH LEAD REVIEW. NOT EXECUTED.**
>
> No forecasts have been produced under this design. No GPU time has been spent.
> Execution is gated on sign-off of this document. Items below are marked
> **FROZEN** (fixed by this draft, not to be changed without an amendment) or
> **AWAITING SIGN-OFF** (a proposal, explicitly not yet decided).

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

### 3.2 The long-horizon hazard — a genuine open risk

**FROZEN as a hard stop. The outcome is UNKNOWN until verified on the model.**

`H+g` must fit in the model's **single-shot** output,
`max_output_patches × output_patch_size`. `Chronos2Pipeline.predict` only
**warns** when `prediction_length` exceeds that (its `limit_prediction_length`
defaults to `False`) and then silently falls back to
`_autoregressive_unroll_for_long_horizon` — a different inference regime, and a
contract violation under the original pilot's rules.

At `g=128` this asks for **224 steps**. Whether the real checkpoint can serve
that single-shot is **not known from this environment** and must be verified
before any GPU time:

```bash
python scripts/verify_model_contract.py        # prints max_output_patches, output_patch_size
```

If `max_output_patches × output_patch_size < 224`, **`g=128` is not executable
as specified** and this document requires an amendment — dropping that gap
length, or accepting unrolling as a documented regime change with its own
justification. `experiments/trailing_gap.preflight()` raises
`TrailingGapContractViolation` naming the offending gap rather than proceeding,
and the pipeline is additionally called with `limit_prediction_length=True` so
the library raises instead of warning.

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

Interpretation — **AWAITING SIGN-OFF** as a set of readings, not yet a decision:

| Reading | Consistent with |
|---|---|
| `R_g` inside the equivalence band | An **effective-horizon-only** story: trailing NaNs cost nothing beyond the lost recency |
| `R_g > 0`, outside the band | An **extra penalty** specific to the explicit trailing-NaN representation, masking, or position handling |
| `R_g < 0`, outside the band | The explicit NaN framing **outperforms** the truncated-autoregressive framing |

"Consistent with" is the strongest available reading. None of these is a
demonstrated mechanism, and no report may upgrade them to one.

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

`configs/trailing_gap_config.yaml` does not restate it; tests assert the `0.03`
literal appears in neither `stats/mechanism_decision.py` nor the new config.

**Equivalence is checked before significance.** A result both statistically
significant and practically negligible reads as *equivalent*, not as an effect.
Non-significance alone never establishes equivalence.

---

## 7. Dose-response across the four gap lengths

**METHOD FROZEN. THE MONOTONIC-VS-NONLINEAR CHOICE AWAITS SIGN-OFF.**

Per origin, the OLS slope of `R_g` against `g` across that origin's four gap
values; then the across-origin mean slope bootstrapped with the same
moving-block procedure used everywhere else. This mirrors the internal-only
trend analysis exactly (`stats.internal_only.per_origin_trend_slopes`), so the
two are directly comparable.

**Proposal, needing sign-off:** test the **linear slope as primary**, and assess
nonlinearity only as a **secondary descriptive check** (a quadratic term plus
the per-gap reading pattern already reported in §5.1).

*Reasoning, offered for the Research Lead to accept or overrule.* An
effective-horizon-only story predicts `R_g ≈ 0` at every `g` — a flat line, and
the linear test is the sharpest instrument against it. A representation-specific
penalty most plausibly grows with the number of unobserved trailing patches,
which is again monotonic in `g`. Four points is too few to fit a shape with any
confidence: a quadratic on four points has 1 residual degree of freedom, so a
nonlinearity test here would be badly underpowered and prone to
over-interpretation. The per-gap readings in §5.1 already surface a non-monotone
pattern qualitatively — and the classification rule treats direction
disagreement across gaps as `INCONSISTENT_ACROSS_GAPS` rather than averaging it
away.

**The counter-argument the Research Lead should weigh:** if the effect is a
threshold — nothing until the trailing gap exceeds some number of patches, then
a jump — a linear slope could dilute it toward zero and read as equivalence.
Making nonlinearity primary, or adding gap lengths, would address that at the
cost of power and GPU time. **This is the single item in this document most
worth overruling, and it is not decided.**

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

**DRAFT PROPOSAL — THRESHOLDS AWAIT SIGN-OFF. NOT AUTHORITATIVE.**

Implemented as a deterministic function, `stats/mechanism_decision.py::classify`,
unit-tested in the same style as `stats/decision.py`. Every decision it emits
carries `authoritative: False` and a status string saying it awaits sign-off.

Precedence, fixed in advance:

1. `R_g` readings **disagree in direction** across gaps → `INCONSISTENT_ACROSS_GAPS`.
   A mechanism that reverses sign with gap length is not one mechanism, and this
   deliberately **outranks** a majority reading (3 positive + 1 negative is not
   a positive result).
2. At least `min_consistent_gaps` (**proposed: 3 of 4**) agree on one reading →
   `EFFECTIVE_HORIZON_ONLY`, `EXTRA_TRAILING_NAN_PENALTY`, or
   `NAN_FRAMING_OUTPERFORMS`.
3. Otherwise → `INCONCLUSIVE`.

`B_g` is recorded in the criteria but **never** determines the label.

**Open for sign-off:** the value of `min_consistent_gaps`; whether
`INCONSISTENT_ACROSS_GAPS` should outrank a 3-of-4 majority (proposed: yes);
and whether `INCONCLUSIVE` should be split by *why* it was inconclusive.

---

## 10. Preconditions for execution

**FROZEN as requirements. None performed yet.**

1. **Re-verify the model contract.** `python scripts/verify_model_contract.py`,
   confirming the pinned revision, the patch grid, and — critically — the
   single-shot horizon limit against §3.2.
2. **Re-run `clean` and audit it against the original pilot.** Before any new
   result is trusted, the clean condition must be re-run and its predictions
   checksum/tolerance-audited against the original pilot's clean predictions, to
   catch environment drift since the original run: model revision, dependency
   versions, hardware. A `clean` that has moved invalidates cross-run
   comparison, and the audit must run **before** any `R_g` or `B_g` is read.
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

## 12. Summary of what is frozen vs. open

| Item | Status |
|---|---|
| Hypothesis provenance statement (§0) | **FROZEN** |
| Conditions and `internal_block` at `d=16` (§2) | **FROZEN** |
| Corrected `truncated_long(g)` (§3) | **FROZEN** |
| Unrolling as a hard stop (§3.2) | **FROZEN** — but whether `g=128` is executable is **UNVERIFIED** |
| Matrix, 2,314 forecasts (§4) | **FROZEN** |
| `R_g` / `B_g` definitions and Holm families (§5) | **FROZEN** |
| `R_g` interpretation readings (§5.1) | **AWAITING SIGN-OFF** |
| `B_g` confounding label (§5.2) | **FROZEN** |
| SESOI ±3%, inherited (§6) | **FROZEN** |
| Dose-response method (§7) | **FROZEN** |
| Linear primary vs. nonlinear (§7) | **AWAITING SIGN-OFF** — most worth overruling |
| Bootstrap and multiplicity (§8) | **FROZEN** |
| Classification rule structure (§9) | **DRAFT** |
| Classification thresholds (§9) | **AWAITING SIGN-OFF** — not authoritative |
| Execution preconditions (§10) | **FROZEN** |
| Results-delivery requirement (§10.1) | **FROZEN** |
