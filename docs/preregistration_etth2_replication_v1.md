# Preregistration v1 — ETTh2/OT trailing-gap replication

> **STATUS: FROZEN AND PREREGISTERED. IMPLEMENTED, NOT YET RUN.**
>
> **Signed off** by the Research Lead against reference commit
> `5b62949d5faff25d9dc712ee8f711b19f24fceed`, on **2026-09-04**.
>
> The scientific design — sections 0 to 9 as revised in that commit — is now
> **frozen**, pending only implementation. It is changeable from here only by
> **numbered amendment**, as in
> `docs/preregistration_trailing_gap_mechanism_v1.md`. That covers the gap
> lengths, the SESOI, the bootstrap settings, the primary intersection
> hypothesis, the classification thresholds, the multiplicity treatment, the
> clean control and the three replication outcomes.
>
> **What has changed since sign-off is implementation and recorded fact, not
> design.** ETTh2 has been downloaded and validated; its measured properties
> are recorded in **factual amendment A1** below. The config, fetcher, runner,
> analysis and decision rule exist. **No ETTh2 forecast has been produced.**
> Real execution requires a separate, explicit Research Lead approval.

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

## 1. Two things that must not be conflated

This round inherits one thing and introduces another, and the document is
careful to keep them apart:

| | Status |
|---|---|
| **The execution design** — context length, horizon, gaps, conditions, model revision, patch grid, isolation settings, bootstrap | **Inherited unchanged from F3.** Not restated here, so it cannot drift. |
| **The replication decision criterion** — the primary intersection hypothesis of §5, its multiplicity treatment, and the REPLICATED / CONTRADICTED / INCONCLUSIVE outcomes of §5.1 | **NEW, and specific to this round.** It did not exist in F3 and is not inherited. |

**Do not read this document as "only the dataset differs."** The dataset is the
only change to the *execution design*; the *decision rule for what counts as a
replication* is new, was written after seeing the ETTh1 result, and is stated
below rather than inherited.

### 1.1 The inherited execution design

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

### 2.1 Native missingness > 0 is a HARD STOP

**If ETTh2's `OT` column contains ANY native missing values, stop and escalate
to the Research Lead before proceeding.** Do not silently drop the affected
rows, impute them, or fold them in as another missingness condition.

ETTh1's **zero** native-missing property was load-bearing for the entire
target-integrity framework: every guarantee about masks touching only the
context, and about the scored target being byte-identical across conditions,
was built assuming the source series is complete. A dataset with native gaps
breaks that assumption in ways this design has not been checked against — for
instance, a native gap inside a forecast target would make the target neither
clean nor byte-comparable across conditions.

Handling native missingness is a **design question requiring its own
preregistered decision**, not an implementation detail.

**Done, and recorded in amendment A1 (2026-09-04).** ETTh2 has been fetched and
validated. Its `OT` column carries **zero** native missing values, so the hard
stop did not fire — it was **executed and passed**, not skipped. The checksum,
row count, timestamp range and derived origin count are in A1.

The gate stays armed at run time. It is not a one-off acceptance check: it runs
at the top of **every** phase of the runner, before any model is loaded, and it
**raises** rather than logging. A swapped, truncated or corrupted ETTh2 halts
the pipeline at that point. See `data/fetch_etth2.assert_no_native_missingness`
and `experiments/run_etth2.native_missingness_gate`.

---

## 3. Contrasts — unchanged

```
R_g = MAE(trailing_nan(g)) − MAE(truncated_long(g))
B_g = MAE(trailing_nan(g)) − MAE(internal_block(g, d=16))
```

Two **separately corrected** Holm families of 4, `{R_16..R_128}` and
`{B_16..B_128}`, corrected separately from each other and from every family
in the ETTh1 work. "Separately corrected" is a multiplicity-control choice
and **is not a claim that the families are statistically independent** — they
are not assumed statistically independent because they share origins and the
`trailing_nan` arm. No direction is asserted for that dependence; nothing here
derives its sign. See §5.1.

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

### 5.1 Multiplicity — RESOLVED

**Each component is checked against its nominal 90% CI lying strictly inside
`±delta_ETTh2`. No additional Holm or Bonferroni correction is applied to the
intersection.**

This is the standard intersection-union test (IUT) treatment of an
intersection (AND) claim. The argument is:

> Under the global null, at least one component equivalence null is true.
> Rejecting the global null requires rejecting every component null, including
> that true null. Therefore the probability of a false global rejection is no
> greater than the size of any true component test, regardless of dependence.

The last clause is the point: the bound holds whatever the dependence structure
among the three components, so no assumption about their correlation is needed
and none is made. A union (OR) claim would need correction; an intersection
does not.

**Holm correction is retained across the four `R` gaps for material directional
readings only** — `MATERIAL_POSITIVE` / `MATERIAL_NEGATIVE`. The equivalence
check that feeds the primary intersection claim does **not** carry that
correction.

`B` remains a **separately corrected** secondary family, with its own Holm
correction across its four gaps.

> **`R` and `B` are separately corrected. They are NOT statistically
> independent, and nothing in this document may describe them as such.** They
> are not assumed statistically independent because they share origins and the
> `trailing_nan` arm. Separate correction is a multiplicity-control choice, not
> an independence claim. **No sign is claimed for the dependence** — this
> document asserts non-independence only, never its direction, because nothing
> here derives one. The two must not be conflated.

### 5.2 Primary reporting outcomes

Exactly one of three, decided from the three primary gaps' readings:

| Outcome | Condition |
|---|---|
| **REPLICATED** | all three of `R_16`, `R_32`, `R_64` read `EQUIVALENT` |
| **CONTRADICTED** | at least one of the three reads a material directional result (`MATERIAL_POSITIVE` or `MATERIAL_NEGATIVE`) |
| **INCONCLUSIVE** | no material directional result among the three, but at least one reads `UNRESOLVED` |

`CONTRADICTED` takes precedence over `INCONCLUSIVE`: a material directional
reading is a contradiction of the equivalence claim even if another gap is
merely unresolved.

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

1. **Sign-off on this document.** The §5 multiplicity question and the clean
   control below are resolved in this version; sign-off is on the resolutions as
   written, not on an open question.
2. **Amendment recording ETTh2's checksum, row count, native-missing count and
   derived origin count**, before any forecast.
3. **Model contract re-verification** — same revision, same patch grid, same
   single-shot capacity — as in the ETTh1 runbook step 2.
4. **The clean-condition control — RESOLVED.** ETTh1's audit compared against a
   prior ETTh1 run. No prior ETTh2 run exists, so the control is a comparison of
   **two independently generated runs against each other**:

   * Run an all-origin **`clean_reference`** in its **own separate directory**,
     its **own separate process**, and its **own separate model load**.
   * Run the formal run's own **clean-only pass** in a **second, independent
     process**.
   * **Before any corrupted-condition forecast runs**, require **exact
     equality** between the two on: cardinality, `(origin_id, step_index)` keys,
     forecast timestamps, ground truth, per-origin target digests, dataset
     checksum, model revision, and the canonical float32 prediction SHA256 —
     the same identity-before-values discipline as the ETTh1 clean audit.
   * **No tolerance of any kind. Any discrepancy is a hard stop.**

   **`clean_reference` is QC-only and is explicitly EXCLUDED from the
   statistical analysis.** Only the formal run's own clean-only pass feeds
   `R_g` / `B_g`. The reference run exists to detect nondeterminism between
   independent model loads; it is never a data source.
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

## A. Factual amendments

Dated, additive records of facts measured after sign-off. **An amendment records
what was measured; it never rewrites a design section.** Sections 0 to 9 above
are unchanged by anything here.

### A1 — ETTh2 dataset contract and derived origin count (2026-09-04)

Recorded after fetching and validating the canonical file. Every value was
**measured**, none assumed.

| Fact | Value |
|---|---|
| Source | `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv` |
| Local path | `data/raw/ETTh2.csv` (uncommitted, per `.gitignore`) |
| sha256 | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` |
| Row count | 17,420 |
| Target column | `OT` |
| Timestamp range | 2016-07-01 00:00:00 → 2018-06-26 19:00:00 |
| Timestamps monotonic increasing | true |
| Timestamps unique | true |
| Gaps in the hourly grid | 0 |
| Inferred frequency | `h` (matches the expected `1h`) |
| **Native missing values in `OT`** | **0** |
| Native missing values, all other columns | 0 |
| `OT` min / max / mean / std | −2.6465 / 58.8770 / 26.6094 / 11.8879 |

**The native-missingness hard stop did not fire.** ETTh2 was **confirmed** to
share ETTh1's zero-native-missingness property; §2.1's hard stop was executed
and passed, not skipped. The gate remains armed at run time — it runs at the top
of every phase, before any model load, and raises rather than warning.

**Derived origin count: 178**, from ETTh2's own 17,420 rows at L=320, H=96,
stride=96:

```
last_start = 17420 - 320 - 96 = 17004
origins    = len(range(0, 17004 + 1, 96)) = 178
```

computed with `runner.windows.enumerate_origins` — the same enumeration the
runner uses at execution time, not a parallel formula.

**That this equals ETTh1's 178 is a consequence, not an inheritance.** The two
files happen to have the same row count. The derivation is sensitive to length:
17,408 rows still yields 178, 17,407 yields 177, and 17,311 yields 176.
`tests/test_etth2_dataset.py` asserts both the derived value against the real
file and that sensitivity, so the number cannot be right by coincidence.

**Matrix size implied by A1**, asserted at run time and unchanged from the
inherited design: 178 origins × 13 conditions = **2,314 forecasts**, of which
178 are clean. The `clean_reference` run adds a further 178 clean forecasts that
are **QC-only and excluded from the analysis**.

**Dataset distinctness.** ETTh2's sha256 differs from ETTh1's
(`f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`), and the
two target series differ elementwise. A replication run against a copy of ETTh1
would be worthless; `tests/test_etth2_dataset.py` checks this rather than
assuming it.

**Model contract: unchanged, and confirmed rather than assumed.** The pinned
revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, the patch grid
(16/16/16, 64 output patches) and the 1,024-step single-shot horizon are
inherited from `configs/pilot_config.yaml` and re-asserted by
`tests/test_etth2_config.py`, including that `H + g` still fits a single forward
pass for every gap. The **real-checkpoint** re-verification
(`tests/test_model_contract.py`, `requires_model`) could not run in the
implementation environment — its network policy denies `huggingface.co` — and
**must be run on the GPU host before execution**.

---

### A2 — Pre-execution hardening (2026-09-04)

Recorded after the Research Lead declined to approve E1
(`881f1c5e04980309aafd5929ece06cbd62ea1ff0`) for execution and required
defence-in-depth first. **E2 (`c10beaf0ab7c10019c15415f5e85a65ab2cf0007`) is the
current execution commit; E1 must not be executed.**

**This amendment changes no design decision.** The gap lengths, SESOI, bootstrap
settings, primary intersection hypothesis and classification thresholds are
untouched. What changed is the machinery that enforces §2.1's hard stop, §8's
clean control and §8.1's delivery obligations, plus the correctness of what the
pipeline writes down.

| # | Hardening | Why it was needed |
|---|---|---|
| 1 | The native-missingness hard stop moved into the **shared runner**, after `load_series` and before any window or forecast. Opt-in via `dataset.native_missing_is_a_hard_stop`. | E1's gate lived only in the ETTh2 wrapper. Importing the shared runner and calling it directly walked around it — the hole the E1 report flagged. |
| 2 | ETTh2 manifests carry their **own** experiment identity, preregistration path, rule version, sign-off commit, entrypoint, phase, mock flag, and the real git commit and dirty status. The enriched summary is persisted atomically. | E1's manifests inherited ETTh1's experiment name from `trailing_gap_config.meta`, misattributing the run in the one file a reviewer uses to establish what executed. |
| 3 | **Family-correct `B_g` reporting built in**, with an explicit `comparison_arm` on every row, verified against a pre-correction numeric snapshot, staged and promoted atomically. No ETTh2 artifact carries the ETTh1 erratum id. | ETTh1 needed erratum `ERRATUM-2026-09-04-bg-comparison-arm` after the fact. ETTh2 must not repeat it, and must not claim to have been corrected for a defect it never had. |
| 4 | Mock runs are **refused by default**; under the explicit escape hatch every artifact carries `mock_model: true`, `scientifically_valid: false`, `authoritative_result: false`, a plain-language note, and a `MOCK_NOT_A_FINDING__` label prefix. | A mock classification was structurally indistinguishable from a real one. This is a constraint on the data shape, not on prose around it. |
| 5 | Clean-reference gate: distinct directories enforced, provenance columns required, both sides anchored to the checksum and revision loaded **now**, clean-only verified before the audit, and the gate bound to the **bytes** of the clean artifacts with a re-check at `formal-full` startup. | Two runs can agree with each other and both be stale, and a gate valid when written said nothing about files edited afterwards. |
| 6 | `data/fetch_etth2.py` returns a **truthful exit code** on every contract violation, tested through the real CLI as a subprocess. | A script that printed "VALIDATION NOTES" and exited 0 would let a runbook's `$?` check conclude the dataset was fine. |
| 7 | Every runbook pipeline preserves the Python command's status via `PIPESTATUS[0]`, verified by a behavioural test that runs each block against a deliberately failing command. | `cmd \| tee log` makes `$?` tee's status, which is 0 even when `cmd` failed — the same class of bug fixed in the trailing-gap runbook's §3.0 guard. |

**ETTh1 is unaffected.** The shared-runner change is additive and opt-in;
ETTh1's config does not set the policy key. The full 178-origin ETTh1 mock
matrix was hashed before and after: every column except `run_id` (a UTC
timestamp) and `runtime_seconds` (wall clock) is identical, and all seven
derived analysis files are byte-identical. Those two columns differ between any
two runs of unchanged code. The hashes are pinned in
`tests/test_native_missingness_policy.py`, so a future change that moves ETTh1's
behaviour fails the suite. **No already-produced ETTh1 result is altered by
this amendment** — the frozen classification
`INCONCLUSIVE / mixed_equivalent_and_unresolved` stands.

**Still not run.** No ETTh2 forecast has been produced. Real execution requires
a separate, explicit Research Lead approval.

---

### A3 — Three interaction bugs found in E2, and what the gate now binds to (2026-09-04)

Recorded after the Research Lead ruled E2
(`c10beaf0ab7c10019c15415f5e85a65ab2cf0007`) **NO-GO for execution**.
**E3 (`75fe27445e5c6c7fd28b3b9cca28a5d000a2fe4c`) is the current execution
commit; E1 and E2 must not be executed.**

**This amendment changes no design decision.** Gap lengths, SESOI, bootstrap
settings, the primary intersection hypothesis, the classification thresholds and
every model setting are untouched. All three fixes are structural or procedural.
Each was **reproduced before being fixed**, as every prior round in this project
has required.

#### A3.1 The clean-content invariant — what "the gate" now binds to

This is the substantive change and it needs stating precisely, because *what the
gate is a statement about* has changed.

**Before (E2).** The gate recorded a SHA256 of the whole bytes of
`predictions_long.csv` and `window_results.csv` on each side.

**The failure.** `formal-full` appends corrupted-condition rows to those same
files while building the 2,314-row matrix. A whole-file hash cannot distinguish
"a clean row was tampered with" from "the run legitimately appended its own
progress", so **any** append invalidated the gate. Reproduced: a `formal-full`
limited to 10 origins wrote 298 cells; the resume then hard-stopped with
`formal/predictions_long.csv has CHANGED since the audit passed`. **Resumability
was broken outright.**

**After (E3).** The gate records, per side and per file, a SHA256 of a
**canonical projection of the clean rows only**:

1. read the file as text;
2. keep only rows whose `condition_id` is `clean`;
3. sort by keys — `(origin_id, step_index)` for predictions,
   `(origin_id, condition_id)` for window results — with a stable sort;
4. order the columns, render to CSV, hash.

**What this permits:** anything outside the projection. Appended
corrupted-condition rows are invisible to it, so a gate written before an
interruption still authorises the resume.

**What it still forbids, with zero tolerance:** the projection contains **every
column of every clean row**, so a changed value, a changed key, a **removed**
clean row and an **added** clean row each change the digest. One differing
character is a different hash and a hard stop. Row *order* is not part of the
invariant, because the projection sorts; that is the only degree of freedom
added, and it is deliberate.

**Not a loosening.** What changed is *what the invariant is about* — the audited
clean content — not how strictly it is enforced. A gate in the old whole-file
format is **refused**, not silently accepted, since this version cannot verify
it; re-run the audit.

#### A3.2 Provenance is established positively, never by absence

**The failure.** The mock check returned "real" whenever it failed to find a
mock marker. Reproduced: five of six broken directories were accepted as a real
ETTh2 `formal-full` result — an empty `{}` manifest, a manifest carrying only
`mock_model: false`, a run with no summary, a manifest naming ETTh1's
experiment, and one naming the `clean-reference` phase.

**After.** A run is `REAL` only when the manifest **and** the summary explicitly
and consistently establish all of: `mock_model` false, the ETTh2 experiment
identity, the `formal-full` phase, the expected entrypoint, and a real
non-placeholder model revision. Missing, partial, malformed or contradictory
evidence is **`INVALID_PROVENANCE`** — a distinct third verdict, a distinct
exception, exit code 5.

**`--allow-mock-analysis` does not bypass it.** That flag analyses a run whose
mockness is *established*. Broken or ambiguous provenance is a different failure
class, and deciding to salvage it is not a call this tool may make.

#### A3.3 A broken matrix fails the process

**The failure.** A forecaster raising on one cell produced
`forecasts_failed_this_process: 1` and the process **exited 0**. The failure
lived as a status value inside an output file, invisible to anything reading an
exit code, and would have been absorbed into a "resume later" path.

**After.** On a full `formal-full` pass the runner asserts every frozen
completion invariant — 178 origins, 2,314 rows, 2,314 distinct cells,
`rows_by_status == {"ok": 2314}`, max target digests per origin == 1, zero
failures — and the CLI **exits 6** on any violation. **No cell is retried**, by
design: a retry would hide both the failure and its cause. The summary is
persisted before the raise, so the evidence survives. A pass deliberately
limited with `--limit-origins` is a partial run by instruction, not a violation.

#### A3.4 Corrected hard-stop numbering

The runbook previously listed "**4.0**" among its hard stops. There is no §4.0.
The initialization guard is **§3.0** and the clean-reference audit is **§5**. The
current hard stops are steps **2, 3, 3.0, 5, 6 and 7**, plus the
native-missingness check inside every runner invocation, and the
quick-reference table now lists twelve.

**Still not run.** No ETTh2 forecast has been produced. Real execution requires
a separate, explicit Research Lead approval.

---

### A4 — REAL provenance requires the exact pinned revision (2026-09-04)

Recorded after the Research Lead independently confirmed a defect in E3
(`75fe27445e5c6c7fd28b3b9cca28a5d000a2fe4c`), which **remains NO-GO**.
**E4 (`e67a5858b370d844bfcc90328c50cb0d08b6c5ac`) is the current execution
commit; E1, E2 and E3 must not be executed.**

**This amendment changes no design decision.** Gap lengths, SESOI, bootstrap
settings, the primary hypothesis, the classification thresholds and every model
setting are untouched. Nothing numerical or execution-related changed. The fix
is confined to how a completed run's provenance is judged.

#### The defect

`assess_provenance` asked only whether `model_contract.revision` was long enough
and did not begin with a known placeholder prefix. **"Not obviously fake" is not
"the pinned checkpoint."** Reproduced against E3 — every one of these returned
`REAL`:

| Recorded revision | E3 verdict |
|---|---|
| `deadbeef` | **REAL** |
| `0000…0000` (40 chars) | **REAL** |
| `ffff…ffff` (40 chars) | **REAL** |
| pinned value + one leading character | **REAL** |
| pinned value, last character changed | **REAL** |
| pinned value + a trailing space | **REAL** |
| pinned value, uppercased | **REAL** |

A run produced against **unknown weights** would therefore have been analysed,
reported and compared against ETTh1 as a real ETTh2 result. For a replication
whose entire claim rests on running the *same* model on a *different* dataset,
that is not a cosmetic failure — the pinned checkpoint is half the comparison.

#### The fix

`REAL` now requires `model_contract.revision` to **EQUAL** the pinned value
`29ec3766d36d6f73f0696f85560a422f50e8498c`.

* **Exact comparison.** Not stripped, not case-folded, not prefix- or
  substring-matched. A trailing space and an uppercased hash are different
  strings, and a run whose manifest recorded one did not record the pinned
  checkpoint. This is the same zero-tolerance discipline §8's clean audit
  applies to its identity fields, and it is deliberate: normalising here would
  be choosing a tolerance after seeing a discrepancy.
* **One source of truth.** The expected value is passed in, read from
  `pilot_config["model"]["revision"]` at call time. It is **not** duplicated in
  `experiments/analyze_etth2.py`; a test asserts the string does not appear in
  that file at all, so the pin cannot drift from the config it is pinned in.
* **No default pass.** A caller that omits the expected revision cannot obtain
  `REAL` — the omission is itself recorded as a reason and the verdict is
  `INVALID_PROVENANCE`.
* **Not bypassable.** A wrong revision is `INVALID_PROVENANCE`, exit code 5, and
  `--allow-mock-analysis` does not reach it, exactly as A3.2 established for the
  other invalid-provenance cases.

The placeholder helper was inverted and renamed to state what it is for. It now
recognises a **mock** run's revision only, and can no longer be mistaken for
evidence that a revision is the *right* one — which is how the defect arose.

#### Coverage

The reproduction case; three well-formed but wrong 40-character SHAs, so this
cannot be a length or format check; seven near misses including prefix and
suffix truncation and single-character changes at either end; seven whitespace
and case variants; the exact pinned revision as a positive control; a declared
mock left unaffected; a caller omitting the pin; a caller whose config moves the
pin; and both settings of `--allow-mock-analysis`.

**Still not run.** No ETTh2 forecast has been produced. Real execution requires
a separate, explicit Research Lead approval.

---

## 10. Sign-off status of every item

Signed off against commit `5b62949d5faff25d9dc712ee8f711b19f24fceed` on
2026-09-04. Every design item below is **FROZEN**: changeable only by numbered
amendment. Nothing in this table is open.

| Item | Status |
|---|---|
| Design inherited from F3, dataset changed only (§1) | **FROZEN** |
| ETTh2 dataset contract and origin count | **RECORDED — amendment A1.** 17,420 rows, sha256 `a3dc2c59…`, 0 native missing in `OT`, **178 derived** origins. Measured, not assumed. |
| Primary hypothesis: `R_16 ∧ R_32 ∧ R_64` equivalent (§5) | **FROZEN** |
| Provenance label as an exploratorily generated hypothesis (§0, §5) | **FROZEN, and not negotiable in substance** |
| Multiplicity treatment of the three-gap intersection (§5.1) | **FROZEN.** No correction on the intersection itself; Holm within `R` over material directional readings only; `B` corrected separately. `R` and `B` are not assumed statistically independent, and no sign is claimed for the dependence. |
| Replication decision criterion: REPLICATED / CONTRADICTED / INCONCLUSIVE (§5.2) | **FROZEN.** Implemented as `stats/replication_decision.py`, `rule_version: etth2-replication-v1`. CONTRADICTED takes precedence. |
| Clean-condition control in the absence of a prior ETTh2 run (§8, precondition 4) | **FROZEN.** Independent `clean_reference` run; exact equality on the identity fields; no tolerance; QC-only and excluded from analysis. |
| Native ETTh2 missingness > 0 is a hard stop (§2.1) | **FROZEN, and EXERCISED.** A1 records the measured zero. The gate runs at the top of every phase, before any model load, and raises. |
| `R_128` mandatory secondary (§6) | **FROZEN.** Measured and reported; excluded from the primary decision. |
| `B_g` secondary and content-confounded (§3.1) | **FROZEN — carried forward verbatim** |
| No pooled analysis; no new model/imputation/baseline (§7) | **FROZEN** |

### 10.1 What remains before execution

Not design decisions — operational gates. See
`docs/gpu_execution_runbook_etth2.md`.

| Gate | State |
|---|---|
| Real-checkpoint contract re-verification on the GPU host | **OUTSTANDING.** Could not run where this was implemented; that environment's network policy denies `huggingface.co`. |
| `clean_reference` run, all origins, own process and model load | **NOT RUN** |
| Formal run's clean-only pass, second independent process | **NOT RUN** |
| Exact clean-reference audit — the hard stop before any corrupted forecast | **NOT RUN** |
| The remaining corrupted-condition forecasts | **NOT RUN** |
| **Research Lead approval for real execution** | **REQUIRED, and separate from this sign-off.** Implementation sign-off is not execution approval. E1 was declined, E2 and E3 ruled NO-GO; E4 is the current candidate. |
