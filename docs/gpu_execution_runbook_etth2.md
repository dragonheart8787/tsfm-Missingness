# GPU-host execution runbook — ETTh2 trailing-gap replication

**For a human to run on the GPU host, in order. Do not skip or reorder steps.**

> **This runbook is a DELIVERABLE, not an instruction to run now.** Real ETTh2
> execution requires a **separate, explicit Research Lead approval**. Sign-off on
> the preregistration froze the design; it did not authorise a forecast.

Steps **2**, **3**, **3.0**, **5**, **6** and **7**, plus the
native-missingness check inside every runner invocation, are **hard stops**. If
any fails, stop and report. **Do not select a tolerance after seeing a
discrepancy** — a mismatch is a stop, not a tuning problem, and that applies to
the identity fields as much as to the predictions.

> **Corrected numbering.** An earlier revision of this document listed "4.0" as
> a hard stop. There is no §4.0; the clean-reference audit is **§5** and the
> initialization guard is **§3.0**. The list above is the current, correct one
> and matches the quick-reference table at the end.

**Execution commit (E4): `e67a5858b370d844bfcc90328c50cb0d08b6c5ac`** — the current ETTh2
implementation freeze. This is what step 1 checks out and what every run
manifest must record as executed.

**E4 supersedes E3 (`75fe27445e5c6c7fd28b3b9cca28a5d000a2fe4c`), which remains
NO-GO, and E2 and E1 before it.** E4 fixes one narrow but serious defect
recorded in preregistration amendment A4: `REAL` provenance checked only that a
model revision looked non-placeholder, not that it was the pinned one, so a run
produced against unknown weights read as a real ETTh2 result. Do **not** execute
E1, E2 or E3.

This runbook ships in a later documentation commit (D8), which names E4 above.
**D8 is never the executed commit** — no documentation commit ever is.
If `git rev-parse HEAD` during a run does not equal E4, stop: the manifests
would otherwise attribute results to the wrong tree.

---

## 0. What is different from the ETTh1 runbook

Read this before anything else. Three things changed; everything else is the
same discipline.

| | ETTh1 (`gpu_execution_runbook_trailing_gap.md`) | ETTh2 (this document) |
|---|---|---|
| Clean control | one clean pass, audited against the **prior** `results/pilot_v1` run | **two independently generated runs**, audited against **each other** — there is no prior ETTh2 run |
| Native missingness | known zero, checked once at fetch | **hard stop armed at the top of every phase**, before any model load |
| Origin count | 178, preregistered | 178, **derived** from ETTh2's own row count (amendment A1) — the runner re-derives it and refuses a mismatch |

**Four invocations, four processes.** The separation is the control, not an
implementation convenience: `clean_reference` and the formal clean pass must be
**separate processes with separate model loads**, or they cannot detect
nondeterminism between loads. Do not merge them, and do not run them in one
Python session.

---

## 1. Check out the frozen commit

```bash
git clone https://github.com/dragonheart8787/tsfm-Missingness.git
cd tsfm-Missingness
git fetch origin claude/etth2-replication-v1
git checkout e67a5858b370d844bfcc90328c50cb0d08b6c5ac    # E4, the current implementation freeze
git status --porcelain            # must print nothing
git rev-parse HEAD                # must equal the execution commit; record it

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --torch-backend cu124
uv pip install --python .venv/bin/python -e ".[dev]"

mkdir -p handoff
```

---

## 2. Fetch and validate ETTh2 — HARD STOP on native missingness

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python data/fetch_etth2.py \
    --config configs/etth2_config.yaml 2>&1 | tee handoff/01_fetch_etth2.txt
  status=${PIPESTATUS[0]}
  echo "fetch_etth2 exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

Expected, and each is asserted rather than eyeballed:

```
sha256                       a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b
n_rows                       17420
native_missing_in_target     0
n_gaps_in_hourly_grid        0
timestamp range              2016-07-01 00:00:00 .. 2018-06-26 19:00:00
DERIVED origin count         178
```

> **HARD STOP.** A non-zero exit, a checksum mismatch, or **any** native missing
> value in `OT` stops the run. Do **not** impute, drop, interpolate, or widen a
> tolerance. How to handle native missingness is a **design decision reserved to
> the Research Lead**; report and wait. The check raises inside
> `data/fetch_etth2.py` and again at the top of every runner phase, so it cannot
> be bypassed by starting at a later step.

---

## 3. Full test suite, including the real-model tests

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python -m pytest tests/ -v 2>&1 | tee handoff/02_tests.txt
  status=${PIPESTATUS[0]}
  echo "pytest exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

**All tests must pass here**, including the three `requires_model` tests that
could not run in the implementation environment (its network policy denied
`huggingface.co`):

* `tests/test_model_contract.py::test_real_checkpoint_matches_the_preregistered_contract`
* `tests/test_model_contract.py::test_real_model_accepts_nan_context_and_returns_finite_forecasts`
* `tests/test_isolation.py::test_9e_real_model_isolation`

> **HARD STOP.** The first of these re-verifies that revision
> `29ec3766d36d6f73f0696f85560a422f50e8498c` still reports the pinned patch grid
> (16/16/16, 64 output patches, 1,024-step single-shot horizon). The model is
> unchanged since ETTh1, so this **should** pass — confirm it rather than assume
> it. A contract drift invalidates the comparison to ETTh1 and is a stop.

### 3.0 Initialization guard — HARD STOP

**Run once, before the first execution below.** Neither run directory may exist:
a leftover from an abandoned attempt would be resumed into, silently mixing
cells from two different runs.

```bash
# Wrapped in a subshell so the `exit` cannot close your login shell when pasted.
# The subshell is the FINAL command: its status is the block's status, so a
# script wrapping this in `|| abort` actually halts. Do not append anything
# after it -- a trailing `echo` would mask the status with its own 0.
(
  status=0
  for d in results/etth2_replication_v1/clean_reference \
           results/etth2_replication_v1/formal; do
    if test -e "$d"; then
      echo "HARD STOP: $d already exists."
      ls -la "$d"
      status=1
    fi
  done
  if test "$status" -eq 0; then
    echo "OK: no existing run directories; safe to initialize."
  else
    echo "Do not delete, overwrite, or reuse them. Report and await instruction."
  fi
  exit "$status"
)
```

Returns **0** when both are absent and **1** when either exists, and leaves your
interactive shell alive either way. Read the status with `echo $?` **as a
separate command**.

> **This guard applies at INITIALIZATION ONLY.** Once step 4 has created the
> directories they are the correct ones: step 6's full-matrix pass and every
> interruption-recovery resume must continue using **those same directories** and
> the same execution commit. Do not re-run the guard before step 6 or before a
> resume — resumption at origin × condition granularity is the intended
> behaviour there, and re-running the guard would wrongly block it.

---

## 4. The two-pass clean design — two separate processes

### 4.1 `clean_reference` — all 178 origins, clean only, QC ONLY

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python experiments/run_etth2.py \
    --phase clean-reference 2>&1 | tee handoff/03_clean_reference.txt
  status=${PIPESTATUS[0]}
  echo "clean-reference exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

Expect **178 rows**, all `kind == clean`, zero failures.

> `clean_reference` is **QC-only**. It exists to detect nondeterminism between
> independent model loads. It is **excluded from the statistical analysis** and
> is **never** a data source for `R_g` or `B_g`. The audit refuses a reference
> directory that contains any non-clean condition, so it cannot be mistaken for
> an analysable run.

### 4.2 The formal run's own clean-only pass — a SECOND, independent process

**Start a new shell, or at minimum a new Python process.** This is the control.

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python experiments/run_etth2.py \
    --phase formal-clean 2>&1 | tee handoff/04_formal_clean.txt
  status=${PIPESTATUS[0]}
  echo "formal-clean exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

Expect **178 rows**, all `kind == clean`, zero failures.

---

## 5. Clean-reference exact audit — HARD STOP

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python experiments/run_etth2.py \
    --phase audit 2>&1 | tee handoff/05_clean_audit.txt
  status=${PIPESTATUS[0]}
  echo "audit exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

Every one of these must match **exactly**, with **no tolerance of any kind**:

1. row cardinality — 178 × 96 = **17,088** clean rows on each side
2. `(origin_id, step_index)` key sets
3. `forecast_timestamp`, cell by cell
4. `ground_truth`, cell by cell
5. per-origin `target_sha256`, on both sides and between them
6. `dataset_sha256`
7. `model_revision`
8. the canonical origin/step-sorted **float32 sha256** of the median predictions

> **HARD STOP.** Any discrepancy stops the run. Do not proceed, and **do not
> select a tolerance now** — not for the predictions and not for any identity
> field. There is no tolerance mechanism, by design. A one-ULP difference is a
> stop, and the tests assert that it is.

On success this writes `results/etth2_replication_v1/formal/clean_reference_audit.json`,
the **gate file**, recording `passed: true` together with the `dataset_sha256`
and the `model_revision` the audit was run against. Step 6 refuses to start
without it, and refuses it if either field fails to match what step 6 itself
loads — so a stale gate cannot authorise a run against different data or
different weights.

### 5.1 What the gate binds to — the clean-content invariant

The gate also records, under `clean_content_digests`, a SHA256 per side per file
of a **canonical projection of the clean rows only**:

1. read `predictions_long.csv` and `window_results.csv` as text;
2. keep only rows whose `condition_id` is `clean`;
3. sort them by their keys — `(origin_id, step_index)` for predictions,
   `(origin_id, condition_id)` for window results — with a stable sort;
4. put the columns in sorted order, render to CSV, and hash.

**Why a projection and not the whole file.** Step 6 appends
corrupted-condition rows to those same files while building the 2,314-row
matrix. A whole-file hash could not tell a tampered clean row from a
legitimately appended corrupted one, so **every** append invalidated the gate —
and an interrupted step 6 found its own prior legitimate progress reported as
tampering. Resumability was broken outright. Everything outside the projection
is now free to vary.

**This is not a loosening of what the gate protects.** The projection contains
every column of every clean row, so a changed value, a changed key, a removed
clean row and an added clean row each change the digest. There is **no
tolerance**: one differing character is a different hash and a hard stop. What
changed is *what the invariant is about* — the audited clean content — not how
strictly it is enforced.

**A gate written before this change is refused, not silently accepted.** It
records whole-file digests, which this version cannot verify; re-run step 5.

---

## 6. Only if step 5 passed exactly — the remaining forecasts

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python experiments/run_etth2.py \
    --phase formal-full 2>&1 | tee handoff/06_formal_full.txt
  status=${PIPESTATUS[0]}
  echo "formal-full exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

The 178 clean cells are already written and are skipped at origin × condition
granularity; this pass produces the remaining **2,136** corrupted-condition
forecasts.

**On completion the runner asserts, and the summary must show:**

```
result_rows                          2314
expected_result_rows                 2314
distinct_cells                       2314
expected_distinct_cells              2314
origins_fully_completed              178
rows_by_status                       {"ok": 2314}
max_distinct_target_digests_per_origin  1
```

2,314 = 178 origins × 13 conditions. **Distinct** cells, not merely 2,314 rows,
so a duplicated cell cannot mask a missing one.

**If interrupted**, re-run the same command in the same directory. Resumption is
decided from `window_results.csv` at origin × condition granularity; the
checkpoint file is advisory and never gates it. Do **not** re-run the step 3.0
guard. The gate written in step 5 remains valid across the interruption and the
resume — that is what §5.1's clean-content projection is for.

> **HARD STOP — the process exit code.** On completion the runner asserts every
> invariant above and **exits 6** if any is violated, including a single failed
> cell (`rows_by_status` would then read `{"ok": 2313, "failed": 1}`). **No cell
> is ever retried automatically.** A failure is surfaced at the process level
> and left for you: re-running would hide both the failure and its cause. Read
> the recorded `error_message` in `window_results.csv` and report it.
>
> Check the status explicitly — the pipeline below preserves it via
> `PIPESTATUS[0]`, so a non-zero exit is visible rather than masked by `tee`.

---

## 7. Analysis

```bash
# `| tee` makes $? the status of TEE, which is 0 even when python failed.
# PIPESTATUS[0] is python's own status; the subshell exits on it, so the
# block's status is the command's. The subshell is the FINAL thing here:
# do not append an `echo` after it, or its 0 would mask the failure.
(
  .venv/bin/python experiments/analyze_etth2.py 2>&1 | tee handoff/07_analysis.txt
  status=${PIPESTATUS[0]}
  echo "analyze_etth2 exit status: $status"
  exit "$status"
)
```

Read the status with `echo $?` **as a separate command**. A non-zero
status is a stop.

Writes into `results/etth2_replication_v1/formal/analysis/`:

* `R_contrasts.csv`, `B_contrasts.csv` — per-gap readings and CIs
* `classification_etth2.json` — **the replication outcome**
* `etth2_analysis.json` — the full payload
* the dose-response and paired-difference outputs, as on ETTh1

`analyze_etth2.py` establishes the run's provenance from **positive evidence**
before doing anything: the manifest and the summary must both explicitly and
consistently carry `mock_model`, the ETTh2 experiment identity, the
`formal-full` phase, the expected entrypoint, and a real model revision.

* **`INVALID_PROVENANCE` → exit 5.** Missing, partial, malformed or
  contradictory provenance — **including a `model_contract.revision` that is
  not exactly the pinned one.** **`--allow-mock-analysis` does NOT bypass
  this** — that flag analyses a run whose mockness is *established*; broken
  provenance is a different failure class, and salvaging it is not a decision
  this tool may make. Re-run the phase or report the directory.
* **Mock run → exit 3**, unless `--allow-mock-analysis` is passed.
* A real run needs no flag. If you ever pass `--allow-mock-analysis` for a pipeline exercise, every
artifact it writes is stamped `mock_model: true`,
`scientifically_valid: false`, `authoritative_result: false`, and its outcome
label is prefixed `MOCK_NOT_A_FINDING__` — such output must never be reported.

Outputs are written to `analysis.staging`, verified, then promoted with an
atomic rename. A failure leaves **no** official `analysis/` directory rather
than a half-correct one, and exits 4.

#### 7.1 The revision pin

`REAL` provenance requires `model_contract.revision` to **equal** the pinned
value `29ec3766d36d6f73f0696f85560a422f50e8498c`, read from
`configs/pilot_config.yaml` at call time — the single source of truth for it.

The comparison is **exact**: not stripped, not case-folded, not prefix-matched.
A trailing space, an uppercased hash, a one-character difference and a plausible
but different 40-character SHA are each a different checkpoint claim, and each
is `INVALID_PROVENANCE`. This is the same zero-tolerance discipline §5 applies
to the audit's identity fields.

If step 3's real-checkpoint verification passed, this will pass too — the run
loaded the pinned revision and recorded it. A failure here means the manifest
does not say what you think it says: **stop and report it**, do not edit the
manifest.

The outcome is exactly one of **REPLICATED** / **CONTRADICTED** /
**INCONCLUSIVE**, decided from `R_16`, `R_32`, `R_64` only. `R_128` is measured
and reported but excluded; `B_g` is reported alongside and never decisive.

> Any of the three is a valid result. **CONTRADICTED is not a failure of the
> run**, and INCONCLUSIVE is neither a replication nor a contradiction. Do not
> re-run, re-tune, or re-cut anything in response to the outcome.

---

## 8. Deliver to the reviewer

Per `docs/results_delivery_policy.md`. The Research Lead cannot recompute values
that live only in an uncommitted `results/` directory — **summaries, not just
code, must reach the reviewer**:

* `git rev-parse HEAD` from the run, and both run manifests
* `data/raw/ETTh2_validation.json` — the dataset contract as measured on the host
* `handoff/01..07` transcripts
* `clean_reference_audit.json` — the gate, with both canonical hashes
* both `run_summary.json` files
* `R_contrasts.csv`, `B_contrasts.csv`, `classification_etth2.json`,
  `etth2_analysis.json`, `dose_response.json`
* bootstrap sensitivity across block lengths 4 / 8 / 12

Raw per-forecast files may stay uncommitted. Summaries may not.

---

## Quick reference — the hard stops

| # | Step | Condition | Action |
|---|---|---|---|
| 1 | 2 | any native missing value in ETTh2 `OT` | stop; do not impute or drop; escalate to the Research Lead |
| 2 | 2 | dataset sha256 or row count mismatch | stop; the file is not the recorded one |
| 3 | 2 | derived origin count ≠ 178 | stop; the count is derived, never assumed |
| 4 | 3 | any test fails, incl. the 3 real-model tests | stop; a contract drift invalidates the ETTh1 comparison |
| 5 | 3.0 | either run directory already exists | stop; do not delete, overwrite or reuse |
| 6 | 5 | **any** clean-audit field differs | stop; do not select a tolerance, for any field |
| 7 | 6 | gate file absent, failed, or mismatched | stop; the audit does not authorise this run |
| 8 | 6 | the audited **clean rows** changed since the audit passed (§5.1) | stop; the gate no longer describes the clean data. Appended corrupted rows are NOT this |
| 9 | 6 | gate predates the clean-content projection | stop; re-run step 5 to write a verifiable gate |
| 10 | 6 | any completion invariant violated, incl. one failed cell | stop; **exit 6**. No cell is retried — read the recorded error |
| 11 | 7 | `INVALID_PROVENANCE`, incl. a revision that is not exactly the pinned one (§7.1) | stop; **exit 5**. Not bypassable with `--allow-mock-analysis` |
| 12 | 7 | staged reporting fails verification | stop; nothing was promoted, and exit 4 says so |

**No step below a hard stop may be run until that stop is cleared by the
Research Lead. Clearing means a decision, not a retry.**
