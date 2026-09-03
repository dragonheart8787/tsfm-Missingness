# GPU-host execution runbook — trailing-gap mechanism experiment

**For a human to run on the GPU host, in order. Do not skip or reorder steps.**

Steps 2 and 4 are **hard stops**: if either fails, stop and report. Do not
select a tolerance after seeing a discrepancy — a mismatch is a stop, not a
tuning problem.

**Execution commit (F): `<FROZEN_COMMIT_F2>`** — the implementation freeze. This
is what step 1 checks out and what the run manifest must record as executed.

This runbook ships in a later documentation commit (D), which names F above.
**D is never the executed commit.** If `git rev-parse HEAD` during the run does
not equal F, stop: the manifest would otherwise attribute results to the wrong
tree.

---

## 1. Check out the frozen commit

```bash
git clone https://github.com/dragonheart8787/tsfm-Missingness.git
cd tsfm-Missingness
git fetch origin claude/chronos2-missingness-pilot-xwx5y7
git checkout <FROZEN_COMMIT_F2>    # F, the implementation-freeze commit
git status --porcelain            # must print nothing
git rev-parse HEAD                # must equal F; record it in the run manifest

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch --torch-backend cu124
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python data/fetch_etth1.py --config configs/pilot_config.yaml

.venv/bin/python -m pytest tests/ -v      # includes the 3 real-model tests
```

---

## 2. Reproduce the existing analyses and compare against history

```bash
mkdir -p handoff

.venv/bin/python scripts/verify_model_contract.py 2>&1 | tee handoff/01_model_contract.txt

.venv/bin/python scripts/analyze_run.py \
  --run-dir results/pilot_v1 \
  --out-dir  results/pilot_v1/post_hoc_reanalysis_v2 2>&1 | tee handoff/02_decision_v2.txt

.venv/bin/python scripts/analyze_internal_only.py \
  --run-dir results/pilot_v1 2>&1 | tee handoff/03_internal_only.txt
```

**Check `handoff/01_model_contract.txt` reports:**

| Field | Expected |
|---|---|
| revision | `29ec3766d36d6f73f0696f85560a422f50e8498c` |
| input_patch_size / stride | 16 / 16 |
| output_patch_size | 16 |
| max_output_patches | 64 |
| single-shot capacity | 1024 (≥ the matrix's largest request, 224) |

**Compare `handoff/02` and `handoff/03` against the values already reported in
this project's history** — `BJ1_20`, `BJ2_40`, `IC1_20`, `IC2_40`.

> **HARD STOP.** If any reproduced summary disagrees with the reported history,
> stop and report. **Do not proceed to step 3.** A disagreement means the
> environment has drifted and every downstream comparison is invalid.

---

## 3. Run ONLY the 178 clean forecasts

### 3.0 Initialization guard — HARD STOP

**Run this once, before the first execution below.** The run directory must not
already exist: a leftover directory from an abandoned attempt would be resumed
into, silently mixing cells from two different runs.

```bash
# Wrapped in a subshell so the `exit` cannot close your login shell when pasted.
( test -e results/trailing_gap_v1 && {
    echo "HARD STOP: results/trailing_gap_v1 already exists."
    echo "Do not delete, overwrite, or reuse it. Report it and await instruction."
    ls -la results/trailing_gap_v1
    exit 1
  }
  echo "OK: no existing run directory; safe to initialize." )
echo "guard exit status: $?    # 0 = safe to proceed, 1 = HARD STOP"
```

> **HARD STOP.** If the directory exists, stop and report. **Do not delete it,
> do not overwrite it, and do not silently reuse it** — it may hold results
> produced under a different commit or a partially completed attempt, and only
> the Research Lead can decide what it is.

**This guard applies at INITIALIZATION ONLY.** Once step 3 has created the
directory, it is the correct one: step 5's full-matrix pass and every
interruption-recovery resume must continue using **that same directory** and the
**same execution commit F**. Do not re-run this guard before step 5 or before a
resume — resumption at origin × condition granularity is the intended behavior
there, and re-running the guard would wrongly block it.

### 3.1 Execute

```bash
.venv/bin/python experiments/run_trailing_gap.py \
  --run-dir results/trailing_gap_v1 \
  --only-conditions clean 2>&1 | tee handoff/04_clean_run.txt
```

Expect 178 rows, `kind == clean`, zero failures. Nothing else runs yet.

---

## 4. Clean-prediction checksum audit — HARD STOP

```bash
.venv/bin/python scripts/audit_clean_predictions.py \
  --new-run-dir       results/trailing_gap_v1 \
  --reference-run-dir results/pilot_v1 2>&1 | tee handoff/05_clean_audit.txt
```

Verifies **identity before values**, each check a hard stop:

1. no duplicate or missing cells; row count exactly 178 x 96 = 17,088
2. identical `(origin_id, step_index)` key sets
3. exact forecast timestamps
4. exact `ground_truth`
5. per-origin `target_sha256` agreeing with `window_results.csv` on both sides
6. matching dataset checksum and model revision
7. canonical origin/step-sorted float32 SHA256 of the predictions

> **HARD STOP.** Every check must pass. Identical prediction numbers are not
> sufficient: they would pass even with origin keys, timestamps, ground truth or
> target digests wrong underneath them, which is exactly why identity is checked
> first. The script exits non-zero and prints every failure with diagnostics.
> **There is no tolerance mechanism for any field, and none may be added after
> seeing a discrepancy.**

---

## 5. Only if step 4 passed exactly — run the remaining 2,136 forecasts

```bash
tmux new -d -s tgap '.venv/bin/python experiments/run_trailing_gap.py \
  --run-dir results/trailing_gap_v1 >> results/trailing_gap_v1.stdout.log 2>&1'
```

The runner resumes at **origin x condition** granularity: the 178 clean cells
already written are skipped individually, and this adds the remaining **2,136**
(12 conditions × 178 origins) for **2,314** total. Resumption is decided from
`window_results.csv`; the checkpoint file is advisory and never gates it.

Progress from a fresh login:

```bash
tail -f results/trailing_gap_v1/run.log
python -c "import json;print(json.load(open('results/trailing_gap_v1/checkpoint.json'))['n_completed'],'/178')"
```

Resume after an interruption — the identical command, which is what
`test_resume_does_not_duplicate_or_overwrite` validates:

```bash
.venv/bin/python experiments/run_trailing_gap.py --run-dir results/trailing_gap_v1
```

Then analyse:

```bash
.venv/bin/python experiments/analyze_trailing_gap.py \
  --run-dir results/trailing_gap_v1 2>&1 | tee handoff/06_trailing_gap_analysis.txt
```

---

## 6. Deliver to the reviewer

Per `docs/results_delivery_policy.md`, summaries must reach the reviewer — raw
per-forecast predictions may stay gitignored.

```bash
cp configs/pilot_config.yaml configs/trailing_gap_config.yaml handoff/
cp results/trailing_gap_v1/run_manifest.json                  handoff/
cp results/trailing_gap_v1/run_summary.json                   handoff/
cp results/trailing_gap_v1/analysis/R_contrasts.csv           handoff/
cp results/trailing_gap_v1/analysis/B_contrasts.csv           handoff/
cp results/trailing_gap_v1/analysis/dose_response.json        handoff/
cp results/trailing_gap_v1/analysis/classification.json       handoff/
cp results/trailing_gap_v1/analysis/trailing_gap_analysis.json handoff/
cp results/pilot_v1/post_hoc_reanalysis_v2/decision.json      handoff/decision_v2.json

tar czf trailing_gap_handoff.tar.gz handoff/
```

The delivery must contain: executed config, run manifest, the clean-audit
result, summary tables, bootstrap sensitivity across block lengths 4/8/12, the
per-gap `R_g`/`B_g` readings, the dose-response output, and the final
classification.

---

## Quick reference — the two hard stops

| Step | Condition | Action on failure |
|---|---|---|
| 2 | Reproduced summaries disagree with reported history | Stop. Report. Do not run step 3. |
| 3.0 | `results/trailing_gap_v1` already exists at initialization | Stop. Report. Do not delete, overwrite or reuse it. (Initialization only — never re-run before step 5 or a resume.) |
| 4 | Clean-prediction hash differs from the original pilot | Stop. Report full diagnostics. Do not tune a tolerance. Do not run step 5. |
