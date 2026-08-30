# tsfm-Missingness

Chronos-2 missingness geometry pilot — a small, preregistered TAAI 2026 pilot
measuring whether Chronos-2's forecast error on ETTh1 depends on **where** and
**how** missing history sits in the context window.

The pilot's output is a deterministic **GO / PIVOT / NO-GO** decision about
whether a larger study is warranted. **A NO-GO is a complete, successful
outcome.**

No novelty is claimed for TSFM missingness benchmarking, for evaluating
Chronos-2 under missing history, for imputation harming forecasts, for forecast
proximity as a causal mechanism, or for any proximity-weighted severity score.

---

## Design (all constants live in `configs/pilot_config.yaml`)

* **Data:** canonical ETTh1, target column `OT`, univariate, 17,420 hourly rows.
* **Windows:** context L = 320, horizon H = 96, stride = 96 → **178 origins**.
* **Conditions per origin: 17** → **3,026 forecasts**.
  * 1 clean
  * 6 exact-count point-random (20% = 64 missing, 40% = 128 missing; seeds 42, 123, 2026)
  * 10 contiguous block (20%: length 64 at d ∈ {0, 64, 128, 192, 256}; 40%: length 128 at d ∈ {0, 48, 96, 144, 192})
* **Model:** `amazon/chronos-2`, revision-pinned, `eval()`, float32, explicit q=0.5, one origin per call, `cross_learning=False`.
* **Missingness representation:** float NaN in place — Chronos-2's supported convention. No row or timestamp is ever dropped.

`d = L - block_end_exclusive` is the distance from the block's exclusive end to
the forecast boundary, so d=0 is flush against it.

## The four preregistered contrasts

| ID | Contrast | Kind |
|---|---|---|
| `C1_loc_20` | 20%: boundary block (d=0) − far block (d=256) | within-pattern |
| `C2_loc_40` | 40%: boundary block (d=0) − far block (d=192) | within-pattern |
| `C3_geom_20` | 20%: random (seed mean within origin) − centred block (d=128) | **pipeline comparison** |
| `C4_geom_40` | 40%: random (seed mean within origin) − centred block (d=96) | **pipeline comparison** |

C3 and C4 are **not** pure causal estimates of contiguity — the arms differ in
position, patch occupancy and which observations are removed, all at once. That
label travels with them through the code, the output schema and every figure.

---

## Running it

```bash
# 1. Environment (see docs/ENVIRONMENT.md for why uv)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"

# 2. Dataset (already validated; re-runs are idempotent)
.venv/bin/python data/fetch_etth1.py --config configs/pilot_config.yaml

# 3. Tests — all pass on CPU with the model mocked at the inference boundary
.venv/bin/python -m pytest tests/ -v -m "not requires_model"

# 4. Model contract gate — REQUIRED, needs Hugging Face access.
#    Pins the exact revision and verifies the real patch grid. Exits non-zero
#    and refuses the run if the preregistered 16-step grid does not hold.
.venv/bin/python scripts/verify_model_contract.py --write-revision

# 5. Smoke test (3 origins = 51 forecasts) — confirms per-forecast timing
.venv/bin/python runner/run_pilot.py --run-dir results/smoke --limit-origins 3

# 6. The pilot
.venv/bin/python runner/run_pilot.py --run-dir results/pilot_v1

# 7. Analysis + decision, then figures
.venv/bin/python scripts/analyze_run.py --run-dir results/pilot_v1
.venv/bin/python scripts/make_figures.py --run-dir results/pilot_v1
```

### Unattended execution

```bash
tmux new -d -s pilot '.venv/bin/python runner/run_pilot.py --run-dir results/pilot_v1 \
  >> results/pilot_v1.stdout.log 2>&1'
```

Progress from a fresh login:

```bash
tail -f results/pilot_v1/run.log
python -c "import json;print(json.load(open('results/pilot_v1/checkpoint.json'))['n_completed'],'/178')"
```

Resume after an interruption — the **same** command, which is what Test 13
validates. It skips completed origins and never duplicates or overwrites them:

```bash
.venv/bin/python runner/run_pilot.py --run-dir results/pilot_v1
```

`--mock-model` runs the whole pipeline with a deterministic stand-in and no
weights. Its output is a pipeline exercise, **not** a Chronos-2 result.

---

## Layout

```
configs/pilot_config.yaml   every numeric constant; nothing hardcoded elsewhere
data/fetch_etth1.py         acquisition + validation (checksum, gaps, native NaNs)
masks/                      rng.py, base.py, point_random.py, contiguous_block.py, plan.py
model/chronos2_runner.py    revision pin, contract verification, q=0.5, isolation
metrics/                    pointwise.py (MAE/MSE/RMSE/MASE), aggregate.py (RED)
stats/                      bootstrap.py, contrasts.py, decision.py, analyze.py
diagnostics/confounders.py  per-mask confounder table
runner/                     windows.py (read-only targets), run_pilot.py (resumable)
scripts/                    verify_model_contract.py, analyze_run.py, make_figures.py
tests/                      93 CPU tests + 3 marked requires_model
report/audit_report.md      interpretation-discipline report
docs/                       ENVIRONMENT.md, SCHEMA.md
```

`results/` is gitignored; only summaries are committed.

## Documentation

* [`report/audit_report.md`](report/audit_report.md) — the decision, the live alternative explanations, and what this design can and cannot separate.
* [`docs/SCHEMA.md`](docs/SCHEMA.md) — output file schemas.
* [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) — environment, pinning, determinism.

## Scope exclusions

Deliberately not implemented, pending Research Lead audit: 10% missingness;
periodic missingness; linear interpolation; forward-fill; ARIMA; XGBoost;
DLinear; TimesFM; MOMENT; TS-ICL; additional datasets; hyperparameter searches;
any fitted forecast-proximity weighting parameter.
