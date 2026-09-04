# Erratum — 2026-09-04 — `B_g` comparison arm mislabelled in reported prose

**ID:** `ERRATUM-2026-09-04-bg-comparison-arm`
**Affects:** reported prose only — `B_contrasts.csv`'s `interpretation` column and the
`b_contrasts[].interpretation` entries of `trailing_gap_analysis.json`
**Execution commit:** `e5f1183b54591856285ebc0fe7b0e2ac49ed6fcb` (F3)
**Corrected artifact:** `analysis_reporting_v2/` (the original `analysis/` is untouched)

> **Zero effect on any reported number, interval, p-value, per-gap reading, the
> SESOI, the dose-response result, or the classification.** The frozen ETTh1
> result — `INCONCLUSIVE / mixed_equivalent_and_unresolved` — is unchanged, and
> is neither relabelled nor reinterpreted by this erratum.

---

## 1. What was wrong

`experiments/analyze_trailing_gap.py` builds its output rows through a single
helper and writes `INTERPRETATION[reading]` into the `interpretation` column for
**both** families:

```python
r_rows, b_rows = rows(r_stats, r_blocks, "R"), rows(b_stats, b_blocks, "B")
```

`stats.mechanism_decision.INTERPRETATION` was authored for the **R** family, and
two of its four texts name or imply R's comparison arm:

| Reading | Original text (R-specific fragment) |
|---|---|
| `EQUIVALENT` | "…consistent with a **horizon-dominant** explanation…" |
| `MATERIAL_POSITIVE` | "…higher mean error than **shorter-context/longer-horizon single-shot forecasting**…" |

That arm is `truncated_long(g)`. It is **R_g's** comparison, not **B_g's**. So a
`B_g` row was described as though it had been compared against the truncated
arm.

The other two texts (`MATERIAL_NEGATIVE`, `UNRESOLVED`) name no arm and were not
wrong.

## 2. What was always correct

**The computation.** `B_g` has always been computed against `internal_block`:

```python
# experiments/analyze_trailing_gap.py
r_series = {f"R_{g}": (matrix[f"trailing_nan_g{g}"] - matrix[f"truncated_long_g{g}"]) ...}
b_series = {f"B_{g}": (matrix[f"trailing_nan_g{g}"] - matrix[f"internal_block_g{g}"]) ...}
```

giving the definitions frozen in the preregistration (§5.1, §5.2):

```
R_g = MAE(trailing_nan(g)) − MAE(truncated_long(g))
B_g = MAE(trailing_nan(g)) − MAE(internal_block(g, d=16))
```

**Every other description was also correct.** `configs/trailing_gap_config.yaml`
labels all four `B_g` contrasts against `internal_block`; preregistration §5.2
states the definition correctly; `stats.mechanism_decision`'s `b_note` names no
arm and carries the confounding caveat accurately. A repository-wide search
found **no line** that mentions a `B_g` identifier and truncation together — the
fault was only reachable through the shared `INTERPRETATION` dict at output
time, which is why it survived earlier reviews.

## 3. What changed

A new, additive reporting layer, `experiments/reporting_v2.py`, emits
`analysis_reporting_v2/` beside the original `analysis/`:

* **B rows** get family-correct text naming `internal_block(g, d=16)`, carrying
  the project's existing "CONFOUNDED WITH DIFFERENCES IN REMOVED CONTENT"
  caveat forward verbatim rather than re-deriving it.
* **Every row, R and B**, gains an explicit `comparison_arm` column, so the arm
  can never again be left implicit.
* `reporting_version` and `erratum_id` markers are added.

**Nothing that produces a number was modified.** `analyze_trailing_gap.py` and
`stats/mechanism_decision.py` are untouched, so the numeric behaviour of F3 is
unchanged by construction, not merely by inspection. The original `analysis/`
directory is never written to.

Three of four B rows change on the frozen ETTh1 result: `B_16`, `B_32`, `B_64`
read `EQUIVALENT`, whose text was arm-specific. `B_128` reads `UNRESOLVED`,
whose text names no arm and is byte-identical in both families — so it correctly
does not change.

## 4. How "no number changed" is proved

`compare_reporting_versions()` walks every value in both directories — the
contrast tables, the analysis payload, `classification.json` and
`dose_response.json` — and reports any difference outside B's `interpretation`
as a violation. CSV cells are read with `dtype=str`, so non-description cells
are compared byte-for-byte and cannot be perturbed by a float round-trip.

```bash
python scripts/build_reporting_v2.py \
  --analysis-dir results/trailing_gap_v1/analysis
```

Exits non-zero if anything numeric moved, and writes `identity_check.json`
recording what it compared.

`tests/test_reporting_v2.py` (28 tests) covers this, including adversarial
cases that perturb — one at a time — the raw difference, both Holm CI bounds,
the Holm-adjusted p-value, both 90% CI bounds, the SESOI, the fraction of
positive origins, the median paired difference, a block-length estimate, a
per-gap reading, the dose-response slope, the classification label, and R's
interpretation, plus a dropped row and a dropped column. **Each must fail the
identity check**, so a passing check is evidence rather than decoration.

## 5. Scope

* No number, interval, p-value, reading, SESOI, dose-response result or
  classification changed.
* The frozen ETTh1 result is unaltered, unrelabelled and uninterpreted.
* The original `analysis/` outputs are retained unmodified; the correction is a
  versioned parallel artifact, following this project's
  `decision_rule_version` / `post_hoc_reanalysis_v2` convention.
* No inference was run and no model was called to produce this erratum.

## 6. Reviewer note

The mislabelled text lives in generated output, not in source prose. Reviewing
`configs/`, the preregistration, or the code comments would not have surfaced
it — only reading a produced `B_contrasts.csv` row would. The added
`comparison_arm` column exists so that this class of error is visible on the
face of every future output row.
