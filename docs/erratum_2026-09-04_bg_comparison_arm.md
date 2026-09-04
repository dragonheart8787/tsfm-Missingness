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

**All four B rows change on the real ETTh1 result.** `B_16`, `B_32`, `B_64` and
`B_128` all read `MATERIAL_POSITIVE`, whose text is arm-specific — so **8
interpretation cells** are rewritten: four in `B_contrasts.csv` and four again
in `trailing_gap_analysis.json`.

> **Correction to this erratum's first draft.** It stated "6, not 8", claiming
> `B_128` read `UNRESOLVED` and so did not change. That was wrong. It was a
> property of a *synthetic test fixture* whose B rows had been given R's reading
> distribution — which contains an `UNRESOLVED` at `g=128` — and it was
> incorrectly generalised as if it described the real data. On the real ETTh1
> output every B reading is `MATERIAL_POSITIVE`. The count is **8**.
>
> The test suite now keeps `R_READINGS` and `B_READINGS` as separate fixtures,
> and the row builder looks a family's distribution up by family rather than
> accepting one as a parameter, so R's distribution can no longer stand in for
> B's. That substitution was the entire cause of the error.

## 4. How "no number changed" is proved

`compare_reporting_versions()` validates the **complete permitted
transformation**, not merely that the originals survived. Preservation alone is
necessary but not sufficient: a v2 could keep every number and still assert the
wrong comparison arm, carry the wrong interpretation, plant a rogue field, or
quietly edit a copied-through file. All three of those classes previously
slipped through. The check now rejects:

| Rejected | Why it matters |
|---|---|
| B interpretation ≠ `INTERPRETATION_B[reading]` | Right dictionary, wrong key is still wrong |
| A rewritten **R** interpretation | Only B may change |
| A wrong `comparison_arm`, either family | Every number preserved, arm still misstated |
| A wrong `reporting_version` / `erratum_id` | Provenance markers must be exact |
| Any CSV column not on the whitelist | Editorial commentary cannot be smuggled in |
| Any JSON key not on the whitelist, at root or in a row | A planted numeric field has nothing to compare against |
| A missing, extra, or modified pass-through file | Byte-identity required on anything meant to pass through |
| A pre-existing, non-empty output directory | Stale files from an earlier build would survive |

The whitelist is exactly: the contrast-row `comparison_arm`,
`reporting_version` and `erratum_id` columns, and the root `reporting_version`,
`erratum_id` and `erratum_note` keys. CSV cells are read with `dtype=str`, so
non-description cells are compared byte-for-byte and cannot be perturbed by a
float round-trip.

Two categories are reported separately, because they are not the same thing: an
**interpretation change** rewrites a pre-existing value (this is the erratum),
whereas a **whitelisted addition** adds a new field and alters nothing.

```bash
python scripts/build_reporting_v2.py \
  --analysis-dir results/trailing_gap_v1/analysis
```

Exits non-zero if anything moved outside the whitelist, and writes
`identity_check.json` recording what it compared.

**Expected output on the real ETTh1 analysis:**

```
values compared         : 536
numeric fields compared : 363
pass-through files      : 4   (byte-identical)
interpretation changes  : 8   (B_16, B_32, B_64, B_128 — in the CSV and the payload)
whitelisted additions   : 8   (comparison_arm on all 8 contrast rows)
classification          : INCONCLUSIVE / mixed_equivalent_and_unresolved (unchanged)
```

### 4.1 Provenance of those figures — read before citing them

`results/trailing_gap_v1/` is uncommitted and **was not present in the
environment this erratum was authored in**. The block above was therefore
produced against a **structural replica**, not against the real ETTh1 artifact:
`experiments/run_trailing_gap.py --mock-model` (178 origins, 2,314 cells) followed
by the real `analyse()`, with the real frozen reading distribution — `R`:
`EQUIVALENT`, `EQUIVALENT`, `EQUIVALENT`, `UNRESOLVED`; `B`: `MATERIAL_POSITIVE`
×4 — and the frozen classification forced onto the result. The field structure
and the readings are real; the numeric values are mock.

The counts are structural, not numeric — they depend on the schema and the
readings, both of which the replica reproduces exactly — so the match is strong
evidence. It is **not** a run against the real artifact, and this document does
not claim it is. **Re-run the command above on the host holding
`results/trailing_gap_v1/analysis/` and confirm 536 / 363 / 8 / 8 there.** A
different count is a real finding about the artifact, not a number to reconcile
against this page.

`tests/test_reporting_v2.py` (44 tests) covers this. Adversarial cases perturb,
one at a time: the raw difference, both Holm CI bounds, the Holm-adjusted
p-value, both 90% CI bounds, the SESOI, the fraction of positive origins, the
median paired difference, a block-length estimate, a per-gap reading, the
dose-response slope, the classification label, R's interpretation, a wrong
`comparison_arm` on either family, a wrong `INTERPRETATION_B` key, a wrong
`reporting_version` or `erratum_id`, an unwhitelisted CSV column, an
unwhitelisted root JSON key, a rogue numeric field planted at root and inside a
row, a modified pass-through file, a missing pass-through file, an unexpected
extra file, a dropped row, and a dropped column. **Each must fail the identity
check**, so a passing check is evidence rather than decoration.

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
