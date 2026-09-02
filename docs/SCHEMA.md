# Output schemas

All outputs are CSV with a header row. CSV was chosen over Parquet so that the
Research Lead can audit the raw files with `head`, `awk` and a spreadsheet
without a pyarrow dependency; `pyarrow` is nonetheless in the lock file if a
Parquet conversion is wanted later.

Everything below lives in `results/<run-dir>/`. `results/` is gitignored; only
summaries are committed.

---

## `window_results.csv` — one row per origin x condition (3,026 rows)

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | Run identifier, `<config name>-<UTC timestamp>` |
| `dataset_sha256` | str | sha256 of the ETTh1 CSV actually loaded |
| `model_revision` | str | Pinned Hugging Face commit hash of `amazon/chronos-2` |
| `model_id` | str | `amazon/chronos-2` |
| `origin_id` | int | 0..177, chronological |
| `origin_timestamp` | str | Timestamp of the FIRST forecast step |
| `context_start` | int | Inclusive index of the context start in the source series |
| `context_end_exclusive` | int | Exclusive context end == `target_start` (the forecast boundary) |
| `target_start` | int | Inclusive index of the scored target start |
| `target_end_exclusive` | int | Exclusive index of the scored target end |
| `condition_id` | str | `clean`, `random_r{20,40}_s{42,123,2026}`, `block_r{20,40}_d{...}` |
| `pattern` | str | `clean` / `point_random` / `contiguous_block` |
| `rate` | float | Nominal missing rate (0.0, 0.20, 0.40) |
| `seed` | int/null | Point-random base seed; null otherwise |
| `block_start` | int/null | Inclusive block start within the context |
| `block_end_exclusive` | int/null | Exclusive block end within the context |
| `distance_to_boundary` | int/null | `d = context_length - block_end_exclusive` |
| `missing_count` | int | Exact number of removed observations (0, 64 or 128) |
| `mask_descriptor` | str | Compact human-readable mask description |
| `mask_indices_sha256` | str | sha256 of the int64 mask index array — full mask reproducibility witness |
| `target_sha256` | str | sha256 of the scored target bytes — the integrity witness. **Must be identical across all 17 conditions of an origin.** |
| `mae`, `mse`, `rmse` | float | Forecast errors; NaN if the forecast failed |
| `mase` | float/null | Secondary only; excluded from the decision rule |
| `mase_denominator` | float/null | Seasonal-naive denominator from the clean context only |
| `clean_mae`, `clean_mse`, `clean_rmse` | float | The origin's paired clean metrics |
| `paired_diff_mae_from_clean` | float | `mae - clean_mae` |
| `abs_paired_diff_mae_from_clean` | float | `abs(mae - clean_mae)` |
| `status` | str | `ok` or `failed` |
| `error_message` | str | Exception type and message when `status == "failed"`; empty otherwise |
| `runtime_seconds` | float | Wall-clock time of that single forecast call |

Failures are **recorded**, never skipped: a failed forecast still produces its
row, with NaN metrics and a populated `error_message`.

---

## `predictions_long.csv` — one row per origin x condition x forecast step

3,026 x 96 = 290,496 rows for a complete run.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | Run identifier |
| `origin_id` | int | 0..177 |
| `condition_id` | str | As above |
| `seed` | int/null | Point-random base seed |
| `step_index` | int | 0..95 |
| `forecast_timestamp` | str | Timestamp of that forecast step |
| `ground_truth` | float | Actual value (identical across conditions for an origin) |
| `median_prediction` | float | Chronos-2's explicit q=0.5 forecast |

---

## `mask_diagnostics.csv` — one row per origin x condition

Carries the identifying columns (`run_id`, `origin_id`, `condition_id`,
`pattern`, `rate`, `seed`, `distance_to_boundary`, `block_start`,
`block_end_exclusive`) plus every confounder diagnostic:

`missing_count`, `missing_rate_realised`, `n_missing_runs`,
`longest_missing_run`, `trailing_run_length`, `nearest_missing_to_boundary`,
`mean_missing_distance_to_boundary`, `missing_frac_last_24`,
`missing_frac_last_48`, `missing_frac_last_96`, `n_patches`,
`n_patches_fully_missing`, `n_patches_partially_missing`,
`n_patches_any_missing`, `clean_context_mean`, `clean_context_std`,
`retained_context_mean`, `retained_context_std`, `retained_mean_change`,
`retained_std_change`, `retained_mean_change_frac_of_clean_std`,
`retained_std_ratio`, `removed_mean`, `removed_variance`, `removed_min`,
`removed_max`, `removed_slope`, `removed_abs_diff_mean`,
`removed_hour_entropy_normalised`, `removed_hour_max_share`.

These are computed at mask-generation time for every mask, not post-hoc. **No
severity metric is fitted from them in this pilot.**

---

## Analysis outputs

| File | Contents |
|---|---|
| `mae_matrix.csv` | 178 x 17 origins-by-conditions MAE matrix — the object the bootstrap resamples |
| `condition_summary.csv` | Per-condition aggregate MAE, pooled RMSE, RED vs. clean |
| `contrast_results.csv` | The four contrasts: raw difference, % of mean clean MAE, Holm 95% CI, adjusted p, uncorrected 90% CI, median paired difference, fraction of origins positive, per-block-length estimates and intervals, confounder rank correlations |
| `paired_differences.csv` | Per-origin paired difference for each contrast |
| `bootstrap_results.csv` | Every bootstrap result at every block length (4, 8, 12) |
| `analysis.json` | Everything above plus the decision payload, `decision_rule_version`, and `distance_relative_check_status` |
| `decision.json` | The GO / PIVOT / NO-GO label with its full triggering criteria and `decision_rule_version` |
| `run_manifest.json` | Config snapshot, dataset validation, environment, model contract |
| `checkpoint.json` | Resume state: the list of completed origins |
| `run_summary.json` | Completion counts and the full-run assertions |


---

## Decision rule versioning

Every decision output carries `decision_rule_version`:

| Version | Behaviour |
|---|---|
| `v1` | Coerced an undefined `rho_distance` (NaN) to `0.0` before comparing confounder correlations against it. On the preregistered two-arm contrasts distance is constant, so this manufactured a comparison baseline that does not exist. |
| `v2` | Distance-relative comparisons report an explicit status and are `not_evaluable` when distance is constant within the contrast. |

### `distance_relative_check_status`

Three PIVOT checks compare a confounder's rank correlation against distance's.
Each publishes a per-contrast status under this key in `analysis.json` (and
inside `decision.criteria.pivot`), alongside — not instead of — the raw
`rho_distance` / `rho_confounders` / `rho_scaling` dumps:

```json
"confounder_out_tracks_distance_status": {
  "C1_loc_20": {
    "status": "not_evaluable",
    "reason": "distance_constant_within_contrast",
    "n_unique_finite_distance": 1,
    "rho_distance": NaN,
    "compared": [],
    "exceeding": []
  }
}
```

| Status | Meaning | Can contribute a trigger? |
|---|---|---|
| `not_evaluable` | Fewer than 2 unique finite distance values in the contrast, so distance's correlation is undefined. **The question could not be asked — not the same as "no effect".** | No |
| `invalid_input` | Distance genuinely varies but `rho_distance` is non-finite. Something upstream is wrong. | No |
| `false` | Evaluated; no confounder exceeded distance by the margin. | No |
| `true` | Evaluated; at least one confounder did. | Yes |

### Not overwriting an earlier run's record

`stats.analyze.analyse()` reads from `run_dir` and writes to `out_dir`
(defaulting to `run_dir`). Writing over an existing `decision.json` produced
under a *different* rule version raises `FileExistsError` unless
`allow_overwrite=True`. For a post-hoc re-analysis, point `--out-dir` at a
separate directory so the original run's preregistered outputs stay intact.
