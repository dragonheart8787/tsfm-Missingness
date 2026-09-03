# DRAFT next-round preregistration: trailing-gap mechanism experiment

> **SUPERSEDED by `docs/preregistration_trailing_gap_mechanism_v1.md`.**
>
> Kept for the record only. Its `truncated(g)` arm contained a fatal bug: it
> supplied a context of `L-g` and asked for `H` steps, so the model would
> forecast `[L-g, L-g+H)` while the pilot scored `[L, L+H)` — a `g`-step
> timestamp misalignment. The corrected `truncated_long(g)` in the
> preregistration requests `H+g` steps and drops the first `g`. **Do not build
> from this file.**
>
> **STATUS: DRAFT DESIGN ONLY. NOT IMPLEMENTED, NOT RUN.**
>
> No inference code, runner change, mask generator, or config entry exists for
> this design. It requires **new forecasts** — a new GPU run — and is
> deliberately sequenced to happen **after** the boundary jump is confirmed
> robust under the formal `BJ1_20` / `BJ2_40` treatment, not concurrently with
> it. Nothing here should be built until the Research Lead says the boundary
> jump has cleared that bar.

## What it would test

The `BJ1_20` / `BJ2_40` contrasts (audit report §9.3) measure how much error the
trailing-boundary condition adds over the nearest internal position. They do not
say **why**. At least three explanations remain live and this pilot cannot
separate them:

1. **Loss of recent observations.** The information in the final steps before
   the forecast is simply the most valuable, whatever the model.
2. **Effective horizon extension.** With the final `g` steps gone, the model is
   in effect forecasting `g + H` steps ahead from its last real observation,
   not `H`. Nothing model-specific is required.
3. **Chronos-2-specific trailing-missingness handling.** The NaN-aware instance
   normalisation, the patch/mask channel, and the time encoding all behave
   differently when the trailing patches are unobserved.

Explanations 2 and 3 are the ones that would change what a practitioner should
do, and they are exactly the pair this pilot's design cannot tell apart.

## Design sketch

Patch-aligned gap lengths **g ∈ {16, 32, 64, 128}**, and per origin three
conditions at each `g`:

| Condition | Construction | Isolates |
|---|---|---|
| `trailing_nan(g)` | Context length L; the final `g` steps set to NaN. | The full effect: recency loss + horizon extension + model handling |
| `truncated(g)` | Context of length `L - g` ending `g` steps earlier; **no NaN anywhere**. | Recency loss + horizon extension, with NO trailing-missingness handling involved |
| `internal_block(g)` | Context length L; a block of `g` NaNs placed internally, away from the boundary. | Information loss of the same size and patch alignment, with the boundary intact |

The contrast that carries the weight is **`trailing_nan(g) − truncated(g)`**.
Both arms lose the same `g` most-recent observations and both face the same
extended effective horizon; they differ only in whether the model is *shown* a
trailing run of NaNs or simply given a shorter context. A non-zero difference
there is attributable to Chronos-2's trailing-missingness handling in a way the
present design cannot achieve. `internal_block(g)` supplies the same-sized
information loss at a position where the boundary is untouched.

## What would have to be specified before running

Preregistered, in the config, before any forecast:

- context length, horizon and stride (matching the current pilot unless there is
  a stated reason to change them, so results remain comparable);
- exact origin count, asserted at runtime as the current runner already does;
- the contrast set and its Holm family, fixed in advance;
- the same paired-by-origin discipline and moving-block bootstrap
  (lengths 4/8/12, 5,000 replicates, joint resampling);
- a decision rule written as a deterministic function before seeing results, as
  `stats/decision.py` is;
- a pre-specified confounder family, fixed before inspection.

Care needed on two points the current pilot already learned the hard way:

- **`truncated(g)` changes the context length**, so Chronos-2's instance
  normalisation is computed over a different number of observations than in
  `trailing_nan(g)`. That is a real difference between the arms and must be
  recorded as a diagnostic, not assumed away.
- **Patch alignment**: `g` must be a whole number of patches on the model's
  verified patch grid, and `L - g` must remain a whole number of patches, or
  Chronos-2 will left-pad and shift every position. The existing contract check
  in `model/chronos2_runner.py` already enforces the grid; it would need
  extending to the truncated arm.

## Cost

Three conditions × four gap lengths = 12 forecasts per origin, plus a clean
control. At the current 178 origins that is roughly 2,300 forecasts — the same
order as the completed pilot's 3,026, so a comparable GPU run.

## Sequencing

Do not start this until the Research Lead confirms the boundary jump is stable
under `BJ1_20` / `BJ2_40`. If the jump is not robust there, this experiment is
measuring the mechanism of something that may not need explaining.
