# Results-delivery policy

**Adopted repo-wide.** Generalised from §10.1 of
`docs/preregistration_trailing_gap_mechanism_v1.md`, which was written for one
experiment but describes a failure this project has already hit repeatedly.

## The problem it fixes

`results/` is gitignored. Raw per-forecast outputs are large and do not belong
in version control — that part is right. But the consequence, as actually
experienced:

| Round | Built and tested | Could report numbers? |
|---|---|---|
| Decision-rule v2 correction | yes | **no** — `results/pilot_v1/` never left the execution host |
| Internal-only analysis | yes | **no** — same |
| BJ1/BJ2 boundary-jump contrasts | yes | **no** — same |

Three consecutive rounds shipped correct, tested analysis code that **could not
be turned into a result**, because the inputs existed only on one machine. The
reviewer cannot recompute values they cannot reach, and neither can any later
session. Every one of those rounds ended with a "run this command and send me
the output" handoff that has not yet closed.

## The policy

After any execution round, the following must **reach the reviewer** — committed
to the repository, or delivered by another durable route:

1. **The executed config**, with the model revision as actually resolved (not
   `null`, not a tag).
2. **The run manifest** — environment, dependency versions, CUDA/GPU, git
   commit, dataset checksum, model contract.
3. **Summary statistics** — condition summaries, contrast tables, bootstrap
   intervals, and any decision or classification output, with its rule version.
4. **Any precondition audit** the round required (for example the `clean`
   re-run checksum audit in the trailing-gap preregistration §10).

**Raw per-forecast files may stay uncommitted. Summaries may not.**

## Why summaries are safe to commit

They are small — kilobytes against the ~40 MB of `predictions_long.csv` — and
they are the artefacts that carry scientific content. `.gitignore` already
distinguishes the two cases; the intent was always "commit only summaries", and
this policy makes that a requirement rather than an option.

## Practical note

`results/` is ignored wholesale, so a summary has to be added deliberately:

```bash
git add -f results/pilot_v1/condition_summary.csv \
            results/pilot_v1/contrast_results.csv \
            results/pilot_v1/decision.json \
            results/pilot_v1/run_manifest.json
```

Or copy them somewhere tracked. Either is fine; leaving them only on the
execution host is not.
