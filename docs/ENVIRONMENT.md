# Environment and reproducibility

## Dependency manager: `uv`

`uv` was chosen over Poetry for three reasons specific to this pilot:

1. **Split build/run environments.** The pilot is built and tested on a CPU-only
   host and executed later on a GPU host. `uv pip install` resolves against the
   same `pyproject.toml` on both, and `uv.lock` / `requirements.lock.txt` pins
   the exact resolution, so the GPU host does not silently get different wheels.
2. **Torch wheel handling.** `uv` understands PyTorch's index layout and can be
   pointed at a CUDA or CPU wheel index without restructuring the project.
3. **Speed.** Full resolution and install of the torch/transformers stack takes
   a couple of minutes, which matters when the operator is provisioning a
   short-lived GPU box.

Poetry would have worked equally well scientifically; the choice is operational,
not methodological.

## Creating the environment

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

On a CUDA host, install the matching torch build first if the default PyPI wheel
is not the one you want:

```bash
uv pip install --python .venv/bin/python torch --torch-backend cu124
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Exact resolution used to build and test this pilot

The full, exact resolution is committed as `requirements.lock.txt`
(`uv pip freeze` output). Key versions:

| Component | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.13.0 |
| transformers | 5.16.1 |
| chronos-forecasting | 2.3.1 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scipy | 1.17.1 |
| CUDA | none on the build host (CPU-only) |
| GPU | none on the build host |

The GPU host's actual versions, CUDA build and GPU model are captured at run
time into `results/<run>/run_manifest.json` under `environment`, alongside the
git commit and the resolved model revision. That manifest, not this document, is
the provenance record for a given run.

## Model revision pinning

`configs/pilot_config.yaml` ships with `model.revision: null`, and the inference
wrapper **refuses to run** in that state. Before the pilot may execute:

```bash
python scripts/verify_model_contract.py --write-revision
```

This resolves `amazon/chronos-2`'s current HEAD commit hash, writes it into the
config, loads that exact revision, and verifies the preregistered patch grid.
The pinned hash is then recorded in every result row (`model_revision`) and in
the run manifest.

## Determinism

* Mask generation is seeded by a fixed BLAKE2b digest over
  `"{namespace}|{base_seed}|{origin_id}|{rate_ppm}"`. Python's builtin `hash()`
  is never used (it is salted per process for `str`), and
  `tests/test_masks.py::test_6c_derived_seeds_are_reproducible_across_processes`
  recomputes a digest in a fresh interpreter to prove it.
* The bootstrap uses `numpy.random.default_rng` with a seed derived from the
  config's `statistics.bootstrap.seed` plus the block length and contrast index,
  so every interval is reproducible.
* Inference is `eval()`-mode, `inference_mode`, float32, batch size 1. GPU
  floating-point non-determinism may still perturb the low-order bits of a
  forecast between machines; the pilot does not depend on bitwise
  reproducibility of forecasts across hardware, only within a run.
