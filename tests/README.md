# Tests

The suite is a **contract / API-doc** layer, not a numerical-correctness oracle.
Its job is to pin the public API and framework conventions so that changing or
adding code that breaks an existing contract fails fast, and to act as
executable documentation of how HiMuon and the training framework are meant to
be used. Tests target HiMuon + the framework only — baseline optimizers (Muon,
SOAP, AdamW) are not covered.

## Tiers

| Dir | Tier | Needs | Runs in CI |
|-----|------|-------|------------|
| `unit/` | Pure-CPU contracts / API-doc | CPU only | ✅ yes |
| `integration/` | GPU behavior (kernels, `step()`, eager↔graph, FSDP) | CUDA GPU + Triton | ❌ no (manual) |

`unit/` never calls `Optimizer.step()` (which requires Triton/CUDA). It
introspects signatures, param-group schemas, the optimizer factory's grouping
convention, and the dataset/model/util contracts — all on synthetic data, no
mocks, no network, no HF downloads.

`integration/` is auto-marked `gpu` (see `integration/conftest.py`) and is
skipped/ignored entirely when no CUDA device is present.

## Running

```bash
# CPU contract tier — exactly what CI runs:
uv run pytest tests/unit

# Equivalent marker selector (works from the repo root on a CPU box too —
# the integration dir is auto-ignored without a GPU):
uv run pytest -m "not gpu and not slow and not distributed"

# GPU tier on a compute node:
uv run pytest tests/integration -m "gpu and not slow and not distributed"

# GPU + distributed/slow (needs 2–4 GPUs, real HF model download):
uv run pytest tests/integration -m "gpu"
```

## Markers

- `gpu` — requires a CUDA GPU + Triton (auto-applied to everything in `integration/`).
- `distributed` — requires a `torchrun` multi-process launch (FSDP tests).
- `slow` — long-horizon / full-pipeline / real-model tests.

## Layout

```
tests/
├── conftest.py                       # CUDA cleanup (no-op on CPU), dynamo cache limit
├── unit/                             # CPU contract tier (CI)
│   ├── conftest.py                   # synthetic model / fake tokenizer / text stream
│   ├── test_optimizer_api.py         # HiMuon public API: signature, group schema, tile norm, reconfigure, state_dict, train-script call surface
│   ├── test_optimizer_factory.py     # get_optimizer grouping convention + kwarg filtering
│   ├── test_fsdp_bank.py             # parameter-bank scheme: wrap/copy/delete, interleave, padding, unbank round-trip
│   ├── test_dataset.py               # ToyDataset schema/labels contract
│   ├── test_model.py                 # param-count / token-formula helpers
│   └── test_utils.py                 # seed/kv-args/TimeBudget real behavior
└── integration/                      # GPU tier (manual)
    ├── conftest.py                   # gpu auto-marking + model/grad fixtures
    ├── test_himuon_smoke.py          # step() runs finite across shape/dtype/tile
    ├── test_himuon_self_consistency.py  # b_hw-invariance, eager == cuda_graph
    ├── test_himuon_dispatch.py       # xlayer_plan() completeness / chunking
    ├── test_himuon_reconfigure.py    # cache/graph invalidation, math no-op
    ├── test_himuon_memory.py         # no peak-memory growth after warmup
    ├── test_kernels.py               # NS seamless-replacement + Triton kernels vs torch reference
    ├── test_himuon_fsdp_smoke.py     # Qwen3-0.6B FSDP2 end-to-end (slow)
    ├── test_himuon_fsdp_parity.py    # per-bank shard == single-process (slow)
    └── _fsdp_worker.py               # torchrun worker (not collected)
```
