# HiMuon Benchmark Suite

Microbenchmarks for evaluating HiMuon's tiled Newton-Schulz preconditioning.

Experiments live in `experiments/` (one file each, prefixed `exp_`). Shared infrastructure
(`bench_utils.py`, `plot_style.py`, `assoc_mem.py`, `publication.mplstyle`) and the
orchestrators (`run.py`, `plot.py`) stay at the top level. Each experiment module exposes
`run(...)` and `plot(data, out_dir=None)`; results go to `data/*.json`, figures to `plots/`.

## Available Tests

### Run via `run.py`

| Test | Module | Description |
|---|---|---|
| `shapes` | `exp_real_shapes_bench.py` | Per-layer speedup on real Qwen3 weight shapes |
| `ns_convergence` | `exp_ns_convergence.py` | NS iteration-count K convergence sweep (K × tile size) |

### Standalone scripts (own CLI, `--plot-only` supported)

| Module | Description |
|---|---|
| `exp_ns_kernel_microbench.py` | NS kernel comparison at T=128: eager vs torch.compile vs 3-kernel vs ns5_smem |
| `exp_optimizer_step_microbench.py` | Optimizer step comparison: Muon vs HiMuon vs HiMuon+x-layer vs HiMuon+x-layer+graph |
| `exp_ns5_rect_bench.py` | Fused `ns5_smem` rectangular-tile + grid-strategy sweep across (M, N) shapes |
| `exp_tile_shape_quality.py` | HiMuon tile-shape training-quality sweep (trains llama_220m, 300 steps × multiple tile shapes) |
| `exp_precision_descent.py` | Alignment γ and descent-potential φ of low-precision / tiled NS on a teacher-student least-squares proxy, vs training step |
| `exp_capacity_scaling.py` | One-step associative-memory capacity vs `d`/`alpha`/tile (N=1e5, recovered items) |
| `exp_topk_recovery.py` | One-step top-k recovery on the unique sampled support (rate + gap-to-Muon) |
| `exp_precision_recovery.py` | Low-precision NS recovery (fp64/fp32/bf16/native-fp8); emits a LaTeX table |

The last three are stateless, operator-only associative-memory probes built on the shared
`assoc_mem.py` engine (no momentum / optimizer state); they need a GPU and, for native FP8,
an sm89+ card (L40S / H100).

## Quick Start

```bash
# Run a single test (GPU required)
uv run python microbench/run.py --test shapes

# Run multiple / all tests
uv run python microbench/run.py --test shapes ns_convergence
uv run python microbench/run.py --test all

# Standalone scripts with their own CLI
uv run python microbench/experiments/exp_optimizer_step_microbench.py --batches 32 128 512
uv run python microbench/experiments/exp_capacity_scaling.py
uv run python microbench/experiments/exp_topk_recovery.py
uv run python microbench/experiments/exp_precision_recovery.py
```

## Re-plotting Without Re-running

Experiments save results to `data/*.json`. To regenerate figures:

```bash
uv run python microbench/plot.py --test shapes
uv run python microbench/experiments/exp_capacity_scaling.py --plot-only
uv run python microbench/experiments/exp_precision_recovery.py --plot-only
```

## File Structure

```
microbench/
├── run.py                      # Unified experiment runner (registered tests)
├── plot.py                     # Unified plot generator (reads data/)
├── bench_utils.py              # Shared: GPU timing, memory, JSON I/O
├── plot_style.py               # Shared: matplotlib style, colors, helpers
├── assoc_mem.py                # Shared: associative-memory operator engine (NS / tiling / recovery)
├── publication.mplstyle        # matplotlib style sheet
├── experiments/                # one file per experiment, prefixed exp_
│   ├── exp_real_shapes_bench.py
│   ├── exp_ns_convergence.py
│   ├── exp_ns_kernel_microbench.py
│   ├── exp_optimizer_step_microbench.py
│   ├── exp_ns5_rect_bench.py
│   ├── exp_tile_shape_quality.py
│   ├── exp_precision_descent.py     # low-precision / tiled NS alignment + descent potential
│   ├── exp_capacity_scaling.py      # associative-memory capacity scaling
│   ├── exp_topk_recovery.py         # support-restricted top-k recovery
│   └── exp_precision_recovery.py    # low-precision NS recovery (LaTeX table only)
├── data/                       # JSON results (auto-created)
└── plots/                      # Generated figures (+ precision_recovery_table.tex)
```
