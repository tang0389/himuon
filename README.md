<h1 align="center">HiMuon</h1>

<p align="center">
  <strong>A tile-local Newton-Schulz orthogonalized optimizer for LLM pre-training.</strong> Full training stack (single-GPU / DDP / FSDP with parameter-bank sharding) across Llama and Qwen model families, with baseline optimizers (AdamW, Muon, SOAP) for comparison.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"/></a>
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.6"/>
  <img src="https://img.shields.io/badge/Triton-required-orange" alt="Triton required"/>
</p>

---

## ✨ Highlights

- **HiMuon optimizer** — tile-local Newton-Schulz orthogonalization; ships with a Triton super-kernel (SRAM-resident, FP32-accumulating), cross-layer batched GEMM, and full-step CUDA graph capture.
- **FSDP with parameter-bank sharding** — scales HiMuon and Muon to large models with zero cross-rank collectives inside the optimizer.
- **Mid-training reconfigure** — swap HiMuon's tile size on the fly, e.g., full-matrix warmup → tile-local body, without restarting; the captured CUDA graph auto re-captures after the change.
- **Three training entry points, one CLI** — `train.py` / `train_ddp.py` / `train_fsdp.py` share the same flags, so switching between single-GPU, DDP, and FSDP is a one-line change.
- **Built-in optimizer zoo** — AdamW, SOAP, Muon, HiMuon all reachable via `--optimizer <name>` through a single factory; enables clean head-to-head comparisons.

<p align="center">
  <img src="assets/tile_local_ns.svg" alt="Full-matrix Newton-Schulz (Muon) vs tile-local NS (HiMuon)" width="100%"/>
</p>

---

## 📦 Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                # core dependencies
uv sync --extra dev    # + pytest, matplotlib
uv sync --all-extras   # everything
```

Triton (bundled with PyTorch) is required for the HiMuon kernels. GPU training assumes CUDA 12.x with bf16 support (A40 / L40S / A100 / H100).

---

## 🚀 Quick Start

### Single GPU

```bash
uv run train.py \
    --model Qwen3-0.6B \
    --bs 1 \
    --grad-acc 16 \
    --optimizer adamw \
    --lr 1e-3 \
    --wd 0.1 \
    --save-checkpoint
```

### DDP (multi-GPU, single node)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun \
    --standalone --nnodes=1 --nproc_per_node=2 \
    train_ddp.py
```

### FSDP (multi-GPU, sharded)

Parameter-bank sharding is required; see [Parameter-Bank Sharding](#parameter-bank-sharding).

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun \
    --standalone --nnodes=1 --nproc_per_node=4 \
    train_fsdp.py \
    --model Qwen3-4B \
    --bs 2 \
    --grad-acc 8 \
    --max-length 1024 \
    --optimizer himuon \
    --lr 1e-2
```

---

## ⚡ Advanced Features

### Reconfigure schedule

Change HiMuon's `tile_size` mid-training via `--reconfigure-schedule`, a JSON list of `{"step": N, ...}` dicts. Works in single-GPU, DDP, and FSDP alike — it's an optimizer-level feature, not tied to any distributed mode. A common pattern is to start with `tile_size` larger than any matrix (each matrix becomes a single tile, mathematically equivalent to Muon but retaining HiMuon's cross-layer batching) and switch to tile-local HiMuon after the warmup:

```bash
uv run train.py \
    --model Qwen3-0.6B \
    --optimizer himuon \
    --lr 1e-2 \
    --reconfigure-schedule '[{"step":0,"tile_size":1000000000},{"step":200,"tile_size":512}]'
```

### CUDA graph capture

HiMuon captures the full optimizer step (momentum → tile → NS → scatter → untile) into a CUDA graph after a short warmup, replacing hundreds of per-parameter kernel launches with one `cudaGraphLaunch`. LR and weight decay are applied outside the graph so schedulers take effect without recapture. Enabled by default — pass `--no-cuda-graph` to disable. `reconfigure()` clears the graph; it's re-captured on the next step.

### HF-format checkpoint save

Pass `--save-checkpoint` to any training script (`train.py` / `train_ddp.py` / `train_fsdp.py`) to save the trained model in standard Hugging Face format, round-trippable with `AutoModelForCausalLM.from_pretrained`. Under FSDP with parameter-bank sharding, an internal un-bank step re-keys the gathered state dict to per-layer HF param names (`model.layers.{i}.self_attn.q_proj.weight`, …) so the output remains HF-compatible.

---

## 🔧 Optimizers

| CLI key | Class | Distributed modes | Notes |
|---|---|---|---|
| `adamw` | `AdamW` | single / DDP / FSDP | standard AdamW |
| `soap` | `SOAP` | single / DDP | diagonal preconditioner |
| `muon` | `Muon` | single / DDP | Newton-Schulz orthogonalization |
| `muon-fsdp` | `MuonFsdp` | FSDP + banks | Muon ported to bank DTensors; baseline for comparison with HiMuon |
| `himuon` | `HiMuon` | single / DDP / FSDP + banks | tile-local NS, cross-layer batched, CUDA graph |
| `himuon-legacy` | `HiMuonLegacy` | single / DDP | earlier reference implementation |

The factory in `src/himuon/optim.py` groups parameters into decay / no-decay buckets and further splits Muon-eligible parameters (2D, excluding embeddings and `lm_head`) from their AdamW fallback.

---

<a id="parameter-bank-sharding"></a>
## 🧩 Parameter-Bank Sharding

Standard `fully_shard` row-splits each matrix across ranks, which breaks tile-local optimizers: a tile would straddle shard boundaries. `src/himuon/fsdp_bank/` solves this by grouping same-shape per-layer projections (e.g., all `q_proj.weight` across layers) into a single 3D tensor of shape `(num_layers, H, W)` and sharding on the layer axis. Each rank then holds complete `(H, W)` matrices — just different layer indices — so per-matrix NS needs no cross-rank collectives.

<p align="center">
  <img src="assets/bank_sharding.svg" alt="Per-layer FSDP vs parameter-bank sharding" width="100%"/>
</p>

Minimal usage:

```python
from himuon.fsdp_bank import Qwen3BankScheme, wrap_with_banks, release_pre_shard_refs
from torch.distributed.fsdp import fully_shard

model, banks = wrap_with_banks(model, Qwen3BankScheme(), world_size=ws)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)
release_pre_shard_refs(banks)  # free pre-shard full-size references
```

`Qwen3BankScheme` produces 5 banks: `q_bank`, `kv_bank`, `o_bank`, `gate_up_bank`, `down_bank`. Other model families need their own `BankScheme` implementation.

---

## 📁 Project Structure

```
himuon/
├── src/himuon/
│   ├── fsdp_bank/          # parameter-bank sharding for FSDP (Qwen3 scheme)
│   ├── optimizers/         # AdamW, Muon, SOAP, HiMuon
│   ├── triton_kernels/     # fused NS kernels
│   ├── dataset.py          # FineWeb streaming dataset
│   ├── logger.py           # Loguru + W&B logging
│   ├── model.py            # model loading (Llama, Qwen)
│   ├── optim.py            # optimizer factory + parameter grouping
│   └── utils.py            # kv args parsing, time budget hook
├── train.py        # single-GPU training
├── train_ddp.py    # DDP training
└── train_fsdp.py   # FSDP training (bank-sharded)
```

---

## 🧪 Tests

```bash
uv run pytest tests/
```

`tests/unit/` (contract / API-doc checks) runs on CPU; `tests/integration/` requires a CUDA GPU (Triton kernels and DTensor/FSDP checks) and is auto-skipped without one.

---

## 📖 Citation

```bibtex
@misc{himuon,
  title   = {Hierarchical Muon: Tile-Local Newton-Schulz Orthogonalization for Scalable LLM Pre-Training},
  author  = {TBD},
  year    = {2026},
  note    = {Preprint in preparation.}
}
```

---

## 🙏 Acknowledgements

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — HiMuon LR scaling and NS implementation reference
- [Muon](https://github.com/KellerJordan/Muon) — Newton-Schulz orthogonalization optimizer

---

## 📄 License

MIT — see [LICENSE](LICENSE).
