# himuon

Core library for LLM pre-training and fine-tuning.

| File | Description |
|------|-------------|
| `model.py` | Model loading (`get_model_and_tokenizer`), parameter counting, Chinchilla token estimation. Supports Llama (local configs) and Qwen (HF Hub). |
| `dataset.py` | `ToyDataset`: iterable streaming dataset over FineWeb with on-the-fly tokenization. |
| `optim.py` | Optimizer factory. Groups parameters into decay/no-decay and Muon-eligible (2D, excluding embeddings/lm_head) vs AdamW fallback. |
| `logger.py` | `LoguruLogger` (file) and `WandbLogger` (cloud) wrappers. |
| `utils.py` | `TimeBudget`, `seed_everything`, `merge_kv_args`. |

## Subdirectories

- **`optimizers/`** — Optimizer implementations (AdamW, Muon, SOAP, HiMuon).
- **`triton_kernels/`** — Triton GPU kernels used by HiMuon's Newton-Schulz iterations.
- **`fsdp_bank/`** — Parameter-bank sharding for FSDP2 (Qwen3 scheme) — groups same-shape per-layer projections into 3D banks sharded on the layer axis so HiMuon tiles never straddle shard boundaries.
