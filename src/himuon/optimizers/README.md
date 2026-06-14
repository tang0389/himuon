# Optimizers

Optimizer implementations used by the training scripts. Selected via `--optimizer <key>` through the factory in `src/himuon/optim.py`.

| File | Class | Description |
|------|-------|-------------|
| `adamw.py` | `AdamW` | Standard decoupled-weight-decay AdamW |
| `soap.py` | `SOAP` | Shampoo-based diagonal preconditioner |
| `muon.py` | `Muon` | Newton-Schulz orthogonalized momentum (Moonlight LR scaling) |
| `muon_fsdp.py` | `MuonFsdp` | Plain Muon ported to FSDP2 bank-sharded DTensors; baseline for HiMuon |
| `himuon.py` | `HiMuon` | Tile-local Newton-Schulz with cross-layer batched GEMM and CUDA-graph capture |
| `himuon_legacy.py` | `HiMuonLegacy` | Reference HiMuon implementation kept for regression tests |

For algorithmic details (tile-separable polar surrogate, NS arithmetic intensity, fused-kernel design, and end-to-end results), see the paper.
