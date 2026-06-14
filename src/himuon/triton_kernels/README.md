# Triton Kernels

Fused GPU kernels for HiMuon's batched Newton-Schulz iteration. These replace PyTorch matmul calls with Triton to reduce kernel launch overhead on many small tiles.

| File | Operation | Used in NS step |
|------|-----------|-----------------|
| `XXT.py` | `A @ A.T` (batched) | Computing the Gram matrix per tile |
| `ba_plus_cAA.py` | `b*A + c*A@A` (fused) | NS polynomial evaluation `X = aX + bX@X.T@X` |
| `fused_bmm_add.py` | `B@X + a*X` (fused) | NS epilogue: matmul + scalar add in one pass |
| `ns5_smem.py` | All-in-one fused 5-iteration NS (SRAM-resident, FP32-accumulating) | Small-tile fast path (T ≤ 128) — full K-iteration NS without HBM round-trips |
| `utils.py` | Shared helpers (block size selection, etc.) | — |
