"""Fused Newton-Schulz 5-iter kernel for rectangular tiles.

Single-CTA-per-tile fused kernel. Accepts (B, M, N) with M ≤ N. Each
tile's 5 NS iterations run in SMEM/registers with zero HBM intermediate
traffic. Dispatch rule (M · N ≤ 16384) is enforced by callers.

Algorithm (same quintic as the 3-kernel baseline):
    for _ in range(5):
        A = X @ X.T           # (M, N) × (N, M) → (M, M)
        B = c*A@A + b*A       # (M, M) × (M, M) → (M, M)
        X = a*X + B @ X       # (M, M) × (M, N) → (M, N)

Grid strategies (controlled by launcher via ``tl.num_programs``):
  * standard   — grid = (B,)        → one CTA per tile, loop trip = 1
  * persistent — grid = (NUM_SMS,)  → each CTA strides through tiles
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Quintic coefficients (modded-nanogpt).
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315
_NS_STEPS = 5
_NORM_EPS = 1e-7


def _autotune_configs():
    """Autotune only num_warps / num_stages; BLOCK_M, BLOCK_N are fixed at
    the tile shape. Keep the config grid small so compile cost stays
    manageable when sweeping many (M, N) pairs."""
    configs = []
    for num_warps in (4, 8):
        for num_stages in (2, 3, 4):
            configs.append(
                triton.Config({}, num_warps=num_warps, num_stages=num_stages)
            )
    return configs


@triton.jit
def _ns5_tile_body(
    tile_id,
    X_ptr,
    OUT_ptr,
    stride_xb,
    stride_xr,
    stride_xc,
    stride_ob,
    stride_or,
    stride_oc,
    offs_m,
    offs_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    A_COEF: tl.constexpr,
    B_COEF: tl.constexpr,
    C_COEF: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    """Load one (M, N) tile, run 5 NS iterations, store. Shared by the
    single-tile and persistent kernel entry points."""
    x_ptrs = (
        X_ptr
        + tile_id * stride_xb
        + offs_m[:, None] * stride_xr
        + offs_n[None, :] * stride_xc
    )
    out_ptrs = (
        OUT_ptr
        + tile_id * stride_ob
        + offs_m[:, None] * stride_or
        + offs_n[None, :] * stride_oc
    )

    X_raw = tl.load(x_ptrs).to(ACC_DTYPE)
    norm = tl.sqrt(tl.sum(X_raw * X_raw))
    X_norm = X_raw / (norm + NORM_EPS)
    X = X_norm.to(DTYPE)

    for _ in tl.static_range(_NS_STEPS):
        X_t = tl.trans(X)
        A_acc = tl.dot(X, X_t, out_dtype=ACC_DTYPE)
        A_bf = A_acc.to(DTYPE)

        B_acc = tl.dot(A_bf, A_bf, out_dtype=ACC_DTYPE)
        B_acc = C_COEF * B_acc + B_COEF * A_bf.to(ACC_DTYPE)
        B_bf = B_acc.to(DTYPE)

        X_acc = tl.dot(B_bf, X, out_dtype=ACC_DTYPE)
        BX_bf = X_acc.to(DTYPE)
        X_sum = BX_bf.to(ACC_DTYPE) + A_COEF * X.to(ACC_DTYPE)
        X = X_sum.to(DTYPE)

    out_dtype = OUT_ptr.dtype.element_ty
    tl.store(out_ptrs, X.to(out_dtype))


@triton.autotune(
    configs=_autotune_configs(),
    key=["BLOCK_M", "BLOCK_N", "DTYPE_ID", "PERSISTENT"],
)
@triton.jit
def ns5_smem_kernel(
    X_ptr,
    OUT_ptr,
    B,
    stride_xb,
    stride_xr,
    stride_xc,
    stride_ob,
    stride_or,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE_ID: tl.constexpr,  # 0 = bf16 (reserved for future fp16 key)
    DTYPE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    A_COEF: tl.constexpr,
    B_COEF: tl.constexpr,
    C_COEF: tl.constexpr,
    NORM_EPS: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    """Kernel launcher.
    * ``PERSISTENT = False`` (standard): grid = (B,). Each CTA processes
      tile_id = pid. No runtime loop — matches the single-tile codegen
      of ``ns5_smem`` at T=128.
    * ``PERSISTENT = True``: grid = (NUM_SMS,). Each CTA strides over
      tiles via ``tl.range(pid, B, num_programs)``.
    """
    pid = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    if PERSISTENT:
        num_programs = tl.num_programs(axis=0)
        for tile_id in tl.range(pid, B, num_programs):
            _ns5_tile_body(
                tile_id,
                X_ptr,
                OUT_ptr,
                stride_xb,
                stride_xr,
                stride_xc,
                stride_ob,
                stride_or,
                stride_oc,
                offs_m,
                offs_n,
                BLOCK_M,
                BLOCK_N,
                DTYPE,
                ACC_DTYPE,
                A_COEF,
                B_COEF,
                C_COEF,
                NORM_EPS,
            )
    else:
        _ns5_tile_body(
            pid,
            X_ptr,
            OUT_ptr,
            stride_xb,
            stride_xr,
            stride_xc,
            stride_ob,
            stride_or,
            stride_oc,
            offs_m,
            offs_n,
            BLOCK_M,
            BLOCK_N,
            DTYPE,
            ACC_DTYPE,
            A_COEF,
            B_COEF,
            C_COEF,
            NORM_EPS,
        )


def ns5_smem(
    X: torch.Tensor,
    *,
    persistent: bool = False,
    grid_size: int | None = None,
) -> torch.Tensor:
    """Fused 5-iter Newton-Schulz on (B, M, N) rectangular tiles.

    Args:
        X: (B, M, N) or (M, N) tensor, dtype bf16. Requires M ≤ N; caller
           must transpose if the natural shape has M > N.
        persistent: if True, grid = (num_SMs,) and each CTA loops through
           tiles. Else grid = (B,), one CTA per tile (classic ns5_smem).
        grid_size: explicit grid override (used for ablations). Takes
           precedence over ``persistent``.

    Returns:
        Fresh (B, M, N) bf16 tensor with orthogonalised tiles.
    """
    assert X.ndim in (2, 3), f"ns5_smem expects 2D or 3D, got {X.ndim}D"
    unsqueezed = False
    if X.ndim == 2:
        X = X.unsqueeze(0)
        unsqueezed = True

    B_total, M, N = X.shape
    assert M <= N, f"ns5_smem expects M ≤ N, got ({M}, {N})"
    assert X.dtype == torch.bfloat16, f"ns5_smem is bf16-only, got {X.dtype}"
    assert X.is_contiguous(), "ns5_smem requires contiguous input"

    OUT = torch.empty_like(X)

    if grid_size is not None:
        G = grid_size
        is_persistent = G != B_total
    elif persistent:
        G = torch.cuda.get_device_properties(X.device).multi_processor_count
        is_persistent = True
    else:
        G = B_total
        is_persistent = False
    grid = (G,)

    ns5_smem_kernel[grid](
        X,
        OUT,
        B_total,
        X.stride(0),
        X.stride(1),
        X.stride(2),
        OUT.stride(0),
        OUT.stride(1),
        OUT.stride(2),
        BLOCK_M=M,
        BLOCK_N=N,
        DTYPE_ID=0,  # bf16 only for now
        DTYPE=tl.bfloat16,
        ACC_DTYPE=tl.float32,
        A_COEF=_NS_A,
        B_COEF=_NS_B,
        C_COEF=_NS_C,
        NORM_EPS=_NORM_EPS,
        PERSISTENT=is_persistent,
    )

    if unsqueezed:
        OUT = OUT.squeeze(0)
    return OUT
