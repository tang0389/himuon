import torch
import triton
import triton.language as tl

from .utils import get_bmm_autotune_configs


@triton.autotune(
    configs=get_bmm_autotune_configs(),
    key=["M", "N", "K"],
)
@triton.jit
def bmm_add_kernel(
    b_ptr,
    x_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_bb,
    stride_bm,
    stride_bk,
    stride_xb,
    stride_xk,
    stride_xn,
    stride_cb,
    stride_cm,
    stride_cn,
    a_scalar,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid_batch = tl.program_id(1)
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    b_ptrs = (
        b_ptr
        + pid_batch * stride_bb
        + (offs_am[:, None] * stride_bm + offs_k[None, :] * stride_bk)
    )
    x_ptrs = (
        x_ptr
        + pid_batch * stride_xb
        + (offs_k[:, None] * stride_xk + offs_bn[None, :] * stride_xn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K-dim mask: last chunk may be partial
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = offs_k < K - k * BLOCK_SIZE_K
        b = tl.load(b_ptrs, mask=k_mask[None, :], other=0.0)
        x = tl.load(x_ptrs, mask=k_mask[:, None], other=0.0)
        accumulator = tl.dot(b, x, accumulator)
        b_ptrs += BLOCK_SIZE_K * stride_bk
        x_ptrs += BLOCK_SIZE_K * stride_xk

    # Dynamic output dtype (matches XXT / ba_plus_cAA pattern)
    out_dtype = c_ptr.dtype.element_ty
    c = accumulator.to(out_dtype)

    # Epilogue: load X and fuse add.  Unmasked positions get 0 so they
    # don't corrupt in-bounds values; store mask below prevents writing them.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    epilogue_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    x_out_ptrs = (
        x_ptr
        + pid_batch * stride_xb
        + (offs_am[:, None] * stride_xk + offs_bn[None, :] * stride_xn)
    )
    x_val = tl.load(x_out_ptrs, mask=epilogue_mask, other=0.0)
    c = c + a_scalar * x_val

    # Store with boundary mask
    c_ptrs = (
        c_ptr
        + pid_batch * stride_cb
        + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    )
    tl.store(c_ptrs, c, mask=epilogue_mask)


def fused_bmm_add(
    B: torch.Tensor, X: torch.Tensor, a: float, out: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes Out = B @ X + a * X in a single fused Triton kernel.
    """
    Batch, M, K = B.shape
    _, _, N = X.shape
    assert M == K, "fused_bmm_add requires M == K for epilogue B@X + a*X"

    if out is None:
        out = torch.empty_like(X)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        Batch,
    )

    bmm_add_kernel[grid](
        B,
        X,
        out,
        M,
        N,
        K,
        B.stride(0),
        B.stride(1),
        B.stride(2),
        X.stride(0),
        X.stride(1),
        X.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        a,
    )
    return out
