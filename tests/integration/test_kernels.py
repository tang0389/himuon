"""Kernel-correctness contract: the Triton kernels seamlessly replace NS.

Two layers, both oracle'd against pure torch (no frozen-kernel reference):

  * **Building blocks** — each kernel (``XXT``, ``ba_plus_cAA``, ``fused_bmm_add``)
    equals the exact torch op it stands in for, at the shapes/dtypes/symmetry the
    Newton-Schulz iteration feeds it.
  * **Seamless NS replacement** (``TestNewtonSchulzReplacement``) — the thing that
    actually matters: ``HiMuon.newton_schulz()`` reproduces a pure-torch quintic
    NS, on *both* dispatch backends (fused ``ns5_smem`` and the 3-kernel path) and
    across the tall/wide-transpose and dtype cases. If a kernel rewrite breaks the
    orthogonalization, this fails even when the building-block tests still pass.

Bf16 tolerance, not bitwise: NS runs in bf16 and Triton's reduction order differs
from cuBLAS by a few ulps. Benchmarks live in ``microbench/``, not here.
"""

import pytest
import torch

from himuon.optimizers.himuon import HiMuon
from himuon.triton_kernels.ba_plus_cAA import ba_plus_cAA
from himuon.triton_kernels.fused_bmm_add import fused_bmm_add
from himuon.triton_kernels.XXT import XXT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")


# ============================================================================
# Correctness Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_XXT():
    M, K = 128, 64
    A = torch.randn((M, K), device=DEVICE, dtype=torch.bfloat16)

    # Triton result
    C_triton = torch.empty((M, M), device=DEVICE, dtype=torch.bfloat16)
    XXT(A, C_triton)

    # Torch result
    C_torch = A @ A.T

    assert torch.allclose(C_triton, C_torch, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not TORCH_HAS_FP8, reason="FP8 not available")
def test_XXT_fp8():
    M, K = 128, 64
    # Note: Using random fp8 can lead to precision issues, so we use small integers
    A_float = torch.randint(-5, 5, (M, K), device=DEVICE, dtype=torch.float16)
    A = A_float.to(torch.float8_e5m2)

    # Triton result (Output usually in float16 for stability even with FP8 input)
    # The kernel casts accumulator to C.dtype.
    C_triton = torch.empty((M, M), device=DEVICE, dtype=torch.float16)
    XXT(A, C_triton)

    # Torch result (compute in high precision)
    C_torch = A_float @ A_float.T

    # Relax tolerance for FP8
    assert torch.allclose(C_triton, C_torch, atol=0.5, rtol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ba_plus_cAA():
    M = 128
    # The kernel ba_plus_cAA assumes the input matrix A is symmetric because
    # it writes the output symmetrically (storing both C[i,j] and C[j,i]).
    # In the Polar Express algorithm, the input to this kernel is always X @ X.T, which is symmetric.
    # Therefore, we must test with a symmetric matrix.
    A_rand = torch.randn((M, M), device=DEVICE, dtype=torch.bfloat16)
    A = A_rand @ A_rand.T  # Make A symmetric

    alpha, beta = 0.5, 0.3

    # Triton result
    C_triton = torch.empty((M, M), device=DEVICE, dtype=torch.bfloat16)
    ba_plus_cAA(A, alpha, beta, C_triton)

    # Torch result
    C_torch = alpha * (A @ A.T) + beta * A

    assert torch.allclose(C_triton, C_torch, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or not TORCH_HAS_FP8, reason="FP8 not available")
def test_ba_plus_cAA_fp8():
    M = 128
    A_rand = torch.randn((M, M), device=DEVICE, dtype=torch.float16)
    # Normalize by spectral norm to ensure values stay small (<= 1)
    # This mimics the behavior in Polar Express algorithm
    A_float = A_rand @ A_rand.T
    A_float = A_float / (A_float.norm() + 1e-6)

    A = A_float.to(torch.float8_e5m2)
    # We should use the casted A converted back to float16 as reference.
    A_ref = A.to(torch.float16)

    alpha, beta = 0.5, 0.3

    # Triton result
    C_triton = torch.empty((M, M), device=DEVICE, dtype=torch.float16)
    ba_plus_cAA(A, alpha, beta, C_triton)

    # Torch result (using the actual FP8 values)
    # The triton kernel accumulates in float32, so we should do the same for reference if possible,
    # or just use float16 which has enough precision for these small values.
    C_torch = alpha * (A_ref @ A_ref.T) + beta * A_ref

    # FP8 e5m2 has very low precision (2 mantissa bits).
    # A value like 0.1 might be rounded significantly.
    # We use a relaxed tolerance.
    assert torch.allclose(C_triton, C_torch, atol=0.2, rtol=0.2)


# -- fused_bmm_add tests ----------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "Batch,M,N",
    [
        (4, 100, 70),  # neither M nor N divisible by 32
        (2, 50, 130),  # M < smallest block, N not aligned
        (3, 200, 100),  # M not power-of-2
    ],
)
def test_fused_bmm_add_boundary_mask(Batch, M, N):
    """Non-aligned dims — exposes missing boundary masks in tl.load/tl.store."""
    torch.manual_seed(42)
    K = M
    B = torch.randn((Batch, M, K), device=DEVICE, dtype=torch.bfloat16)
    X = torch.randn((Batch, K, N), device=DEVICE, dtype=torch.bfloat16)
    a = 3.4445

    C_triton = fused_bmm_add(B, X, a)
    C_ref = torch.bmm(B, X) + a * X

    assert torch.allclose(C_triton, C_ref, atol=1e-1, rtol=1e-2), (
        f"max diff = {(C_triton - C_ref).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_bmm_add_dtype(dtype):
    """Output dtype follows the out tensor — verifies dynamic dtype cast."""
    torch.manual_seed(0)
    Batch, M, N, K = 4, 128, 128, 128
    B = torch.randn((Batch, M, K), device=DEVICE, dtype=dtype)
    X = torch.randn((Batch, K, N), device=DEVICE, dtype=dtype)
    a = 3.4445

    out = torch.empty((Batch, M, N), device=DEVICE, dtype=dtype)
    C_triton = fused_bmm_add(B, X, a, out=out)
    C_ref = torch.bmm(B, X) + a * X

    assert C_triton.dtype == dtype
    # Triton and cuBLAS use different reduction order → normal floating-point divergence
    atol = 1e-1 if dtype != torch.float32 else 5e-2
    assert torch.allclose(C_triton, C_ref, atol=atol, rtol=1e-2), (
        f"dtype={dtype}, max diff = {(C_triton - C_ref).abs().max().item()}"
    )


# ============================================================================
# Seamless NS replacement
# ============================================================================


def _ns_reference(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Pure-torch quintic Newton-Schulz — the oracle the kernels must match.

    Mirrors the Muon/Polar-Express iteration ``HiMuon.newton_schulz`` implements
    (same coefficients, bf16 compute, tall-matrix transpose, spectral-norm
    seeding) but built only from ``torch`` matmuls — no Triton. This is the
    independent ground truth: whichever backend ``newton_schulz`` dispatches to
    (``ns5_smem`` or the 3-kernel path) must reproduce *this*.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    original_dtype = G.dtype
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(original_dtype)


def _well_conditioned(shape, dtype, seed: int = 0) -> torch.Tensor:
    """Random G whose singular spectrum sits in the NS stability band [0.8, 1.2].

    The Muon quintic holds singular values in a band around 1 but is *chaotic*
    on near-zero ones: two correct NS impls (Triton vs torch) legitimately
    diverge there purely from reduction order. Seeding the spectrum away from 0
    makes the iteration well-behaved, so a tight Triton-vs-torch match isolates
    kernel correctness from that intrinsic sensitivity.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    A = torch.randn(*shape, device=DEVICE, dtype=torch.float32, generator=g)
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    k = U.shape[-1]
    spectrum = torch.linspace(0.8, 1.2, k, device=DEVICE)
    return ((U * spectrum.unsqueeze(-2)) @ Vh).to(dtype)


# Well-conditioned NS reproduces the torch reference to <3e-2 across every
# dispatch branch; 4e-2 leaves margin while still catching a real kernel break.
NS_RTOL, NS_ATOL = 4e-2, 4e-2

# Shapes chosen to hit each dispatch branch of HiMuon.newton_schulz
# (see _NS5_DIMS / the 16384-element area cap).
_NS_SHAPES = [
    # --- fused ns5_smem branch: 3-D, dims in _NS5_DIMS, area <= 16384 ---
    ("ns5_square_128", (4, 128, 128)),
    ("ns5_square_64", (8, 64, 64)),
    ("ns5_wide_64x256", (4, 64, 256)),
    ("ns5_tall_128x64", (4, 128, 64)),  # tall → transpose branch
    # --- 3-kernel branch: 2-D, or 3-D too big for ns5_smem ---
    ("k3_3d_big", (4, 512, 512)),  # area > 16384 → 3-kernel
    ("k3_2d_square", (512, 512)),
    ("k3_2d_nondiv", (768, 513)),  # non-divisible dims
    ("k3_2d_tall", (513, 768)),  # tall 2-D → transpose branch
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("label, shape", _NS_SHAPES, ids=[s[0] for s in _NS_SHAPES])
def test_newton_schulz_matches_torch_reference(label, shape):
    """``HiMuon.newton_schulz`` (Triton, bf16) == pure-torch NS, on both dispatch
    backends, across tall/wide — on well-conditioned inputs."""
    G = _well_conditioned(shape, torch.bfloat16)

    out = HiMuon.newton_schulz(G, steps=5)
    ref = _ns_reference(G, steps=5)

    assert out.shape == G.shape
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), f"{label}: non-finite NS output"
    assert torch.allclose(out.float(), ref.float(), rtol=NS_RTOL, atol=NS_ATOL), (
        f"{label} ({tuple(shape)}): NS diverged from torch reference, "
        f"max|Δ|={(out.float() - ref.float()).abs().max().item():.3e}"
    )
