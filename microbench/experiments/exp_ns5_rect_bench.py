"""Rect-shape + grid-strategy sweep for the fused NS5 kernel.

Exercises ``ns5_smem`` across all (M, N) tile shapes drawn from
DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048] with M ≤ N, filtered by
an estimated SMEM budget of ``M*N + 2*M² ≤ SMEM_BUDGET`` bf16 elements
(49152 is the known-working 128×128 working set).

For each admitted shape, two launch strategies are measured:
  * standard   — grid = (B,)        → one CTA per tile
  * persistent — grid = (NUM_SMS,)  → each CTA strides through tiles

The batch B is chosen so the total working set B*M*N is ≈ constant
(default 2^21 elements ≈ 128 tiles of 128×128), so that wall-clock
numbers across shapes are directly comparable as throughput.

Ground truth for the cosine-similarity column is fp32 eager NS on the
same input. The intent is to characterise the fused kernel itself —
3-kernel / eager baselines are out of scope here.

Usage (GPU node):
  uv run python microbench/ns5_rect_bench.py
  uv run python microbench/ns5_rect_bench.py --total-elements 524288
  uv run python microbench/ns5_rect_bench.py --plot-only
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from bench_utils import cosine_sim, load_results, save_results  # noqa: E402
from plot_style import apply_style, clean_spines, save_fig  # noqa: E402

from himuon.triton_kernels import (  # noqa: E402
    XXT,
    ba_plus_cAA,
    fused_bmm_add,
    ns5_smem,
)

# 3-kernel path is torch.compile'd with dynamic=False → specialised per
# (B, M, N). 19 shapes blow past the default dynamo cache limit of 8.
torch._dynamo.config.cache_size_limit = 128


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
NS_STEPS = 5
NS_COEFS = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------------------
# 3-kernel baseline (torch.compile'd, same algorithm as HiMuon's NS path)
# ---------------------------------------------------------------------------
@torch.compile(dynamic=False, fullgraph=True)
def ns_3kernel(X: torch.Tensor) -> torch.Tensor:
    """Pre-norm + 5 iterations of (XXT, ba_plus_cAA, fused_bmm_add) Triton
    kernels, wrapped in torch.compile. Mirrors HiMuon._newton_schulz_3kernel
    but defined locally to avoid depending on the optimizer class."""
    a, b, c = NS_COEFS
    X = X.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    Bbuf = torch.empty_like(A)
    C = torch.empty_like(X)
    for _ in range(NS_STEPS):
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=Bbuf)
        fused_bmm_add(Bbuf, X, a, out=C)
        X, C = C, X
    return X


DIMS = [16, 32, 64, 128, 256, 512, 1024, 2048]
# bf16 elements. Matches the known-working 128×128 tile (3 · 128² = 49152).
SMEM_BUDGET = 49152
# Keep total tile working set ≈ 128 tiles of 128×128 for throughput parity.
TOTAL_ELEMENTS_TARGET = 128 * 128 * 128


# ---------------------------------------------------------------------------
# Shape enumeration
# ---------------------------------------------------------------------------
def enumerate_shapes(dims=DIMS, budget=SMEM_BUDGET):
    """All (M, N) with M ≤ N, passing the SMEM heuristic."""
    shapes = []
    for M in dims:
        for N in dims:
            if N < M:
                continue
            if M * N + 2 * M * M > budget:
                continue
            shapes.append((M, N))
    return shapes


# ---------------------------------------------------------------------------
# Reference NS (fp32 eager) — quality anchor only, not timed
# ---------------------------------------------------------------------------
def ns_reference_fp32(X: torch.Tensor) -> torch.Tensor:
    a, b, c = NS_COEFS
    Xf = X.float()
    Xf = Xf / (Xf.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(NS_STEPS):
        A = Xf @ Xf.mT
        B = b * A + c * A @ A
        Xf = a * Xf + B @ Xf
    return Xf


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def make_input(B: int, M: int, N: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)


def measure(fn, warmup=10, repeats=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)


def run(total_elements: int, warmup: int, repeats: int):
    shapes = enumerate_shapes()
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        f"Running {len(shapes)} shapes × 3 configs (fused-std, fused-per, "
        f"3kernel) on {torch.cuda.get_device_name(0)} (SMs={num_sms})"
    )
    print(f"Target working set: {total_elements} bf16 elements per call\n")

    results = {
        "config": {
            "dims": DIMS,
            "smem_budget": SMEM_BUDGET,
            "total_elements": total_elements,
            "dtype": str(DTYPE),
            "ns_steps": NS_STEPS,
            "device": torch.cuda.get_device_name(0),
            "num_sms": num_sms,
            "warmup": warmup,
            "repeats": repeats,
        },
        "shapes": [],
    }

    for M, N in shapes:
        B = max(1, total_elements // (M * N))
        X = make_input(B, M, N)
        ref = ns_reference_fp32(X)

        entry = {"M": M, "N": N, "B": B, "grids": {}}

        configs = [
            ("standard", lambda: ns5_smem(X, persistent=False)),
            ("persistent", lambda: ns5_smem(X, persistent=True)),
            ("3kernel", lambda: ns_3kernel(X)),
        ]
        for label, fn in configs:
            try:
                out = fn()
                cos = cosine_sim(out, ref)
                times = measure(fn, warmup=warmup, repeats=repeats)
                median_ms = float(np.median(times))
                bytes_processed = B * M * N * 2  # bf16 read
                gbps = bytes_processed / (median_ms * 1e-3) / 1e9
                entry["grids"][label] = {
                    "status": "ok",
                    "median_ms": median_ms,
                    "p25_ms": float(np.percentile(times, 25)),
                    "p75_ms": float(np.percentile(times, 75)),
                    "cos_vs_fp32": cos,
                    "throughput_GBps": gbps,
                }
            except Exception as err:
                entry["grids"][label] = {
                    "status": "fail",
                    "error": repr(err)[:300],
                }

        # Speedup vs 3-kernel baseline (t_3kernel / t_fused). > 1 means
        # fused is faster.
        t3k = entry["grids"]["3kernel"]
        for label in ("standard", "persistent"):
            g = entry["grids"][label]
            if g["status"] == "ok" and t3k["status"] == "ok":
                g["speedup_vs_3kernel"] = t3k["median_ms"] / g["median_ms"]

        results["shapes"].append(entry)

        std = entry["grids"]["standard"]
        per = entry["grids"]["persistent"]
        t3k = entry["grids"]["3kernel"]

        def fmt_ms(g):
            return f"{g['median_ms']:7.3f}" if g["status"] == "ok" else "   FAIL"

        def fmt_spd(g):
            return (
                f"{g['speedup_vs_3kernel']:5.2f}×"
                if g.get("speedup_vs_3kernel")
                else "   --"
            )

        print(
            f"  ({M:4d}, {N:4d}) B={B:5d}  "
            f"std={fmt_ms(std)}ms  per={fmt_ms(per)}ms  "
            f"3k={fmt_ms(t3k)}ms  "
            f"spd(std)={fmt_spd(std)}"
        )

    save_results(results, "ns5_rect_bench")
    return results


# ---------------------------------------------------------------------------
# Plot — single heatmap: speedup fused-std vs 3-kernel, with Pareto boundary
# ---------------------------------------------------------------------------
# M axis only needs values actually attainable in the bench (M ≤ 128, since
# M > 128 already blows the SMEM budget: 2·M² > 16384 → 32768).
M_DIMS = [16, 32, 64, 128]
# Pareto boundary: cells with M·N ≤ 16384 are the "dispatch to fused" region.
PARETO_PRODUCT = 16384


def _build_matrix(shapes, key, grid_label, m_dims, n_dims):
    """Build (len(m_dims), len(n_dims)) matrix indexed [i=M, j=N]."""
    M = np.full((len(m_dims), len(n_dims)), np.nan)
    for e in shapes:
        g = e["grids"].get(grid_label, {})
        if g.get("status") != "ok" or key not in g:
            continue
        if e["M"] not in m_dims or e["N"] not in n_dims:
            continue
        i = m_dims.index(e["M"])
        j = n_dims.index(e["N"])
        M[i, j] = g[key]
    return M


def _pareto_boundary_segments(m_dims, n_dims, predicate):
    """Stair-step line segments around cells satisfying predicate.

    Returns list of ((x0, x1), (y0, y1)) segments in cell-index coords
    suitable for ax.plot after origin="lower".
    """
    nm, nn = len(m_dims), len(n_dims)
    segments = []
    for i in range(nm):
        for j in range(nn):
            if not predicate(m_dims[i], n_dims[j]):
                continue
            # Right edge: boundary if neighbour doesn't satisfy predicate.
            if j + 1 >= nn or not predicate(m_dims[i], n_dims[j + 1]):
                segments.append(((j + 0.5, j + 0.5), (i - 0.5, i + 0.5)))
            # Top edge.
            if i + 1 >= nm or not predicate(m_dims[i + 1], n_dims[j]):
                segments.append(((j - 0.5, j + 0.5), (i + 0.5, i + 0.5)))
    return segments


def _diverging_cmap():
    """Custom navy ↔ warm-cream ↔ vermillion diverging map.

    Navy (#2B5EA7) marks speedup > 1×, warm cream at 1×, deep orange
    (#A03A1A) for < 1×. Matches the paper's HIMUON-navy / FAIL-vermillion
    palette and is friendlier to colorblind readers than RdYlGn.
    Positions hand-tuned so the near-1× band is a narrow warm/cool
    transition, while mid-range values (2-5×) land in a saturated
    mid-blue rather than a washed pastel.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "navy_orange",
        [
            (0.00, "#A03A1A"),
            (0.30, "#D4621A"),
            (0.44, "#F4E8D5"),
            (0.56, "#D8E0EC"),
            (0.70, "#4A7FB0"),
            (1.00, "#1E4884"),
        ],
        N=256,
    )


def _draw_speedup_heatmap(ax, mat, m_dims, n_dims):
    # Local import — top-level `from matplotlib.colors import TwoSlopeNorm`
    # keeps getting stripped by the project formatter.
    from matplotlib.colors import TwoSlopeNorm

    masked = np.ma.masked_invalid(mat)
    # Diverging scale centred at 1.0.
    if masked.count() > 0:
        vmin = min(float(masked.min()), 0.5)
        vmax = max(float(masked.max()), 1.5)
    else:
        vmin, vmax = 0.5, 2.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    im = ax.imshow(
        masked, cmap=_diverging_cmap(), aspect="equal", origin="lower", norm=norm
    )

    ax.set_xticks(range(len(n_dims)))
    ax.set_yticks(range(len(m_dims)))
    ax.set_xticklabels([str(d) for d in n_dims])
    ax.set_yticklabels([str(d) for d in m_dims])
    ax.set_xlabel("N (long dim)")
    ax.set_ylabel("M (short dim)")

    # Cell values. Choose text colour by background luminance.
    for i in range(len(m_dims)):
        for j in range(len(n_dims)):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            # Text colour: grayscale brightness of cell colour.
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "black" if lum > 0.55 else "white"
            ax.text(
                j,
                i,
                f"{v:.2f}×",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    # Pareto / dispatch boundary: outline the M·N ≤ PARETO_PRODUCT region.
    segs = _pareto_boundary_segments(
        m_dims, n_dims, lambda m, n: m * n <= PARETO_PRODUCT
    )
    for (x0, x1), (y0, y1) in segs:
        ax.plot(
            [x0, x1], [y0, y1], color="#1a1a1a", linewidth=2.2, solid_capstyle="round"
        )

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("single-call speedup (×)")
    return im


def plot(data, out_dir=None):
    """Single-panel figure: fused-kernel speedup heatmap vs the 3-kernel
    baseline across (M, N) tile shapes."""
    apply_style()
    # publication.mplstyle enables constrained_layout, which is
    # incompatible with the subplots_adjust used below.
    plt.rcParams["figure.constrained_layout.use"] = False

    shapes = data["shapes"]
    dims_all = data["config"]["dims"]
    m_dims = [d for d in M_DIMS if d in dims_all]
    n_dims = list(dims_all)
    spd_std = _build_matrix(shapes, "speedup_vs_3kernel", "standard", m_dims, n_dims)

    # figsize/margins give the equal-aspect heatmap axes box a fixed
    # 0.93 * 8.9 = 8.28" width x 0.78 * 5.4 = 4.21" height.
    fig, ax = plt.subplots(figsize=(8.9, 5.4))
    _draw_speedup_heatmap(ax, spd_std, m_dims, n_dims)
    ax.set_title("Kernel speedup vs 3-kernel", loc="left", fontweight="bold")
    clean_spines(ax)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.10)
    save_fig(fig, "ns5_rect_bench", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-elements",
        type=int,
        default=TOTAL_ELEMENTS_TARGET,
        help="Target total bf16 elements per call (B*M*N).",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        data = load_results("ns5_rect_bench")
    else:
        data = run(args.total_elements, args.warmup, args.repeats)

    plot(data)
