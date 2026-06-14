"""
Newton-Schulz kernel microbench at T=128.

Compares four NS implementations on batched (B, T, T) bf16 tile inputs:
  * NS (eager)    — plain PyTorch loop (X@X.T, matmuls via torch)
  * torch.compile — same algorithm with @torch.compile
  * 3-kernel      — XXT + ba_plus_cAA + fused_bmm_add Triton kernels,
                    no compile, no fusion across iterations
  * ns5_smem      — fused single-kernel 5-iter (HiMuon dev ships this)

Varies batch size B. T fixed at 128 (the regime where ns5_smem ships).

Usage:
  uv run python microbench/ns_kernel_microbench.py
  uv run python microbench/ns_kernel_microbench.py --batches 32 128 512 2048
  uv run python microbench/ns_kernel_microbench.py --plot-only
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from bench_utils import load_results, save_results  # noqa: E402
from plot_style import (  # noqa: E402
    COLORS,
    FIGSIZE_WIDE,
    HATCH_BASELINE,
    apply_style,
    panel_label,
    save_fig,
    styled_bar,
)

from himuon.triton_kernels import XXT, ba_plus_cAA, fused_bmm_add, ns5_smem  # noqa: E402


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
T = 128
NS_STEPS = 5
NS_COEFS = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------------------
# NS implementations
# ---------------------------------------------------------------------------
def _ns_core(X, steps):
    """Shared eager NS core used by both eager and compiled variants."""
    a, b, c = NS_COEFS
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def ns_eager(X: torch.Tensor) -> torch.Tensor:
    """Plain torch NS — no compile, no Triton."""
    return _ns_core(X.clone(), NS_STEPS)


_ns_compiled_fn = torch.compile(_ns_core, dynamic=False, fullgraph=True)


def ns_compiled(X: torch.Tensor) -> torch.Tensor:
    """@torch.compile'd NS. Same algorithm as ns_eager."""
    return _ns_compiled_fn(X.clone(), NS_STEPS)


def ns_three_kernel(X: torch.Tensor) -> torch.Tensor:
    """3-kernel Triton path (XXT + ba_plus_cAA + fused_bmm_add), no compile.

    Mirrors the T > 128 path in HiMuon.newton_schulz but runs at T=128 for
    a fair comparison against ns5_smem on the same tile size.
    """
    a, b, c = NS_COEFS
    X = X.clone()
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


def ns_smem(X: torch.Tensor) -> torch.Tensor:
    """Fused single-kernel 5-iter NS (ns5_smem)."""
    return ns5_smem(X.clone())


KERNEL_CONFIGS = [
    ("NS (eager)", ns_eager, COLORS.GRAY, HATCH_BASELINE),
    ("torch.compile", ns_compiled, "#6B8FBF", ""),
    ("3-kernel", ns_three_kernel, "#E8734A", ""),
    ("ns5_smem", ns_smem, COLORS.HIMUON, ""),
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def make_input(B: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, T, device=DEVICE, dtype=DTYPE)


def measure(fn, arg, warmup=10, repeats=50):
    for _ in range(warmup):
        fn(arg)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = fn(arg)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)


def cosine_sim(a, b):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float((a_f @ b_f) / (a_f.norm() * b_f.norm() + 1e-12))


def run(batches, warmup, repeats):
    results = {
        "config": {
            "T": T,
            "ns_steps": NS_STEPS,
            "batches": batches,
            "dtype": str(DTYPE),
            "device": torch.cuda.get_device_name(0),
        },
        "per_B": {},
    }

    for B in batches:
        print(f"\n--- B={B}, T={T} ---")
        results["per_B"][str(B)] = {}
        X = make_input(B)
        # Reference for correctness: use ns_eager (the most straightforward)
        # at fp32 precision to anchor cosine comparisons.
        ref = ns_eager(X).float()

        for label, fn, _color, _hatch in KERNEL_CONFIGS:
            times = measure(fn, X, warmup=warmup, repeats=repeats)
            out = fn(X).float()
            cos = cosine_sim(out, ref)
            r = dict(
                label=label,
                B=B,
                median_ms=float(np.median(times)),
                p25_ms=float(np.percentile(times, 25)),
                p75_ms=float(np.percentile(times, 75)),
                cos_vs_eager=cos,
            )
            results["per_B"][str(B)][label] = r
            print(f"  {label:18s}  median={r['median_ms']:7.3f} ms  cos={cos:.6f}")
    save_results(results, "ns_kernel_microbench")
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(data, out_dir=None):
    apply_style()
    per_B = data["per_B"]
    batches = data["config"]["batches"]
    labels = [label for label, *_ in KERNEL_CONFIGS]
    x = np.arange(len(batches))
    n = len(labels)
    w = 0.8 / n  # slot per kernel
    bar_w = w * 0.88  # bar shrunk within slot -> slight gap (matches real_shapes_bench)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # (a) time
    ax = axes[0]
    for i, (label, _fn, color, hatch) in enumerate(KERNEL_CONFIGS):
        medians = [per_B[str(B)][label]["median_ms"] for B in batches]
        p25 = [per_B[str(B)][label]["p25_ms"] for B in batches]
        p75 = [per_B[str(B)][label]["p75_ms"] for B in batches]
        yerr = [
            [m - lo for m, lo in zip(medians, p25)],
            [hi - m for m, hi in zip(medians, p75)],
        ]
        xs = x + (i - (n - 1) / 2) * w
        styled_bar(
            ax,
            xs,
            medians,
            width=bar_w,
            label=label,
            color=color,
            hatch=hatch,
            yerr=yerr,
        )
    ax.set_yscale("log")
    ax.set_ylabel("NS 5-iter time (ms, median)")
    ax.set_title(f"NS kernel comparison at T={T}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(B) for B in batches])
    ax.set_xlabel(r"Batch size $B_{\mathrm{tile}}$")
    ax.legend(fontsize=9, loc="upper left")
    for _side in ("top", "right"):
        ax.spines[_side].set_visible(True)
    panel_label(ax, "a")

    # (b) speedup vs NS (eager). Include the baseline as a hatched 1.0× bar so
    # this panel mirrors optimizer_step_microbench (a).
    ax = axes[1]
    ref_label = "NS (eager)"
    for i, (label, _fn, color, hatch) in enumerate(KERNEL_CONFIGS):
        if label == ref_label:
            ratios = [1.0 for _ in batches]
        else:
            ratios = [
                per_B[str(B)][ref_label]["median_ms"]
                / per_B[str(B)][label]["median_ms"]
                for B in batches
            ]
        xs = x + (i - (n - 1) / 2) * w
        styled_bar(
            ax,
            xs,
            ratios,
            width=bar_w,
            label=label,
            color=color,
            hatch=hatch,
        )
        for xi, v in zip(xs, ratios):
            ax.text(xi, v * 1.02, f"{v:.1f}×", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color="#333333", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_ylabel("Speedup vs NS (eager)")
    ax.set_title("Relative speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([str(B) for B in batches])
    ax.set_xlabel(r"Batch size $B_{\mathrm{tile}}$")
    ax.legend(fontsize=9, loc="upper left")
    for _side in ("top", "right"):
        ax.spines[_side].set_visible(True)
    panel_label(ax, "b")

    plt.tight_layout()
    save_fig(fig, "ns_kernel_microbench", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[32, 128, 512, 2048],
        help="Batch sizes B for (B, T, T) NS input",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        data = load_results("ns_kernel_microbench")
        plot(data)
    else:
        data = run(args.batches, args.warmup, args.repeats)
        plot(data)
