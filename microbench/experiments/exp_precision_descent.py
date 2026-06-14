"""
Does low-precision / tiled Newton-Schulz degrade the Muon update?

Two quantities from Shumaylov, Da Costa, Zaika, Mucsanyi et al., "Muon is Not
That Special" (arXiv:2605.11181), Sec. 3.1:

    alignment          gamma = <G, D> / <G~, D>
    descent potential  phi   = <G~, D>^2 / <D, H D>

G = full-batch gradient, G~ = mini-batch gradient, D = update, H = Hessian.
Both are closed-form on the paper's own proxy (Sec. 3.2, after Davis &
Drusvyatskiy 2026): teacher-student matrix least-squares with anisotropic input
covariance, f(W) = 1/2 E_x ||(W - W*) x||^2, x ~ N(0, Sigma), giving
G = (W - W*) Sigma, G~ = (W - W*) Sigma_hat_b, H D = D Sigma -- no transformer.

A from-scratch fp32 trajectory generates W; at each step every tile x precision
operator runs on the shared (G, G~). The figure plots gamma and phi vs training
step, one line per tile size, bf16 / fp8 as markers on the fp32 line.

Usage:
  uv run python microbench/experiments/exp_precision_descent.py [--plot-only]
"""

import argparse
import os as _os
import sys as _sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from bench_utils import load_results, save_results  # noqa: E402
from plot_style import FIGSIZE_WIDE, apply_style, save_fig  # noqa: E402

PRECS = ("fp32", "bf16", "fp8e4m3")
TILE_COLOR = ["#9ECAE1", "#4292C6", "#2171B5", "#084594"]  # light=128 -> dark=full

NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
_FP8_MAX = 448.0  # e4m3 max finite magnitude
_TINY = torch.finfo(torch.float32).tiny


# torch._scaled_mm is 2D-only, so loop the batch and CUDA-graph the loop.
def _scaled_mm_loop(af, bf, sa, sb):
    out = torch.empty(af.shape[0], af.shape[-2], bf.shape[-1], device=af.device)
    for i in range(af.shape[0]):
        out[i] = torch._scaled_mm(
            af[i],
            bf[i].mT.contiguous().mT,
            scale_a=sa[i],
            scale_b=sb[i],
            out_dtype=torch.float32,
        )
    return out


_scaled_mm_loop_c = torch.compile(_scaled_mm_loop, mode="reduce-overhead")


def _bmm_fp8(a, b):
    """Per-tile dynamic-scaled fp8 GEMM, fp32 accumulation."""
    bc = b.contiguous()
    sa = (a.abs().amax(dim=(-2, -1)) / _FP8_MAX).clamp_min(_TINY)
    sb = (bc.abs().amax(dim=(-2, -1)) / _FP8_MAX).clamp_min(_TINY)
    af = (a / sa[:, None, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    bf = (bc / sb[:, None, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    try:
        return _scaled_mm_loop_c(af, bf, sa, sb).clone()  # compiled buffer is reused
    except Exception:
        return _scaled_mm_loop(af, bf, sa, sb)


def _mm(a, b, fp8):
    return _bmm_fp8(a, b) if fp8 else a @ b


# Square tiling by reshape (dim divisible by tile).
def _tile(g, t):
    n = g.shape[0] // t
    return g.reshape(n, t, n, t).permute(0, 2, 1, 3).reshape(n * n, t, t)


def _untile(x, D, t):
    n = D // t
    return x.reshape(n, n, t, t).permute(0, 2, 1, 3).reshape(D, D)


def _ns(g, tile, prec):
    """Newton-Schulz orthogonalize g at the given precision and tile size
    (tile=None -> full matrix). Returns the fp32 update D."""
    a, b, c = NS_COEFFS
    D = g.shape[0]
    fp8 = prec == "fp8e4m3"
    dtype = torch.bfloat16 if prec == "bf16" else torch.float32
    x = (_tile(g, tile) if tile else g[None]).to(dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(NS_STEPS):
        A = _mm(x, x.mT, fp8)
        B = b * A + c * _mm(A, A, fp8)
        x = a * x + _mm(B, x, fp8)
    x = x.float()
    return _untile(x, D, tile) if tile else x[0]


def _make_problem(dim, alpha, dev, seed):
    """Teacher-student quadratic with an anisotropic power-law input spectrum."""
    g = torch.Generator(device=dev).manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, device=dev))
    lam = (torch.arange(1, dim + 1, device=dev, dtype=torch.float32)) ** (-alpha)
    lam = lam / lam.mean()  # unit average eigenvalue
    sigma = (q * lam) @ q.mT
    sigma_half = (q * lam.sqrt()) @ q.mT
    wstar = torch.randn(dim, dim, generator=g, device=dev)
    w0 = torch.randn(dim, dim, generator=g, device=dev)
    return sigma, sigma_half, wstar / wstar.norm(), w0 / w0.norm()  # unit-Frobenius


def _ip(x, y):
    return float((x.double() * y.double()).sum())


def run(
    dim=1024,
    alpha=1.0,
    steps=40,
    lr=0.03,
    batch=256,
    tiles=(128, 256, 512),
    precisions=PRECS,
    seed=0,
):
    dev = "cuda"
    print(f"Device: {torch.cuda.get_device_name(dev)}  dim={dim} alpha={alpha}")
    sigma, sigma_half, wstar, w = _make_problem(dim, alpha, dev, seed)
    cats = [f"t{t}" for t in tiles] + ["full"]
    tile_of = {f"t{t}": t for t in tiles} | {"full": None}
    curves = {
        f"{cat}/{p}": {"gamma": [], "phi": []} for cat in cats for p in precisions
    }
    losses = []
    mb_gen = torch.Generator(device=dev).manual_seed(seed + 1)

    for k in range(steps):
        err = w - wstar
        g_full = err @ sigma
        z = torch.randn(batch, dim, generator=mb_gen, device=dev)
        sigma_b = (z @ sigma_half).mT @ (z @ sigma_half) / batch
        g_mb = err @ sigma_b
        losses.append(0.5 * _ip(err @ sigma_half, err @ sigma_half))

        for cat in cats:
            for p in precisions:
                d = _ns(g_mb, tile_of[cat], p)
                gd = _ip(g_mb, d)
                hd = _ip(d, d @ sigma)
                curves[f"{cat}/{p}"]["gamma"].append(_ip(g_full, d) / (gd + 1e-20))
                curves[f"{cat}/{p}"]["phi"].append(gd * gd / (hd + 1e-20))

        d_ref = _ns(g_mb, None, "fp32")
        w = w - lr * d_ref / (d_ref.norm() + 1e-7)

    print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    data = {
        "config": {
            "problem": "teacher-student-quadratic",
            "dim": dim,
            "alpha": alpha,
            "steps": steps,
            "lr": lr,
            "batch": batch,
            "tiles": list(tiles),
            "device": torch.cuda.get_device_name(dev),
        },
        "curves": curves,
        "loss": losses,
    }
    save_results(data, "precision_descent")
    return data


def plot(data, out_dir=None):
    apply_style()
    curves, cfg = data["curves"], data["config"]
    cats = [(f"t{t}", str(t)) for t in cfg["tiles"]] + [("full", "full")]
    colors = {cat: TILE_COLOR[i] for i, (cat, _) in enumerate(cats)}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    every = 4

    for ax, metric in ((axa, "gamma"), (axb, "phi")):
        for cat, _ in cats:
            c = colors[cat]
            steps = np.arange(len(curves[f"{cat}/fp32"][metric]))
            ax.plot(steps, curves[f"{cat}/fp32"][metric], color=c, lw=2.0, zorder=2)
            ms = steps[::every]
            for prec, mk, sz in (("bf16", "o", 7), ("fp8e4m3", "+", 7)):
                y = np.asarray(curves[f"{cat}/{prec}"][metric])[::every]
                ax.plot(
                    ms,
                    y,
                    color="#3a3a3a",
                    marker=mk,
                    ls="none",
                    markersize=sz,
                    markerfacecolor="none",
                    markeredgewidth=1.1,
                    zorder=4,
                )

    axa.set_ylabel(r"Alignment  $\gamma$")
    axa.set_title("Update alignment")
    axb.set_ylabel(r"Descent potential  $\Phi$")
    axb.set_title("Descent potential")
    for ax in (axa, axb):
        ax.set_xlabel("Training step")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)

    # one in-plot legend, two columns: colour -> tile (squares), marker -> precision
    sq = lambda c, lab: plt.Line2D(
        [], [], color=c, marker="s", ls="none", markersize=9, label=lab
    )
    tile_h = [sq(colors[cat], lab) for cat, lab in cats]
    prec_h = [
        plt.Line2D([], [], color="#3a3a3a", lw=1.8, label="fp32"),
        plt.Line2D(
            [],
            [],
            color="#3a3a3a",
            marker="o",
            ls="none",
            markerfacecolor="none",
            markeredgewidth=1.2,
            markersize=7,
            label="bf16",
        ),
        plt.Line2D(
            [],
            [],
            color="#3a3a3a",
            marker="+",
            ls="none",
            markeredgewidth=1.2,
            markersize=8,
            label="fp8",
        ),
    ]
    blank = plt.Line2D([], [], color="none", label="")
    prec_h += [blank] * (len(tile_h) - len(prec_h))
    handles = tile_h + prec_h  # column-major fill: col0 tiles, col1 precision
    axa.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        ncol=2,
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        columnspacing=1.4,
        handletextpad=0.5,
        labelspacing=0.4,
    )
    plt.tight_layout()
    save_fig(fig, "precision_descent", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--tiles", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot(load_results("precision_descent"))
    else:
        plot(
            run(
                dim=args.dim,
                alpha=args.alpha,
                steps=args.steps,
                lr=args.lr,
                batch=args.batch,
                tiles=tuple(args.tiles),
            )
        )
