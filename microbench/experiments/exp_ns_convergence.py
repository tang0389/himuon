"""
M2: Newton-Schulz Iteration Count K Convergence

Sweeps K ∈ {1, ..., 10} × tileSize ∈ {128, 256, 512} and measures how
close the NS output is to its intended target (the polar factor):

  - Full-NS:  cos(output_K, polar_fp64(G))           — global polar factor
  - Tile-NS:  mean over tiles of cos(tile_K, polar_fp64(tile_input))
              — each tile compared to its own polar factor

This directly measures convergence quality and justifies the K=5 default.

Usage (compute node with GPU):
  uv run python microbench/ns_convergence.py
  uv run python microbench/ns_convergence.py --max-k 8 --tiles 256 512
  uv run python microbench/ns_convergence.py --plot-only
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from bench_utils import save_results
from plot_style import (
    COLORS,
    FIGSIZE_WIDE,
    apply_style,
    clean_spines,
    panel_label,
    save_fig,
)
from himuon.optimizers.himuon import HiMuon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _polar_fp64(M):
    """Polar factor of M via fp64 SVD: U @ Vh."""
    M64 = M.double()
    U, _, Vh = torch.linalg.svd(M64, full_matrices=False)
    return (U @ Vh).float()


def _full_ns_at_k(G, k):
    """Full-matrix NS with exactly k iterations."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16().clone()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(k):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.to(G.dtype)


def _tile_ns_at_k(G, tile_size, k):
    """Tile-local NS with exactly k iterations.

    Returns (untiled_output, list_of_per_tile_outputs, list_of_per_tile_inputs).
    """
    ts = (tile_size, tile_size)
    tiled, info = HiMuon._tile(G, ts)
    R, C, Th, Tw = tiled.shape

    # Save tile inputs for per-tile polar reference
    tile_inputs = tiled.view(-1, Th, Tw).clone()

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = tile_inputs.bfloat16().clone()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(k):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    tile_outputs = X.float()
    return tile_outputs, tile_inputs


def _cosine_sim(a, b):
    a_f = a.double().flatten()
    b_f = b.double().flatten()
    return float((a_f @ b_f) / (a_f.norm() * b_f.norm() + 1e-15))


def _relative_error(a, ref):
    return float((a.double() - ref.double()).norm() / (ref.double().norm() + 1e-15))


def _batched_polar_fp64(tiles):
    """Batched polar factor: (N, H, W) -> (N, H, W) via fp64 SVD."""
    T64 = tiles.double()
    U, _, Vh = torch.linalg.svd(T64, full_matrices=False)
    return (U @ Vh).float()


def _per_tile_polar_similarity_with_refs(tile_outputs, refs):
    """Same as _per_tile_polar_similarity but with pre-computed polar refs."""
    out_flat = tile_outputs.double().flatten(1)
    ref_flat = refs.double().flatten(1)
    dots = (out_flat * ref_flat).sum(dim=1)
    norms_out = out_flat.norm(dim=1)
    norms_ref = ref_flat.norm(dim=1)
    cos_vals = dots / (norms_out * norms_ref + 1e-15)
    err_vals = (out_flat - ref_flat).norm(dim=1) / (norms_ref + 1e-15)
    return (
        float(cos_vals.mean().item()),
        float(cos_vals.min().item()),
        float(err_vals.mean().item()),
        float(err_vals.max().item()),
    )


def _per_tile_polar_similarity(tile_outputs, tile_inputs):
    """Per-tile cosine sim / relative error vs each tile's own fp64 polar factor.

    Uses batched SVD for speed. Returns (mean_cos, min_cos, mean_err, max_err).
    """
    refs = _batched_polar_fp64(tile_inputs)  # (N, Th, Tw)
    # Flatten each tile for vectorized cosine sim / error
    out_flat = tile_outputs.double().flatten(1)  # (N, Th*Tw)
    ref_flat = refs.double().flatten(1)
    # Per-tile cosine similarity
    dots = (out_flat * ref_flat).sum(dim=1)
    norms_out = out_flat.norm(dim=1)
    norms_ref = ref_flat.norm(dim=1)
    cos_vals = dots / (norms_out * norms_ref + 1e-15)
    # Per-tile relative error
    err_vals = (out_flat - ref_flat).norm(dim=1) / (norms_ref + 1e-15)
    return (
        float(cos_vals.mean().item()),
        float(cos_vals.min().item()),
        float(err_vals.mean().item()),
        float(err_vals.max().item()),
    )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
def run(max_k, tiles, D=4096, n_seeds=3):
    """Sweep K × tile_size. Returns JSON-serializable dict."""
    device = torch.device("cuda")
    ks = list(range(1, max_k + 1))

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"D={D}, K∈[1..{max_k}], tiles={tiles}, seeds={n_seeds}\n")

    hdr = (
        f"{'Seed':>5} | {'Method':>10} | {'K':>3} | "
        f"{'cos mean':>10} | {'cos min':>10} | {'err mean':>10} | {'err max':>10}"
    )
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    curves = {}
    for T in tiles:
        curves[str(T)] = {
            k: {"cos": [], "cos_min": [], "err": [], "err_max": []} for k in ks
        }
    curves["full"] = {
        k: {"cos": [], "cos_min": [], "err": [], "err_max": []} for k in ks
    }

    for seed in range(n_seeds):
        torch.manual_seed(seed * 137)
        G = torch.randn(D, D, device=device, dtype=torch.bfloat16)
        U, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
        S = torch.logspace(0, 2, D, device=device)
        G = ((U * S.unsqueeze(0)) @ Vh).bfloat16()

        # Full-NS: compare to global polar factor (single matrix, min=mean)
        polar_global = _polar_fp64(G)
        for k in ks:
            out = _full_ns_at_k(G, k)
            cs = _cosine_sim(out, polar_global)
            re = _relative_error(out, polar_global)
            curves["full"][k]["cos"].append(cs)
            curves["full"][k]["cos_min"].append(cs)
            curves["full"][k]["err"].append(re)
            curves["full"][k]["err_max"].append(re)
            print(
                f"{seed:>5} | {'full':>10} | {k:>3} | "
                f"{cs:>10.6f} | {'—':>10} | {re:>10.6f} | {'—':>10}"
            )

        # Tile-NS: compare each tile to its own polar factor
        for T in tiles:
            # Pre-compute per-tile polar refs (same for all K)
            _, tile_inputs_for_ref = _tile_ns_at_k(G, T, 1)
            tile_polar_refs = _batched_polar_fp64(tile_inputs_for_ref)

            for k in ks:
                tile_outputs, _ = _tile_ns_at_k(G, T, k)
                cs_mean, cs_min, re_mean, re_max = _per_tile_polar_similarity_with_refs(
                    tile_outputs, tile_polar_refs
                )
                curves[str(T)][k]["cos"].append(cs_mean)
                curves[str(T)][k]["cos_min"].append(cs_min)
                curves[str(T)][k]["err"].append(re_mean)
                curves[str(T)][k]["err_max"].append(re_max)
                print(
                    f"{seed:>5} | {'T=' + str(T):>10} | {k:>3} | "
                    f"{cs_mean:>10.6f} | {cs_min:>10.6f} | {re_mean:>10.6f} | {re_max:>10.6f}"
                )

        print(sep)
        del G, polar_global
        torch.cuda.empty_cache()

    # Aggregate
    summary = {}
    for label, k_dict in curves.items():
        summary[label] = {}
        for k, vals in k_dict.items():
            cos_arr = np.array(vals["cos"])
            cos_min_arr = np.array(vals["cos_min"])
            err_arr = np.array(vals["err"])
            err_max_arr = np.array(vals["err_max"])
            summary[label][str(k)] = {
                "cos_mean": float(cos_arr.mean()),
                "cos_std": float(cos_arr.std()),
                "cos_min_mean": float(cos_min_arr.mean()),
                "cos_min_std": float(cos_min_arr.std()),
                "err_mean": float(err_arr.mean()),
                "err_std": float(err_arr.std()),
                "err_max_mean": float(err_max_arr.mean()),
                "err_max_std": float(err_max_arr.std()),
            }

    data = {
        "config": {
            "max_k": max_k,
            "tiles": tiles,
            "D": D,
            "n_seeds": n_seeds,
            "device": torch.cuda.get_device_name(device),
        },
        "summary": summary,
    }
    save_results(data, "ns_convergence")
    return data


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------
_LABEL_COLORS = {
    "128": "#C4541A",
    "256": "#D4900D",
    "512": "#2A9D6E",
    "1024": "#7B5EA7",
    "full": "#1A6FA0",
}

_LABEL_MARKERS = {
    "128": "s",
    "256": "D",
    "512": "o",
    "1024": "^",
    "full": "x",
}


def plot(data, out_dir=None):
    """Convergence to polar factor: cosine sim and relative error vs K."""
    from plot_style import LINE_WIDTH, MARKER_SIZE

    apply_style()

    config = data["config"]
    summary = data["summary"]
    max_k = config["max_k"]
    ks = np.arange(1, max_k + 1)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    labels_order = [str(t) for t in config["tiles"]]

    # --- (A) Cosine similarity to per-tile polar factor ---
    # Worst-case tile only (solid with error bars)
    ax = axes[0]
    for label in labels_order:
        if label not in summary:
            continue
        color = _LABEL_COLORS.get(label, COLORS.GRAY)
        marker = _LABEL_MARKERS.get(label, "o")
        display = f"T={label}"

        mins = [summary[label][str(k)]["cos_min_mean"] for k in ks]
        min_stds = [summary[label][str(k)]["cos_min_std"] for k in ks]
        ax.errorbar(
            ks,
            mins,
            yerr=min_stds,
            label=f"{display} (worst tile)",
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            capsize=2,
            capthick=0.6,
        )

    ax.axhline(1.0, color=COLORS.LIGHT_GRAY, linestyle="--", linewidth=0.6)
    ax.axvline(5, color=COLORS.GRAY, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("NS iterations K")
    ax.set_ylabel("Cosine similarity to per-tile polar factor")
    ax.set_title("Per-Tile Convergence: Cosine Similarity")
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=None, top=1.002)
    clean_spines(ax)
    panel_label(ax, "a")

    # --- (B) Relative Frobenius error ---
    # Worst-case tile only (solid with error bars)
    ax = axes[1]
    for label in labels_order:
        if label not in summary:
            continue
        color = _LABEL_COLORS.get(label, COLORS.GRAY)
        marker = _LABEL_MARKERS.get(label, "o")
        display = f"T={label}"

        maxs = [summary[label][str(k)]["err_max_mean"] for k in ks]
        max_stds = [summary[label][str(k)]["err_max_std"] for k in ks]
        ax.errorbar(
            ks,
            maxs,
            yerr=max_stds,
            label=f"{display} (worst tile)",
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            capsize=2,
            capthick=0.6,
        )

    ax.axvline(5, color=COLORS.GRAY, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("NS iterations K")
    ax.set_ylabel("Relative Frobenius error to polar factor")
    ax.set_title("Per-Tile Convergence: Relative Error")
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    clean_spines(ax)
    panel_label(ax, "b")

    plt.tight_layout()
    save_fig(fig, "ns_convergence", out_dir)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2: NS iteration count K convergence")
    parser.add_argument("--max-k", type=int, default=10, help="Max K to sweep")
    parser.add_argument(
        "--tiles",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Tile sizes to test",
    )
    parser.add_argument("--D", type=int, default=4096, help="Matrix size")
    parser.add_argument("--seeds", type=int, default=3, help="Random seeds to average")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        from bench_utils import load_results

        apply_style()
        plot(load_results("ns_convergence"))
    else:
        data = run(
            max_k=args.max_k,
            tiles=args.tiles,
            D=args.D,
            n_seeds=args.seeds,
        )
        plot(data)
