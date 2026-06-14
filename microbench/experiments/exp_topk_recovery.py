"""
Top-K Frequency Recovery on the unique sampled support.

Question (fig:topk_recovery): which items are recovered after one step -- only the
high-frequency head, or also the tail? Recovery is evaluated over the unique sampled
support S (items with nonzero empirical frequency in the minibatch), sorted by
descending population frequency. Restricting to S avoids confounding update-map
quality with finite-sample coverage: an unsampled item has zero gradient and cannot
be recovered after one step.

Setup: d=1024, alpha=1.5, B/d=10, N=1e5, 3 seeds, normalization=none. The reported
curve goes up to the common cutoff k_max = min|S| across seeds; recovery is invariant
to (positive) eta, so a single eta is used.

Usage:
  uv run python microbench/experiments/exp_topk_recovery.py
  uv run python microbench/experiments/exp_topk_recovery.py --plot-only
"""

import argparse

import matplotlib.pyplot as plt
import torch

# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from assoc_mem import (  # noqa: E402
    apply_one_step,
    build_gradient,
    evaluate_recovery,
    sample_empirical_frequencies,
    sample_problem,
)
from bench_utils import load_results, save_results  # noqa: E402
from plot_style import COLORS, apply_style, clean_spines, save_fig  # noqa: E402

SERIES_COLORS = {
    "SGD": COLORS.FAIL,
    "Muon": COLORS.BLACK,
    "HiMuon-512": "#1F4E79",
    "HiMuon-256": COLORS.HIMUON,  # #2B5EA7
    "HiMuon-128": "#7FA9D6",
}
LEFT_ORDER = ["SGD", "Muon", "HiMuon-512", "HiMuon-256", "HiMuon-128"]
GAP_ORDER = ["HiMuon-128", "HiMuon-256", "HiMuon-512"]


def _methods(tiles):
    return [("SGD", "sgd", None), ("Muon", "muon", None)] + [
        (f"HiMuon-{t}", "himuon", t) for t in tiles
    ]


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
def run(
    d,
    alpha,
    seeds,
    batch_scale,
    num_items,
    tiles,
    normalization,
    ns_steps,
    device="cuda",
):
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"d={d}, alpha={alpha}, B/d={batch_scale}, N={num_items}, seeds={list(seeds)}, tiles={list(tiles)}\n"
    )
    methods = _methods(tiles)
    support_sizes = {}
    per_label = {lab: [] for lab, _, _ in methods}

    for seed in seeds:
        problem = sample_problem(
            d=d,
            minimum_num_items=num_items,
            frequency_mode="power_law",
            alpha=alpha,
            seed=seed,
            device=device,
        )
        bs = max(1, int(round(batch_scale * d)))
        gseed = seed + bs
        gradient, _ = build_gradient(
            problem, gradient_mode="true_minibatch", batch_size=bs, seed=gseed
        )
        # Recompute the empirical frequencies build_gradient used -> unique sampled support.
        freqs = sample_empirical_frequencies(problem.p, batch_size=bs, seed=gseed)
        support = torch.nonzero(freqs > 0, as_tuple=False).squeeze(1).cpu()
        p_cpu = problem.p.to(torch.float32).cpu()
        support_sorted = support[torch.argsort(p_cpu[support], descending=True)]
        support_sizes[seed] = int(support_sorted.numel())

        for label, opt, tile in methods:
            update = apply_one_step(
                gradient,
                optimizer=opt,
                eta=1.0,
                ns_steps=ns_steps,
                tile_size=tile if tile is not None else 256,
                normalization=normalization,
            )
            recovered, _ = evaluate_recovery(problem, update)
            rec_s = recovered.to(torch.float32)[support_sorted]
            ks = torch.arange(1, rec_s.numel() + 1, dtype=torch.float32)
            per_label[label].append((torch.cumsum(rec_s, 0) / ks))
        print(f"seed={seed} |S|={support_sizes[seed]}")

    k_max = min(support_sizes.values())
    series = {}
    for label, _, tile in methods:
        stacked = torch.stack([c[:k_max] for c in per_label[label]], dim=0)
        series[label] = {
            "tile_size": tile,
            "mean_recovery_rate": [float(v) for v in stacked.mean(dim=0).tolist()],
            "std_recovery_rate": [
                float(v) for v in stacked.std(dim=0, unbiased=False).tolist()
            ],
        }

    data = {
        "config": {
            "d": d,
            "alpha": alpha,
            "seeds": list(seeds),
            "batch_scale": batch_scale,
            "num_items": num_items,
            "tiles": list(tiles),
            "normalization": normalization,
            "ns_steps": ns_steps,
            "k_max": k_max,
            "support_sizes": support_sizes,
            "mean_support_size": sum(support_sizes.values()) / len(support_sizes),
            "k_values": list(range(1, k_max + 1)),
            "device": torch.cuda.get_device_name(device),
        },
        "series": series,
    }
    save_results(data, "topk_recovery")
    return data


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------
def plot(data, out_dir=None, gap_min_k=50):
    apply_style()
    series = data["series"]
    ks = data["config"]["k_values"]
    k_max = data["config"]["k_max"]

    fig, (ax_rate, ax_gap) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: support-restricted top-k recovery rate.
    for label in LEFT_ORDER:
        if label not in series:
            continue
        mean = series[label]["mean_recovery_rate"]
        std = series[label]["std_recovery_rate"]
        dashed = label == "SGD"
        ax_rate.plot(
            ks,
            mean,
            color=SERIES_COLORS[label],
            label=label,
            linewidth=2.0,
            linestyle="--" if dashed else "-",
        )
        ax_rate.fill_between(
            ks,
            [max(0.0, m - s) for m, s in zip(mean, std)],
            [min(1.0, m + s) for m, s in zip(mean, std)],
            color=SERIES_COLORS[label],
            alpha=0.12,
        )
    ax_rate.set_xscale("log")
    ax_rate.set_xlim(1, k_max)
    ax_rate.set_ylim(0.0, 1.02)
    ax_rate.set_xlabel(r"Rank prefix $k$ (log scale)")
    ax_rate.set_ylabel(r"Top-$k$ recovery rate")
    ax_rate.grid(alpha=0.2, which="both")
    ax_rate.legend(loc="lower left", fontsize=9)
    clean_spines(ax_rate)

    # Right: gap to Muon in percentage points.
    muon = series["Muon"]["mean_recovery_rate"]
    ax_gap.axhline(0.0, color=COLORS.GRAY, linewidth=1.2)
    for label in GAP_ORDER:
        if label not in series:
            continue
        gap = [
            (h - m) * 100.0 for h, m in zip(series[label]["mean_recovery_rate"], muon)
        ]
        ax_gap.plot(ks, gap, color=SERIES_COLORS[label], label=label, linewidth=2.0)
    ax_gap.set_xscale("log")
    ax_gap.set_xlim(gap_min_k, k_max)
    ax_gap.set_xlabel(r"Rank prefix $k$ (log scale)")
    ax_gap.set_ylabel("Recovery gap to Muon (pp)")
    ax_gap.grid(alpha=0.2, which="both")
    ax_gap.legend(loc="lower right", fontsize=9)
    clean_spines(ax_gap)

    plt.tight_layout()
    save_fig(fig, "topk_recovery", out_dir)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-step top-k recovery on the unique sampled support"
    )
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-scale", type=float, default=10.0)
    parser.add_argument("--num-items", type=int, default=100000)
    parser.add_argument("--tiles", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--normalization", default="none")
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot(load_results("topk_recovery"))
    else:
        data = run(
            d=args.d,
            alpha=args.alpha,
            seeds=args.seeds,
            batch_scale=args.batch_scale,
            num_items=args.num_items,
            tiles=args.tiles,
            normalization=args.normalization,
            ns_steps=args.ns_steps,
        )
        plot(data)
