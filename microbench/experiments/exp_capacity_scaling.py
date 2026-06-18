"""
Tile-Size Capacity Diagnostic: one-step associative-memory capacity scaling.

Question (sec:capacity_diagnostic): as d, alpha, and tile size vary, does tiled
HiMuon behave more like full-matrix Muon or like SGD? HiMuon replaces the global
zeropower map with a tile-local one, so singular directions spanning multiple tiles
cannot be amplified coherently.

Setup: one update from W0=0 on a linear associative memory with N=1e5 keys, Zipf
frequencies (alpha in {1.25,1.5,1.75}), B/d=10, d in {256,512,1024}, 3 seeds. Metric
is the number of recovered items (argmax-correct keys); recovery is invariant to the
(positive) learning rate, so a single eta is used. Stateless operator-only benchmark
(no momentum / optimizer state) on the shared assoc_mem engine.

Usage:
  uv run python microbench/experiments/exp_capacity_scaling.py
  uv run python microbench/experiments/exp_capacity_scaling.py --alphas 1.25 1.5 1.75 --ds 256 512 1024
  uv run python microbench/experiments/exp_capacity_scaling.py --plot-only
"""

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from assoc_mem import apply_one_step, build_gradient, evaluate_recovery, sample_problem  # noqa: E402
from bench_utils import load_results, save_results  # noqa: E402
from plot_style import COLORS, FIGSIZE_TRIPLE, apply_style, save_fig  # noqa: E402

# Muon/SGD anchors plus the three HiMuon tiles, ordered as in the reference figure.
SERIES_ORDER = ["Muon", "HiMuon-512", "HiMuon-256", "HiMuon-128", "SGD"]
SERIES_COLORS = {
    "Muon": COLORS.BLACK,  # #333333
    "HiMuon-512": "#1F4E79",  # dark navy
    "HiMuon-256": "#4292C6",  # medium blue
    "HiMuon-128": "#9ECAE1",  # light blue
    "SGD": "#D62728",  # red
}


def _methods(tiles):
    return [("Muon", "muon", None), ("SGD", "sgd", None)] + [
        (f"HiMuon-{t}", "himuon", t) for t in tiles
    ]


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
def run(ds, alphas, batch_scale, num_items, seeds, tiles, ns_steps, device="cuda"):
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"ds={list(ds)}, alphas={list(alphas)}, B/d={batch_scale}, N={num_items}, seeds={list(seeds)}, tiles={list(tiles)}\n"
    )
    methods = _methods(tiles)
    records = []
    for d in ds:
        for alpha in alphas:
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
                gradient, meta = build_gradient(
                    problem,
                    gradient_mode="true_minibatch",
                    batch_size=bs,
                    seed=seed + bs,
                )
                for label, opt, tile in methods:
                    update = apply_one_step(
                        gradient,
                        optimizer=opt,
                        eta=1.0,
                        ns_steps=ns_steps,
                        tile_size=tile if tile is not None else 256,
                    )
                    recovered, _ = evaluate_recovery(problem, update)
                    records.append(
                        {
                            "d": d,
                            "alpha": alpha,
                            "seed": seed,
                            "label": label,
                            "tile_size": tile,
                            "recovered_count": int(recovered.sum().item()),
                            "effective_support": meta["effective_support"],
                        }
                    )
                print(
                    f"d={d:5d} alpha={alpha:.2f} seed={seed} | "
                    + " ".join(
                        f"{r['label']}={r['recovered_count']}"
                        for r in records[-len(methods) :]
                    )
                )
            torch.cuda.empty_cache()

    data = {
        "config": {
            "ds": list(ds),
            "alphas": list(alphas),
            "batch_scale": batch_scale,
            "num_items": num_items,
            "seeds": list(seeds),
            "tiles": list(tiles),
            "ns_steps": ns_steps,
            "gradient_mode": "true_minibatch",
            "device": torch.cuda.get_device_name(device),
        },
        "records": records,
    }
    save_results(data, "capacity_scaling")
    return data


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------
def plot(data, out_dir=None):
    apply_style()
    records = data["records"]
    alphas = sorted({r["alpha"] for r in records})
    ds = sorted({r["d"] for r in records})

    agg = defaultdict(list)
    for r in records:
        agg[(r["alpha"], r["d"], r["label"])].append(r["recovered_count"])

    x = np.arange(len(ds))
    w = 0.16  # slot per series; bars drawn at 88% leaving a slight gap (matches real_shapes_bench)
    offsets = {lab: (i - 2) * w for i, lab in enumerate(SERIES_ORDER)}

    fig, axes = plt.subplots(
        1, len(alphas), figsize=(FIGSIZE_TRIPLE[0], 4.0), sharey=False
    )
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        for label in SERIES_ORDER:
            means, stds, xpos = [], [], []
            for i, d in enumerate(ds):
                vals = agg.get((alpha, d, label))
                if not vals:
                    continue
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                xpos.append(i + offsets[label])
            if means:
                ax.bar(
                    xpos,
                    means,
                    width=w * 0.88,
                    color=SERIES_COLORS[label],
                    label=label,
                    yerr=stds,
                    capsize=2.5,
                    error_kw=dict(elinewidth=0.7, ecolor="#444444", capthick=0.6),
                    edgecolor="#333333",
                    linewidth=0.4,
                    zorder=3,
                )
        ax.set_yscale("log")
        ax.set_title(rf"$\alpha = {alpha:g}$")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_xlabel("Embedding dimension $d$")
        ax.grid(axis="y", which="both", alpha=0.2, zorder=0)
        for side in ("top", "right"):  # full box
            ax.spines[side].set_visible(True)
    axes[0].set_ylabel("One-step recovered items")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(SERIES_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
    )
    plt.tight_layout()
    save_fig(fig, "capacity_scaling", out_dir)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-step associative-memory capacity scaling"
    )
    parser.add_argument("--ds", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.25, 1.5, 1.75])
    parser.add_argument("--batch-scale", type=float, default=10.0)
    parser.add_argument("--num-items", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tiles", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot(load_results("capacity_scaling"))
    else:
        data = run(
            ds=args.ds,
            alphas=args.alphas,
            batch_scale=args.batch_scale,
            num_items=args.num_items,
            seeds=args.seeds,
            tiles=args.tiles,
            ns_steps=args.ns_steps,
        )
        plot(data)
