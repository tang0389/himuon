"""
Optimizer step-time + peak-memory microbench.

Compares four optimizers on the same synthetic workload:
  * Muon          — full-matrix NS, no tiling
  * HiMuon        — per-param tiled NS (main-branch baseline)
  * HiMuon+       — cross-layer batched NS + @torch.compile post-NS
  * HiMuon+ (graph) — HiMuon+ with CUDA graph capture

Workload: a synthetic nn.Module of K Linear(H, H) layers in bf16 on CUDA.
Varying H exercises NS at different tile densities.

Usage (on a GPU node):
  uv run python microbench/optimizer_step_microbench.py
  uv run python microbench/optimizer_step_microbench.py --sizes 1024 2048 4096
  uv run python microbench/optimizer_step_microbench.py --plot-only
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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

from himuon.optimizers.himuon_legacy import HiMuonLegacy as HiMuon  # noqa: E402
from himuon.optimizers.himuon import HiMuon as HiMuonPlus  # noqa: E402
from himuon.optimizers.muon import Muon  # noqa: E402


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Optimizer configs. Ordering drives plot x-axis grouping.
# CUDA graph is treated as part of the x-layer implementation (workspace
# pooling) — not a separate variant. The eager x-layer entry is kept in
# saved JSON for ablation but not plotted.
OPTIM_CONFIGS = [
    ("Muon", Muon, dict(lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1)),
    (
        "HiMuon",
        HiMuon,
        dict(
            lr=0.02,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.1,
            tile_size=512,
            ns_steps=5,
        ),
    ),
    (
        "HiMuon + x-layer",
        HiMuonPlus,
        dict(
            lr=0.02,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.1,
            tile_size=512,
            ns_steps=5,
            cuda_graph=True,
            cuda_graph_warmup=3,
        ),
    ),
]

# Display label → JSON key.
_DATA_KEY = {
    "Muon": "Muon",
    "HiMuon": "HiMuon",
    "HiMuon + x-layer": "HiMuon + x-layer",
}

# Color mapping for plots. HiMuon family shares the HIMUON color; the two
# variants are distinguished by saturation / hatching. Muon stays gray.
_OPTIM_COLORS = {
    "Muon": COLORS.MUON,
    "HiMuon": "#6B8FBF",  # lighter HIMUON tone
    "HiMuon + x-layer": COLORS.HIMUON,
}
_OPTIM_HATCH = {
    "Muon": HATCH_BASELINE,
    "HiMuon": "",
    "HiMuon + x-layer": "",
}


def build_param_groups(model, supports_muon: bool):
    """Match optim.py's param grouping. Non-muon optimizers (Muon, HiMuon,
    HiMuon+) all share the same (2D → muon, 1D → AdamW fallback) scheme."""
    param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    muon_params = [p for n, p in param_dict.items() if p.ndim >= 2]
    adamw_params = [p for n, p in param_dict.items() if p.ndim < 2]
    groups = []
    if supports_muon and muon_params:
        groups.append({"params": muon_params, "weight_decay": 0.0, "use_muon": True})
    if adamw_params:
        groups.append(
            {"params": adamw_params, "lr": 1e-3, "weight_decay": 0.0, "use_muon": False}
        )
    if not supports_muon:
        # Plain AdamW/Muon path: single group, no use_muon flag.
        return [{"params": list(param_dict.values())}]
    return groups


def make_model(H: int, K: int = 8) -> nn.Module:
    """K stacked Linear(H, H) layers, bf16 on CUDA."""
    return nn.Sequential(
        *[nn.Linear(H, H, bias=False, dtype=DTYPE) for _ in range(K)]
    ).to(DEVICE)


def fill_random_grads(model, seed=1234):
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.empty_like(p)
        p.grad.normal_(generator=gen)


def measure_step(opt, model, warmup=20, repeats=50, seed_base=5000):
    """Median per-step wall-clock after warmup. Setup amortizes to zero."""
    for i in range(warmup):
        fill_random_grads(model, seed=seed_base + i)
        opt.step()
    torch.cuda.synchronize()

    times = []
    for i in range(repeats):
        fill_random_grads(model, seed=seed_base + warmup + i)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        opt.step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)


def measure_peak(opt, model, n_steps=5, seed_base=9000):
    for i in range(max(5, n_steps)):
        fill_random_grads(model, seed=seed_base + i)
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for i in range(n_steps):
        fill_random_grads(model, seed=seed_base + 100 + i)
        opt.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def bench_one(H: int, K: int, label: str, cls, kwargs, warmup: int, repeats: int):
    torch.cuda.empty_cache()
    torch.manual_seed(7)
    model = make_model(H, K=K)
    supports_muon = (
        "use_muon" in cls.__init__.__code__.co_varnames
        or "muon" in cls.__name__.lower()
    )
    # Muon itself uses its own param-grouping convention; HiMuon / HiMuonPlus
    # take the use_muon flag. Keep it simple: build groups per class.
    if cls is Muon:
        groups = [{"params": list(model.parameters())}]
    else:
        groups = build_param_groups(model, supports_muon=True)
    opt = cls(groups, **kwargs)

    times = measure_step(opt, model, warmup=warmup, repeats=repeats)
    peak = measure_peak(opt, model)

    del opt, model, groups
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return dict(
        label=label,
        H=H,
        median_ms=float(np.median(times)),
        p25_ms=float(np.percentile(times, 25)),
        p75_ms=float(np.percentile(times, 75)),
        peak_bytes=int(peak),
    )


def run(sizes, K, warmup, repeats):
    results = {
        "config": {
            "sizes": sizes,
            "K": K,
            "dtype": str(DTYPE),
            "device": torch.cuda.get_device_name(0),
        },
        "per_H": {},
    }
    for H in sizes:
        print(f"\n--- H={H}, K={K} layers ---")
        results["per_H"][str(H)] = {}
        for label, cls, kwargs in OPTIM_CONFIGS:
            r = bench_one(H, K, label, cls, kwargs, warmup, repeats)
            results["per_H"][str(H)][label] = r
            print(
                f"  {label:20s}  step={r['median_ms']:7.2f} ms  "
                f"peak={r['peak_bytes'] / 1e6:7.1f} MB"
            )
    save_results(results, "optimizer_step_microbench")
    return results


def plot(data, out_dir=None):
    apply_style()
    config = data["config"]
    per_H = data["per_H"]
    sizes = config["sizes"]
    labels = [label for label, _, _ in OPTIM_CONFIGS]
    x = np.arange(len(sizes))
    n_opt = len(labels)
    # Slot width drives bar offsets; bar_w shrinks the rendered bar within
    # the slot so adjacent bars don't touch.
    w = 0.6 / n_opt  # slot per optimizer (narrower group -> thinner bars overall)
    bar_w = w * 0.88  # 12% gap within the slot

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # (a) Step-time speedup vs Muon (linear, anchored at 1×).
    ax = axes[0]
    for i, label in enumerate(labels):
        key = _DATA_KEY[label]
        vals = [
            per_H[str(H)]["Muon"]["median_ms"] / per_H[str(H)][key]["median_ms"]
            for H in sizes
        ]
        xs = x + (i - (n_opt - 1) / 2) * w
        styled_bar(
            ax,
            xs,
            vals,
            width=bar_w,
            label=label,
            color=_OPTIM_COLORS[label],
            hatch=_OPTIM_HATCH[label],
        )
        for xi, v in zip(xs, vals):
            ax.text(xi, v * 1.02, f"{v:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="#888", linewidth=0.6, linestyle="--", zorder=2)
    ax.set_ylabel("Speedup vs Muon  (×, higher = faster)")
    ax.set_title("Step-time speedup")
    ax.set_ylim(0, 6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Hidden dim H")
    ax.legend(fontsize=9, loc="upper left")
    for _side in ("top", "right"):
        ax.spines[_side].set_visible(True)
    panel_label(ax, "a")

    # (b) Peak memory in absolute MB; bars annotated with +X% vs Muon.
    ax = axes[1]
    for i, label in enumerate(labels):
        key = _DATA_KEY[label]
        peaks_mb = [per_H[str(H)][key]["peak_bytes"] / 1e6 for H in sizes]
        ratios = [
            per_H[str(H)][key]["peak_bytes"] / per_H[str(H)]["Muon"]["peak_bytes"]
            for H in sizes
        ]
        xs = x + (i - (n_opt - 1) / 2) * w
        styled_bar(
            ax,
            xs,
            peaks_mb,
            width=bar_w,
            label=label,
            color=_OPTIM_COLORS[label],
            hatch=_OPTIM_HATCH[label],
        )
        if label == "Muon":
            continue
        for xi, mb, r in zip(xs, peaks_mb, ratios):
            pct = (r - 1.0) * 100
            txt = f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%"
            ax.text(xi, mb * 1.02, txt, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Peak memory per step")
    ax.set_ylim(0, 1500)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Hidden dim H")
    for _side in ("top", "right"):
        ax.spines[_side].set_visible(True)
    panel_label(ax, "b")

    plt.tight_layout()
    save_fig(fig, "optimizer_step_microbench", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="Hidden dims H for the synthetic K × Linear(H,H) model",
    )
    parser.add_argument(
        "--layers", type=int, default=8, help="Number of Linear layers K"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        data = load_results("optimizer_step_microbench")
        plot(data)
    else:
        data = run(args.sizes, args.layers, args.warmup, args.repeats)
        plot(data)
