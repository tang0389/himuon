"""
Real weight shape benchmark.

Extracts all Muon-eligible parameter shapes from Qwen3 models, then benchmarks
tile-NS vs full-NS on each unique shape.  Reports per-shape time, memory,
speedup, padding overhead, and output quality (SV std, cosine similarity).

Usage:
  uv run python microbench/real_shapes_bench.py
  uv run python microbench/real_shapes_bench.py --models Qwen3-0.6B Qwen3-1.7B
  uv run python microbench/real_shapes_bench.py --tile 256
  uv run python microbench/real_shapes_bench.py --plot-only
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

from bench_utils import measure, measure_paired_speedup, save_results
from plot_style import (
    COLORS,
    apply_style,
    clean_spines,
    save_fig,
)
from himuon.model import get_model_and_tokenizer
from himuon.optimizers.himuon import HiMuon
from himuon.optimizers.muon import zeropower_via_newtonschulz5 as muon_ns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_tile_ns(G, tile_size, ns_steps):
    ts = (tile_size, tile_size)
    tiled, info = HiMuon._tile(G, ts)
    R, C, Th, Tw = tiled.shape
    orth = HiMuon.newton_schulz(tiled.view(-1, Th, Tw), steps=ns_steps).view(
        R, C, Th, Tw
    )
    return HiMuon._untile(orth, info)


def padding_overhead(H, W, tile_size):
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size
    original = H * W
    padded = (H + pad_h) * (W + pad_w)
    return (padded - original) / original


def get_muon_eligible_shapes(model_name):
    """Return list of (name, shape) for Muon-eligible parameters."""
    model, _ = get_model_and_tokenizer(model_name)
    shapes = []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name:
            shapes.append((name, tuple(p.shape)))
    del model
    return shapes


def dedupe_shapes(shapes):
    groups = defaultdict(list)
    for name, shape in shapes:
        groups[shape].append(name)
    return [(names[0], shape, len(names)) for shape, names in groups.items()]


def get_block_layers(shapes, block_idx=0):
    prefix = f"model.layers.{block_idx}."
    block_layers = []
    for name, shape in shapes:
        if name.startswith(prefix):
            short = name[len(prefix) :]
            short = short.replace(".weight", "")
            block_layers.append((short, shape))
    return block_layers


# ---------------------------------------------------------------------------
# run() -- experiment, returns JSON-serializable dict
# ---------------------------------------------------------------------------
def run(models, tile, ns_steps, warmup=3, repeats=30):
    """Run real-shapes benchmark. Returns serializable dict."""
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"tile_size={tile}, ns_steps={ns_steps}, repeats={repeats}\n")

    all_model_results = {}

    for model_name in models:
        print(f"=== {model_name} ===")
        raw_shapes = get_muon_eligible_shapes(model_name)
        block_layers = get_block_layers(raw_shapes, block_idx=0)
        print(
            f"  {len(raw_shapes)} Muon-eligible params total, "
            f"{len(block_layers)} layers in block 0\n"
        )

        sep = "-" * 140
        print(
            f"  {'Layer':>25} | {'Shape':>14} | {'Pad %':>6} | "
            f"{'Tile med(ms)':>12} | {'p25-p75':>14} | "
            f"{'Full med(ms)':>12} | {'p25-p75':>14} | "
            f"{'Speedup':>8} | {'p25-p75':>12} | "
            f"{'Tile Mem':>10} | {'Full Mem':>10}"
        )
        print(f"  {sep}")

        model_results = []
        for name, shape in block_layers:
            count = sum(1 for _, s in raw_shapes if s == shape)
            H, W = shape[0], shape[1]
            pad_pct = padding_overhead(H, W, tile) * 100
            G = torch.randn(H, W, device=device, dtype=torch.bfloat16)

            # Full NS
            mem_f = time_f = p25_f = p75_f = -1
            try:
                mem_f, time_f, p25_f, p75_f, _, _ = measure(
                    lambda: muon_ns(G.clone(), steps=ns_steps),
                    warmup,
                    repeats,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

            # Tile NS
            mem_t = time_t = p25_t = p75_t = -1
            try:
                mem_t, time_t, p25_t, p75_t, _, _ = measure(
                    lambda: run_tile_ns(G, tile, ns_steps),
                    warmup,
                    repeats,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

            # Paired speedup
            speedup_median = speedup_p25 = speedup_p75 = float("nan")
            if time_t > 0 and time_f > 0:
                try:
                    speedup_median, speedup_p25, speedup_p75 = measure_paired_speedup(
                        fn_a=lambda: muon_ns(G.clone(), steps=ns_steps),
                        fn_b=lambda: run_tile_ns(G, tile, ns_steps),
                        warmup=warmup,
                        repeats=repeats,
                    )
                except Exception:
                    if time_t > 0 and time_f > 0:
                        speedup_median = time_f / time_t

            ok_t = time_t > 0
            ok_f = time_f > 0
            t_str = f"{time_t:>10.2f}ms" if ok_t else f"{'OOM':>11}"
            f_str = f"{time_f:>10.2f}ms" if ok_f else f"{'OOM':>11}"
            t_iqr = f"[{p25_t:.1f},{p75_t:.1f}]" if ok_t else "--"
            f_iqr = f"[{p25_f:.1f},{p75_f:.1f}]" if ok_f else "--"
            sp_str = (
                f"{speedup_median:>7.2f}x"
                if not np.isnan(speedup_median)
                else f"{'--':>8}"
            )
            sp_iqr = (
                f"[{speedup_p25:.2f},{speedup_p75:.2f}]"
                if not np.isnan(speedup_p25)
                else "--"
            )
            mem_t_str = f"{mem_t / 1e6:>8.1f}MB" if ok_t else f"{'--':>10}"
            mem_f_str = f"{mem_f / 1e6:>8.1f}MB" if ok_f else f"{'--':>10}"

            print(
                f"  {name:>25} | {f'{H}x{W}':>14} | {pad_pct:>5.1f}% | "
                f"{t_str} | {t_iqr:>14} | "
                f"{f_str} | {f_iqr:>14} | "
                f"{sp_str} | {sp_iqr:>12} | "
                f"{mem_t_str} | {mem_f_str}"
            )

            model_results.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "count": count,
                    "pad_pct": pad_pct,
                    "time_tile": time_t,
                    "time_full": time_f,
                    "p25_tile": p25_t,
                    "p75_tile": p75_t,
                    "p25_full": p25_f,
                    "p75_full": p75_f,
                    "mem_tile": mem_t,
                    "mem_full": mem_f,
                    "speedup": speedup_median if not np.isnan(speedup_median) else None,
                    "speedup_p25": speedup_p25 if not np.isnan(speedup_p25) else None,
                    "speedup_p75": speedup_p75 if not np.isnan(speedup_p75) else None,
                }
            )

            del G
            torch.cuda.empty_cache()

        # Aggregate
        total_tile = sum(
            r["time_tile"] * r["count"] for r in model_results if r["time_tile"] > 0
        )
        total_full = sum(
            r["time_full"] * r["count"] for r in model_results if r["time_full"] > 0
        )
        if total_tile > 0:
            print(
                f"\n  Total NS time (all layers): tile={total_tile:.1f}ms, full={total_full:.1f}ms, "
                f"speedup={total_full / total_tile:.2f}x\n"
            )

        all_model_results[model_name] = model_results

    data = {
        "config": {
            "models": models,
            "tile": tile,
            "ns_steps": ns_steps,
            "device": torch.cuda.get_device_name(device),
        },
        "results": all_model_results,
    }
    save_results(data, "real_shapes_bench")
    return data


# ---------------------------------------------------------------------------
# plot() -- dot plot from saved data
# ---------------------------------------------------------------------------
def plot(data, out_dir=None):
    """Generate per-layer speedup grouped bar chart (DualPath style)."""
    import numpy as np

    apply_style()

    config = data["config"]
    all_model_results = data["results"]
    models = config["models"]

    # Determine layer names from first model with valid results
    layer_names = []
    for model_results in all_model_results.values():
        names = [
            r["name"]
            for r in model_results
            if r["time_tile"] > 0 and r["time_full"] > 0
        ]
        if not layer_names:
            layer_names = names
    if not layer_names:
        return

    short_labels = [
        n.replace("self_attn.", "").replace("mlp.", "") for n in layer_names
    ]
    n_layers = len(layer_names)
    n_models = len(models)
    x = np.arange(n_layers)

    # Navy gradient: light → dark = small → large model
    _navy_gradient = ["#A8CBE0", "#4A7DB8", "#2B5EA7"]
    model_colors_list = [
        _navy_gradient[i] if i < len(_navy_gradient) else COLORS.GRAY
        for i in range(n_models)
    ]
    w = 0.24

    fig, ax = plt.subplots(figsize=(10, 3.5))

    for mi, model_name in enumerate(models):
        model_results = all_model_results.get(model_name, [])
        speedups = []
        for layer_name in layer_names:
            valid = [
                r
                for r in model_results
                if r["name"] == layer_name and r.get("speedup") is not None
            ]
            speedups.append(valid[0]["speedup"] if valid else 0)

        offset = (mi - (n_models - 1) / 2) * w
        bars = ax.bar(
            x + offset,
            speedups,
            width=w * 0.88,
            color=model_colors_list[mi],
            edgecolor="#444",
            linewidth=0.4,
            label=model_name,
            zorder=3,
        )
        # Value labels above bars
        for xi, s in zip(x + offset, speedups):
            if s > 0:
                ax.text(
                    xi,
                    s + 0.04,
                    f"{s:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#444",
                )

    ax.axhline(1.0, color="#BBBBBB", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Per-Layer Speedup", pad=25)
    ax.legend(
        fontsize=9,
        ncol=n_models,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        borderaxespad=0,
        frameon=False,
    )
    clean_spines(ax)

    plt.tight_layout()
    save_fig(fig, "real_shapes_bench", out_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real weight shape benchmark")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"],
    )
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip experiment, only plot from saved data",
    )
    args = parser.parse_args()

    if args.plot_only:
        from bench_utils import load_results

        plot(load_results("real_shapes_bench"))
    else:
        data = run(
            models=args.models,
            tile=args.tile,
            ns_steps=args.ns_steps,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        plot(data)
