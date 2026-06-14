"""HiMuon tile-shape training-quality sweep.

Trains llama_220m for 300 steps under 9 tile-size configurations and
compares loss curves. REF is (512, 512) — the largest square that still
tiles on attn (640, 640). Candidate tiles span the fused-NS / 3-kernel
dispatch boundary (product = 16384 / 32768) plus two smaller squares.

Each run seeds deterministically, replays a preloaded batch list, and
asserts every Muon param routes through the bucketed tiled path (i.e.
no full-NS fallback) before step 2.

Usage (GPU node):
  uv run python microbench/tile_shape_quality.py
  uv run python microbench/tile_shape_quality.py --plot-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_wsd_schedule,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from bench_utils import load_results, save_results  # noqa: E402
from plot_style import apply_style, clean_spines, save_fig  # noqa: E402

from himuon.dataset import ToyDataset  # noqa: E402
from himuon.optim import get_optimizer  # noqa: E402
from himuon.utils import seed_everything  # noqa: E402

# HiMuon's _post_ns_per_param is @torch.compile(dynamic=True) and
# specialises per (R, C, Th, Tw, h, w, target_dtype). Across 9 tile shapes
# with 2 distinct param shapes each, the default cache_size_limit=8 is
# hit on the 4th run.
torch._dynamo.config.cache_size_limit = 128


MODEL = "llama_220m"
DATASET = "sample-10BT"
NUM_STEPS = 300
NUM_WARMUP = 20
NUM_DECAY = 60
MIN_LR_RATIO = 0.1
BATCH_SIZE = 8
MAX_LENGTH = 1024
SEED = 42
LR = 0.02

# llama_220m model config (hidden 640 / 12 layers / 10 heads, vocab 128256).
_LLAMA_220M_CONFIG = {
    "architectures": ["LLaMAForCausalLM"],
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "silu",
    "hidden_size": 640,
    "intermediate_size": 1708,
    "initializer_range": 0.02,
    "max_sequence_length": 1024,
    "model_type": "llama",
    "num_attention_heads": 10,
    "num_hidden_layers": 12,
    "rms_norm_eps": 1e-5,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.2",
    "use_cache": True,
    "vocab_size": 128256,
}


def build_model_and_tokenizer():
    """Build the llama_220m model + tokenizer."""
    config = AutoConfig.for_model(
        "llama",
        **{k: v for k, v in _LLAMA_220M_CONFIG.items() if k != "model_type"},
    )
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B", trust_remote_code=True, use_fast=False
    )
    return model, tokenizer


# Tile shapes (tile_h, tile_w). First entry is the reference square.
# Three squares + three fused boundary rect (area 16384) + three 3-kernel
# boundary rect (area 32768).
SHAPES: list[tuple[int, int]] = [
    (512, 512),  # REF, 3-kernel, area 262144
    (256, 256),  # square, 3-kernel, area 65536
    (128, 128),  # square, fused,    area 16384
    (16, 1024),  # rect,   fused,    area 16384
    (32, 512),  # rect,   fused,    area 16384
    (64, 256),  # rect,   fused,    area 16384
    (32, 1024),  # rect,   3-kernel, area 32768
    (64, 512),  # rect,   3-kernel, area 32768
    (128, 256),  # rect,   3-kernel, area 32768
    # Appended sweep 2: (M, N) ↔ (N, M) swaps for every rect above, to
    # probe whether tall (M > N) behaves like wide (M < N) at matched
    # area / aspect.
    (1024, 16),
    (512, 32),
    (256, 64),
    (1024, 32),
    (512, 64),
    (256, 128),
    # Sweep 3: tile_area ∈ {2^16, 2^17} — fills the gap between the
    # 3-kernel boundary (2^15) and the 2^18 reference (512, 512).
    # 2048-side tiles are excluded because llama_220m's max param dim
    # is 1708, which would force the full-NS fallback.
    # area 2^16 = 65,536 (tall/wide, no new square — (256,256) already present)
    (64, 1024),
    (128, 512),
    (1024, 64),
    (512, 128),
    # area 2^17 = 131,072 (no perfect square in the dim grid)
    (128, 1024),
    (256, 512),
    (1024, 128),
    (512, 256),
]


def shape_label(ts: tuple[int, int]) -> str:
    return f"({ts[0]},{ts[1]})"


# ---------------------------------------------------------------------------
# Tiling assertion
# ---------------------------------------------------------------------------
def bucket_summary(optimizer, muon_params) -> tuple[str, int]:
    """Return a one-line bucket summary plus the count of Muon params that
    are NOT in the bucketed plan (i.e. routed through the full-NS fallback
    in ``_step_muon_full_ns_and_dtensor``).
    """
    plan = optimizer.xlayer_plan()
    plan_ids = {pid for b in plan for pid in b["param_ids"]}
    missing = sum(1 for p in muon_params if id(p) not in plan_ids)
    parts = []
    for b in plan:
        th, tw, _full_ns, _dt = b["bucket_key"]
        parts.append(
            f"({th},{tw}) n_params={len(b['param_ids'])} "
            f"B_total={b['B_total']} n_calls={b['n_calls']}"
        )
    return " | ".join(parts), missing


# ---------------------------------------------------------------------------
# Data preloading — one deterministic batch list shared across all runs
# ---------------------------------------------------------------------------
def preload_batches(tokenizer, n_batches: int) -> list[dict]:
    ds = load_dataset(
        "HuggingFaceFW/fineweb", name=DATASET, split="train", streaming=True
    )
    ds = ds.shuffle(seed=SEED, buffer_size=10000)
    toy = ToyDataset(ds, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(toy, batch_size=BATCH_SIZE, num_workers=0)
    it = iter(loader)
    batches = []
    for _ in range(n_batches):
        batches.append(next(it))
    return batches


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one_tile(tile_size: tuple[int, int], batches: list[dict], device) -> dict:
    seed_everything(SEED)
    torch.cuda.empty_cache()

    model, _ = build_model_and_tokenizer()
    model.to(device)
    model.train()

    args = SimpleNamespace(
        lr=LR,
        tile_size=tile_size,
        cuda_graph=False,  # disabled: quality test, no graph noise
    )
    optimizer = get_optimizer("himuon", model, args)
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=NUM_WARMUP,
        num_stable_steps=0,
        num_decay_steps=NUM_DECAY,
        num_training_steps=NUM_STEPS,
        min_lr_ratio=MIN_LR_RATIO,
        decay_type="cosine",
        warmup_type="linear",
    )
    muon_params = [
        p for g in optimizer.param_groups if g.get("use_muon") for p in g["params"]
    ]

    losses: list[float] = []
    bucket_log = None
    t_start = time.time()
    for step_idx, batch in enumerate(batches):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().item()))

        if step_idx == 0:
            summary, missing = bucket_summary(optimizer, muon_params)
            bucket_log = summary
            assert missing == 0, (
                f"tile {tile_size}: {missing} muon params went to full-NS "
                f"fallback; aborting"
            )
            print(f"  plan: {summary}")

        if step_idx + 1 >= NUM_STEPS:
            break
    elapsed = time.time() - t_start

    # Free model + optimizer before the next run
    del optimizer, scheduler, model
    torch.cuda.empty_cache()

    return {
        "tile_size": list(tile_size),
        "losses": losses,
        "bucket_log": bucket_log,
        "elapsed_sec": elapsed,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run() -> dict:
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "this bench requires CUDA"

    # Load tokenizer once (model is reinstantiated per run so that init
    # state matches across runs under the same seed).
    _, tokenizer = build_model_and_tokenizer()
    print(f"Preloading {NUM_STEPS} batches from FineWeb ({DATASET})...")
    t0 = time.time()
    batches = preload_batches(tokenizer, NUM_STEPS)
    print(f"  preloaded in {time.time() - t0:.1f}s\n")

    print(
        f"Sweep: {MODEL}, {NUM_STEPS} steps, bs={BATCH_SIZE}, "
        f"seq={MAX_LENGTH}, lr={LR}, seed={SEED}"
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Resume from existing JSON if present — skip tile_sizes already done.
    try:
        results = load_results("tile_shape_quality")
        done = {tuple(r["tile_size"]) for r in results.get("runs", [])}
        if done:
            print(f"Resuming: {len(done)} tile_sizes already done, skipping\n")
    except FileNotFoundError:
        results = {
            "config": {
                "model": MODEL,
                "dataset": DATASET,
                "num_steps": NUM_STEPS,
                "num_warmup": NUM_WARMUP,
                "batch_size": BATCH_SIZE,
                "max_length": MAX_LENGTH,
                "seed": SEED,
                "lr": LR,
                "device": torch.cuda.get_device_name(0),
            },
            "runs": [],
        }
        done = set()

    for ts in SHAPES:
        if tuple(ts) in done:
            continue
        print(f"--- tile_size = {shape_label(ts)} ---")
        entry = run_one_tile(ts, batches, device)
        print(
            f"  step0 loss={entry['losses'][0]:.3f}  "
            f"step{NUM_STEPS - 1} loss={entry['losses'][-1]:.3f}  "
            f"elapsed={entry['elapsed_sec']:.1f}s"
        )
        results["runs"].append(entry)
        # Persist after every run so a mid-sweep crash leaves partial
        # results on disk (plot handles len(runs) < len(SHAPES) fine).
        save_results(results, "tile_shape_quality")
        print()

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _tile_color(ts: tuple[int, int], ref: tuple[int, int]) -> tuple:
    """Color by (path, aspect); linestyle distinguishes wide (M<N) vs tall (M>N)."""
    if tuple(ts) == tuple(ref):
        return "#2B5EA7", 2.6, 1.0, "-"  # REF: thick navy solid
    area = ts[0] * ts[1]
    fused = area <= 16384
    aspect = max(ts) / min(ts)
    if fused:
        palette = ["#2B7A4B", "#4A9060", "#6BA570", "#8EBA85"]
    else:
        palette = ["#C4541A", "#D4621A", "#E0884A", "#E8A875"]
    idx = min(int(np.log2(aspect)), len(palette) - 1) if aspect > 1 else 0
    # Solid for square / wide (M≤N); dashed for tall (M>N).
    linestyle = "--" if ts[0] > ts[1] else "-"
    return palette[idx], 1.4, 0.95, linestyle


def _rolling_mean(x: np.ndarray, window: int = 25) -> np.ndarray:
    """Centered rolling mean with edge padding. Window must be odd."""
    assert window % 2 == 1, "window must be odd for centered alignment"
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def plot(data: dict, out_dir: str | None = None) -> None:
    apply_style()
    runs = data["runs"]
    ref_ts = tuple(runs[0]["tile_size"])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )

    # Left: EMA-smoothed loss vs step, log-y
    ax = axes[0]
    for r in runs:
        ts = tuple(r["tile_size"])
        losses = np.asarray(r["losses"])
        smoothed = _rolling_mean(losses, window=75)
        color, lw, alpha, ls = _tile_color(ts, ref_ts)
        ax.plot(
            np.arange(1, len(losses) + 1),
            smoothed,
            label=shape_label(ts),
            color=color,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (rolling mean, w=75, log)")
    ax.set_title("Loss vs step", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncols=2, fontsize=8.5)
    clean_spines(ax)

    # Middle: EMA-smoothed loss vs cumulative wall-clock time, log-y.
    # Per-run timing data: we only have total elapsed, so distribute
    # uniformly across steps (ok for relative comparison; steady-state
    # per-step time is ~constant after the first few steps).
    ax = axes[1]
    for r in runs:
        ts = tuple(r["tile_size"])
        losses = np.asarray(r["losses"])
        smoothed = _rolling_mean(losses, window=75)
        elapsed = float(r.get("elapsed_sec", 0.0))
        t_per_step = elapsed / len(losses) if len(losses) > 0 else 0.0
        times = np.arange(1, len(losses) + 1) * t_per_step
        color, lw, alpha, ls = _tile_color(ts, ref_ts)
        ax.plot(
            times,
            smoothed,
            label=shape_label(ts),
            color=color,
            linewidth=lw,
            alpha=alpha,
            linestyle=ls,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Loss (rolling mean, w=75, log)")
    ax.set_title("Loss vs wall-clock time", loc="left", fontweight="bold")
    ax.legend(loc="upper right", ncols=2, fontsize=8.5)
    clean_spines(ax)

    # Right: final-100-step mean loss vs tile area, grouped by path.
    ax = axes[2]
    by_label = []
    for r in runs:
        ts = tuple(r["tile_size"])
        losses = np.asarray(r["losses"])
        final = float(np.mean(losses[-100:]))
        area = ts[0] * ts[1]
        by_label.append((ts, area, final))
    ref_final = next(f for ts, _, f in by_label if ts == ref_ts)

    for ts, area, final in by_label:
        color, _, _, _ = _tile_color(ts, ref_ts)
        if ts[0] == ts[1]:
            marker = "s"  # square
        elif ts[0] < ts[1]:
            marker = "o"  # wide (M<N)
        else:
            marker = "D"  # tall (M>N)
        ax.scatter(
            area,
            final,
            s=80 if ts == ref_ts else 60,
            color=color,
            marker=marker,
            edgecolor="#333333",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            shape_label(ts),
            (area, final),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )
    ax.axhline(ref_final, color="#2B5EA7", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xscale("log", base=2)
    # Tighten x to the data (2^14 .. 2^18) with a small buffer each side.
    xmin = min(a for _, a, _ in by_label)
    xmax = max(a for _, a, _ in by_label)
    ax.set_xlim(xmin / 1.5, xmax * 1.5)
    ax.set_xlabel("Tile area (log₂)")
    ax.set_ylabel("Loss (mean of final 100 steps)")
    ax.set_title("Final loss vs tile area", loc="left", fontweight="bold")

    from matplotlib.lines import Line2D

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="#333333",
            markersize=8,
            label="square",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="#333333",
            markersize=8,
            label="wide (M<N)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#888888",
            markeredgecolor="#333333",
            markersize=8,
            label="tall (M>N)",
        ),
    ]
    ax.legend(handles=marker_handles, loc="upper right", fontsize=9)
    clean_spines(ax)

    save_fig(fig, "tile_shape_quality", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        data = load_results("tile_shape_quality")
        plot(data)
    else:
        data = run()
        plot(data)
