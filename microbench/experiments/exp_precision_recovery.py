"""
Capacity Robustness to Low-Precision Newton-Schulz.

Question (tab:precision_recovery): does lowering only the Newton-Schulz inner-GEMM
precision change one-step recovery? The problem, minibatch, and gradient are built in
fp64 and held bit-identical across precision variants; only the GEMM precision inside
the NS iteration differs. fp8e4m3 uses native FP8 via torch._scaled_mm (per-tensor
dynamic scaling) and requires an sm89+ GPU (e.g. L40S / H100).

Setup: d=1024, N=4096, alpha=1.5, B/d=10, K=5 NS steps,
5 seeds. Recovery is invariant to (positive) eta, so a single eta is used. The operator
gap delta_F is the relative Frobenius distance of the produced one-step update to its
own fp64 reference, averaged over seeds.

Output: a LaTeX table (plots/precision_recovery_table.tex).

Usage:
  uv run python microbench/experiments/exp_precision_recovery.py
  uv run python microbench/experiments/exp_precision_recovery.py --plot-only
"""

import argparse
from collections import defaultdict

import numpy as np
import torch

# --- suite path bootstrap: make top-level utils importable when run standalone ---
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from assoc_mem import apply_one_step, build_gradient, evaluate_recovery, sample_problem  # noqa: E402
from bench_utils import load_results, save_results  # noqa: E402
from plot_style import get_plots_dir  # noqa: E402

# Table rows (Muon family) and precision columns, matching the reference table.
TABLE_ROWS = ["Muon", "HiMuon-512", "HiMuon-256", "HiMuon-128"]
PRECISIONS = ["fp64_ref", "fp32", "bf16", "fp8e4m3_native"]
PREC_LABEL = {
    "fp64_ref": "fp64",
    "fp32": "fp32",
    "bf16": "bf16",
    "fp8e4m3_native": "fp8e4m3",
}
METHOD_TEX = {
    "Muon": "Muon",
    "HiMuon-512": "HiMuon-$512$",
    "HiMuon-256": "HiMuon-$256$",
    "HiMuon-128": "HiMuon-$128$",
}


def _methods(tiles):
    return (
        [("Muon", "muon", None)]
        + [(f"HiMuon-{t}", "himuon", t) for t in sorted(tiles, reverse=True)]
        + [("SGD", "sgd", None)]
    )


def _rel_fro_gap(ref, cand):
    ref = ref.to(torch.float32)
    cand = cand.to(torch.float32)
    return float(
        torch.norm(ref - cand, p="fro").item()
        / (torch.norm(ref, p="fro").item() + 1e-12)
    )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------
def run(
    d,
    alpha,
    seeds,
    batch_scale,
    num_items,
    precisions,
    tiles,
    ns_steps,
    problem_dtype,
    device="cuda",
):
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"d={d}, alpha={alpha}, N={num_items}, B/d={batch_scale}, seeds={list(seeds)}, "
        f"precisions={precisions}, tiles={list(tiles)}\n"
    )
    methods = _methods(tiles)
    records = []
    for seed in seeds:
        problem = sample_problem(
            d=d,
            minimum_num_items=num_items,
            frequency_mode="power_law",
            alpha=alpha,
            seed=seed,
            device=device,
            dtype=problem_dtype,
        )
        bs = max(1, int(round(batch_scale * d)))
        gradient, _ = build_gradient(
            problem, gradient_mode="true_minibatch", batch_size=bs, seed=seed + bs
        )
        for label, opt, tile in methods:
            ref = apply_one_step(
                gradient,
                optimizer=opt,
                eta=1.0,
                ns_steps=ns_steps,
                tile_size=tile if tile is not None else 256,
                ns_precision="fp64_ref",
            )
            for precision in precisions:
                update = (
                    ref
                    if precision == "fp64_ref"
                    else apply_one_step(
                        gradient,
                        optimizer=opt,
                        eta=1.0,
                        ns_steps=ns_steps,
                        tile_size=tile if tile is not None else 256,
                        ns_precision=precision,
                    )
                )
                recovered, _ = evaluate_recovery(problem, update)
                records.append(
                    {
                        "seed": seed,
                        "label": label,
                        "tile_size": tile,
                        "ns_precision": precision,
                        "recovered_count": int(recovered.sum().item()),
                        "rel_fro_gap_to_fp64": _rel_fro_gap(ref, update),
                    }
                )
        print(f"seed={seed} done")

    data = {
        "config": {
            "d": d,
            "alpha": alpha,
            "seeds": list(seeds),
            "batch_scale": batch_scale,
            "num_items": num_items,
            "precisions": list(precisions),
            "tiles": list(tiles),
            "ns_steps": ns_steps,
            "problem_dtype": problem_dtype,
            "device": torch.cuda.get_device_name(device),
        },
        "records": records,
    }
    save_results(data, "precision_recovery")
    return data


# ---------------------------------------------------------------------------
# plot() + table
# ---------------------------------------------------------------------------
def _aggregate(records):
    rec = defaultdict(list)
    gap = defaultdict(list)
    for r in records:
        rec[(r["label"], r["ns_precision"])].append(r["recovered_count"])
        gap[(r["label"], r["ns_precision"])].append(r["rel_fro_gap_to_fp64"])
    rec_ms = {k: (float(np.mean(v)), float(np.std(v))) for k, v in rec.items()}
    gap_m = {k: float(np.mean(v)) for k, v in gap.items()}
    return rec_ms, gap_m


def _write_table(rec_ms, gap_m, config, out_dir=None):
    precs = [p for p in PRECISIONS if p in config["precisions"]]
    lines = [
        f"% Low-precision one-step recovery, generated on {config['device']} "
        f"(d={config['d']}, N={config['num_items']}, alpha={config['alpha']}, "
        f"B/d={config['batch_scale']}, K={config['ns_steps']}, {len(config['seeds'])} seeds).",
        r"\begin{tabular}{l" + "c" * len(precs) + " cc}",
        r"  \toprule",
        r"  & \multicolumn{%d}{c}{Recovered items} & \multicolumn{2}{c}{Op.\ gap $\delta_F$ to fp64} \\"
        % len(precs),
        r"  \cmidrule(lr){2-%d} \cmidrule(lr){%d-%d}"
        % (1 + len(precs), 2 + len(precs), 3 + len(precs)),
        r"  Method        & "
        + " & ".join(PREC_LABEL[p] for p in precs)
        + r" & bf16 & fp8e4m3 \\",
        r"  \midrule",
    ]
    for label in TABLE_ROWS:
        if (label, precs[0]) not in rec_ms:
            continue
        cells = " & ".join(
            rf"${rec_ms[(label, p)][0]:.1f}\!\pm\!{rec_ms[(label, p)][1]:.1f}$"
            for p in precs
        )
        bf16 = gap_m.get((label, "bf16"), float("nan"))
        fp8 = gap_m.get((label, "fp8e4m3_native"), float("nan"))
        lines.append(
            f"  {METHOD_TEX[label]:13s} & {cells} & {bf16:.3f} & {fp8:.3f} \\\\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}", ""]

    out = _os.path.join(out_dir or get_plots_dir(), "precision_recovery_table.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Table saved to {out}")


def plot(data, out_dir=None):
    """Write the LaTeX recovery table from aggregated records."""
    rec_ms, gap_m = _aggregate(data["records"])
    _write_table(rec_ms, gap_m, data["config"], out_dir)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Low-precision one-step recovery (NS inner-GEMM precision sweep)"
    )
    parser.add_argument("--d", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-scale", type=float, default=10.0)
    parser.add_argument("--num-items", type=int, default=4096)
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp64_ref", "fp32", "bf16", "fp8e4m3_native"],
    )
    parser.add_argument("--tiles", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument(
        "--problem-dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot(load_results("precision_recovery"))
    else:
        data = run(
            d=args.d,
            alpha=args.alpha,
            seeds=args.seeds,
            batch_scale=args.batch_scale,
            num_items=args.num_items,
            precisions=args.precisions,
            tiles=args.tiles,
            ns_steps=args.ns_steps,
            problem_dtype=args.problem_dtype,
        )
        plot(data)
