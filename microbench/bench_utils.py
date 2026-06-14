"""
Shared measurement utilities for the HiMuon benchmark suite.

Provides GPU timing, memory measurement, quality metrics, and JSON I/O
used by all individual benchmark modules.
"""

import json
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Timing & memory
# ---------------------------------------------------------------------------
def measure(fn, warmup=5, repeats=30):
    """Measure peak GPU memory and wall-clock time of fn().

    Returns (peak_memory_bytes, median_ms, p25_ms, p75_ms, raw_times, last_output).

    Memory: measured 3 times, minimum taken.
    """
    device = torch.device("cuda")

    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times_arr = np.array(times)
    median_time = float(np.median(times_arr))
    p25 = float(np.percentile(times_arr, 25))
    p75 = float(np.percentile(times_arr, 75))

    mem_samples = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated(device)
        out = fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        mem_samples.append(peak - baseline)
    peak_delta = min(mem_samples)

    return peak_delta, median_time, p25, p75, times_arr, out


def measure_paired_speedup(fn_a, fn_b, warmup=5, repeats=30):
    """True paired speedup: fn_a and fn_b timed back-to-back per trial.

    Returns (median_ratio, p25_ratio, p75_ratio) where ratio = t_a / t_b.
    """
    for _ in range(warmup):
        fn_a()
        fn_b()
    torch.cuda.synchronize()

    ratios = []
    for _ in range(repeats):
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)

        s1.record()
        fn_a()
        e1.record()
        torch.cuda.synchronize()

        s2.record()
        fn_b()
        e2.record()
        torch.cuda.synchronize()

        t_a = s1.elapsed_time(e1)
        t_b = s2.elapsed_time(e2)
        if t_b > 0:
            ratios.append(t_a / t_b)

    ratios_arr = np.array(ratios)
    return (
        float(np.median(ratios_arr)),
        float(np.percentile(ratios_arr, 25)),
        float(np.percentile(ratios_arr, 75)),
    )


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def measure_quality(output, reference=None):
    """Measure output quality via singular values and optional cosine similarity."""
    svs = torch.linalg.svdvals(output.float())
    result = {
        "sv_mean": svs.mean().item(),
        "sv_std": svs.std().item(),
        "sv_min": svs.min().item(),
        "sv_max": svs.max().item(),
    }
    if reference is not None:
        cos = (output.float() * reference.float()).sum() / (
            output.float().norm() * reference.float().norm() + 1e-8
        )
        result["cosine_sim"] = cos.item()
    return result


def cosine_sim(a, b):
    """Flat cosine similarity between two same-shape tensors."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float((a_f @ b_f) / (a_f.norm() * b_f.norm() + 1e-12))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def stats(values):
    """Return (median, p25, p75) for a list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return s[0], s[0], s[0]
    idx_med = 0.5 * (n - 1)
    idx_p25 = 0.25 * (n - 1)
    idx_p75 = 0.75 * (n - 1)

    def _interp(idx):
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] + frac * (s[hi] - s[lo])

    return _interp(idx_med), _interp(idx_p25), _interp(idx_p75)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def get_data_dir(script_file=None):
    """Return the data/ directory next to the calling script, creating it if needed."""
    if script_file is None:
        base = os.path.dirname(os.path.abspath(__file__))
    else:
        base = os.path.dirname(os.path.abspath(script_file))
    data_dir = os.path.join(base, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def save_results(data, name, script_file=None):
    """Save benchmark results dict to data/{name}.json."""
    data_dir = get_data_dir(script_file)
    path = os.path.join(data_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    print(f"Results saved to {path}")
    return path


def load_results(name, script_file=None):
    """Load benchmark results from data/{name}.json."""
    data_dir = get_data_dir(script_file)
    path = os.path.join(data_dir, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
