from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import torch
import torch.nn.functional as F


NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_PRECISIONS = ("fp64_ref", "fp32", "bf16", "fp8e4m3_sim", "fp8e4m3_native")
DEFAULT_SCORE_CHUNK = 256
DEFAULT_TOPK = (32, 128, 512)
DEFAULT_FREQUENCY_RANK_FRACTIONS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
BYTES_PER_GIB = float(1 << 30)


@dataclass(frozen=True)
class ProblemSpec:
    d: int
    num_items: int
    frequency_mode: str
    alpha: float
    seed: int
    device: str
    dtype: str


@dataclass
class ProblemInstance:
    spec: ProblemSpec
    u: torch.Tensor
    v: torch.Tensor
    p: torch.Tensor
    mean_u: torch.Tensor

    @property
    def d(self) -> int:
        return self.spec.d

    @property
    def num_items(self) -> int:
        return self.spec.num_items


def resolve_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device(device)


def resolve_dtype(dtype: str | torch.dtype = "float32") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def default_num_items(
    d: int,
    num_items: int | None = None,
    multiplier: int = 4,
    minimum: int = 512,
) -> int:
    if num_items is not None:
        return int(num_items)
    return int(max(minimum, multiplier * d))


def _dtype_bytes(dtype: str | torch.dtype = "float32") -> int:
    return torch.empty((), dtype=resolve_dtype(dtype)).element_size()


def estimate_peak_memory_bytes(
    *,
    d: int,
    num_items: int,
    dtype: str | torch.dtype = "float32",
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
    tile_size: int | tuple[int, int] | None = None,
    num_weight_matrices: int = 1,
) -> int:
    """Coarse upper bound for peak tensor RAM during one benchmark configuration."""
    working_bytes = max(_dtype_bytes(dtype), 4)
    chunk = min(score_chunk_size, num_items)

    problem_bytes = (2 * num_items * d + 2 * num_items + d) * working_bytes
    gradient_bytes = d * d * 4
    weights_bytes = num_weight_matrices * d * d * 4
    score_bytes = (num_items * chunk + 2 * chunk * d + 2 * num_items) * 4

    if tile_size is None:
        operator_bytes = 2 * d * d * 4
    else:
        tile_h, tile_w = _normalize_tile_size(tile_size)
        pad_h = (tile_h - d % tile_h) % tile_h
        pad_w = (tile_w - d % tile_w) % tile_w
        padded_h = d + pad_h
        padded_w = d + pad_w
        operator_bytes = 2 * padded_h * padded_w * 4

    return int(problem_bytes + gradient_bytes + weights_bytes + score_bytes + operator_bytes)


def estimate_peak_memory_gib(**kwargs: Any) -> float:
    return estimate_peak_memory_bytes(**kwargs) / BYTES_PER_GIB


def assert_memory_budget(
    *,
    d: int,
    num_items: int,
    context: str,
    memory_budget_gib: float | None = None,
    dtype: str | torch.dtype = "float32",
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
    tile_size: int | tuple[int, int] | None = None,
    num_weight_matrices: int = 1,
) -> dict[str, Any]:
    estimated_peak_ram_gib = estimate_peak_memory_gib(
        d=d,
        num_items=num_items,
        dtype=dtype,
        score_chunk_size=score_chunk_size,
        tile_size=tile_size,
        num_weight_matrices=num_weight_matrices,
    )
    if memory_budget_gib is not None and estimated_peak_ram_gib > memory_budget_gib:
        raise ValueError(
            f"{context}: estimated peak RAM {estimated_peak_ram_gib:.2f} GiB exceeds "
            f"budget {memory_budget_gib:.2f} GiB. Reduce d, num_items, or tile size."
        )
    return {
        "estimated_peak_ram_gib": estimated_peak_ram_gib,
        "memory_budget_gib": memory_budget_gib,
    }


def make_frequencies(
    num_items: int,
    mode: str = "power_law",
    alpha: float = 1.5,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    device = device or torch.device("cpu")
    if mode == "uniform":
        return torch.full((num_items,), 1.0 / num_items, device=device, dtype=dtype)
    if mode == "power_law":
        ranks = torch.arange(1, num_items + 1, device=device, dtype=dtype)
        weights = ranks.pow(-alpha)
        return weights / weights.sum()
    raise ValueError(f"Unsupported frequency mode: {mode}")


def sample_problem(
    *,
    d: int,
    num_items: int | None = None,
    num_items_multiplier: int = 4,
    minimum_num_items: int = 512,
    frequency_mode: str = "power_law",
    alpha: float = 1.5,
    seed: int = 0,
    device: str | None = None,
    dtype: str | torch.dtype = "float32",
) -> ProblemInstance:
    dev = resolve_device(device)
    dt = resolve_dtype(dtype)
    n = default_num_items(d, num_items=num_items, multiplier=num_items_multiplier, minimum=minimum_num_items)

    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)

    u = torch.randn((n, d), generator=generator, device=dev, dtype=dt) / math.sqrt(d)
    v = torch.randn((n, d), generator=generator, device=dev, dtype=dt) / math.sqrt(d)
    p = make_frequencies(n, mode=frequency_mode, alpha=alpha, device=dev, dtype=dt)
    mean_u = u.mean(dim=0)

    spec = ProblemSpec(
        d=d,
        num_items=n,
        frequency_mode=frequency_mode,
        alpha=alpha,
        seed=seed,
        device=str(dev),
        dtype=str(dt).replace("torch.", ""),
    )
    return ProblemInstance(spec=spec, u=u, v=v, p=p, mean_u=mean_u)


def sample_empirical_frequencies(
    p: torch.Tensor,
    batch_size: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=p.device)
    generator.manual_seed(seed)
    idx = torch.multinomial(p, num_samples=batch_size, replacement=True, generator=generator)
    counts = torch.bincount(idx, minlength=p.numel()).to(p.dtype)
    return counts / batch_size


def _true_gradient_from_weight(
    problem: ProblemInstance,
    *,
    freqs: torch.Tensor,
    weight: torch.Tensor,
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
) -> torch.Tensor:
    u = problem.u.to(torch.float32)
    v = problem.v.to(torch.float32)
    w = weight.to(torch.float32)
    grad = torch.zeros((problem.d, problem.d), dtype=torch.float32, device=w.device)

    for start in range(0, problem.num_items, score_chunk_size):
        end = min(problem.num_items, start + score_chunk_size)
        v_chunk = v[start:end]
        logits = u @ (v_chunk @ w.transpose(0, 1)).transpose(0, 1)
        logits = logits - logits.max(dim=0, keepdim=True).values
        probs = torch.softmax(logits, dim=0)
        expected_u = probs.transpose(0, 1) @ u
        centered_u = u[start:end] - expected_u
        grad = grad + (freqs[start:end, None].to(torch.float32) * centered_u).transpose(0, 1) @ v_chunk

    return grad


def build_gradient(
    problem: ProblemInstance,
    *,
    gradient_mode: str,
    batch_size: int | None = None,
    seed: int | None = None,
    weight: torch.Tensor | None = None,
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if gradient_mode == "approx_population":
        freqs = problem.p
        centered_u = problem.u
    elif gradient_mode == "true_population":
        freqs = problem.p
        centered_u = problem.u - problem.mean_u
    elif gradient_mode in {"approx_minibatch", "true_minibatch"}:
        if batch_size is None:
            raise ValueError(f"batch_size is required for {gradient_mode}")
        if seed is None:
            raise ValueError(f"seed is required for {gradient_mode}")
        freqs = sample_empirical_frequencies(problem.p, batch_size=batch_size, seed=seed)
        centered_u = problem.u if gradient_mode == "approx_minibatch" else (problem.u - problem.mean_u)
    else:
        raise ValueError(f"Unsupported gradient mode: {gradient_mode}")

    if gradient_mode.startswith("true_") and weight is not None:
        grad = _true_gradient_from_weight(problem, freqs=freqs, weight=weight, score_chunk_size=score_chunk_size)
    else:
        grad = (freqs[:, None] * centered_u).transpose(0, 1) @ problem.v
    meta = {
        "gradient_mode": gradient_mode,
        "batch_size": batch_size,
        "gradient_seed": seed,
        "gradient_depends_on_weight": bool(gradient_mode.startswith("true_") and weight is not None),
        "effective_support": int((freqs > 0).sum().item()),
        "freq_max": float(freqs.max().item()),
        "freq_min_nonzero": float(freqs[freqs > 0].min().item()),
    }
    return grad, meta


def _manual_fp8e4m3_quantize(x: torch.Tensor) -> torch.Tensor:
    """Portable fallback for approximate E4M3 finite-only quantization."""
    x32 = x.to(torch.float32)
    sign = torch.sign(x32)
    ax = x32.abs()
    min_subnormal = 2.0 ** -9
    nonzero = ax >= min_subnormal
    safe_ax = ax.clamp(min=min_subnormal)
    exponent = torch.floor(torch.log2(safe_ax)).clamp(-6, 8)
    step = torch.pow(torch.full_like(exponent, 2.0), exponent - 3)
    quantized = torch.round(safe_ax / step) * step
    quantized = torch.clamp(quantized, max=448.0)
    quantized = torch.where(nonzero, quantized, torch.zeros_like(quantized))
    return sign * quantized


def fp8e4m3_dynamic_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-tensor dynamic-scaled fp8e4m3 storage, returning float32."""
    x32 = x.to(torch.float32)
    max_abs = torch.max(x32.abs())
    if not torch.isfinite(max_abs) or float(max_abs.item()) == 0.0:
        return torch.zeros_like(x32)

    scale = max_abs / 448.0
    scaled = torch.clamp(x32 / scale, min=-448.0, max=448.0)
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale
        except RuntimeError:
            pass
    return _manual_fp8e4m3_quantize(scaled) * scale


def _to_column_major(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1).contiguous().transpose(-2, -1)


def _fp8e4m3_scaled_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not left.is_cuda or not right.is_cuda:
        raise RuntimeError("fp8e4m3_native requires CUDA tensors.")
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("fp8e4m3_native requires torch._scaled_mm and torch.float8_e4m3fn.")

    left32 = left.to(torch.float32).contiguous()
    right32 = right.to(torch.float32)
    left_scale = torch.clamp(torch.max(left32.abs()) / 448.0, min=torch.finfo(torch.float32).tiny)
    right_scale = torch.clamp(torch.max(right32.abs()) / 448.0, min=torch.finfo(torch.float32).tiny)

    left_fp8 = torch.clamp(left32 / left_scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
    right_scaled = torch.clamp(right32 / right_scale, min=-448.0, max=448.0)
    right_fp8 = _to_column_major(right_scaled).to(torch.float8_e4m3fn)

    return torch._scaled_mm(
        left_fp8,
        right_fp8,
        scale_a=left_scale,
        scale_b=right_scale,
        out_dtype=torch.float32,
    )


def _round_for_ns_precision(x: torch.Tensor, ns_precision: str) -> torch.Tensor:
    if ns_precision == "fp64_ref":
        return x.to(torch.float64)
    if ns_precision == "fp32":
        return x.to(torch.float32)
    if ns_precision == "bf16":
        return x.to(torch.float32)
    if ns_precision == "fp8e4m3_sim":
        return fp8e4m3_dynamic_roundtrip(x)
    if ns_precision == "fp8e4m3_native":
        return x.to(torch.float32)
    raise ValueError(f"Unsupported NS precision: {ns_precision}")


def _matmul_for_ns(left: torch.Tensor, right: torch.Tensor, ns_precision: str) -> torch.Tensor:
    if ns_precision == "bf16":
        return (left.to(torch.bfloat16) @ right.to(torch.bfloat16)).to(torch.float32)
    if ns_precision == "fp8e4m3_native":
        return _fp8e4m3_scaled_mm(left, right)
    return left @ right


def quintic_ns_operator(
    g: torch.Tensor,
    steps: int = 5,
    *,
    ns_precision: str = "fp32",
) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError(f"Expected a rank-2 matrix, got shape {tuple(g.shape)}")
    if ns_precision not in NS_PRECISIONS:
        raise ValueError(f"Unsupported NS precision: {ns_precision}")
    a, b, c = NS_COEFFICIENTS
    x = _round_for_ns_precision(g, ns_precision)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.transpose(0, 1)
        transposed = True
    x = x / (x.norm() + 1e-7)
    x = _round_for_ns_precision(x, ns_precision)
    for _ in range(steps):
        a_mat = _matmul_for_ns(x, x.transpose(0, 1), ns_precision)
        a_mat = _round_for_ns_precision(a_mat, ns_precision)
        b_mat = b * a_mat + c * _matmul_for_ns(a_mat, a_mat, ns_precision)
        b_mat = _round_for_ns_precision(b_mat, ns_precision)
        x = a * x + _matmul_for_ns(b_mat, x, ns_precision)
        x = _round_for_ns_precision(x, ns_precision)
    if transposed:
        x = x.transpose(0, 1)
    return x.to(torch.float32)


def _normalize_tile_size(tile_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(tile_size, int):
        return tile_size, tile_size
    return int(tile_size[0]), int(tile_size[1])


def _descending_norm_permutation(g: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 0:
        norms = torch.linalg.vector_norm(g.to(torch.float32), ord=2, dim=1)
    elif dim == 1:
        norms = torch.linalg.vector_norm(g.to(torch.float32), ord=2, dim=0)
    else:
        raise ValueError(f"Unsupported dimension {dim} for permutation.")
    return torch.argsort(norms, descending=True, stable=True)


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inverse


def tile_matrix(
    tensor: torch.Tensor,
    tile_size: int | tuple[int, int],
    *,
    shift: tuple[int, int] = (0, 0),
) -> tuple[torch.Tensor, dict[str, Any]]:
    tile_h, tile_w = _normalize_tile_size(tile_size)
    shifted = torch.roll(tensor, shifts=(-shift[0], -shift[1]), dims=(0, 1))

    h, w = shifted.shape
    pad_h = (tile_h - h % tile_h) % tile_h
    pad_w = (tile_w - w % tile_w) % tile_w
    padded = F.pad(shifted, (0, pad_w, 0, pad_h))
    h_pad, w_pad = padded.shape
    rows, cols = h_pad // tile_h, w_pad // tile_w
    tiled = padded.view(rows, tile_h, cols, tile_w).permute(0, 2, 1, 3).contiguous()
    info = {
        "orig_h": h,
        "orig_w": w,
        "pad_h": pad_h,
        "pad_w": pad_w,
        "tile_h": tile_h,
        "tile_w": tile_w,
        "rows": rows,
        "cols": cols,
        "shift": shift,
    }
    return tiled, info


def untile_matrix(tiled: torch.Tensor, info: dict[str, Any]) -> torch.Tensor:
    restored = (
        tiled.permute(0, 2, 1, 3)
        .contiguous()
        .view(info["rows"] * info["tile_h"], info["cols"] * info["tile_w"])
    )
    restored = restored[: info["orig_h"], : info["orig_w"]]
    shift = info["shift"]
    if shift != (0, 0):
        restored = torch.roll(restored, shifts=(shift[0], shift[1]), dims=(0, 1))
    return restored


def _compute_tile_scale(grid: torch.Tensor, mode: str) -> torch.Tensor:
    rows, cols, tile_h, tile_w = grid.shape
    if mode == "tile":
        return grid.reshape(rows, cols, -1).norm(dim=2) / math.sqrt(tile_h * tile_w)
    if mode == "tile_row":
        return grid.reshape(rows, -1).norm(dim=1) / math.sqrt(cols * tile_h * tile_w)
    raise ValueError(f"Unsupported tile scale mode: {mode}")


def _apply_normalization(
    orth_tiles: torch.Tensor,
    raw_tiles: torch.Tensor,
    normalization: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    if normalization == "none":
        return orth_tiles

    if normalization in {"tile_raw", "tile_divide"}:
        scale = _compute_tile_scale(raw_tiles, "tile").unsqueeze(-1).unsqueeze(-1)
    elif normalization in {"tile_row_raw", "tile_row_divide"}:
        scale = _compute_tile_scale(raw_tiles, "tile_row").view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")

    if normalization.endswith("_raw"):
        return orth_tiles * scale
    return orth_tiles / (scale + eps)


def himuon_operator(
    g: torch.Tensor,
    *,
    tile_size: int | tuple[int, int] = 256,
    normalization: str = "none",
    steps: int = 5,
    shift: tuple[int, int] = (0, 0),
    ns_precision: str = "fp32",
) -> torch.Tensor:
    tile_h, tile_w = _normalize_tile_size(tile_size)
    if g.numel() <= tile_h * tile_w:
        return quintic_ns_operator(g, steps=steps, ns_precision=ns_precision)

    tiles, info = tile_matrix(g, (tile_h, tile_w), shift=shift)
    orth_tiles = torch.empty_like(tiles, dtype=torch.float32)
    for r in range(info["rows"]):
        for c in range(info["cols"]):
            orth_tiles[r, c] = quintic_ns_operator(
                tiles[r, c],
                steps=steps,
                ns_precision=ns_precision,
            )
    orth_tiles = _apply_normalization(orth_tiles, tiles.to(torch.float32), normalization)
    return untile_matrix(orth_tiles, info)


def reordered_himuon_operator(
    g: torch.Tensor,
    *,
    tile_size: int | tuple[int, int] = 256,
    normalization: str = "none",
    steps: int = 5,
    shift: tuple[int, int] = (0, 0),
    ns_precision: str = "fp32",
) -> torch.Tensor:
    row_perm = _descending_norm_permutation(g, dim=0)
    col_perm = _descending_norm_permutation(g, dim=1)
    row_inv = _inverse_permutation(row_perm)
    col_inv = _inverse_permutation(col_perm)

    reordered = g[row_perm][:, col_perm]
    reordered_update = himuon_operator(
        reordered,
        tile_size=tile_size,
        normalization=normalization,
        steps=steps,
        shift=shift,
        ns_precision=ns_precision,
    )
    return reordered_update[row_inv][:, col_inv]


def apply_one_step(
    g: torch.Tensor,
    *,
    optimizer: str,
    eta: float,
    ns_steps: int = 5,
    tile_size: int | tuple[int, int] = 256,
    normalization: str = "none",
    shift: tuple[int, int] = (0, 0),
    ns_precision: str = "fp32",
) -> torch.Tensor:
    if optimizer == "sgd":
        update = g.to(torch.float32)
    elif optimizer == "muon":
        update = quintic_ns_operator(g, steps=ns_steps, ns_precision=ns_precision)
    elif optimizer == "himuon":
        update = himuon_operator(
            g,
            tile_size=tile_size,
            normalization=normalization,
            steps=ns_steps,
            shift=shift,
            ns_precision=ns_precision,
        )
    elif optimizer == "reordered_himuon":
        update = reordered_himuon_operator(
            g,
            tile_size=tile_size,
            normalization=normalization,
            steps=ns_steps,
            shift=shift,
            ns_precision=ns_precision,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    return eta * update


def top_singular_values(matrix: torch.Tensor, topk: int = 6) -> list[float]:
    vals = torch.linalg.svdvals(matrix.to(torch.float32))
    return [float(v) for v in vals[: min(topk, vals.numel())].tolist()]


def _format_percent_label(fraction: float) -> str:
    value = 100.0 * fraction
    return f"{int(value)}%" if float(value).is_integer() else f"{value:g}%"


def _fraction_to_rank_bound(num_items: int, fraction: float, previous: int) -> int:
    if fraction <= 0.0:
        return previous
    return min(num_items, max(previous, int(math.ceil(fraction * num_items))))


def _frequency_rank_recovery_summary(
    problem: ProblemInstance,
    recovered: torch.Tensor,
    *,
    rank_fractions: Iterable[float] = DEFAULT_FREQUENCY_RANK_FRACTIONS,
) -> dict[str, Any]:
    fractions = tuple(float(value) for value in rank_fractions)
    if not fractions or fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("rank_fractions must start at 0.0 and end at 1.0")

    p_cpu = problem.p.to(torch.float32).cpu()
    recovered_cpu = recovered.to(torch.float32).cpu()
    order = torch.argsort(p_cpu, descending=True)
    p_sorted = p_cpu[order]
    recovered_sorted = recovered_cpu[order]

    total_probability_mass = float(p_cpu.sum().item())
    total_recovered_probability_mass = float((p_cpu * recovered_cpu).sum().item())

    rank_bounds = [0]
    previous = 0
    for fraction in fractions[1:]:
        previous = _fraction_to_rank_bound(problem.num_items, fraction, previous)
        rank_bounds.append(previous)

    bins = []
    prefixes = []
    for left_fraction, right_fraction, start, end in zip(
        fractions[:-1],
        fractions[1:],
        rank_bounds[:-1],
        rank_bounds[1:],
        strict=True,
    ):
        p_slice = p_sorted[start:end]
        recovered_slice = recovered_sorted[start:end]
        item_count = int(end - start)
        recovered_count = int(recovered_slice.sum().item()) if item_count else 0
        bin_probability_mass = float(p_slice.sum().item()) if item_count else 0.0
        recovered_probability_mass = (
            float((p_slice * recovered_slice).sum().item()) if item_count else 0.0
        )
        bins.append(
            {
                "label": f"{_format_percent_label(left_fraction)}-{_format_percent_label(right_fraction)}",
                "start_fraction": left_fraction,
                "end_fraction": right_fraction,
                "start_index": int(start),
                "end_index": int(end),
                "item_count": item_count,
                "recovered_count": recovered_count,
                "recovered_fraction_within_bin": (
                    float(recovered_count / item_count) if item_count else None
                ),
                "bin_probability_mass": bin_probability_mass,
                "recovered_probability_mass": recovered_probability_mass,
                "recovered_probability_mass_fraction_within_bin": (
                    float(recovered_probability_mass / bin_probability_mass)
                    if bin_probability_mass > 0.0
                    else None
                ),
            }
        )

        prefix_p = p_sorted[:end]
        prefix_recovered = recovered_sorted[:end]
        prefix_probability_mass = float(prefix_p.sum().item()) if end else 0.0
        prefix_recovered_probability_mass = (
            float((prefix_p * prefix_recovered).sum().item()) if end else 0.0
        )
        prefix_recovered_count = int(prefix_recovered.sum().item()) if end else 0
        prefixes.append(
            {
                "label": f"top_{_format_percent_label(right_fraction)}",
                "end_fraction": right_fraction,
                "end_index": int(end),
                "item_count": int(end),
                "recovered_count": prefix_recovered_count,
                "recovered_fraction_within_prefix": (
                    float(prefix_recovered_count / end) if end else None
                ),
                "prefix_probability_mass": prefix_probability_mass,
                "recovered_probability_mass": prefix_recovered_probability_mass,
                "recovered_probability_mass_fraction_within_prefix": (
                    float(prefix_recovered_probability_mass / prefix_probability_mass)
                    if prefix_probability_mass > 0.0
                    else None
                ),
            }
        )

    return {
        "rank_order": "descending_probability",
        "num_items": problem.num_items,
        "rank_fraction_edges": list(fractions),
        "recovered_probability_mass": total_recovered_probability_mass,
        "total_probability_mass": total_probability_mass,
        "bins": bins,
        "prefixes": prefixes,
    }


def evaluate_recovery(
    problem: ProblemInstance,
    weight: torch.Tensor,
    *,
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = problem.u.to(torch.float32)
    v = problem.v.to(torch.float32)
    w = weight.to(torch.float32)
    n = problem.num_items

    recovered = torch.zeros(n, dtype=torch.bool, device=w.device)
    margins = torch.empty(n, dtype=torch.float32, device=w.device)

    for start in range(0, n, score_chunk_size):
        end = min(n, start + score_chunk_size)
        idx = torch.arange(start, end, device=w.device)
        proj = v[start:end] @ w.transpose(0, 1)
        scores = u @ proj.transpose(0, 1)
        diag_scores = scores[idx, torch.arange(end - start, device=w.device)]
        top_vals, top_idx = torch.topk(scores, k=min(2, scores.size(0)), dim=0)
        top_idx0 = top_idx[0]
        max_offdiag = torch.where(
            top_idx0 == idx,
            top_vals[1] if top_vals.size(0) > 1 else torch.full_like(diag_scores, float("-inf")),
            top_vals[0],
        )
        recovered[start:end] = diag_scores > max_offdiag
        margins[start:end] = diag_scores - max_offdiag

    return recovered.cpu(), margins.cpu()


def topk_recovery_curve(
    problem: ProblemInstance,
    weight: torch.Tensor,
    *,
    max_k: int | None = None,
    order: str = "frequency_desc",
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
) -> dict[str, Any]:
    recovered_cpu, _ = evaluate_recovery(problem, weight, score_chunk_size=score_chunk_size)
    if order == "frequency_desc":
        order_idx = torch.argsort(problem.p.to(torch.float32).cpu(), descending=True)
    elif order == "index":
        order_idx = torch.arange(problem.num_items, dtype=torch.int64)
    else:
        raise ValueError(f"Unsupported top-k curve order: {order}")

    recovered_sorted = recovered_cpu.to(torch.float32)[order_idx]
    limit = problem.num_items if max_k is None else min(problem.num_items, int(max_k))
    prefix_counts = torch.cumsum(recovered_sorted[:limit], dim=0)
    ks = torch.arange(1, limit + 1, dtype=torch.float32)
    prefix_fraction = prefix_counts / ks
    return {
        "order": order,
        "max_k": int(limit),
        "k_values": [int(v) for v in range(1, limit + 1)],
        "topk_recovered_count": [float(v) for v in prefix_counts.tolist()],
        "topk_recovered_fraction": [float(v) for v in prefix_fraction.tolist()],
    }


def _topk_prefix_metrics(
    recovered: torch.Tensor,
    *,
    topk_values: Iterable[int],
    prefix_name: str,
    order_idx: torch.Tensor | None = None,
) -> dict[str, Any]:
    recovered_cpu = recovered.cpu()
    if order_idx is not None:
        recovered_cpu = recovered_cpu[order_idx.cpu()]
    n = recovered_cpu.numel()

    metrics: dict[str, Any] = {}
    requested_topk = tuple(dict.fromkeys(int(k) for k in topk_values))
    for k in requested_topk:
        effective_k = min(k, n)
        prefix_recovered = recovered_cpu[:effective_k]
        metrics[f"{prefix_name}_{k}_effective_k"] = effective_k
        metrics[f"{prefix_name}_{k}_recovered_count"] = int(prefix_recovered.sum().item())
        metrics[f"{prefix_name}_{k}_recovered_fraction"] = (
            float(prefix_recovered.float().mean().item()) if effective_k else None
        )
    return metrics


def recovery_metrics(
    problem: ProblemInstance,
    weight: torch.Tensor,
    *,
    topk_values: Iterable[int] = DEFAULT_TOPK,
    score_chunk_size: int = DEFAULT_SCORE_CHUNK,
) -> dict[str, Any]:
    w = weight.to(torch.float32)
    n = problem.num_items

    recovered_cpu, margins_cpu = evaluate_recovery(problem, weight, score_chunk_size=score_chunk_size)
    recovered_count = int(recovered_cpu.sum().item())
    failure_idx = torch.nonzero(~recovered_cpu, as_tuple=False)
    first_failure_rank = None if failure_idx.numel() == 0 else int(failure_idx[0, 0].item() + 1)
    frequency_rank_summary = _frequency_rank_recovery_summary(problem, recovered_cpu)

    metrics: dict[str, Any] = {
        "recovered_count": recovered_count,
        "recovered_fraction": float(recovered_count / n),
        "recovered_probability_mass": frequency_rank_summary["recovered_probability_mass"],
        "first_failure_rank": first_failure_rank,
        "mean_margin_all": float(margins_cpu.mean().item()),
        "mean_margin_recovered": float(margins_cpu[recovered_cpu].mean().item()) if recovered_count else None,
        "weight_fro_norm": float(w.norm().item()),
        "weight_spectral_norm": float(torch.linalg.matrix_norm(w, ord=2).item()),
        "weight_top_singular_values": top_singular_values(w),
        "frequency_rank_summary": frequency_rank_summary,
    }
    metrics.update(
        _topk_prefix_metrics(
            recovered_cpu,
            topk_values=topk_values,
            prefix_name="top",
        )
    )
    freq_order_idx = torch.argsort(problem.p.to(torch.float32).cpu(), descending=True)
    metrics.update(
        _topk_prefix_metrics(
            recovered_cpu,
            topk_values=topk_values,
            prefix_name="freq_top",
            order_idx=freq_order_idx,
        )
    )
    return metrics


def operator_gap_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.to(torch.float32)
    cand = candidate.to(torch.float32)
    denom = (ref.norm() * cand.norm()).item()
    cosine = float(torch.sum(ref * cand).item() / denom) if denom > 0 else None
    return {
        "operator_cosine_similarity": cosine,
        "operator_fro_gap": float(torch.norm(ref - cand, p="fro").item()),
        "operator_relative_fro_gap": float(torch.norm(ref - cand, p="fro").item() / (torch.norm(ref, p="fro").item() + 1e-12)),
        "operator_spectral_norm_gap": float(abs(torch.linalg.matrix_norm(ref, ord=2).item() - torch.linalg.matrix_norm(cand, ord=2).item())),
    }


def benchmark_operator_runtime(
    operator_fn,
    *args,
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, Any]:
    for _ in range(warmup):
        operator_fn(*args)
    if torch.cuda.is_available() and any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            operator_fn(*args)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            operator_fn(*args)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return {"runtime_ms": float(latency_ms), "warmup_iters": warmup, "timed_iters": iters}


def make_result_record(
    *,
    problem: ProblemInstance,
    gradient_meta: dict[str, Any],
    optimizer: str,
    eta: float,
    metrics: dict[str, Any],
    tile_size: int | tuple[int, int] | None = None,
    normalization: str | None = None,
    ns_steps: int = 5,
    shift: tuple[int, int] = (0, 0),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "seed": problem.spec.seed,
        "d": problem.d,
        "num_items": problem.num_items,
        "frequency_mode": problem.spec.frequency_mode,
        "alpha": problem.spec.alpha,
        "device": problem.spec.device,
        "dtype": problem.spec.dtype,
        "optimizer": optimizer,
        "eta": eta,
        "ns_steps": ns_steps,
        "tile_size": tile_size,
        "normalization": normalization,
        "tile_shift": shift,
    }
    record.update(gradient_meta)
    record.update(metrics)
    if extra:
        record.update(extra)
    return record


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]], *, append: bool = True) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_records": 0}
    recovered = torch.tensor([r["recovered_count"] for r in records], dtype=torch.float32)
    summary = {
        "num_records": len(records),
        "recovered_mean": float(recovered.mean().item()),
        "recovered_std": float(recovered.std(unbiased=False).item()),
        "recovered_min": int(recovered.min().item()),
        "recovered_max": int(recovered.max().item()),
    }
    return summary
