"""Peak memory leak gate for HiMuon.

Asserts that ``torch.cuda.max_memory_allocated()`` is flat across
steady-state steps. A growing peak indicates a buffer is being
re-allocated each step instead of reused (e.g. a stale plan-cache
entry, an `_ensure_*_buffer` miss).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
import torch.nn as nn

from himuon.optimizers.himuon import HiMuon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_marker():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_toy_model(shapes: Sequence[tuple[int, int]], dtype=torch.float32):
    layers: list[nn.Module] = []
    for H, W in shapes:
        layers.append(nn.Linear(W, H, bias=False, dtype=dtype))
    return nn.ModuleList(layers).to(DEVICE)


def _build_param_groups(model: nn.Module) -> list[dict]:
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    return [{"params": muon_params, "weight_decay": 0.0, "use_muon": True}]


def _seed_grads_inplace(model, seed):
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.empty(p.shape, device=DEVICE, dtype=p.dtype)
        p.grad.normal_(generator=gen)


class TestNoGrowth:
    @_cuda_marker()
    def test_no_growth_over_10_steps(self):
        shapes = [(1024, 1024), (512, 2048)]
        kwargs = dict(
            lr=0.02,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.1,
            tile_size=512,
            ns_steps=5,
        )
        model = _make_toy_model(shapes)
        opt = HiMuon(_build_param_groups(model), **kwargs)

        # Warm up past the one-time CUDA-graph capture (default warmup=3, so
        # capture lands on step 4). Measuring before capture completes would
        # mis-read its one-time allocation as a per-step leak.
        for i in range(8):
            _seed_grads_inplace(model, 300 + i)
            opt.step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        peaks = []
        for i in range(10):
            torch.cuda.reset_peak_memory_stats()
            _seed_grads_inplace(model, 400 + i)
            opt.step()
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated())
        print(f"\n[no_growth] peaks={[f'{p:_}' for p in peaks]}")
        growth = (max(peaks) - min(peaks)) / max(min(peaks), 1)
        assert growth < 0.01, f"peak grew {growth * 100:.3f}% across 10 steps — leak suspected"
