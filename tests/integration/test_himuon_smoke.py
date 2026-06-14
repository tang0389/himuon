"""Smoke tier: HiMuon.step() runs and produces finite updates.

No numerical oracle — just the behavioral guarantee that the documented
shape / dtype / tile_size matrix dispatches, launches its Triton kernels, and
leaves parameters finite and changed. Catches "the optimizer crashes / NaNs"
regressions cheaply.
"""

from __future__ import annotations

import pytest
import torch

from himuon.optimizers.himuon import HiMuon


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile_size", [128, 256, 512])
def test_step_runs_and_is_finite(make_model, build_groups, seed_grads, dtype, tile_size):
    # Mix of aligned, non-divisible, and below-tile (full-NS) shapes.
    shapes = [(512, 1024), (768, 513), (64, 64)]
    model = make_model(shapes, dtype=dtype)
    opt = HiMuon(build_groups(model), tile_size=tile_size, cuda_graph=False)

    before = [p.detach().clone() for p in model.parameters()]
    seed_grads(model, seed=1)
    opt.step()

    changed = False
    for p, b in zip(model.parameters(), before):
        assert torch.isfinite(p).all(), f"non-finite param after step ({p.shape})"
        if not torch.equal(p, b):
            changed = True
    assert changed, "step() left every parameter unchanged"


def test_multi_step_stays_finite(make_model, build_groups, seed_grads):
    model = make_model([(512, 512), (1024, 512)])
    opt = HiMuon(build_groups(model), tile_size=256, cuda_graph=False)
    for k in range(5):
        seed_grads(model, seed=10 + k)
        opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_cuda_graph_mode_runs(make_model, build_groups, seed_grads):
    """The CUDA-graph path captures after warmup and replays without error."""
    model = make_model([(512, 1024)])
    opt = HiMuon(build_groups(model), tile_size=512, cuda_graph=True, cuda_graph_warmup=2)
    for k in range(5):  # warmup(2) + capture(1) + replay(2)
        seed_grads(model, seed=20 + k)
        opt.step()
    assert opt._cuda_graph is not None, "graph never captured"
    for p in model.parameters():
        assert torch.isfinite(p).all()
