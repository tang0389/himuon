"""Coverage for HiMuon.reconfigure() — mid-training param-group updates and the
cache/CUDA-graph invalidation it triggers.

``train.py`` drives a reconfigure schedule (e.g. growing ``tile_size`` mid-run),
but nothing in the suite exercised it. The risky part is ``_invalidate_plan_cache``:
a stale bucket plan, concat buffer, or — worst — a captured CUDA graph whose
pending-update tensors point into a dropped graph pool would silently corrupt
training after a reconfigure.

Pinned here:

  1. **Group mutation**: reconfigure writes through to every param group and
     normalises ``tile_size`` to a 2-tuple.
  2. **Cache invalidation**: after a step has populated the plan / concat buffer
     (and, with ``cuda_graph=True``, captured a graph + pending updates),
     reconfigure clears all of it.
  3. **Takes effect**: the bucket plan rebuilt on the next step reflects the new
     ``tile_size``, not the old one.
  4. **No corruption**: reconfiguring to the *same* tile_size is a math no-op —
     splitting a run with a mid-stream reconfigure matches an uninterrupted run,
     in both eager and CUDA-graph modes (this exercises graph recapture).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest
import torch
import torch.nn as nn

from himuon.optimizers.himuon import HiMuon

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_RTOL, BF16_ATOL = 1e-2, 1e-2


def _cuda_marker():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_toy_model(
    shapes: Sequence[tuple[int, int]],
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    layers: list[nn.Module] = []
    for H, W in shapes:
        layers.append(nn.Linear(W, H, bias=False, dtype=dtype))
    return nn.ModuleList(layers).to(DEVICE)


def _muon_groups(model: nn.Module) -> list[dict]:
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    return [{"params": muon_params, "weight_decay": 0.0, "use_muon": True}]


def _seed_grads_inplace(model: nn.Module, seed: int) -> None:
    # set_to_none=False semantics: reuse stable grad buffers (graph contract).
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.empty(p.shape, device=DEVICE, dtype=p.dtype)
        p.grad.normal_(generator=gen)


def _tile_dims_in_plan(opt: HiMuon) -> set:
    """Set of (tile_h, tile_w) over the bucket plan. bucket_key is
    (tile_h, tile_w, full_ns_flag, dtype)."""
    return {(b["bucket_key"][0], b["bucket_key"][1]) for b in opt.xlayer_plan()}


# Group-mutation contract (pure CPU, no step) lives in
# tests/unit/test_optimizer_api.py::TestReconfigure. This file covers the
# GPU-only behaviors: cache/graph invalidation and the math no-op guarantee.


# ===========================================================================
# 1 — cache / graph invalidation
# ===========================================================================


@_cuda_marker()
class TestCacheInvalidation:
    def _model_opt(self, cuda_graph: bool, warmup: int = 2):
        # (512, 1024) with tile 512 → 2 tiles → goes through the bucketed
        # concat path, so a concat buffer is actually allocated.
        torch.manual_seed(0)
        model = _make_toy_model([(512, 1024)])
        opt = HiMuon(
            _muon_groups(model),
            tile_size=512,
            cuda_graph=cuda_graph,
            cuda_graph_warmup=warmup,
        )
        return model, opt

    def test_eager_invalidates_plan_and_buffers(self):
        model, opt = self._model_opt(cuda_graph=False)
        _seed_grads_inplace(model, 1)
        opt.step()
        # Step populated the bucket plan + concat buffer.
        assert opt.xlayer_plan(), "plan not cached after a step"
        assert opt._concat_buffers, "concat buffer not allocated after a step"

        opt.reconfigure(tile_size=256)
        assert opt.xlayer_plan() == [], "plan cache not cleared"
        assert opt._concat_buffers == {}, "concat buffers not cleared"
        assert opt._cached_plan is None

    def test_graph_invalidates_capture_state(self):
        warmup = 2
        model, opt = self._model_opt(cuda_graph=True, warmup=warmup)
        # warmup eager steps + 1 capture step + 1 replay step.
        for k in range(warmup + 2):
            _seed_grads_inplace(model, 10 + k)
            opt.step()
        assert opt._cuda_graph is not None, "graph never captured"
        assert opt._graph_step_count >= warmup
        assert opt._pending_updates, "pending updates empty after capture"
        assert opt.xlayer_plan(), "plan not cached"

        opt.reconfigure(tile_size=256)
        assert opt._cuda_graph is None, "captured graph not dropped"
        assert opt._graph_step_count == 0, "graph step count not reset"
        assert opt._pending_updates == [], "stale pending updates not cleared"
        assert opt.xlayer_plan() == [], "plan cache not cleared"
        assert opt._concat_buffers == {}, "concat buffers not cleared"


# ===========================================================================
# 3 — reconfigure takes effect on the next step
# ===========================================================================


@_cuda_marker()
class TestTakesEffect:
    def test_plan_reflects_new_tile_size(self):
        torch.manual_seed(0)
        model = _make_toy_model([(512, 1024)])
        opt = HiMuon(_muon_groups(model), tile_size=512, cuda_graph=False)

        _seed_grads_inplace(model, 1)
        opt.step()
        assert _tile_dims_in_plan(opt) == {(512, 512)}, _tile_dims_in_plan(opt)

        opt.reconfigure(tile_size=256)
        _seed_grads_inplace(model, 2)
        opt.step()
        assert _tile_dims_in_plan(opt) == {(256, 256)}, (
            f"plan still on old tiling: {_tile_dims_in_plan(opt)}"
        )


# ===========================================================================
# 4 — reconfigure(same) is a math no-op (also exercises graph recapture)
# ===========================================================================


@_cuda_marker()
class TestNoCorruption:
    @pytest.mark.parametrize("cuda_graph", [False, True])
    def test_reconfigure_same_tile_matches_uninterrupted(self, cuda_graph):
        shapes = [(512, 1024), (256, 512)]
        tile = 256
        K = 6  # > warmup so the graph path actually captures + replays

        torch.manual_seed(0)
        ref_model = _make_toy_model(shapes)
        rec_model = copy.deepcopy(ref_model)

        kwargs = dict(
            lr=0.02,
            weight_decay=0.05,
            tile_size=tile,
            cuda_graph=cuda_graph,
            cuda_graph_warmup=2,
        )
        opt_ref = HiMuon(_muon_groups(ref_model), **kwargs)
        opt_rec = HiMuon(_muon_groups(rec_model), **kwargs)

        for k in range(2 * K):
            seed = 200 + k
            _seed_grads_inplace(ref_model, seed)
            _seed_grads_inplace(rec_model, seed)
            opt_ref.step()
            if k == K:
                # Reconfigure to the SAME tile_size: forces a full cache /
                # graph teardown + rebuild that must not change the math.
                opt_rec.reconfigure(tile_size=tile)
            opt_rec.step()

        for pr, pc in zip(ref_model.parameters(), rec_model.parameters()):
            assert torch.allclose(pr.data, pc.data, rtol=BF16_RTOL, atol=BF16_ATOL), (
                "reconfigure(same) diverged from uninterrupted run: max|Δ|="
                f"{(pr.data.float() - pc.data.float()).abs().max().item():.3e}"
            )
