"""Self-consistency tier: HiMuon must agree with itself across equivalent paths.

These pin the invariants that survive any legitimate refactor (no frozen-legacy
oracle, no math ground truth): the per-param result must not depend on how tiles
are *batched* (chunk size ``b_hw``) nor on whether the step runs eager or via a
captured CUDA graph. Cross-layer batching re-orders tiles within a torch.compile'd
NS call, so the bar is bf16 tolerance, not bitwise.
"""

from __future__ import annotations

import copy

import torch

from himuon.optimizers.himuon import HiMuon

BF16_RTOL, BF16_ATOL = 1e-2, 1e-2

# Mixed shapes so a bucket holds several tiles (b_hw chunking is exercised).
SHAPES = [(512, 1024), (256, 512), (1024, 2048)]


def _assert_params_close(model_a, model_b, msg):
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(pa, pb, rtol=BF16_RTOL, atol=BF16_ATOL), (
            f"{msg}: max|Δ|={(pa.float() - pb.float()).abs().max().item():.3e}"
        )


def test_result_invariant_to_b_hw(make_model, build_groups, seed_grads):
    """Chunk-size cap must not change the math: one big NS call (huge b_hw)
    vs many tiny calls (b_hw=1) land at the same params."""
    kwargs = dict(lr=0.02, weight_decay=0.1, tile_size=512, ns_steps=5, cuda_graph=False)

    torch.manual_seed(0)
    model_big = make_model(SHAPES, dtype=torch.bfloat16)
    model_small = copy.deepcopy(model_big)

    opt_big = HiMuon(build_groups(model_big), b_hw=100_000, **kwargs)
    opt_small = HiMuon(build_groups(model_small), b_hw=1, **kwargs)

    for k in range(5):
        seed_grads(model_big, seed=100 + k)
        seed_grads(model_small, seed=100 + k)
        opt_big.step()
        opt_small.step()

    _assert_params_close(model_big, model_small, "b_hw chunking changed the result")


def test_eager_matches_cuda_graph(make_model, build_groups, seed_grads):
    """Captured-graph replay must match eager within bf16 tolerance."""
    kwargs = dict(lr=0.02, weight_decay=0.0, tile_size=512, ns_steps=5)

    torch.manual_seed(0)
    model_eager = make_model(SHAPES, dtype=torch.bfloat16)
    model_graph = copy.deepcopy(model_eager)

    opt_eager = HiMuon(build_groups(model_eager), cuda_graph=False, **kwargs)
    opt_graph = HiMuon(build_groups(model_graph), cuda_graph=True, cuda_graph_warmup=3, **kwargs)

    for k in range(8):  # > warmup so the graph path captures + replays
        seed_grads(model_eager, seed=200 + k)
        seed_grads(model_graph, seed=200 + k)
        opt_eager.step()
        opt_graph.step()

    assert opt_graph._cuda_graph is not None, "graph never captured"
    _assert_params_close(model_eager, model_graph, "graph replay diverged from eager")
