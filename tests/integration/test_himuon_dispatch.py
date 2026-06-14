"""Dispatcher-completeness contract for HiMuon's cross-layer batched NS.

Pins the introspection API ``HiMuon.xlayer_plan() -> list[dict]`` and the
guarantee that every tile of every muon param lands in exactly one bucket chunk,
chunk sizes respect ``b_hw``, and the ``b_hw`` override is honoured. This is a
structural contract on the plan (no NS math oracle), so it stays valid across
kernel/numerics changes. Ported from the original xlayer pack with the
legacy-parity oracle removed.
"""

from __future__ import annotations

import copy

from himuon.optimizers.himuon import HiMuon

# Mixed shapes → a bucket with several tiles across multiple params.
MULTI_MIXED = [(512, 1024), (256, 512), (1024, 2048)]


def _plan(opt):
    assert hasattr(opt, "xlayer_plan"), "HiMuon must expose xlayer_plan()"
    return opt.xlayer_plan()


def test_every_tile_accounted_for(make_model, build_groups, seed_grads):
    """Sum of tiles in the plan equals the tile count over all muon params,
    except params routed through the full-NS path (numel <= tile area)."""
    tile = 256
    model = make_model(MULTI_MIXED)
    opt = HiMuon(build_groups(model), tile_size=tile, ns_steps=5, cuda_graph=False)
    seed_grads(model, seed=12)
    opt.step()

    expected = {}
    for p in model.parameters():
        if p.ndim < 2 or not p.requires_grad:
            continue
        H, W = p.shape
        R = (H + tile - 1) // tile
        C = (W + tile - 1) // tile
        expected[id(p)] = R * C

    plan_tiles = {}
    for bucket in _plan(opt):
        for pid, tc in zip(bucket["param_ids"], bucket["tile_count_per_param"]):
            plan_tiles[pid] = plan_tiles.get(pid, 0) + tc

    for pid, tc in plan_tiles.items():
        assert tc == expected[pid], f"param id={pid}: {tc} tiles, expected {expected[pid]}"

    missing = set(expected) - set(plan_tiles)
    for p in model.parameters():
        if id(p) in missing:
            assert p.numel() <= tile * tile, (
                f"param {tuple(p.shape)} missing from plan but numel "
                f"{p.numel()} > tile area — dispatcher dropped tiles"
            )


def test_chunk_sizes_sum_to_b_total(make_model, build_groups, seed_grads):
    model = make_model(MULTI_MIXED)
    opt = HiMuon(build_groups(model), tile_size=512, ns_steps=5, cuda_graph=False)
    seed_grads(model, seed=14)
    opt.step()

    for bucket in _plan(opt):
        assert sum(bucket["chunk_sizes"]) == bucket["B_total"]
        assert all(c > 0 for c in bucket["chunk_sizes"]), "empty chunk"
        assert all(c <= bucket["B_hw"] for c in bucket["chunk_sizes"]), "chunk exceeds b_hw"
        assert bucket["n_calls"] == len(bucket["chunk_sizes"])


def test_split_regimes_below_at_above_b_hw(make_model, build_groups, seed_grads):
    """b_hw below / at / above B_total yields the right number of NS calls."""
    kwargs = dict(tile_size=512, ns_steps=5, cuda_graph=False)
    ref = make_model(MULTI_MIXED)

    m0 = copy.deepcopy(ref)
    opt0 = HiMuon(build_groups(m0), b_hw=10_000, **kwargs)
    seed_grads(m0, seed=18)
    opt0.step()
    plan0 = _plan(opt0)
    assert plan0, "expected at least one bucket"
    b_total_ref = max(b["B_total"] for b in plan0)
    assert b_total_ref >= 2, "fixture too small; want a bucket with B_total >= 2"

    for regime, b_hw in (
        ("below", b_total_ref + 10),
        ("equal", b_total_ref),
        ("above", max(1, b_total_ref // 2)),
    ):
        m = copy.deepcopy(ref)
        opt = HiMuon(build_groups(m), b_hw=b_hw, **kwargs)
        seed_grads(m, seed=19)
        opt.step()
        target = [b for b in _plan(opt) if b["B_total"] == b_total_ref]
        if not target:
            target = [max(_plan(opt), key=lambda b: b["B_total"])]
        b = target[0]
        if regime == "below":
            assert b["n_calls"] == 1
        elif regime == "equal":
            assert b["n_calls"] == 1
            assert b["chunk_sizes"] == [b_total_ref]
        else:
            expected_n = -(-b_total_ref // b_hw)
            assert b["n_calls"] == expected_n
            assert b["chunk_sizes"][-1] <= b_hw


def test_b_hw_override_honoured(make_model, build_groups, seed_grads):
    model = make_model([(1024, 1024)])
    opt = HiMuon(build_groups(model), tile_size=512, ns_steps=5, b_hw=7, cuda_graph=False)
    seed_grads(model, seed=22)
    opt.step()
    for bucket in _plan(opt):
        assert bucket["B_hw"] == 7, f"b_hw override not honoured: {bucket['B_hw']}"
