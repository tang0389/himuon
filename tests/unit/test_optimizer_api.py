"""Executable API doc for the HiMuon optimizer's public surface.

Pure-CPU contract tests: construct ``HiMuon`` and introspect its signature,
param-group schema, ``tile_size`` normalization, ``reconfigure`` behavior, and
``state_dict`` round-trip — without ever calling ``step()`` (which needs
Triton/CUDA). If you change the public API, update the recorded spec below;
that is the point — this file *is* the contract.
"""

from __future__ import annotations

import inspect
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.optim import Optimizer
from transformers import get_cosine_schedule_with_warmup

from himuon.optim import get_optimizer
from himuon.optimizers.himuon import HiMuon

# Recorded public constructor spec. Any change to HiMuon.__init__ must be
# mirrored here deliberately — that mirroring is the "doc update".
EXPECTED_SIGNATURE = {
    "lr": 0.02,
    "momentum": 0.95,
    "nesterov": True,
    "weight_decay": 0.1,
    "tile_size": 512,
    "ns_steps": 5,
    "use_muon": True,
    "lr_adjust": "tile",
    "adamw_betas": (0.95, 0.95),
    "adamw_eps": 1e-8,
    "b_hw": None,
    "cuda_graph": True,
    "cuda_graph_warmup": 3,
}

# Default keys HiMuon must populate on every param group.
GROUP_DEFAULT_KEYS = {
    "lr",
    "momentum",
    "nesterov",
    "weight_decay",
    "tile_size",
    "ns_steps",
    "use_muon",
    "adamw_betas",
    "adamw_eps",
}


def _muon_group(*params):
    return [{"params": list(params), "use_muon": True}]


def test_constructor_signature_is_locked():
    sig = inspect.signature(HiMuon.__init__)
    params = {n: p for n, p in sig.parameters.items() if n not in ("self", "params")}
    assert set(params) == set(EXPECTED_SIGNATURE), (
        "HiMuon.__init__ parameters changed; update EXPECTED_SIGNATURE to match."
    )
    for name, expected_default in EXPECTED_SIGNATURE.items():
        assert params[name].default == expected_default, (
            f"default for {name!r} changed: {params[name].default!r} != {expected_default!r}"
        )


def test_is_torch_optimizer():
    p = nn.Parameter(torch.zeros(8, 8))
    assert isinstance(HiMuon(_muon_group(p)), Optimizer)


def test_param_group_default_schema():
    p = nn.Parameter(torch.zeros(8, 8))
    opt = HiMuon(_muon_group(p))
    for group in opt.param_groups:
        missing = GROUP_DEFAULT_KEYS - set(group)
        assert not missing, f"group missing documented default keys: {missing}"


@pytest.mark.parametrize(
    "given, expected",
    [
        (512, (512, 512)),
        ((128, 256), (128, 256)),
        ([64, 64], (64, 64)),
    ],
)
def test_tile_size_normalized_to_2tuple(given, expected):
    p = nn.Parameter(torch.zeros(8, 8))
    opt = HiMuon(_muon_group(p), tile_size=given)
    for group in opt.param_groups:
        assert group["tile_size"] == expected


def test_tile_size_rejects_nonpositive():
    p = nn.Parameter(torch.zeros(8, 8))
    with pytest.raises(AssertionError):
        HiMuon(_muon_group(p), tile_size=(512, 0))


class TestReconfigure:
    """reconfigure() mutates group defaults in place (pure CPU, no step)."""

    def test_updates_all_groups_and_normalizes_tile(self):
        p_a = nn.Parameter(torch.zeros(16, 16))
        p_b = nn.Parameter(torch.zeros(8))
        opt = HiMuon(
            [
                {"params": [p_a], "use_muon": True, "tile_size": (128, 128)},
                {"params": [p_b], "use_muon": False, "tile_size": (128, 128)},
            ]
        )
        opt.reconfigure(tile_size=256, lr=0.05, weight_decay=0.3)
        for g in opt.param_groups:
            assert g["tile_size"] == (256, 256), "int tile_size not normalised"
            assert g["lr"] == 0.05
            assert g["weight_decay"] == 0.3

    def test_tuple_tile_size_preserved(self):
        p = nn.Parameter(torch.zeros(16, 16))
        opt = HiMuon(_muon_group(p))
        opt.reconfigure(tile_size=(128, 256))
        for g in opt.param_groups:
            assert g["tile_size"] == (128, 256)

    def test_unknown_key_ignored(self):
        p = nn.Parameter(torch.zeros(16, 16))
        opt = HiMuon(_muon_group(p))
        opt.reconfigure(not_a_real_key=123)
        for g in opt.param_groups:
            assert "not_a_real_key" not in g


def test_state_dict_roundtrip_structure():
    p = nn.Parameter(torch.zeros(8, 8))
    opt = HiMuon(_muon_group(p))
    sd = opt.state_dict()
    assert "state" in sd and "param_groups" in sd

    p2 = nn.Parameter(torch.zeros(8, 8))
    opt2 = HiMuon(_muon_group(p2))
    opt2.load_state_dict(sd)  # must not raise
    assert len(opt2.param_groups) == len(opt.param_groups)


class TestTrainScriptCallSurface:
    """Pin the optimizer surface the three ``train*.py`` scripts actually use.

    These are the calls the training loop makes around ``step()`` (which is the
    only GPU-only bit and lives in the integration tier). If any of these breaks,
    every train script breaks — so they are pinned here as the CPU contract:

      * factory build + ``param_groups[i]['lr']`` readable (logging + scheduler),
      * ``get_cosine_schedule_with_warmup(opt, ...)`` + ``scheduler.step()``
        drives the group LRs (the exact scheduler the scripts construct),
      * ``opt.zero_grad(set_to_none=...)`` works in *both* modes — train.py picks
        the mode off ``--cuda-graph`` (set_to_none=False keeps stable grad
        buffers for graph replay; True is the eager path),
      * ``hasattr(opt, 'reconfigure')`` + ``reconfigure`` mutates group defaults
        readable back through ``param_groups`` (the reconfigure-schedule path).
    """

    def _opt(self, model):
        # Built exactly as train*.py do: factory + an args-like namespace.
        return get_optimizer("himuon", model, SimpleNamespace())

    def test_every_group_exposes_numeric_lr(self, tiny_causal_lm):
        opt = self._opt(tiny_causal_lm)
        # train.py logs optimizer.param_groups[0]['lr']; the scheduler reads an
        # initial lr off every group.
        for g in opt.param_groups:
            assert isinstance(g["lr"], (int, float))

    def test_cosine_warmup_scheduler_drives_lr(self, tiny_causal_lm):
        opt = self._opt(tiny_causal_lm)
        base = [g["lr"] for g in opt.param_groups]
        warmup, total = 3, 12
        sched = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup, num_training_steps=total
        )

        # LambdaLR derives LR from its own step count, so we can drive the
        # schedule without a real (GPU-only) optimizer.step() — torch only warns
        # about the call order, which is irrelevant to the LR values here.
        lrs = [opt.param_groups[0]["lr"]]  # step 0 (warmup start ≈ 0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of")
            for _ in range(total):
                sched.step()
                lrs.append(opt.param_groups[0]["lr"])

        assert lrs[0] == pytest.approx(0.0, abs=1e-9), "warmup should start at lr≈0"
        # Peaks at the group's base lr right after warmup, then decays below it.
        assert lrs[warmup] == pytest.approx(base[0], rel=1e-6)
        assert lrs[-1] < lrs[warmup], "cosine phase should decay the lr"

    def test_zero_grad_both_modes(self, tiny_causal_lm):
        opt = self._opt(tiny_causal_lm)
        params = [p for g in opt.param_groups for p in g["params"]]
        for p in params:
            p.grad = torch.ones_like(p)

        # set_to_none=False (cuda_graph path): buffers kept, zeroed in place.
        opt.zero_grad(set_to_none=False)
        for p in params:
            assert p.grad is not None and torch.count_nonzero(p.grad) == 0

        # set_to_none=True (eager path): buffers dropped.
        opt.zero_grad(set_to_none=True)
        for p in params:
            assert p.grad is None

    def test_reconfigure_schedule_surface(self, tiny_causal_lm):
        opt = self._opt(tiny_causal_lm)
        assert hasattr(opt, "reconfigure"), "train*.py guards on hasattr(opt,'reconfigure')"
        opt.reconfigure(tile_size=256, lr=0.05)
        # apply_reconfigure reads new values back off param_groups exactly so:
        assert next(g["tile_size"] for g in opt.param_groups if "tile_size" in g) == (256, 256)
        assert next(g["lr"] for g in opt.param_groups if "lr" in g) == 0.05
