"""Contract for ``himuon.utils`` — real behavior, no clock/RNG mocking."""

from __future__ import annotations

import argparse
import time

import torch

from himuon.utils import TimeBudget, merge_kv_args, parse_kv_args, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = torch.randn(64)
    seed_everything(123)
    b = torch.randn(64)
    assert torch.equal(a, b)

    seed_everything(456)
    c = torch.randn(64)
    assert not torch.equal(a, c)


def test_parse_kv_args_infers_types():
    out = parse_kv_args(["lr=0.02", "steps=5", "name=muon", "flag=True"])
    assert out == {"lr": 0.02, "steps": 5, "name": "muon", "flag": True}


def test_merge_kv_args_sets_attrs():
    args = argparse.Namespace(existing=None)
    merge_kv_args(args, ["existing=0.1", "new=7"])
    assert args.existing == 0.1
    assert args.new == 7


def test_time_budget_none_never_expires():
    assert not TimeBudget(hours=None).is_expired()


def test_time_budget_expires_after_deadline():
    budget = TimeBudget(hours=1e-6)  # ~3.6 ms
    time.sleep(0.02)
    assert budget.is_expired()


def test_time_budget_repr_is_hms():
    # Far-future deadline → "H:MM:SS" with two colons.
    assert TimeBudget(hours=1.0).__repr__().count(":") == 2
