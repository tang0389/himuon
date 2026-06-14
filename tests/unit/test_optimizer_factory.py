"""Contract for the optimizer factory (``himuon.optim``).

Pins the parameter-grouping convention the factory must follow for HiMuon, so
a change to the factory wiring that drops a param, mis-routes embed/lm_head, or
breaks the muon/adamw split fails here. Scope is HiMuon + the framework wiring;
baseline optimizer internals are out of scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from himuon.optim import get_optimizer, get_optimizer_kwargs
from himuon.optimizers.himuon import HiMuon


def _ids(params):
    return {id(p) for p in params}


def _split_by_muon(opt):
    muon = [g for g in opt.param_groups if g.get("use_muon")]
    adamw = [g for g in opt.param_groups if not g.get("use_muon")]
    return muon, adamw


class TestHiMuonGrouping:
    def test_three_documented_groups(self, tiny_causal_lm):
        opt = get_optimizer("himuon", tiny_causal_lm, SimpleNamespace())
        assert isinstance(opt, HiMuon)
        assert len(opt.param_groups) == 3

    def test_param_routing_matches_convention(self, tiny_causal_lm):
        m = tiny_causal_lm
        opt = get_optimizer("himuon", m, SimpleNamespace())
        muon_groups, adamw_groups = _split_by_muon(opt)

        # Exactly one Muon group: 2-D weights that are not embed_tokens/lm_head.
        assert len(muon_groups) == 1
        assert muon_groups[0]["use_muon"] is True
        assert _ids(muon_groups[0]["params"]) == {id(m.mlp.weight)}

        # AdamW side splits into a 1-D no-decay group and a 2-D embed/lm_head group.
        nodecay = [g for g in adamw_groups if g["weight_decay"] == 0.0]
        decay2d = [g for g in adamw_groups if g["weight_decay"] != 0.0]
        assert len(nodecay) == 1 and len(decay2d) == 1

        assert _ids(nodecay[0]["params"]) == {
            id(m.norm.weight),
            id(m.norm.bias),
            id(m.mlp.bias),
        }
        assert _ids(decay2d[0]["params"]) == {
            id(m.embed_tokens.weight),
            id(m.lm_head.weight),
        }

    def test_partition_is_complete_and_disjoint(self, tiny_causal_lm):
        """Framework invariant: every trainable param lands in exactly one
        group — none dropped, none double-counted."""
        m = tiny_causal_lm
        opt = get_optimizer("himuon", m, SimpleNamespace())
        grouped = [id(p) for g in opt.param_groups for p in g["params"]]
        expected = _ids(p for p in m.parameters() if p.requires_grad)
        assert len(grouped) == len(set(grouped)), "a param appears in multiple groups"
        assert set(grouped) == expected, "grouped params != trainable params"


class TestKwargFiltering:
    def test_filters_to_signature_and_drops_none(self):
        ns = SimpleNamespace(lr=0.01, weight_decay=0.2, not_a_param=5, momentum=None)
        out = get_optimizer_kwargs(HiMuon, ns)
        assert out == {"lr": 0.01, "weight_decay": 0.2}

    def test_none_kwargs_yields_empty(self):
        assert get_optimizer_kwargs(HiMuon, None) == {}


def test_unsupported_name_raises(tiny_causal_lm):
    with pytest.raises(ValueError):
        get_optimizer("not-an-optimizer", tiny_causal_lm, SimpleNamespace())
