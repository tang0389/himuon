"""Contract for the FSDP parameter-bank machinery (``himuon.fsdp_bank``).

Pure-CPU and offline: the banking / scheme / unbank logic needs no GPU or
distributed launch — only ``fully_shard`` does, and that is covered in the
integration tier. A synthetic decoder-shaped module (just the submodule names
the schemes key on), run against every scheme, exercises:

  * ``wrap_with_banks`` builds the 5 documented banks at the right shapes, pads
    each layer axis up to a multiple of ``world_size``, copies every source
    weight into its slot (k/v and gate/up interleaved), and deletes the original
    Linear weights so nothing is double-stored or double-counted.
  * the rewired Linear forward reads from the bank and preserves the original
    output.
  * ``unbank_state_dict`` inverts the banking back to HF-keyed weights.
  * ``release_pre_shard_refs`` drops the pre-shard Parameter references.

If a scheme mapping, padding rule, or unbank were wrong, the integration smoke
would still "run + log finite loss"; these tests are what actually pin it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from himuon.fsdp_bank import (
    Llama3BankScheme,
    Qwen3BankScheme,
    release_pre_shard_refs,
    unbank_state_dict,
    wrap_with_banks,
)

# Tiny decoder-shaped model. The schemes key on
# ``model.model.layers[i].self_attn.{q,k,v,o}_proj`` and
# ``.mlp.{gate,up,down}_proj`` (all bias-free nn.Linear). q/o share a shape,
# k/v share a (smaller) shape, gate/up share a shape, down is its transpose.
HID, KV, FFN, L = 16, 8, 24, 3
WS = 2


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HID, HID, bias=False)
        self.k_proj = nn.Linear(HID, KV, bias=False)
        self.v_proj = nn.Linear(HID, KV, bias=False)
        self.o_proj = nn.Linear(HID, HID, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HID, FFN, bias=False)
        self.up_proj = nn.Linear(HID, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, HID, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _MLP()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(L)])


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.embed_tokens = nn.Embedding(32, HID)
        self.lm_head = nn.Linear(HID, 32, bias=False)


def _fake_model() -> _FakeDecoder:
    torch.manual_seed(0)
    return _FakeDecoder()


def _by_name(banks):
    return {b.spec.name: b for b in banks}


# Both schemes key on the same per-layer projection layout the fixture
# exposes, so every contract below runs against each of them.
@pytest.fixture(params=[Qwen3BankScheme, Llama3BankScheme], ids=["qwen3", "llama3"])
def scheme_cls(request):
    return request.param


# Expected per-bank (matrix_shape, n_padded@ws2, n_logical).
_EXPECTED = {
    "q_bank": ((HID, HID), 4, 3),
    "kv_bank": ((KV, HID), 6, 6),
    "o_bank": ((HID, HID), 4, 3),
    "gate_up_bank": ((FFN, HID), 6, 6),
    "down_bank": ((HID, FFN), 4, 3),
}


class TestWrapWithBanks:
    def test_builds_five_banks_with_expected_shapes(self, scheme_cls):
        _, banks = wrap_with_banks(_fake_model(), scheme_cls(), world_size=WS)
        by = _by_name(banks)
        assert set(by) == set(scheme_cls.BANK_NAMES)
        for name, (mshape, n_pad, n_log) in _EXPECTED.items():
            b = by[name]
            assert b.spec.matrix_shape == mshape, name
            assert b.spec.n_padded == n_pad, name
            assert b.spec.n_logical == n_log, name
            assert tuple(b.parameter.shape) == (n_pad, *mshape), name

    @pytest.mark.parametrize("ws", [1, 2, 4])
    def test_pads_layer_axis_to_multiple_of_world_size(self, scheme_cls, ws):
        _, banks = wrap_with_banks(_fake_model(), scheme_cls(), world_size=ws)
        for b in banks:
            assert b.spec.n_padded % ws == 0, b.spec.name
            assert b.spec.n_padded >= b.spec.n_logical
            assert b.spec.n_padded - b.spec.n_logical < ws, "padding should be minimal"

    def test_copies_weights_into_slots_and_deletes_originals(self, scheme_cls):
        m = _fake_model()
        layers = m.model.layers
        orig = {}
        for i, lyr in enumerate(layers):
            orig[("q", i)] = lyr.self_attn.q_proj.weight.detach().clone()
            orig[("k", i)] = lyr.self_attn.k_proj.weight.detach().clone()
            orig[("v", i)] = lyr.self_attn.v_proj.weight.detach().clone()
            orig[("o", i)] = lyr.self_attn.o_proj.weight.detach().clone()
            orig[("gate", i)] = lyr.mlp.gate_proj.weight.detach().clone()
            orig[("up", i)] = lyr.mlp.up_proj.weight.detach().clone()
            orig[("down", i)] = lyr.mlp.down_proj.weight.detach().clone()

        _, banks = wrap_with_banks(m, scheme_cls(), world_size=WS)
        by = _by_name(banks)

        for i, lyr in enumerate(layers):
            assert torch.equal(by["q_bank"].parameter.data[i], orig[("q", i)])
            assert torch.equal(by["kv_bank"].parameter.data[2 * i], orig[("k", i)])
            assert torch.equal(by["kv_bank"].parameter.data[2 * i + 1], orig[("v", i)])
            assert torch.equal(by["o_bank"].parameter.data[i], orig[("o", i)])
            assert torch.equal(by["gate_up_bank"].parameter.data[2 * i], orig[("gate", i)])
            assert torch.equal(by["gate_up_bank"].parameter.data[2 * i + 1], orig[("up", i)])
            assert torch.equal(by["down_bank"].parameter.data[i], orig[("down", i)])
            # original Linear weights are gone (no duplicate storage)
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assert "weight" not in getattr(lyr.self_attn, proj)._parameters
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert "weight" not in getattr(lyr.mlp, proj)._parameters

    def test_padding_rows_are_zero(self, scheme_cls):
        _, banks = wrap_with_banks(_fake_model(), scheme_cls(), world_size=WS)
        by = _by_name(banks)
        # q_bank pads 3 -> 4: the trailing row is padding and must stay zero.
        assert torch.count_nonzero(by["q_bank"].parameter.data[3]) == 0

    def test_name_to_idx_interleaves_kv_and_gate_up(self, scheme_cls):
        _, banks = wrap_with_banks(_fake_model(), scheme_cls(), world_size=WS)
        by = _by_name(banks)
        kv = by["kv_bank"].name_to_idx
        gu = by["gate_up_bank"].name_to_idx
        for i in range(L):
            assert kv[f"model.layers.{i}.self_attn.k_proj.weight"] == 2 * i
            assert kv[f"model.layers.{i}.self_attn.v_proj.weight"] == 2 * i + 1
            assert gu[f"model.layers.{i}.mlp.gate_proj.weight"] == 2 * i
            assert gu[f"model.layers.{i}.mlp.up_proj.weight"] == 2 * i + 1

    def test_named_parameters_swaps_projections_for_banks(self, scheme_cls):
        m = _fake_model()
        before = {n for n, _ in m.named_parameters()}
        proj_names = {
            n for n in before if n.startswith("model.layers") and n.endswith(".weight")
        }
        assert proj_names, "fixture should expose per-layer projection weights"

        model, _ = wrap_with_banks(m, scheme_cls(), world_size=WS)
        after = {n for n, _ in model.named_parameters()}
        for name in scheme_cls.BANK_NAMES:
            assert name in after, f"bank {name} not registered as a parameter"
        assert proj_names.isdisjoint(after), "projection weights still present after banking"
        # untouched params survive
        assert "embed_tokens.weight" in after and "lm_head.weight" in after


class TestForwardSemantics:
    def test_rewired_linear_matches_original_output(self, scheme_cls):
        m = _fake_model()
        x = torch.randn(2, HID)
        w_orig = m.model.layers[0].self_attn.q_proj.weight.detach().clone()
        ref = F.linear(x, w_orig, None)

        model, _ = wrap_with_banks(m, scheme_cls(), world_size=WS)
        out = model.model.layers[0].self_attn.q_proj(x)
        assert torch.equal(out, ref), "rewired forward diverged from original linear"


class TestUnbankStateDict:
    def test_inverts_banking_back_to_hf_keys(self, scheme_cls):
        m = _fake_model()
        orig = {n: p.detach().clone() for n, p in m.named_parameters()}
        model, banks = wrap_with_banks(m, scheme_cls(), world_size=WS)

        banked = {n: p.detach().clone() for n, p in model.named_parameters()}
        out = unbank_state_dict(banked, banks)

        # every original projection weight reconstructed with original values
        for i in range(L):
            for role in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ):
                key = f"model.layers.{i}.{role}.weight"
                assert key in out, key
                assert torch.equal(out[key], orig[key]), key
        # bank keys removed; padding slots dropped (only n_logical emitted)
        for name in scheme_cls.BANK_NAMES:
            assert name not in out
        # non-bank params pass through untouched
        assert torch.equal(out["lm_head.weight"], orig["lm_head.weight"])
        assert torch.equal(out["embed_tokens.weight"], orig["embed_tokens.weight"])

    def test_missing_bank_raises(self, scheme_cls):
        _, banks = wrap_with_banks(_fake_model(), scheme_cls(), world_size=WS)
        with pytest.raises(KeyError):
            unbank_state_dict({}, banks)


def test_release_pre_shard_refs_nulls_parameter_keeps_metadata(scheme_cls):
    _, banks = wrap_with_banks(_fake_model(), scheme_cls(), world_size=WS)
    assert all(b.parameter is not None for b in banks)
    release_pre_shard_refs(banks)
    assert all(b.parameter is None for b in banks)
    # spec / name_to_idx survive for introspection + unbank
    assert all(b.spec.name in scheme_cls.BANK_NAMES for b in banks)
    assert all(b.name_to_idx for b in banks)
