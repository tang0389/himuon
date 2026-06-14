"""Bank scheme for Qwen3 decoder-only models.

Groups each layer's 2D weight matrices into 5 banks:
  * ``q_bank``       — q_proj.weight, per layer
  * ``kv_bank``      — k_proj + v_proj interleaved per layer (same shape)
  * ``o_bank``       — o_proj.weight, per layer
  * ``gate_up_bank`` — gate_proj + up_proj interleaved (same shape)
  * ``down_bank``    — down_proj.weight, per layer

Merging same-shape projections (k+v, gate+up) halves the number of
banks and doubles the cross-layer NS batch size in HiMuon for free.

``embed_tokens``, ``lm_head``, and all LayerNorm weights stay as regular
parameters — 1D LNs run AdamW, and the embeddings are separately large
enough that banking them doesn't help.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bank import BankSpec, ParamBank


class Qwen3BankScheme:
    BANK_NAMES = ("q_bank", "kv_bank", "o_bank", "gate_up_bank", "down_bank")

    # -- collect -----------------------------------------------------------

    def collect(self, model: nn.Module, world_size: int) -> list[BankSpec]:
        layers = _get_layers(model)
        L = len(layers)

        def _role_names(role: str) -> list[str]:
            # ``role`` picks the attribute chain inside each layer, e.g.
            # "self_attn.q_proj" → model.layers.{i}.self_attn.q_proj.weight
            return [f"model.layers.{i}.{role}.weight" for i in range(L)]

        q_names = _role_names("self_attn.q_proj")
        k_names = _role_names("self_attn.k_proj")
        v_names = _role_names("self_attn.v_proj")
        o_names = _role_names("self_attn.o_proj")
        gate_names = _role_names("mlp.gate_proj")
        up_names = _role_names("mlp.up_proj")
        down_names = _role_names("mlp.down_proj")

        # k+v and gate+up share shape; interleave so bank[2i], bank[2i+1]
        # correspond to layer i's two projections. Keeps locality of
        # reference cache-friendly when a layer pulls its weights.
        kv_names = [name for pair in zip(k_names, v_names) for name in pair]
        gate_up_names = [name for pair in zip(gate_names, up_names) for name in pair]

        q_shape = tuple(layers[0].self_attn.q_proj.weight.shape)
        k_shape = tuple(layers[0].self_attn.k_proj.weight.shape)
        v_shape = tuple(layers[0].self_attn.v_proj.weight.shape)
        o_shape = tuple(layers[0].self_attn.o_proj.weight.shape)
        gate_shape = tuple(layers[0].mlp.gate_proj.weight.shape)
        up_shape = tuple(layers[0].mlp.up_proj.weight.shape)
        down_shape = tuple(layers[0].mlp.down_proj.weight.shape)
        assert k_shape == v_shape, f"k/v shape mismatch: {k_shape} vs {v_shape}"
        assert gate_shape == up_shape, (
            f"gate/up shape mismatch: {gate_shape} vs {up_shape}"
        )

        def _pad_up(n: int) -> int:
            return ((n + world_size - 1) // world_size) * world_size

        specs = [
            BankSpec("q_bank", q_shape, q_names, _pad_up(L)),
            BankSpec("kv_bank", k_shape, kv_names, _pad_up(2 * L)),
            BankSpec("o_bank", o_shape, o_names, _pad_up(L)),
            BankSpec("gate_up_bank", gate_shape, gate_up_names, _pad_up(2 * L)),
            BankSpec("down_bank", down_shape, down_names, _pad_up(L)),
        ]
        return specs

    # -- install -----------------------------------------------------------

    def install(self, model: nn.Module, banks: list[ParamBank]) -> nn.Module:
        by_name = {b.spec.name: b for b in banks}
        for b in banks:
            # Attach the bank as a regular named Parameter on the root
            # model. HF keeps ``model.model = Qwen3Model`` so the banks
            # end up on the *outer* ``Qwen3ForCausalLM``. This keeps
            # them visible to ``model.named_parameters()`` for the
            # optimizer factory without colliding with HF's own names.
            setattr(model, b.spec.name, b.parameter)

        layers = _get_layers(model)
        for i, layer in enumerate(layers):
            _rewire_linear(
                layer.self_attn.q_proj, model, "q_bank", by_name["q_bank"], i
            )
            _rewire_linear(
                layer.self_attn.k_proj, model, "kv_bank", by_name["kv_bank"], 2 * i
            )
            _rewire_linear(
                layer.self_attn.v_proj, model, "kv_bank", by_name["kv_bank"], 2 * i + 1
            )
            _rewire_linear(
                layer.self_attn.o_proj, model, "o_bank", by_name["o_bank"], i
            )
            _rewire_linear(
                layer.mlp.gate_proj,
                model,
                "gate_up_bank",
                by_name["gate_up_bank"],
                2 * i,
            )
            _rewire_linear(
                layer.mlp.up_proj,
                model,
                "gate_up_bank",
                by_name["gate_up_bank"],
                2 * i + 1,
            )
            _rewire_linear(
                layer.mlp.down_proj, model, "down_bank", by_name["down_bank"], i
            )

        return model


# -- helpers ---------------------------------------------------------------


def _get_layers(model: nn.Module):
    """Locate the list of decoder layers in a Qwen3 HF model.

    HF wraps the transformer as ``Qwen3ForCausalLM.model (Qwen3Model).
    layers``. Hold the indirection behind this helper so a future HF
    refactor only touches one line.
    """
    return model.model.layers


def _rewire_linear(
    linear: nn.Linear,
    root_model: nn.Module,
    bank_name: str,
    bank: ParamBank,
    layer_idx: int,
) -> None:
    """Replace ``linear.forward`` with one that reads from the bank
    at ``layer_idx``. Copies the original weight into the bank slice,
    then deletes the original ``weight`` Parameter so it's not double-
    counted (and the optimizer factory sees only the bank).

    The closure resolves the bank each forward via
    ``getattr(root_model, bank_name)`` rather than capturing the
    Parameter directly — this is required because ``fully_shard``
    REPLACES the ``root_model.<bank_name>`` attribute with a fresh
    DTensor Parameter at wrap time. A closure that captured the
    original Parameter would dereference an orphan (no grad flow).

    ``root_model`` and ``bank_name`` are cheap ref-plus-string; no
    nn.Module registration side-effect.
    """
    assert isinstance(linear, nn.Linear), f"expected nn.Linear, got {type(linear)}"
    expected = bank.spec.matrix_shape
    assert tuple(linear.weight.shape) == expected, (
        f"shape mismatch for {bank.spec.name}[{layer_idx}]: "
        f"linear.weight={tuple(linear.weight.shape)} vs bank={expected}"
    )

    # Copy into bank slot (honors current dtype/device of the bank).
    with torch.no_grad():
        bank.parameter.data[layer_idx].copy_(linear.weight.data)

    # Drop the original Parameter. We also null the ``weight`` attribute
    # outright so future code paths that still peek at it (e.g. HF
    # ``_tie_weights``) break loudly instead of silently diverging.
    del linear._parameters["weight"]

    linear._bank_name = bank_name  # type: ignore[attr-defined]
    linear._bank_idx = layer_idx  # type: ignore[attr-defined]

    bias = linear.bias  # may be None; captured into closure
    # Wrap root in a list so nn.Module's __setattr__ does not try to
    # auto-register the nested module (which would create a cycle).
    _root = [root_model]
    name = bank_name
    idx = layer_idx

    def _forward(input: torch.Tensor, _root=_root, _name=name, _idx=idx, _bias=bias):
        # getattr resolves to whatever is CURRENTLY attached at
        # root_model.<name> — post-FSDP, this is the live DTensor.
        w = getattr(_root[0], _name)[_idx]
        return F.linear(input, w, _bias)

    linear.forward = _forward  # type: ignore[assignment]
