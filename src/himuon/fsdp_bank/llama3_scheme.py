"""Bank scheme for Llama 3 decoder-only models.

Covers the Llama 3 base text models within reach of single-node FSDP:
``meta-llama/Llama-3.1-8B``, ``meta-llama/Llama-3.2-3B`` and
``meta-llama/Llama-3.2-1B``. All three are ``LlamaForCausalLM`` and
share the per-layer projection layout this scheme keys on, so one
class covers the lot — only head counts / hidden sizes differ, and the
bank shapes follow from the weights themselves.

Groups each layer's 2D weight matrices into 5 banks:
  * ``q_bank``       — q_proj.weight, per layer
  * ``kv_bank``      — k_proj + v_proj interleaved per layer (same shape)
  * ``o_bank``       — o_proj.weight, per layer
  * ``gate_up_bank`` — gate_proj + up_proj interleaved (same shape)
  * ``down_bank``    — down_proj.weight, per layer

Merging same-shape projections (k+v, gate+up) halves the number of
banks and doubles the cross-layer NS batch size in HiMuon for free.
Llama 3 uses GQA, so k/v are narrower than q — but k and v still share
a shape with each other, which is all ``kv_bank`` needs.

``embed_tokens``, ``lm_head``, and all RMSNorm weights stay as regular
parameters — 1D norms run AdamW, and the embeddings are separately
large enough that banking them doesn't help.
"""

from __future__ import annotations

import torch.nn as nn

from .bank import BankSpec, ParamBank
from .wrap import get_layers, rewire_linear


class Llama3BankScheme:
    BANK_NAMES = ("q_bank", "kv_bank", "o_bank", "gate_up_bank", "down_bank")

    # -- collect -----------------------------------------------------------

    def collect(self, model: nn.Module, world_size: int) -> list[BankSpec]:
        layers = get_layers(model)
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
        kv_names = [name for pair in zip(k_names, v_names, strict=True) for name in pair]
        gate_up_names = [name for pair in zip(gate_names, up_names, strict=True) for name in pair]

        q_shape = tuple(layers[0].self_attn.q_proj.weight.shape)
        k_shape = tuple(layers[0].self_attn.k_proj.weight.shape)
        v_shape = tuple(layers[0].self_attn.v_proj.weight.shape)
        o_shape = tuple(layers[0].self_attn.o_proj.weight.shape)
        gate_shape = tuple(layers[0].mlp.gate_proj.weight.shape)
        up_shape = tuple(layers[0].mlp.up_proj.weight.shape)
        down_shape = tuple(layers[0].mlp.down_proj.weight.shape)
        assert k_shape == v_shape, f"k/v shape mismatch: {k_shape} vs {v_shape}"
        assert gate_shape == up_shape, f"gate/up shape mismatch: {gate_shape} vs {up_shape}"

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
            # model. HF keeps ``model.model = LlamaModel`` so the banks
            # end up on the *outer* ``LlamaForCausalLM``. This keeps
            # them visible to ``model.named_parameters()`` for the
            # optimizer factory without colliding with HF's own names.
            setattr(model, b.spec.name, b.parameter)

        layers = get_layers(model)
        for i, layer in enumerate(layers):
            rewire_linear(layer.self_attn.q_proj, model, "q_bank", by_name["q_bank"], i)
            rewire_linear(layer.self_attn.k_proj, model, "kv_bank", by_name["kv_bank"], 2 * i)
            rewire_linear(layer.self_attn.v_proj, model, "kv_bank", by_name["kv_bank"], 2 * i + 1)
            rewire_linear(layer.self_attn.o_proj, model, "o_bank", by_name["o_bank"], i)
            rewire_linear(
                layer.mlp.gate_proj,
                model,
                "gate_up_bank",
                by_name["gate_up_bank"],
                2 * i,
            )
            rewire_linear(
                layer.mlp.up_proj,
                model,
                "gate_up_bank",
                by_name["gate_up_bank"],
                2 * i + 1,
            )
            rewire_linear(layer.mlp.down_proj, model, "down_bank", by_name["down_bank"], i)

        return model
