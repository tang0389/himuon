"""Orchestration: take an HF model + a scheme, build banks, rewire.

The output is a model that reads weights from its banks via the
scheme's install step. The caller then runs ``fully_shard`` on the
(rewired) model to convert each bank into a 3D DTensor sharded on
the layer axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bank import ParamBank
from .scheme import BankScheme


def wrap_with_banks(
    model: nn.Module,
    scheme: BankScheme,
    world_size: int,
) -> tuple[nn.Module, list[ParamBank]]:
    """Build banks from ``model`` and install them.

    Steps:
      1. ``scheme.collect(model, ws)`` → list of ``BankSpec``.
      2. Allocate each bank's ``nn.Parameter`` with shape
         ``(n_padded, H, W)``, device/dtype inherited from model.
         Padding rows are zero-initialised; they'll carry zero grad
         from their source (there is no source), so they stay zero.
      3. ``scheme.install(model, banks)`` — copies weights from each
         original ``nn.Linear`` into the right bank slice, deletes
         the original weight, rewires forward.

    Returns ``(rewired_model, banks)`` for the caller to FSDP2-wrap.
    """
    specs = scheme.collect(model, world_size)

    # Inherit device + dtype from the model's first parameter. A model
    # split across devices would need per-bank device selection; not supported.
    first = next(model.parameters())
    device = first.device
    dtype = first.dtype

    banks: list[ParamBank] = []
    for spec in specs:
        H, W = spec.matrix_shape
        p = nn.Parameter(torch.zeros(spec.n_padded, H, W, device=device, dtype=dtype))
        name_to_idx = {name: i for i, name in enumerate(spec.source_param_names)}
        banks.append(ParamBank(spec=spec, parameter=p, name_to_idx=name_to_idx))

    scheme.install(model, banks)
    return model, banks


def release_pre_shard_refs(banks: list[ParamBank]) -> None:
    """Drop ``ParamBank.parameter`` references after ``fully_shard``.

    ``fully_shard`` replaces ``model.<bank_name>`` in-place with a new
    sharded DTensor Parameter. The old un-sharded nn.Parameter that
    ``ParamBank.parameter`` captured during ``wrap_with_banks`` becomes
    orphaned but stays alive via the ``banks`` list — occupying the
    full un-sharded size on every rank (8 GB for Qwen3-4B).

    Call this **after** ``fully_shard(model, ...)`` to free that memory:

        model, banks = wrap_with_banks(model, scheme, world_size=ws)
        fully_shard(model, mesh=mesh, ...)
        release_pre_shard_refs(banks)
        torch.cuda.empty_cache()  # optional, returns cached blocks

    After release, ``banks[i].parameter`` is ``None``; ``.spec`` and
    ``.name_to_idx`` remain valid for introspection / logging.
    """
    for b in banks:
        b.parameter = None  # type: ignore[assignment]


def unbank_state_dict(
    banked_state_dict: dict,
    banks: list[ParamBank],
) -> dict:
    """Map a bank-keyed state_dict to HF-keyed one for ``save_pretrained``.

    Given the full (gathered) state_dict of a bank-wrapped model, split
    each bank tensor ``(n_padded, H, W)`` into its ``n_logical`` slots
    and re-key them under their original HF param names. Non-bank keys
    (embed, lm_head, LNs, etc.) pass through unchanged.

    The returned dict is loadable by ``AutoModelForCausalLM.from_pretrained``
    since keys match the original (non-banked) HF layout.
    """
    bank_names = {b.spec.name for b in banks}
    out = {k: v for k, v in banked_state_dict.items() if k not in bank_names}
    for b in banks:
        if b.spec.name not in banked_state_dict:
            raise KeyError(
                f"bank {b.spec.name!r} missing from state_dict "
                f"(keys: {sorted(banked_state_dict)[:5]}...)"
            )
        bank_tensor = banked_state_dict[b.spec.name]
        for idx, hf_name in enumerate(b.spec.source_param_names):
            out[hf_name] = bank_tensor[idx].clone()
    return out


# -- shared scheme helpers -------------------------------------------------


def get_layers(model: nn.Module):
    """Locate the list of decoder layers in an HF causal-LM model.

    The decoder-only families this package banks (Qwen3, Llama 3) wrap the
    transformer as ``<Model>ForCausalLM.model.layers``. Hold the indirection
    behind this helper so a future HF refactor only touches one line.
    """
    return model.model.layers


def rewire_linear(
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
