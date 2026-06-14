"""Orchestration: take an HF model + a scheme, build banks, rewire.

The output is a model that reads weights from its banks via the
scheme's install step. The caller then runs ``fully_shard`` on the
(rewired) model to convert each bank into a 3D DTensor sharded on
the layer axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

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
    # split across devices would need per-bank device selection; Qwen3
    # single-device is what we support for now.
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
