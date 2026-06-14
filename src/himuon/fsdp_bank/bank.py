"""Data structures for parameter banks.

A ``BankSpec`` describes a bank at plan time — its name, per-matrix
shape, source parameter names, and any padding to make the first dim
a multiple of ``world_size``.

A ``ParamBank`` is a built bank: the actual ``nn.Parameter`` of shape
``(n_padded, H, W)`` plus the name-to-index map used by the scheme's
``install`` step to rewire each original weight's forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class BankSpec:
    """Plan-time description of a bank.

    Attributes:
        name: Identifier, e.g. ``"q_bank"``. Becomes a root-level
            attribute on the wrapped model (``model.q_bank``).
        matrix_shape: ``(H, W)`` of each matrix in this bank. All
            source params must already match this shape.
        source_param_names: HF-style dotted names in the order they
            occupy bank slots [0, 1, ..., n_logical-1].
        n_padded: First-dim of the allocated bank tensor, after
            padding up to a multiple of ``world_size``. Padding rows
            stay zero and are masked from training via a zero grad.
    """

    name: str
    matrix_shape: tuple[int, int]
    source_param_names: list[str]
    n_padded: int

    @property
    def n_logical(self) -> int:
        """Number of real (non-padding) matrices."""
        return len(self.source_param_names)

    @property
    def n_pad_rows(self) -> int:
        return self.n_padded - self.n_logical


@dataclass
class ParamBank:
    """A built bank: the nn.Parameter plus lookup metadata.

    Not yet FSDP2-wrapped. The caller (``wrap_with_banks``) constructs
    these, populates them from original weights, and hands them off for
    sharding.

    After ``fully_shard`` replaces ``model.<bank_name>`` in-place, the
    caller should call ``release_pre_shard_refs(banks)`` which sets
    ``parameter = None`` on every bank — otherwise each ``ParamBank``
    keeps an orphaned reference to the pre-shard full-size Parameter
    and GC can never reclaim that ~1/ws × ws = full-model memory.
    """

    spec: BankSpec
    parameter: "nn.Parameter | None"
    # Maps HF-style param name → layer index in the bank. Used by the
    # scheme's install step and by any downstream code that needs to
    # trace a bank slice back to an original parameter.
    name_to_idx: dict[str, int]
