"""``BankScheme`` protocol — model-architecture-specific bank builder.

To support a new model family, implement this protocol and drop the
class into ``fsdp_bank``. The rest of the machinery (allocation,
FSDP2 wrapping, HiMuon 3D DTensor branch) is architecture-agnostic.
"""

from __future__ import annotations

from typing import Protocol

import torch.nn as nn

from .bank import BankSpec, ParamBank


class BankScheme(Protocol):
    """Architecture-specific bank builder.

    Implementations collect per-layer 2D weights from a model and
    rewrite each layer's forward to read its weight from a shared
    bank. The bank itself is sharded by FSDP2 on the layer axis.
    """

    def collect(self, model: nn.Module, world_size: int) -> list[BankSpec]:
        """Inspect ``model`` and produce bank specs.

        Implementations should pad each bank's first dim up to a
        multiple of ``world_size`` so ``Shard(0)`` divides evenly.
        """
        ...

    def install(self, model: nn.Module, banks: list[ParamBank]) -> nn.Module:
        """Rewire ``model`` to use the supplied banks.

        Expected side effects:
          * Each bank's nn.Parameter attached as ``model.<bank_name>``.
          * Each source Linear's ``weight`` Parameter deleted (no
            duplicate storage); its ``forward`` rewritten to index
            ``bank[layer_idx]``.
          * Original fwd semantics preserved (modulo numerical noise
            from dtype changes).

        Returns the same ``model`` for chaining.
        """
        ...
