"""Parameter-bank sharding for FSDP2 + HiMuon.

Groups same-shape per-layer weights into 3D "banks", then shards each
bank along its layer axis (dim 0) with FSDP2. Each rank owns a set of
complete ``(H, W)`` matrices; tiles in HiMuon never straddle shard
boundaries because the sharded axis is the bank's layer dim, not the
matrix's H or W.

Typical use::

    from himuon.fsdp_bank import Qwen3BankScheme, wrap_with_banks

    model = ...  # HF Qwen3 model
    scheme = Qwen3BankScheme()
    model, banks = wrap_with_banks(model, scheme, world_size=ws)
    fully_shard(model, mesh=mesh, ...)  # each bank becomes 3D DTensor
"""

from .bank import BankSpec, ParamBank
from .llama3_scheme import Llama3BankScheme
from .qwen3_scheme import Qwen3BankScheme
from .scheme import BankScheme
from .wrap import release_pre_shard_refs, unbank_state_dict, wrap_with_banks

__all__ = [
    "BankSpec",
    "ParamBank",
    "BankScheme",
    "Qwen3BankScheme",
    "Llama3BankScheme",
    "wrap_with_banks",
    "release_pre_shard_refs",
    "unbank_state_dict",
]
