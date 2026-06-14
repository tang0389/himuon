"""Distributed worker for test_himuon_fsdp_parity.py (not collected by pytest —
leading underscore). Launched under torchrun.

Asserts that HiMuon's FSDP per-bank in-place path
(``_step_muon_per_bank_inplace``) — the HiMuon-unique path that *tiles inside
each bank matrix* and batches Newton-Schulz over the rank-local shard with no
cross-rank collective — produces the same parameter trajectory as a
single-process per-matrix reference (the ordinary non-FSDP HiMuon path run on
each matrix of the bank independently).

The bank is sharded on dim 0 (the matrix/layer axis). Because every matrix's
tiles stay within that matrix, sharding the matrices across ranks must be
mathematically transparent: gather(rank-local results) == reference. A wrong
shard index, a tile straddling matrices, a dropped weight-decay, or cross-rank
contamination would break this.

Exit 0 = parity holds (asserted on rank 0); non-zero = mismatch / error.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from himuon.optimizers.himuon import HiMuon

# NS casts to bf16 internally and FSDP batches a different number of tiles per
# NS call than the per-matrix reference (rank-local N_local·R·C vs 8·R·C), so
# the torch.compile reduction tree differs by a few bf16 ulps. Matches the bar
# used elsewhere in the suite for cross-batch NS parity.
RTOL, ATOL = 1e-2, 1e-2


def _full(n: int, H: int, W: int, seed: int) -> torch.Tensor:
    """Deterministic (n, H, W) bf16 tensor, identical on every rank (CPU generator)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(n, H, W, generator=g).bfloat16()


class _Bank(nn.Module):
    def __init__(self, init: torch.Tensor):
        super().__init__()
        self.bank = nn.Parameter(init.clone())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--tile", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    assert args.n % world_size == 0, (
        f"n={args.n} must be divisible by world_size={world_size} so FSDP "
        f"sharding needs no padding (keeps the reference simple)."
    )

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    init = _full(args.n, args.H, args.W, args.seed).to(device)
    grads = [
        _full(args.n, args.H, args.W, args.seed + 1000 + k).to(device) for k in range(args.steps)
    ]

    opt_kwargs = dict(
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        tile_size=args.tile,
        ns_steps=args.ns_steps,
        cuda_graph=False,
    )

    # ---- FSDP per-bank path (the path under test) -------------------------
    model = _Bank(init).to(device)
    fully_shard(model, mesh=mesh)
    p = model.bank
    assert isinstance(p, DTensor), f"expected DTensor param, got {type(p)}"

    opt = HiMuon([{"params": [p], "use_muon": True}], **opt_kwargs)

    for k in range(args.steps):
        # Assign a DTensor grad matching the param's sharding (no autograd
        # needed; the per-bank path only reads p.grad.to_local()).
        p.grad = distribute_tensor(grads[k], mesh, p.placements)
        if k == 0:
            # Guard: confirm we're actually exercising the FSDP per-bank path
            # (needs a DTensor param *with* a grad), not the non-FSDP path.
            assert opt._has_dtensor_muon(), "FSDP per-bank path not selected"
        opt.step()

    full_after = p.full_tensor().to(device)  # (n, H, W), replicated on all ranks

    # ---- Single-process per-matrix reference (non-FSDP HiMuon path) -------
    ref_params = [nn.Parameter(init[i].clone()) for i in range(args.n)]
    ref_opt = HiMuon([{"params": ref_params, "use_muon": True}], **opt_kwargs)
    for k in range(args.steps):
        for i, rp in enumerate(ref_params):
            rp.grad = grads[k][i].clone()
        ref_opt.step()
    ref_after = torch.stack([rp.data for rp in ref_params], dim=0)

    rc = 0
    if rank == 0:
        if not torch.allclose(full_after, ref_after, rtol=RTOL, atol=ATOL):
            diff = (full_after.float() - ref_after.float()).abs()
            rel = diff / (ref_after.float().abs() + ATOL)
            print(
                f"PARITY FAIL n={args.n} H={args.H} W={args.W} tile={args.tile} "
                f"ws={world_size}: max|Δ|={diff.max().item():.3e} "
                f"max rel={rel.max().item():.3e}",
                flush=True,
            )
            rc = 1
        else:
            print(
                f"PARITY OK n={args.n} H={args.H} W={args.W} tile={args.tile} ws={world_size}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
