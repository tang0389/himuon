"""Correctness gate for HiMuon's FSDP per-bank in-place path.

The existing FSDP coverage (test_himuon_fsdp_smoke) only checks the end-to-end
pipeline survives and loss stays finite. It does *not* check the per-bank update
is numerically correct. This file does, via a distributed parity worker
(``_fsdp_worker.py``) launched under torchrun: it runs HiMuon on a
bank sharded across ranks and asserts the gathered result equals a
single-process per-matrix reference.

This targets the HiMuon-unique behaviour: tiling *inside* each bank matrix and
batching Newton-Schulz over the rank-local shard (MuonFsdp only does the
whole-matrix R=C=1 case). Run with ``-m slow``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.distributed,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 GPUs required for FSDP2"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).parent / "_fsdp_worker.py"


def _torchrun() -> str:
    venv_bin = Path(sys.executable).parent / "torchrun"
    return str(venv_bin) if venv_bin.exists() else "torchrun"


# (label, n, H, W, tile) — exercise the HiMuon-unique within-bank tiling:
#   * tiled, tile divides H/W          (the common HiMuon path)
#   * tiled, NON-divisible → zero-pad  (padding logic inside the bank)
# Both keep the per-bank NS on 128x128 tiles, so the per-bank path and the 2D
# reference go through the *same* ns5_smem kernel — the only thing under test is
# whether sharding the matrices across ranks changes the result.
#
# The whole-matrix branch (tile >= H,W, R=C=1) is intentionally NOT covered
# here: it is "equivalent to MuonFsdp" (not HiMuon-unique), and it routes a 3D
# batch through ns5_smem vs the 2D reference's 3-kernel path, so it would be
# measuring NS-kernel parity (covered by test_kernels), not sharding correctness.
#
# n=8 so it shards cleanly across both 2 and 4 ranks (no FSDP padding).
_CONFIGS = [
    ("tiled_divisible", 8, 256, 512, 128),
    ("tiled_padded", 8, 200, 300, 128),
]


def _run(nproc: int, n: int, H: int, W: int, tile: int) -> None:
    cmd = [
        _torchrun(),
        "--standalone",
        f"--nproc-per-node={nproc}",
        str(WORKER),
        "--n",
        str(n),
        "--H",
        str(H),
        "--W",
        str(W),
        "--tile",
        str(tile),
        "--steps",
        "4",
        "--wd",
        "0.05",
    ]
    env = os.environ.copy()
    # Keep NS deterministic enough for the bf16 tolerance bar.
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cp = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    combined = cp.stdout + "\n" + cp.stderr
    if cp.returncode != 0:
        raise AssertionError(
            f"FSDP parity worker (nproc={nproc}) exited {cp.returncode}\n"
            f"--- STDOUT ---\n{cp.stdout}\n--- STDERR ---\n{cp.stderr}"
        )
    assert "PARITY OK" in combined, (
        f"parity marker missing (nproc={nproc})\n"
        f"--- STDOUT ---\n{cp.stdout}\n--- STDERR ---\n{cp.stderr}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("label,n,H,W,tile", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_fsdp_per_bank_parity_ws2(label, n, H, W, tile):
    _run(2, n, H, W, tile)


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="4 GPUs required for ws=4 sharding")
@pytest.mark.parametrize("label,n,H,W,tile", _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_fsdp_per_bank_parity_ws4(label, n, H, W, tile):
    _run(4, n, H, W, tile)
