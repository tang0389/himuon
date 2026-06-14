"""Shared fixtures + auto-marking for the GPU integration tier.

Every test under ``tests/integration/`` exercises ``HiMuon.step()`` (Triton/CUDA)
or a distributed launch, so it is auto-marked ``gpu`` here — the CPU CI tier
(``-m "not gpu"``) skips the whole directory without each file having to
remember the marker. FSDP files add ``distributed`` explicitly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some integration modules import Triton kernel submodules at module load (and
# Triton resolves the GPU driver at import). On a CPU-only host (the CI tier)
# that errors during *collection*, before marker deselection can skip them — so
# refuse to collect this directory at all when there is no CUDA device.
if not torch.cuda.is_available():
    collect_ignore_glob = ["*.py"]


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)


@pytest.fixture
def make_model():
    """Build an ``nn.ModuleList`` of ``Linear(W, H)`` layers on the device."""

    def _make(shapes, dtype: torch.dtype = torch.float32, bias: bool = True):
        layers = [nn.Linear(W, H, bias=bias, dtype=dtype) for (H, W) in shapes]
        return nn.ModuleList(layers).to(DEVICE)

    return _make


@pytest.fixture
def build_groups():
    """Mirror optim.py grouping: 2-D → muon, 1-D → AdamW-fallback no-decay."""

    def _build(model):
        muon = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        one_d = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        groups = []
        if muon:
            groups.append({"params": muon, "weight_decay": 0.0, "use_muon": True})
        if one_d:
            groups.append({"params": one_d, "lr": 1e-3, "weight_decay": 0.0, "use_muon": False})
        return groups

    return _build


@pytest.fixture
def seed_grads():
    """Fill grad buffers in place (stable addresses — CUDA-graph contract)."""

    def _seed(model, seed: int):
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(seed)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.empty(p.shape, device=DEVICE, dtype=p.dtype)
            p.grad.normal_(generator=gen)

    return _seed
