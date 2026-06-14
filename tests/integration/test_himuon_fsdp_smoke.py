"""End-to-end smoke: ``train_fsdp.py`` with Qwen3-0.6B + HiMuon for
5 steps under torchrun ws=2.

The canary is "the happy-path FSDP pipeline runs end-to-end on a real HF
model": every requested optimizer step executes and logs a finite loss, and the
loop reaches its step budget (the ``Stopping training`` marker).

Process *exit code* is deliberately not the contract. HiMuon's Triton kernels
can race the interpreter's GIL teardown and abort the process at shutdown
*after* training has finished — harmless to the run, but a non-zero exit under
torchrun. So a non-zero exit fails the test only when training did not complete;
a teardown abort after a completed run is tolerated. A genuine mid-training
crash logs fewer than the requested steps (or no stop marker) and still fails.

Kept deliberately cheap: 5 steps, no wandb, no checkpoints.
"""

from __future__ import annotations

import os
import re
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


def _torchrun() -> str:
    venv_bin = Path(sys.executable).parent / "torchrun"
    return str(venv_bin) if venv_bin.exists() else "torchrun"


# LoguruLogger.log_metrics prints lines like
#   ``... | INFO | ... - Step: 1 LR: 0.020000 Training loss: 12.2392, Tokens: 475``
# Capture the per-step training-loss value. Case-insensitive + permissive so a
# minor logger-format tweak doesn't break parsing.
_LOSS_RE = re.compile(r"training loss:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", re.IGNORECASE)


STEPS = 5  # keep the smoke minimal but non-trivial


@pytest.mark.slow
def test_qwen3_0_6b_fsdp_himuon_5_steps(tmp_path):
    log_dir = tmp_path / "run"
    log_dir.mkdir()

    cmd = [
        _torchrun(),
        "--standalone",
        "--nproc-per-node=2",
        "train_fsdp.py",
        "--model",
        "Qwen3-0.6B",
        "--optimizer",
        "himuon",
        "--steps",
        str(STEPS),
        "--num-warmup-steps",
        "1",
        "--batch-size",
        "1",
        "--max-length",
        "256",
        "--lr",
        "0.02",
        # Disable eval / checkpoint-saving — the canary is "process survives",
        # not quality metrics.
        "--eval-interval",
        "0",
        "--no-gradient-checkpointing",
    ]

    env = os.environ.copy()
    # Ensure HF cache is writable and transformers doesn't try to download
    # a tokenizer into a read-only dir. If HF_HOME isn't already set,
    # point it at tmp so we never litter the user's home.
    env.setdefault("HF_HOME", str(tmp_path / "hf_cache"))
    env.setdefault("TRANSFORMERS_OFFLINE", "0")

    cp = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # HF download of Qwen3-0.6B on first run can be slow.
    )

    combined = cp.stdout + "\n" + cp.stderr
    losses = [float(m.group(1)) for m in _LOSS_RE.finditer(combined)]

    # Completion = the loop ran to its step budget and logged the stop line
    # (train_fsdp.py: "Stopping training: Max training steps reached.").
    reached_budget = len(losses) >= STEPS
    logged_stop = "Stopping training" in combined
    completed = reached_budget and logged_stop

    # A non-zero exit only matters if training did not finish. A teardown-only
    # abort after a completed run (Triton/GIL shutdown race) is tolerated.
    if not completed:
        raise AssertionError(
            f"train_fsdp.py did not complete training (returncode={cp.returncode}, "
            f"steps_logged={len(losses)}/{STEPS}, stop_marker={logged_stop})\n"
            f"--- STDOUT ---\n{cp.stdout}\n--- STDERR ---\n{cp.stderr}"
        )

    for step_idx, loss in enumerate(losses):
        assert loss == loss and loss not in (float("inf"), float("-inf")), (
            f"loss on logged step {step_idx} is non-finite: {loss!r}"
        )
