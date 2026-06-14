"""Pytest configuration shared across the test suite."""

import pytest
import torch

# HiMuon.newton_schulz is decorated with @torch.compile(dynamic=False,
# fullgraph=True). Each unique (shape, rank) observed at runtime triggers a
# fresh Dynamo compilation. The default per-function cache limit is 8, which
# is exhausted quickly by the locked upgrade-2 parity matrix (hundreds of
# parametrized invocations spanning many shape variants) and by any future
# multi-shape benchmark. Raise the limit globally for the test suite so that
# RecompileLimitExceeded does not mask legitimate test failures.
torch._dynamo.config.cache_size_limit = 1024


@pytest.fixture(autouse=True)
def _cuda_cleanup_between_tests():
    """Drain CUDA caches between tests to prevent state leakage (autotune
    side state, fragmentation, leftover allocations from earlier tests)
    from causing spurious failures later in the session.

    Observed in full GPU runs: tile_size=512 cases can fail in aggregate
    even when each passes in isolation, because autotune side state and
    allocator fragmentation leak across tests. A ``sync + empty_cache``
    after each test restores a stable red/green signal for a few ms cost.
    On CPU-only runs (the default CI tier) this is a no-op.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
