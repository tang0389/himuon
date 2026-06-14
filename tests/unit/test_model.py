"""Contract for model-stat helpers (``himuon.model``).

CPU-only, no HF download/network: ``get_model_and_tokenizer`` (which hits the
Hub) is out of scope; only the pure counting/formatting helpers are pinned.
Large param counts are simulated with ``meta`` tensors (shape/numel without
storage).
"""

from __future__ import annotations

import pytest
import torch

from himuon.model import count_model_parameters, count_num_training_tokens


class _FakeModel:
    """Minimal object exposing only ``parameters()`` (all the helpers use)."""

    def __init__(self, params):
        self._params = list(params)

    def parameters(self):
        return self._params


def _meta_param(numel: int, requires_grad: bool = True):
    return torch.empty(numel, device="meta", requires_grad=requires_grad)


def test_counts_only_trainable():
    model = _FakeModel([_meta_param(10, True), _meta_param(5, False)])
    n, _ = count_model_parameters(model)
    assert n == 10


@pytest.mark.parametrize(
    "numel, suffix",
    [
        (5_000, " trainable parameters"),
        (5_000_000, "M trainable parameters"),
        (500_000_000, "B trainable parameters"),
    ],
)
def test_format_buckets(numel, suffix):
    model = _FakeModel([_meta_param(numel)])
    n, text = count_model_parameters(model)
    assert n == numel
    assert text.endswith(suffix)


def test_training_tokens_chinchilla_formula():
    model = _FakeModel([_meta_param(1000)])
    assert count_num_training_tokens(model, chinchilla_factor=2.0) == int(20 * 2.0 * 1000)
