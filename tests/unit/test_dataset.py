"""Contract for ``ToyDataset`` (``himuon.dataset``) using synthetic data only.

No mocks, no network: a deterministic fake tokenizer + an in-memory text stream
(both from ``conftest``) exercise the real iteration path.
"""

from __future__ import annotations

import pytest
import torch

from himuon.dataset import ToyDataset


def test_requires_eos_token(synthetic_text_stream, fake_tokenizer):
    fake_tokenizer.eos_token_id = None
    with pytest.raises(ValueError):
        ToyDataset(synthetic_text_stream, fake_tokenizer, max_length=8)


def test_sets_pad_token_to_eos(synthetic_text_stream, fake_tokenizer):
    ds = ToyDataset(synthetic_text_stream, fake_tokenizer, max_length=8)
    assert ds.tokenizer.pad_token == fake_tokenizer.eos_token


def test_yield_schema_and_shapes(synthetic_text_stream, fake_tokenizer):
    max_length = 8
    ds = ToyDataset(synthetic_text_stream, fake_tokenizer, max_length=max_length)
    item = next(iter(ds))
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    for key, value in item.items():
        assert isinstance(value, torch.Tensor), key
        assert value.shape == (max_length,), key


def test_labels_are_clone_of_input_ids(synthetic_text_stream, fake_tokenizer):
    ds = ToyDataset(synthetic_text_stream, fake_tokenizer, max_length=8)
    item = next(iter(ds))
    assert torch.equal(item["labels"], item["input_ids"])
    assert item["labels"].data_ptr() != item["input_ids"].data_ptr()
