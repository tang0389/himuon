"""Shared CPU fixtures for the unit (contract) layer.

Everything here is pure-CPU and offline: a tiny causal-LM-shaped module whose
submodule names mirror the framework (``embed_tokens`` / ``lm_head``), a minimal
fake tokenizer exposing only the surface ``ToyDataset`` relies on, and a
synthetic text stream. No HF downloads, no network, no GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


class TinyCausalLM(nn.Module):
    """Minimal stand-in for an HF causal LM.

    Submodule names match what ``himuon.optim.get_optimizer`` keys on:
    ``embed_tokens`` / ``lm_head`` route to the AdamW-2D group, the inner
    ``Linear`` weight to the Muon group, and every 1-D param (norm
    weight/bias, linear bias) to the AdamW-1D no-decay group.
    """

    def __init__(self, vocab: int = 32, dim: int = 16, ffn: int = 24):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, ffn, bias=True)
        self.lm_head = nn.Linear(dim, vocab, bias=False)


@pytest.fixture
def tiny_causal_lm() -> TinyCausalLM:
    torch.manual_seed(0)
    return TinyCausalLM()


class FakeTokenizer:
    """Deterministic offline tokenizer with only what ``ToyDataset`` uses.

    Maps each character to ``ord(c) % vocab`` and supports the keyword
    surface (``truncation`` / ``padding='max_length'`` / ``max_length`` /
    ``return_tensors='pt'``) plus the ``eos``/``pad`` token attributes the
    dataset reads and mutates.
    """

    def __init__(self, vocab: int = 256, eos_token_id: int | None = 0):
        self.vocab = vocab
        self.eos_token = "<eos>"
        self.eos_token_id = eos_token_id
        self.pad_token = None

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=8,
        return_tensors="pt",
    ):
        ids = [ord(c) % self.vocab for c in text][:max_length]
        attn = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [self.eos_token_id] * pad
            attn = attn + [0] * pad
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attn, dtype=torch.long).unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


class _TextStream(IterableDataset):
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        for r in self.rows:
            yield {"text": r}


@pytest.fixture
def synthetic_text_stream() -> _TextStream:
    return _TextStream(["hello world", "himuon optimizer", "synthetic data only"])
