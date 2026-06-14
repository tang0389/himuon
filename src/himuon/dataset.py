from torch.utils.data import IterableDataset
from typing import Any, Iterator


class ToyDataset(IterableDataset):
    def __init__(
        self, dataset: IterableDataset, tokenizer: Any, max_length: int = 1024
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure tokenizer has an eos_token
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an eos_token defined")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self) -> Iterator[dict]:
        for example in self.dataset:
            encoded = self.tokenizer(
                example["text"],
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            yield {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].detach().clone().squeeze(0),
            }
