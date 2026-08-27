"""
Text preprocessing and tokenization for the Financial PhraseBank dataset.

Author: Manuel Corona
"""

import re
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
TICKER_RE = re.compile(r"\$[A-Za-z]{1,5}\b")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Normalize a financial sentence: lowercase, strip URLs/mentions/tickers,
    collapse whitespace. Stop words and financial terms are kept intact.

    Args:
        text: Raw input sentence.

    Returns:
        Cleaned sentence.
    """
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = TICKER_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


class TokenizedFinancialDataset(Dataset):
    """PyTorch Dataset that tokenizes texts eagerly at construction time."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            list(texts),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
