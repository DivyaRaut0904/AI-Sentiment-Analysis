"""Vocabulary, sequence encoding, and PyTorch data utilities."""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def basic_tokenize(text: str) -> list[str]:
    """Lowercase text and extract words and punctuation."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", str(text).lower())


class Vocabulary:
    """A token-to-index mapping built only from training text."""

    def __init__(self, max_size: int = 30_000):
        self.max_size = max_size
        self.token_to_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.index_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts: Iterable[str]) -> None:
        counts = Counter()
        for text in texts:
            counts.update(basic_tokenize(text))
        for token, _ in counts.most_common(self.max_size - 2):
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.index_to_token[index] = token

    def encode(self, text: str, max_length: int) -> list[int]:
        tokens = basic_tokenize(text)
        indices = [self.token_to_index.get(token, self.token_to_index[UNK_TOKEN]) for token in tokens]
        indices = indices[:max_length]
        indices.extend([self.token_to_index[PAD_TOKEN]] * (max_length - len(indices)))
        return indices

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_size": self.max_size,
            "token_to_index": self.token_to_index,
            "index_to_token": {int(k): v for k, v in self.index_to_token.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocabulary = cls(payload["max_size"])
        vocabulary.token_to_index = payload["token_to_index"]
        vocabulary.index_to_token = {int(k): v for k, v in payload["index_to_token"].items()}
        return vocabulary

    def __len__(self) -> int:
        return len(self.token_to_index)


class ReviewSequenceDataset(Dataset):
    """A PyTorch dataset containing padded BiLSTM sequences."""

    def __init__(self, csv_path: Path, vocabulary: Vocabulary, max_length: int = 256):
        frame = pd.read_csv(csv_path)
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"{csv_path} must contain text and label columns.")
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.vocabulary.encode(self.texts[index], self.max_length)
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )
