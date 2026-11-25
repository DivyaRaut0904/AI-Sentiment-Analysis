"""Bidirectional LSTM sentiment classifier implemented in PyTorch."""

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    """Learn review representations with a bidirectional LSTM."""

    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int = 3,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        padding_index: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=padding_index,
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embeddings)
        # Concatenate the final forward and backward LSTM states.
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        representation = self.dropout(representation)
        return self.classifier(representation)
