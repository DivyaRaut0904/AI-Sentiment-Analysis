"""Train the BiLSTM and DistilBERT sentiment classifiers with PyTorch."""

from argparse import ArgumentParser
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.prepare_data import prepare_data
from src.models.bilstm_model import BiLSTMClassifier
from src.models.distilbert_model import DISTILBERT_MODEL
from src.preprocessing.text_preprocessing import (
    ReviewSequenceDataset,
    Vocabulary,
)
from src.utils import (
    LABEL_NAMES,
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    get_device,
    seed_everything,
)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def train_bilstm(
    device: torch.device,
    epochs: int,
    batch_size: int,
    max_length: int,
    vocab_size: int,
) -> dict:
    """Train and validate the bidirectional LSTM."""
    train_frame = pd.read_csv(PROCESSED_DIR / "train.csv")
    vocabulary = Vocabulary(max_size=vocab_size)
    vocabulary.build(train_frame["text"].fillna("").astype(str))
    vocabulary.save(MODEL_DIR / "bilstm_vocab.json")

    train_dataset = ReviewSequenceDataset(
        PROCESSED_DIR / "train.csv", vocabulary, max_length
    )
    val_dataset = ReviewSequenceDataset(
        PROCESSED_DIR / "validation.csv", vocabulary, max_length
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = BiLSTMClassifier(
        vocabulary_size=len(vocabulary),
        num_classes=len(LABEL_NAMES),
        embedding_dim=128,
        hidden_dim=128,
        dropout=0.3,
    ).to(device)

    class_counts = train_frame["label"].value_counts().to_dict()
    weights = torch.tensor(
        [train_frame.shape[0] / (3.0 * class_counts.get(i, 1)) for i in range(3)],
        dtype=torch.float,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = []
    best_val_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_accuracy += accuracy(logits, labels) * len(labels)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                val_accuracy += accuracy(logits, labels) * len(labels)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        train_accuracy /= len(train_dataset)
        val_accuracy /= len(val_dataset)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"BiLSTM epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary_size": len(vocabulary),
                    "num_classes": len(LABEL_NAMES),
                    "embedding_dim": 128,
                    "hidden_dim": 128,
                    "max_length": max_length,
                    "label_names": LABEL_NAMES,
                },
                MODEL_DIR / "bilstm.pt",
            )

    elapsed = time.perf_counter() - started
    (OUTPUT_DIR / "bilstm_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {"training_seconds": elapsed, "history": history}


class DistilBertReviewDataset(torch.utils.data.Dataset):
    """Tokenized review dataset for DistilBERT fine-tuning."""

    def __init__(self, csv_path: Path, tokenizer, max_length: int):
        frame = pd.read_csv(csv_path)
        self.texts = frame["text"].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def train_distilbert(
    device: torch.device,
    epochs: int,
    batch_size: int,
    max_length: int,
) -> dict:
    """Fine-tune DistilBERT using a standard PyTorch training loop."""
    from src.models.distilbert_model import load_distilbert

    tokenizer, model = load_distilbert(DISTILBERT_MODEL)
    model.to(device)
    train_dataset = DistilBertReviewDataset(
        PROCESSED_DIR / "train.csv", tokenizer, max_length
    )
    val_dataset = DistilBertReviewDataset(
        PROCESSED_DIR / "validation.csv", tokenizer, max_length
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_frame = pd.read_csv(PROCESSED_DIR / "train.csv")
    class_counts = train_frame["label"].value_counts().to_dict()
    weights = torch.tensor(
        [train_frame.shape[0] / (3.0 * class_counts.get(i, 1)) for i in range(3)],
        dtype=torch.float,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    history = []
    best_val_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = nn.functional.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                logits = model(**batch).logits
                loss = nn.functional.cross_entropy(logits, labels, weight=weights)
                val_loss += loss.item() * len(labels)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_correct / len(train_dataset),
            "val_loss": val_loss,
            "val_accuracy": val_correct / len(val_dataset),
        }
        history.append(row)
        print(
            f"DistilBERT epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_accuracy={row['val_accuracy']:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = MODEL_DIR / "distilbert"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    elapsed = time.perf_counter() - started
    (OUTPUT_DIR / "distilbert_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    return {"training_seconds": elapsed, "history": history}


def main() -> None:
    parser = ArgumentParser(description="Train sentiment models.")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_250)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=30_000)
    parser.add_argument(
        "--model",
        choices=["all", "bilstm", "distilbert"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.prepare_data or not (PROCESSED_DIR / "train.csv").exists():
        prepare_data(args.train_size, args.test_size, 0.20, args.seed)

    device = get_device()
    print(f"Using device: {device}")
    timings = {}
    if args.model in ("all", "bilstm"):
        print("Training BiLSTM...")
        timings["bilstm"] = train_bilstm(
            device, args.epochs, args.batch_size, args.max_length, args.vocab_size
        )
    if args.model in ("all", "distilbert"):
        print("Training DistilBERT...")
        timings["distilbert"] = train_distilbert(
            device, args.epochs, args.batch_size, args.max_length
        )
    (OUTPUT_DIR / "training_times.json").write_text(
        json.dumps(
            {key: value["training_seconds"] for key, value in timings.items()},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
