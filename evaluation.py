"""Evaluate saved models and generate comparison artifacts."""

from argparse import ArgumentParser
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src.models.bilstm_model import BiLSTMClassifier
from src.models.distilbert_model import load_distilbert
from src.preprocessing.text_preprocessing import ReviewSequenceDataset, Vocabulary
from src.utils import (
    LABEL_NAMES,
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    get_device,
)


def evaluate_bilstm(device: torch.device, test_frame: pd.DataFrame) -> dict:
    """Generate actual test predictions with the saved BiLSTM."""
    checkpoint_path = MODEL_DIR / "bilstm.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Train the BiLSTM first: python train.py --model bilstm")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BiLSTMClassifier(
        vocabulary_size=checkpoint["vocabulary_size"],
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    vocabulary = Vocabulary.load(MODEL_DIR / "bilstm_vocab.json")
    dataset = ReviewSequenceDataset(
        PROCESSED_DIR / "test.csv",
        vocabulary,
        checkpoint["max_length"],
    )
    loader = DataLoader(dataset, batch_size=64)
    predictions = []
    labels = []
    started = time.perf_counter()
    with torch.no_grad():
        for input_ids, batch_labels in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    inference_seconds = time.perf_counter() - started

    return model_metrics(
        labels,
        predictions,
        "BiLSTM",
        inference_seconds,
        len(test_frame),
    )


def evaluate_distilbert(device: torch.device) -> dict:
    """Generate actual test predictions with the fine-tuned DistilBERT."""
    model_dir = MODEL_DIR / "distilbert"
    if not model_dir.exists():
        raise FileNotFoundError(
            "Train DistilBERT first: python train.py --model distilbert"
        )
    tokenizer, model = load_distilbert(str(model_dir))
    model.to(device)
    model.eval()
    test_frame = pd.read_csv(PROCESSED_DIR / "test.csv")

    predictions = []
    labels = test_frame["label"].astype(int).tolist()
    texts = test_frame["text"].fillna("").astype(str).tolist()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start:start + 16],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    inference_seconds = time.perf_counter() - started
    return model_metrics(
        labels,
        predictions,
        "DistilBERT",
        inference_seconds,
        len(test_frame),
    )


def model_metrics(
    labels: list[int],
    predictions: list[int],
    model_name: str,
    inference_seconds: float,
    test_size: int,
) -> dict:
    """Calculate metrics exclusively from labels and model predictions."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(per_precision[i]),
                "recall": float(per_recall[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": matrix.tolist(),
        "inference_seconds": float(inference_seconds),
        "inference_seconds_per_1000": float(inference_seconds / test_size * 1000),
        "test_size": test_size,
    }


def plot_comparison(results: dict) -> None:
    """Plot macro metrics and inference time for both saved models."""
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    rows = []
    for model_name, result in results.items():
        for metric in metrics:
            rows.append({"model": model_name, "metric": metric, "value": result[metric]})
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=frame, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Evaluate trained sentiment models.")
    parser.add_argument(
        "--model",
        choices=["all", "bilstm", "distilbert"],
        default="all",
    )
    args = parser.parse_args()

    device = get_device()
    test_frame = pd.read_csv(PROCESSED_DIR / "test.csv")
    results = {}
    if args.model in ("all", "bilstm"):
        results["BiLSTM"] = evaluate_bilstm(device, test_frame)
    if args.model in ("all", "distilbert"):
        results["DistilBERT"] = evaluate_distilbert(device)

    training_times_path = OUTPUT_DIR / "training_times.json"
    training_times = (
        json.loads(training_times_path.read_text(encoding="utf-8"))
        if training_times_path.exists()
        else {}
    )
    for model_name, result in results.items():
        result["training_seconds"] = training_times.get(model_name.lower(), None)

    (OUTPUT_DIR / "evaluation_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    plot_comparison(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
