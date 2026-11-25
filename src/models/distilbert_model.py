"""DistilBERT fine-tuning helpers."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer


DISTILBERT_MODEL = "distilbert-base-uncased"


def load_distilbert(model_path: str | None = None):
    """Load a pretrained or locally fine-tuned DistilBERT classifier."""
    model_name = model_path or DISTILBERT_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )
    return tokenizer, model
