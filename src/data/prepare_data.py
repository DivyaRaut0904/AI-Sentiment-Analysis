"""Download, inspect, clean, sample, and split the Yelp review dataset."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.utils import (
    DATASET_ID,
    LABEL_NAMES,
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    seed_everything,
)


def rating_to_sentiment(rating: int) -> int:
    """Map star rating (1-5) to Negative=0, Neutral=1, Positive=2."""
    if rating in (1, 2):
        return 0
    if rating == 3:
        return 1
    return 2


def stratified_sample(frame: pd.DataFrame, requested_rows: int, seed: int) -> pd.DataFrame:
    """Select a balanced-as-available stratified sample from a cleaned frame."""
    counts = frame["label"].value_counts()
    if frame.empty:
        raise ValueError("No rows remain after cleaning.")
    target = min(requested_rows, len(frame))
    if len(counts) < 3:
        raise ValueError("The data is missing one or more sentiment classes.")
    sample, _ = train_test_split(
        frame,
        train_size=target,
        random_state=seed,
        stratify=frame["label"],
    )
    return sample.reset_index(drop=True)


def prepare_data(
    train_size: int = 5_000,
    test_size: int = 1_250,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> dict:
    """Download and prepare non-overlapping train, validation, and test sets."""
    seed_everything(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID)
    train_raw = dataset["train"].to_pandas()
    test_raw = dataset["test"].to_pandas()
    train_raw.to_csv(RAW_DIR / "yelp_train.csv", index=False)
    test_raw.to_csv(RAW_DIR / "yelp_test.csv", index=False)

    train_df = train_raw.rename(columns={"text": "text", "label": "rating"})
    test_df = test_raw.rename(columns={"text": "text", "label": "rating"})
    train_df["rating"] = train_df["rating"].astype(int) + 1
    test_df["rating"] = test_df["rating"].astype(int) + 1
    raw_missing = int(train_df["text"].isna().sum() + test_df["text"].isna().sum())
    raw_duplicates = int(
        train_df.duplicated(subset="text").sum()
        + test_df.duplicated(subset="text").sum()
    )

    train_df["text"] = train_df["text"].fillna("").astype(str).str.strip()
    test_df["text"] = test_df["text"].fillna("").astype(str).str.strip()

    train_clean = train_df.loc[train_df["text"].ne("")].copy()
    test_clean = test_df.loc[test_df["text"].ne("")].copy()
    train_clean = train_clean.drop_duplicates(subset="text")
    test_clean = test_clean.drop_duplicates(subset="text")
    test_clean = test_clean.loc[~test_clean["text"].isin(set(train_clean["text"]))]

    train_clean["label"] = train_clean["rating"].map(rating_to_sentiment)
    test_clean["label"] = test_clean["rating"].map(rating_to_sentiment)

    train_sample = stratified_sample(train_clean, train_size, seed)
    test_sample = stratified_sample(test_clean, test_size, seed + 1)

    train_df, val_df = train_test_split(
        train_sample,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_sample["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_sample = test_sample.reset_index(drop=True)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_sample.to_csv(PROCESSED_DIR / "test.csv", index=False)

    write_eda_outputs(train_df, val_df, test_sample)
    summary = {
        "dataset": DATASET_ID,
        "seed": seed,
        "raw_missing_text_rows": raw_missing,
        "raw_duplicate_text_rows": raw_duplicates,
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_sample),
        "train_distribution": Counter(map(int, train_df["label"])).most_common(),
        "validation_distribution": Counter(map(int, val_df["label"])).most_common(),
        "test_distribution": Counter(map(int, test_sample["label"])).most_common(),
    }
    (OUTPUT_DIR / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def write_eda_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Create basic sentiment and review-length charts."""
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined["sentiment"] = combined["label"].map(dict(enumerate(LABEL_NAMES)))
    combined["review_length"] = combined["text"].str.split().str.len()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=combined, x="sentiment", hue="sentiment", order=LABEL_NAMES, ax=axes[0], legend=False)
    axes[0].set_title("Sentiment Distribution")
    sns.histplot(combined["review_length"], bins=50, ax=axes[1])
    axes[1].set_title("Review Length Distribution")
    axes[1].set_xlabel("Word count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150)
    plt.close(fig)

    split_counts = pd.DataFrame({
        "split": ["Train"] * 3 + ["Validation"] * 3 + ["Test"] * 3,
        "sentiment": LABEL_NAMES * 3,
        "count": [
            int((train_df["label"] == i).sum()) for i in range(3)
        ] + [
            int((val_df["label"] == i).sum()) for i in range(3)
        ] + [
            int((test_df["label"] == i).sum()) for i in range(3)
        ],
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=split_counts, x="sentiment", y="count", hue="split", ax=ax)
    ax.set_title("Sentiment Distribution by Split")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sentiment_by_split.png", dpi=150)
    plt.close(fig)

    from collections import Counter
    import re

    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "had", "has", "have", "he", "her", "his", "i", "in", "is", "it", "its",
        "me", "my", "of", "on", "or", "our", "she", "so", "that", "the", "their",
        "them", "they", "this", "to", "was", "we", "were", "with", "you", "your",
    }
    word_counts = Counter()
    for text in combined["text"].astype(str):
        words = re.findall(r"[a-z0-9]+", text.lower())
        word_counts.update(word for word in words if word not in stop_words and len(word) > 2)
    common = word_counts.most_common(20)
    word_frame = pd.DataFrame(common, columns=["word", "count"])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=word_frame, x="count", y="word", ax=ax)
    ax.set_title("Top 20 Words")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_words.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = ArgumentParser(description="Prepare the Yelp sentiment dataset.")
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=1_250)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_data(args.train_size, args.test_size, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
