"""Generate colorful project diagrams for documentation and presentations."""

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.utils import OUTPUT_DIR


def draw_box(ax, xy, width, height, title, subtitle, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height * 0.28,
        subtitle,
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        alpha=0.92,
    )


def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2,
            color="#7c3aed",
        )
    )


def generate_data_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        ((0.25, 1.25), "Yelp Dataset", "650K training rows", "#2563eb"),
        ((2.75, 1.25), "Clean Text", "Missing + duplicate removal", "#0891b2"),
        ((5.25, 1.25), "Sentiment Labels", "Negative / Neutral / Positive", "#7c3aed"),
        ((7.75, 1.25), "Stratified Split", "Train / Validation / Test", "#db2777"),
        ((10.25, 1.25), "Model Input", "Sequences / Tokens", "#059669"),
    ]
    for (xy, title, subtitle, color) in boxes:
        draw_box(ax, xy, 2.1, 1.5, title, subtitle, color)
    for index in range(4):
        draw_arrow(ax, (2.35 + index * 2.5, 2), (2.75 + index * 2.5, 2))
    ax.set_title("Reproducible Data Pipeline", fontsize=18, fontweight="bold", color="#111827", pad=18)
    fig.savefig(OUTPUT_DIR / "data_pipeline_diagram.png", dpi=180, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(6.5, 4.6, "Model Architecture Comparison", ha="center", fontsize=18, fontweight="bold", color="#111827")
    draw_box(ax, (0.4, 1.35), 5.0, 2.5, "BiLSTM", "Embedding -> Forward LSTM + Backward LSTM -> FC", "#0891b2")
    draw_box(ax, (7.6, 1.35), 5.0, 2.5, "DistilBERT", "Pretrained Context -> Transformer Blocks -> Classifier", "#7c3aed")
    ax.text(6.5, 2.9, "VS", ha="center", va="center", fontsize=28, fontweight="bold", color="#f43f5e")
    draw_box(ax, (4.15, 0.15), 4.7, 0.75, "Three sentiment probabilities", "Negative • Neutral • Positive", "#111827")
    draw_arrow(ax, (5.4, 1.35), (5.9, 0.9))
    draw_arrow(ax, (7.6, 1.35), (7.1, 0.9))
    fig.savefig(OUTPUT_DIR / "model_architecture_diagram.png", dpi=180, bbox_inches="tight", facecolor="#f8fafc")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_data_pipeline()
    generate_architecture_diagram()
