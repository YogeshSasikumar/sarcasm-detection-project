"""
evaluation.py - Model evaluation: metrics and confusion matrix
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """
    Compute standard classification metrics.

    Returns:
        {
            "accuracy"  : float,
            "precision" : float,
            "recall"    : float,
            "f1_score"  : float,
            "report"    : dict  (per-class report)
        }
    """
    avg = "binary" if len(set(y_true)) <= 2 else "weighted"
    metrics = {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "f1_score" : round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "report"   : classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
    logger.info(
        "Evaluation — Acc: %.4f | P: %.4f | R: %.4f | F1: %.4f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"],
    )
    return metrics


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    title: str = "Confusion Matrix",
    figsize: tuple = (6, 5),
) -> plt.Figure:
    """
    Generate a styled confusion matrix figure.

    Returns:
        matplotlib Figure object (can be passed to st.pyplot()).
    """
    cm = confusion_matrix(y_true, y_pred, labels=class_names if isinstance(class_names[0], str) else None)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#333",
        cbar=False,
    )
    ax.set_title(title, color="white", fontsize=13, pad=12)
    ax.set_xlabel("Predicted", color="white", fontsize=10)
    ax.set_ylabel("Actual", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    return fig


def evaluate_sarcasm_model(model, texts: list[str], labels: list[int]) -> dict:
    """
    Run evaluation on a SarcasmDetector instance.

    Args:
        model  : Trained SarcasmDetector.
        texts  : Processed combined texts.
        labels : Ground-truth labels.

    Returns:
        Metrics dict.
    """
    preds = [model.predict(t)["label"] for t in texts]
    metrics = compute_metrics(labels, preds)
    logger.info("Sarcasm model evaluation complete.")
    return metrics


def evaluate_emotion_model(model, texts: list[str], emotions: list[str]) -> dict:
    """
    Run evaluation on an EmotionDetector instance.

    Args:
        model   : Trained EmotionDetector.
        texts   : Processed texts.
        emotions: Ground-truth emotion labels.

    Returns:
        Metrics dict.
    """
    preds = [model.predict(t)["emotion"] for t in texts]
    metrics = compute_metrics(emotions, preds)
    logger.info("Emotion model evaluation complete.")
    return metrics


if __name__ == "__main__":
    # Quick smoke test with dummy data
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0]
    m = compute_metrics(y_true, y_pred)
    print("Metrics:", {k: v for k, v in m.items() if k != "report"})

    fig = plot_confusion_matrix(y_true, y_pred, class_names=["Not Sarcastic", "Sarcastic"])
    fig.savefig("/tmp/confusion_matrix_test.png", dpi=100)
    print("Confusion matrix saved to /tmp/confusion_matrix_test.png")
