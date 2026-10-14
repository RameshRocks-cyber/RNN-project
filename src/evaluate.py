"""
Evaluation module for model performance on unseen test data.
Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
and generates diagnostic visualizations (Confusion Matrix, Loss/Accuracy Curves).
"""

import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model

from src.config import MODEL_PATH, TOKENIZER_PATH
from src.preprocessor import load_tokenizer, transform_texts


def evaluate_model(X_test: list, y_test: list, model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """
    Evaluates the trained RNN model on a test set.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first.")

    model = load_model(model_path)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    X_test_pad = transform_texts(tokenizer, X_test)
    y_pred_probs = model.predict(X_test_pad).ravel()
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)

    report = classification_report(y_test, y_pred_labels, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_labels)
    roc_auc = roc_auc_score(y_test, y_pred_probs)

    print("=== Model Evaluation Report ===")
    print(classification_report(y_test, y_pred_labels, target_names=["Negative", "Positive"]))
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    return {
        "report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "y_pred_probs": y_pred_probs,
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: Path = None) -> None:
    """Plots and optionally saves a stylized Confusion Matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative (0)", "Positive (1)"],
        yticklabels=["Negative (0)", "Positive (1)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
