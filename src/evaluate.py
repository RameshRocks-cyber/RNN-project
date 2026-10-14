"""
Evaluation module for model performance on unseen test data.
Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
generates Confusion Matrix heatmap visualization, and runs sanity checks on manual reviews.
Supports SimpleRNN, LSTM, and GRU models for fair benchmarking.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model

from src.config import (
    GRU_CONFUSION_MATRIX_PATH,
    GRU_MODEL_PATH,
    LSTM_CONFUSION_MATRIX_PATH,
    LSTM_MODEL_PATH,
    MAX_SEQUENCE_LENGTH,
    MODEL_PATH,
    SAVED_MODELS_DIR,
    TOKENIZER_PATH,
)
from src.data_loader import load_raw_data, preprocess_dataset, split_data
from src.predict import SentimentPredictor
from src.preprocessor import load_tokenizer, transform_texts

CONFUSION_MATRIX_PATH = SAVED_MODELS_DIR / "confusion_matrix.png"


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, model_name: str = "Model") -> None:
    """Plots and saves a professional Confusion Matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Negative (0)", "Predicted Positive (1)"],
        yticklabels=["Actual Negative (0)", "Actual Positive (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title(f"Test Set Confusion Matrix ({model_name})", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Visualization] Confusion matrix saved to: {save_path}")


def evaluate_test_set(
    X_test_raw: List[str],
    y_test: np.ndarray,
    model_type: str = "gru",
) -> Dict[str, Any]:
    """
    Evaluates the trained model (SimpleRNN, LSTM, or GRU) on held-out test data.
    """
    model_type = model_type.lower()
    if model_type == "gru":
        model_name = "GRU"
        target_model_path = GRU_MODEL_PATH
        target_cm_path = GRU_CONFUSION_MATRIX_PATH
    elif model_type == "lstm":
        model_name = "LSTM"
        target_model_path = LSTM_MODEL_PATH
        target_cm_path = LSTM_CONFUSION_MATRIX_PATH
    else:
        model_name = "SimpleRNN"
        target_model_path = MODEL_PATH
        target_cm_path = CONFUSION_MATRIX_PATH

    if not target_model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {target_model_path}. Train model first.")

    print(f"Loading {model_name} model from {target_model_path}...")
    model = load_model(target_model_path)
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    print(f"Transforming {len(X_test_raw):,} test reviews into padded sequences...")
    X_test_pad = transform_texts(tokenizer, X_test_raw, max_len=MAX_SEQUENCE_LENGTH)

    print(f"Running batch inference on test set with {model_name}...")
    y_pred_probs = model.predict(X_test_pad, batch_size=128, verbose=0).ravel()
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)

    # Calculate metrics
    accuracy = float(np.mean(y_pred_labels == y_test))
    cm = confusion_matrix(y_test, y_pred_labels)
    report = classification_report(
        y_test,
        y_pred_labels,
        target_names=["Negative (0)", "Positive (1)"],
        output_dict=True,
    )
    roc_auc = float(roc_auc_score(y_test, y_pred_probs))

    # Print summary
    print("\n" + "=" * 65)
    print(f"           TEST SET EVALUATION REPORT ({model_name})")
    print("=" * 65)
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%\n")
    print(
        classification_report(
            y_test,
            y_pred_labels,
            target_names=["Negative (0)", "Positive (1)"],
            digits=4,
        )
    )
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("=" * 65)

    # Plot confusion matrix
    plot_confusion_matrix(cm, target_cm_path, model_name=model_name)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
        "y_pred_probs": y_pred_probs,
        "y_pred_labels": y_pred_labels,
    }


def test_manual_reviews(model_type: str = "gru") -> List[Dict[str, Any]]:
    """
    Evaluates model inference on 10 hand-crafted test reviews with varying sentiment,
    length, negation, and nuance.
    """
    model_type = model_type.lower()
    model_name = model_type.upper()

    manual_cases = [
        # 1. Unambiguously Positive
        ("This movie was an absolute masterpiece! Incredible performances and a thrilling plot.", "Positive"),
        # 2. Unambiguously Negative
        ("Total waste of time and money. Terrible directing, awful dialogue, and boring characters.", "Negative"),
        # 3. Short Positive
        ("Loved it! Outstanding film.", "Positive"),
        # 4. Short Negative
        ("Dreadful, painful to watch.", "Negative"),
        # 5. Negation Positive (contains 'not' but positive overall)
        ("The movie was not bad at all, actually quite enjoyable and entertaining.", "Positive"),
        # 6. Negation Negative (contains 'not' with positive words)
        ("This was definitely not good, completely failed to meet any expectations.", "Negative"),
        # 7. Nuanced / Mixed Review (starts positive, ends negative)
        ("The cinematography and music were breathtaking, but the terrible script ruined everything.", "Negative"),
        # 8. Nuanced / Mixed Review (starts negative, ends positive)
        ("It started off very slow and boring, but the thrilling climax made it totally worthwhile.", "Positive"),
        # 9. Subtle / Sarcastic Tone
        ("If you enjoy staring at paint drying for two hours, you will love this masterpiece.", "Negative"),
        # 10. Neutral / Descriptive Review
        ("A standard action movie with plenty of explosions and average pacing.", "Negative"),
    ]

    predictor = SentimentPredictor(model_type=model_type)
    results = []

    print("\n" + "=" * 75)
    print(f"         {model_name}: MANUAL REVIEW INFERENCE & ERROR ANALYSIS")
    print("=" * 75)

    passed_count = 0
    for i, (review, ground_truth) in enumerate(manual_cases, 1):
        res = predictor.predict(review)
        pred_label = res["label"]
        raw_score = res["raw_score"]
        confidence = res["confidence"]
        is_correct = (pred_label == ground_truth)
        if is_correct:
            passed_count += 1

        status = "CORRECT" if is_correct else "INCORRECT"
        icon = "[PASS]" if is_correct else "[FAIL]"

        print(f"\nCase {i:02d}: {icon} {status}")
        print(f"  Review:     \"{review}\"")
        print(f"  Expected:   {ground_truth}")
        print(f"  Predicted:  {pred_label} (Confidence: {confidence * 100:.2f}% | Raw Prob: {raw_score:.4f})")

        results.append({
            "case_id": i,
            "review": review,
            "expected": ground_truth,
            "predicted": pred_label,
            "confidence": confidence,
            "raw_score": raw_score,
            "is_correct": is_correct,
        })

    print(f"\nManual Review Accuracy: {passed_count}/{len(manual_cases)} ({passed_count/len(manual_cases)*100:.1f}%)")
    print("=" * 75)
    return results


def run_full_evaluation(model_type: str = "gru"):
    print("[1/3] Loading dataset for test evaluation...")
    df_raw = load_raw_data()
    df_clean = preprocess_dataset(df_raw)
    _, _, X_test, _, _, y_test = split_data(df_clean)

    print(f"[2/3] Computing quantitative test metrics for {model_type.upper()}...")
    eval_metrics = evaluate_test_set(X_test.tolist(), np.array(y_test, dtype=int), model_type=model_type)

    print(f"[3/3] Running qualitative manual review test suite on {model_type.upper()}...")
    manual_results = test_manual_reviews(model_type=model_type)

    return eval_metrics, manual_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SimpleRNN, LSTM, or GRU Sentiment Model")
    parser.add_argument("--model-type", type=str, default="gru", choices=["gru", "lstm", "simple_rnn"])
    args = parser.parse_args()
    run_full_evaluation(model_type=args.model_type)
