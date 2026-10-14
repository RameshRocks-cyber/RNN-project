"""
Training pipeline for Recurrent Sentiment Classifiers (SimpleRNN & LSTM).
Handles data preparation, model compilation, training with callbacks,
history persistence, learning curve plotting, and runtime benchmark tracking.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LSTM_METRICS_PATH,
    LSTM_MODEL_PATH,
    LSTM_PLOT_PATH,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    SAVED_MODELS_DIR,
    TOKENIZER_PATH,
)
from src.data_loader import load_raw_data, preprocess_dataset, split_data
from src.model import build_lstm_model, build_rnn_model
from src.preprocessor import fit_tokenizer, save_tokenizer, transform_texts

PLOT_PATH = SAVED_MODELS_DIR / "learning_curves.png"


def plot_training_curves(history_dict: dict, save_path: Path, model_name: str = "Model") -> None:
    """
    Plots training vs validation loss and accuracy curves.
    """
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss Curves ---
    ax1.plot(epochs_range, history_dict["loss"], "o-", color="#2563eb", label="Training Loss", linewidth=2)
    ax1.plot(epochs_range, history_dict["val_loss"], "s--", color="#dc2626", label="Validation Loss", linewidth=2)
    ax1.set_title(f"{model_name}: Training Loss vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # --- Plot 2: Accuracy Curves ---
    ax2.plot(epochs_range, history_dict["accuracy"], "o-", color="#16a34a", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs_range, history_dict["val_accuracy"], "s--", color="#d97706", label="Validation Accuracy", linewidth=2)
    ax2.set_title(f"{model_name}: Training Accuracy vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Visualization] Learning curves saved to: {save_path}")


def train_pipeline(model_type: str = "lstm") -> dict:
    """
    Executes the full end-to-end model training workflow for SimpleRNN or LSTM.
    """
    model_type = model_type.lower()
    is_lstm = (model_type == "lstm")
    model_name = "LSTM" if is_lstm else "SimpleRNN"
    target_model_path = LSTM_MODEL_PATH if is_lstm else MODEL_PATH
    target_metrics_path = LSTM_METRICS_PATH if is_lstm else METRICS_PATH
    target_plot_path = LSTM_PLOT_PATH if is_lstm else PLOT_PATH

    print("=" * 65)
    print(f"      {model_name} Sentiment Classifier - Training Pipeline")
    print("=" * 65)

    print("\n[Step 1/6] Loading raw IMDB dataset...")
    df_raw = load_raw_data()

    print("[Step 2/6] Cleaning text and removing duplicates...")
    df_clean = preprocess_dataset(df_raw)

    print("[Step 3/6] Performing stratified Train (80%) / Val (10%) / Test (10%) split...")
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df_clean)
    print(f"  Training samples:   {len(X_train_raw):,}")
    print(f"  Validation samples: {len(X_val_raw):,}")
    print(f"  Test samples:       {len(X_test_raw):,}")

    print(f"\n[Step 4/6] Checking Tokenizer (capped at top {MAX_VOCAB_SIZE:,} words)...")
    if not TOKENIZER_PATH.exists():
        tokenizer = fit_tokenizer(X_train_raw.tolist(), max_words=MAX_VOCAB_SIZE)
        save_tokenizer(tokenizer, TOKENIZER_PATH)
    else:
        from src.preprocessor import load_tokenizer
        tokenizer = load_tokenizer(TOKENIZER_PATH)
    print(f"  Tokenizer ready from: {TOKENIZER_PATH}")

    print("[Step 5/6] Padding sequences to uniform length of 200 tokens...")
    X_train_pad = transform_texts(tokenizer, X_train_raw.tolist(), max_len=MAX_SEQUENCE_LENGTH)
    X_val_pad = transform_texts(tokenizer, X_val_raw.tolist(), max_len=MAX_SEQUENCE_LENGTH)

    y_train_arr = np.array(y_train, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.float32)

    print(f"\n[Step 6/6] Instantiating and Compiling {model_name} Model...")
    model = build_lstm_model() if is_lstm else build_rnn_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(target_model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nTraining {model_name} for up to {EPOCHS} epochs (Batch Size = {BATCH_SIZE})...\n")
    start_time = time.time()
    history = model.fit(
        X_train_pad,
        y_train_arr,
        validation_data=(X_val_pad, y_val_arr),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    training_duration = time.time() - start_time

    # Persist metrics history including runtime
    history_dict = history.history
    history_dict["training_duration_seconds"] = training_duration
    with open(target_metrics_path, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[Artifacts] Training metrics saved to: {target_metrics_path}")

    # Plot & save loss/accuracy curves
    plot_training_curves(history_dict, target_plot_path, model_name=model_name)

    # Print final evaluation metrics
    best_epoch = int(np.argmin(history_dict["val_loss"]))
    print("\n" + "=" * 65)
    print(f"               {model_name} TRAINING COMPLETE")
    print("=" * 65)
    print(f"Total Training Time:    {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")
    print(f"Best Epoch:             {best_epoch + 1}")
    print(f"  Training Loss:        {history_dict['loss'][best_epoch]:.4f}")
    print(f"  Training Accuracy:    {history_dict['accuracy'][best_epoch] * 100:.2f}%")
    print(f"  Validation Loss:      {history_dict['val_loss'][best_epoch]:.4f}")
    print(f"  Validation Accuracy:  {history_dict['val_accuracy'][best_epoch] * 100:.2f}%")
    print(f"Trained model saved to: {target_model_path}")
    print("=" * 65)

    return history_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN or LSTM Sentiment Analysis Model")
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "simple_rnn"])
    args = parser.parse_args()
    train_pipeline(model_type=args.model_type)
