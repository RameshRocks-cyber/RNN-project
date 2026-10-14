"""
Training pipeline for the Simple RNN Sentiment Classifier.
Handles data preparation, model compilation, training with callbacks,
history persistence, and learning curve plotting.
"""

import json
import sys
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
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    METRICS_PATH,
    MODEL_PATH,
    SAVED_MODELS_DIR,
    TOKENIZER_PATH,
)
from src.data_loader import load_raw_data, preprocess_dataset, split_data
from src.model import build_rnn_model
from src.preprocessor import fit_tokenizer, save_tokenizer, transform_texts

PLOT_PATH = SAVED_MODELS_DIR / "learning_curves.png"


def plot_training_curves(history_dict: dict, save_path: Path = PLOT_PATH) -> None:
    """
    Plots training vs validation loss and accuracy curves.
    """
    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Loss Curves ---
    ax1.plot(epochs_range, history_dict["loss"], "o-", color="#2563eb", label="Training Loss", linewidth=2)
    ax1.plot(epochs_range, history_dict["val_loss"], "s--", color="#dc2626", label="Validation Loss", linewidth=2)
    ax1.set_title("Training Loss vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # --- Plot 2: Accuracy Curves ---
    ax2.plot(epochs_range, history_dict["accuracy"], "o-", color="#16a34a", label="Training Accuracy", linewidth=2)
    ax2.plot(epochs_range, history_dict["val_accuracy"], "s--", color="#d97706", label="Validation Accuracy", linewidth=2)
    ax2.set_title("Training Accuracy vs Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Visualization] Learning curves saved to: {save_path}")


def train_pipeline() -> dict:
    """
    Executes the full end-to-end model training workflow.
    """
    print("=" * 65)
    print("      SimpleRNN Sentiment Classifier - Training Pipeline")
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

    print(f"\n[Step 4/6] Fitting Tokenizer on X_train (capped at top {MAX_VOCAB_SIZE:,} words)...")
    tokenizer = fit_tokenizer(X_train_raw.tolist(), max_words=MAX_VOCAB_SIZE)
    save_tokenizer(tokenizer, TOKENIZER_PATH)
    print(f"  Fitted vocabulary saved to: {TOKENIZER_PATH}")

    print("[Step 5/6] Padding sequences to uniform length of 200 tokens...")
    X_train_pad = transform_texts(tokenizer, X_train_raw.tolist(), max_len=MAX_SEQUENCE_LENGTH)
    X_val_pad = transform_texts(tokenizer, X_val_raw.tolist(), max_len=MAX_SEQUENCE_LENGTH)

    y_train_arr = np.array(y_train, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.float32)

    print("\n[Step 6/6] Instantiating and Compiling SimpleRNN Model...")
    model = build_rnn_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs (Batch Size = {BATCH_SIZE})...\n")
    history = model.fit(
        X_train_pad,
        y_train_arr,
        validation_data=(X_val_pad, y_val_arr),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist metrics history
    history_dict = history.history
    with open(METRICS_PATH, "w") as f:
        json.dump(history_dict, f, indent=4)
    print(f"\n[Artifacts] Training metrics saved to: {METRICS_PATH}")

    # Plot & save loss/accuracy curves
    plot_training_curves(history_dict, PLOT_PATH)

    # Print final evaluation metrics
    best_epoch = int(np.argmin(history_dict["val_loss"]))
    print("\n" + "=" * 65)
    print("                    TRAINING COMPLETE")
    print("=" * 65)
    print(f"Best Epoch: {best_epoch + 1}")
    print(f"  Training Loss:        {history_dict['loss'][best_epoch]:.4f}")
    print(f"  Training Accuracy:    {history_dict['accuracy'][best_epoch] * 100:.2f}%")
    print(f"  Validation Loss:      {history_dict['val_loss'][best_epoch]:.4f}")
    print(f"  Validation Accuracy:  {history_dict['val_accuracy'][best_epoch] * 100:.2f}%")
    print(f"Trained model saved to: {MODEL_PATH}")
    print("=" * 65)

    return history_dict


if __name__ == "__main__":
    train_pipeline()
