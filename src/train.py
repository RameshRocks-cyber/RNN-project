"""
Training pipeline for the Simple RNN Sentiment Classifier.
Handles data preparation, model compilation, training with callbacks, and artifact saving.
"""

import json
from pathlib import Path
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    METRICS_PATH,
    MODEL_PATH,
    TOKENIZER_PATH,
)
from src.data_loader import load_raw_data, split_data
from src.model import build_rnn_model
from src.preprocessor import fit_tokenizer, save_tokenizer, transform_texts


def train_pipeline() -> None:
    """
    Executes the end-to-end model training workflow.
    (Invoked in Phase 2 once the dataset and environment are prepared).
    """
    print("[1/5] Loading raw reviews dataset...")
    df = load_raw_data()

    print("[2/5] Creating stratified splits (Train/Val/Test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("[3/5] Fitting tokenizer and transforming text sequences...")
    tokenizer = fit_tokenizer(X_train.tolist())
    save_tokenizer(tokenizer, TOKENIZER_PATH)

    X_train_pad = transform_texts(tokenizer, X_train.tolist())
    X_val_pad = transform_texts(tokenizer, X_val.tolist())

    # Ensure binary integer labels (0: Negative, 1: Positive)
    y_train_arr = np.array(y_train).astype("float32")
    y_val_arr = np.array(y_val).astype("float32")

    print("[4/5] Instantiating and compiling RNN model...")
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

    print(f"[5/5] Training for up to {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    history = model.fit(
        X_train_pad,
        y_train_arr,
        validation_data=(X_val_pad, y_val_arr),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # Save training history for metric plotting
    with open(METRICS_PATH, "w") as f:
        json.dump(history.history, f, indent=4)

    print(f"Training completed successfully! Model weights saved to {MODEL_PATH}")


if __name__ == "__main__":
    # Do NOT execute training during initial project scaffolding (Phase 1)
    print("Project initialized. Run train_pipeline() during Phase 2 after placing raw dataset.")
