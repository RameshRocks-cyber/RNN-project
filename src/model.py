"""
Recurrent Neural Network architecture definitions: SimpleRNN, LSTM, and GRU.
Constructs and compiles TensorFlow/Keras sequential sentiment classifiers.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from src.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    OUTPUT_UNITS,
    RNN_UNITS,
)


def build_rnn_model(
    vocab_size: int = MAX_VOCAB_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    max_len: int = MAX_SEQUENCE_LENGTH,
    rnn_units: int = RNN_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: int = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """
    Builds and compiles the baseline SimpleRNN sentiment classification model.
    """
    layers = [
        Input(shape=(max_len,), name="input_sequence"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding_layer",
        ),
        SimpleRNN(
            units=rnn_units,
            return_sequences=False,
            name="simple_rnn_layer",
        ),
    ]

    if dropout_rate > 0.0:
        layers.append(Dropout(rate=dropout_rate, name="dropout_layer"))

    layers.extend(
        [
            Dense(units=dense_units, activation="relu", name="dense_intermediate"),
            Dense(units=OUTPUT_UNITS, activation="sigmoid", name="sentiment_output"),
        ]
    )

    model = Sequential(layers, name="SimpleRNN_Sentiment_Classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(
    vocab_size: int = MAX_VOCAB_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    max_len: int = MAX_SEQUENCE_LENGTH,
    lstm_units: int = RNN_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: int = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """
    Builds and compiles the LSTM sentiment classification model.
    """
    layers = [
        Input(shape=(max_len,), name="input_sequence"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding_layer",
        ),
        LSTM(
            units=lstm_units,
            return_sequences=False,
            name="lstm_layer",
        ),
    ]

    if dropout_rate > 0.0:
        layers.append(Dropout(rate=dropout_rate, name="dropout_layer"))

    layers.extend(
        [
            Dense(units=dense_units, activation="relu", name="dense_intermediate"),
            Dense(units=OUTPUT_UNITS, activation="sigmoid", name="sentiment_output"),
        ]
    )

    model = Sequential(layers, name="LSTM_Sentiment_Classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru_model(
    vocab_size: int = MAX_VOCAB_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    max_len: int = MAX_SEQUENCE_LENGTH,
    gru_units: int = RNN_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: int = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """
    Builds and compiles the Gated Recurrent Unit (GRU) sentiment classification model.

    Architecture Flow:
    1. Input Layer: (batch_size, 200) integer token sequence.
    2. Embedding Layer: (batch_size, 200, 64) continuous vector representation.
    3. GRU Layer: (batch_size, 64) with 2 gates (update, reset).
    4. Dropout Layer: (batch_size, 64) regularization.
    5. Dense Intermediate Layer: (batch_size, 32) with ReLU activation.
    6. Output Dense Layer: (batch_size, 1) with Sigmoid activation.
    """
    layers = [
        Input(shape=(max_len,), name="input_sequence"),
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="embedding_layer",
        ),
        GRU(
            units=gru_units,
            return_sequences=False,  # Emits only the final hidden state h_T
            name="gru_layer",
        ),
    ]

    if dropout_rate > 0.0:
        layers.append(Dropout(rate=dropout_rate, name="dropout_layer"))

    layers.extend(
        [
            Dense(units=dense_units, activation="relu", name="dense_intermediate"),
            Dense(units=OUTPUT_UNITS, activation="sigmoid", name="sentiment_output"),
        ]
    )

    model = Sequential(layers, name="GRU_Sentiment_Classifier")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    print("--- SimpleRNN Model Summary ---")
    rnn_model = build_rnn_model()
    rnn_model.summary()

    print("\n--- LSTM Model Summary ---")
    lstm_model = build_lstm_model()
    lstm_model.summary()

    print("\n--- GRU Model Summary ---")
    gru_model = build_gru_model()
    gru_model.summary()
