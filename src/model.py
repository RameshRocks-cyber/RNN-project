"""
Recurrent Neural Network (SimpleRNN) architecture definition.
Constructs and compiles the TensorFlow/Keras sequential sentiment classifier.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense, Dropout
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

    Architecture Flow:
    1. Input Layer: Accepts fixed-length 1D sequences of token integers of shape (200,).
    2. Embedding Layer: Projects token IDs (0..9999) into continuous 64-dimensional vectors.
       Output Shape: (batch_size, 200, 64).
    3. SimpleRNN Layer: Processes sequence chronologically, updating its recurrent hidden state
       at each time step t. With return_sequences=False, it emits only the final hidden state h_T.
       Output Shape: (batch_size, 64).
    4. Dropout Layer: Regularization dropping neurons with probability `dropout_rate` during training.
    5. Intermediate Dense Layer: Non-linear feature aggregation with ReLU activation.
       Output Shape: (batch_size, 32).
    6. Output Dense Layer: Single neuron with Sigmoid activation outputting probability p in [0, 1].
       Output Shape: (batch_size, 1).
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
            return_sequences=False,  # Emits only final hidden vector h_T
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

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Print architectural summary and parameter specifications
    model = build_rnn_model()
    model.summary()
