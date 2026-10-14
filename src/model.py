"""
Recurrent Neural Network (SimpleRNN) architecture definition.
Constructs and compiles the TensorFlow/Keras sequential sentiment classifier.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
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
) -> Sequential:
    """
    Builds a SimpleRNN-based sequence classification model.

    Layer Breakdown:
    1. Embedding Layer: Maps discrete token indices (0..V) to dense continuous vectors of dim `embedding_dim`.
    2. SimpleRNN Layer: Processes sequence step-by-step; maintains a recurrent hidden state across time steps.
    3. Dropout Layer: Mitigates co-adaptation and overfitting on vocabulary cues.
    4. Dense Layer: Non-linear feature aggregation with ReLU activation.
    5. Output Layer: Single neuron with Sigmoid activation outputting probability p in [0, 1].
    """
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_len,
                name="embedding_layer",
            ),
            SimpleRNN(
                units=rnn_units,
                return_sequences=False,  # Return only the final hidden state vector
                name="simple_rnn_layer",
            ),
            Dropout(rate=dropout_rate, name="dropout_layer"),
            Dense(units=dense_units, activation="relu", name="dense_intermediate"),
            Dense(units=OUTPUT_UNITS, activation="sigmoid", name="sentiment_output"),
        ],
        name="SimpleRNN_Sentiment_Classifier",
    )

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
