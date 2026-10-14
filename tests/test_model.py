"""
Unit tests for SimpleRNN and LSTM architecture specifications, tensor shapes, and forward pass sanity checks.
"""

import numpy as np
from src.config import (
    DENSE_UNITS,
    EMBEDDING_DIM,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    RNN_UNITS,
)
from src.model import build_lstm_model, build_rnn_model


def test_build_rnn_model_structure():
    model = build_rnn_model(
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        rnn_units=RNN_UNITS,
        dense_units=DENSE_UNITS,
    )
    assert model is not None
    assert model.input_shape == (None, MAX_SEQUENCE_LENGTH)
    assert model.output_shape == (None, 1)

    layer_names = [l.name for l in model.layers]
    assert any("embedding" in name for name in layer_names)
    assert any("simple_rnn" in name for name in layer_names)
    assert any("sentiment_output" in name for name in layer_names)


def test_build_lstm_model_structure():
    model = build_lstm_model(
        vocab_size=MAX_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_SEQUENCE_LENGTH,
        lstm_units=RNN_UNITS,
        dense_units=DENSE_UNITS,
    )
    assert model is not None
    assert model.input_shape == (None, MAX_SEQUENCE_LENGTH)
    assert model.output_shape == (None, 1)

    layer_names = [l.name for l in model.layers]
    assert any("embedding" in name for name in layer_names)
    assert any("lstm" in name for name in layer_names)
    assert any("sentiment_output" in name for name in layer_names)


def test_lstm_forward_pass_predictions():
    model = build_lstm_model(
        vocab_size=1000,
        embedding_dim=32,
        max_len=50,
        lstm_units=32,
        dense_units=16,
    )
    batch_size = 8
    dummy_input = np.random.randint(0, 1000, size=(batch_size, 50))
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (batch_size, 1)
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
