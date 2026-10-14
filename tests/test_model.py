"""
Unit tests for RNN model architecture and tensor shape validation.
"""

import numpy as np
from src.config import MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE
from src.model import build_rnn_model


def test_build_rnn_model_structure():
    model = build_rnn_model(vocab_size=MAX_VOCAB_SIZE, max_len=MAX_SEQUENCE_LENGTH)
    assert model is not None
    # Verify input and output shape
    assert model.output_shape == (None, 1)


def test_model_forward_pass_shape():
    model = build_rnn_model(vocab_size=1000, max_len=50, embedding_dim=16, rnn_units=16)
    dummy_input = np.random.randint(0, 1000, size=(4, 50))
    output = model.predict(dummy_input, verbose=0)
    assert output.shape == (4, 1)
    assert np.all((output >= 0.0) & (output <= 1.0))
