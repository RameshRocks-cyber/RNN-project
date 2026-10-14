"""
Inference module for real-time sentiment prediction on arbitrary review text.
Supports SimpleRNN, LSTM, and GRU models.
"""

import sys
from pathlib import Path
from typing import Dict, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow.keras.models import load_model

from src.config import (
    GRU_MODEL_PATH,
    LSTM_MODEL_PATH,
    MODEL_PATH,
    TOKENIZER_PATH,
)
from src.preprocessor import load_tokenizer, transform_texts


class SentimentPredictor:
    """Encapsulates model and tokenizer inference for real-time predictions."""

    def __init__(self, model_type: str = "gru"):
        self.model_type = model_type.lower()
        self.model = None
        self.tokenizer = None

    def get_model_path(self) -> Path:
        if self.model_type == "gru":
            return GRU_MODEL_PATH
        elif self.model_type == "lstm":
            return LSTM_MODEL_PATH
        else:
            return MODEL_PATH

    def load_artifacts(self) -> None:
        """Loads trained Keras model and fitted tokenizer."""
        target_model_path = self.get_model_path()
        if not target_model_path.exists():
            # Fallback to any trained model available
            for candidate in [GRU_MODEL_PATH, LSTM_MODEL_PATH, MODEL_PATH]:
                if candidate.exists():
                    print(f"[Notice] {self.model_type.upper()} not found, falling back to: {candidate.name}")
                    target_model_path = candidate
                    break
            else:
                raise FileNotFoundError(
                    f"No trained model found. Please train a model first."
                )

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Please train the model first.")

        self.model = load_model(target_model_path)
        self.tokenizer = load_tokenizer(TOKENIZER_PATH)

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predicts sentiment for a single text review.
        Returns:
            dict containing:
                - text: input string
                - label: 'Positive' or 'Negative'
                - confidence: Float percentage [0.0, 1.0]
                - raw_score: Raw sigmoid probability [0.0, 1.0]
        """
        if self.model is None or self.tokenizer is None:
            self.load_artifacts()

        padded_sequence = transform_texts(self.tokenizer, [text])
        raw_prob = float(self.model.predict(padded_sequence, verbose=0)[0][0])

        is_positive = raw_prob >= 0.5
        label = "Positive" if is_positive else "Negative"
        confidence = raw_prob if is_positive else (1.0 - raw_prob)

        return {
            "text": text,
            "label": label,
            "confidence": round(confidence, 4),
            "raw_score": round(raw_prob, 4),
        }


def predict_sentiment(text: str, model_type: str = "gru") -> Dict[str, Union[str, float]]:
    """Convenience helper function for single-call prediction."""
    predictor = SentimentPredictor(model_type=model_type)
    return predictor.predict(text)
