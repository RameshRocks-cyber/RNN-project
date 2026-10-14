"""
Inference module for real-time sentiment prediction on arbitrary review text.
"""

from typing import Dict, Union
from tensorflow.keras.models import load_model

from src.config import MODEL_PATH, TOKENIZER_PATH
from src.preprocessor import load_tokenizer, transform_texts


class SentimentPredictor:
    """Encapsulates model and tokenizer inference for real-time predictions."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_artifacts(self) -> None:
        """Loads trained Keras model and fitted tokenizer."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Please train the model first.")

        self.model = load_model(MODEL_PATH)
        self.tokenizer = load_tokenizer(TOKENIZER_PATH)

    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Predicts sentiment for a single text review.
        Returns:
            dict containing:
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


def predict_sentiment(text: str) -> Dict[str, Union[str, float]]:
    """Convenience helper function for single-call prediction."""
    predictor = SentimentPredictor()
    return predictor.predict(text)
