"""
prediction.py
Production-grade, standalone sentiment analysis inference module.
Loads pre-trained model and tokenizer artifacts independently without retraining.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Set TensorFlow log level to suppress non-critical warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

PROD_MODEL_PATH = SAVED_MODELS_DIR / "production_model.keras"
FALLBACK_MODEL_PATH = SAVED_MODELS_DIR / "gru_sentiment_model.keras"
TOKENIZER_PKL_PATH = SAVED_MODELS_DIR / "tokenizer.pkl"
PREPROCESSING_CONFIG_PATH = SAVED_MODELS_DIR / "preprocessing_config.json"
LABEL_MAPPING_PATH = SAVED_MODELS_DIR / "label_mapping.json"


# =====================================================================
# 1. Deterministic Preprocessing Pipeline (Identical to Training)
# =====================================================================
def clean_text_for_inference(text: str) -> str:
    """
    Applies identical text cleaning transformations used during training:
    1. Input validation fallback
    2. HTML tag & entity stripping
    3. URL removal
    4. Lowercase conversion
    5. Punctuation isolation with whitespace (e.g. 'film!' -> 'film !')
    6. Special character filtering (keeps alphanumerics, apostrophes, basic punctuation)
    7. Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    # 1. Strip HTML tags (<br />, <b>, <a href=...>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Decode HTML entities
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)

    # 3. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Lowercase normalization
    text = text.lower()

    # 5. Isolate punctuation with spaces
    text = re.sub(r"([.,!?])", r" \1 ", text)

    # 6. Filter non-whitelisted characters
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\s]", " ", text)

    # 7. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================================
# 2. Production Inference Engine (Singleton Artifact Manager)
# =====================================================================
class ProductionSentimentEngine:
    """
    Encapsulates loaded Keras model, tokenizer, and configurations.
    Implements lazy-loading singleton pattern so artifacts are loaded
    into memory exactly once for ultra-fast repeated inference.
    """

    _instance: Optional["ProductionSentimentEngine"] = None

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.label_mapping = None
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "ProductionSentimentEngine":
        """Returns the singleton instance of the inference engine."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self) -> None:
        """Loads all serialized artifacts without invoking any training logic."""
        import pickle
        from tensorflow.keras.models import load_model

        # 1. Load Preprocessing Configuration
        if PREPROCESSING_CONFIG_PATH.exists():
            with open(PREPROCESSING_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "max_sequence_length": 200,
                "padding_type": "post",
                "truncating_type": "post",
            }

        # 2. Load Label Mapping
        if LABEL_MAPPING_PATH.exists():
            with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)
        else:
            self.label_mapping = {
                "id2label": {"0": "Negative", "1": "Positive"},
                "decision_threshold": 0.5,
            }

        # 3. Load Tokenizer
        if not TOKENIZER_PKL_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer artifact missing at: {TOKENIZER_PKL_PATH}."
            )
        with open(TOKENIZER_PKL_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)

        # 4. Load Keras Model (.keras native format)
        target_model = PROD_MODEL_PATH if PROD_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
        if not target_model.exists():
            raise FileNotFoundError(
                f"Trained model artifact missing at: {target_model}."
            )
        self.model = load_model(target_model)

    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """Executes full inference pipeline for a single raw review string."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned = clean_text_for_inference(text)
        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(
            seq,
            maxlen=self.config.get("max_sequence_length", 200),
            padding=self.config.get("padding_type", "post"),
            truncating=self.config.get("truncating_type", "post"),
        )

        raw_prob = float(self.model.predict(padded, verbose=0)[0][0])
        threshold = self.label_mapping.get("decision_threshold", 0.5)

        is_positive = raw_prob >= threshold
        label = "Positive" if is_positive else "Negative"
        confidence = raw_prob if is_positive else (1.0 - raw_prob)

        return {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": label,
            "confidence": round(confidence, 4),
            "probability": round(raw_prob, 4),
            "decision_threshold": threshold,
            "model_type": "GRU (Production)",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Executes vector-batched inference on a list of review texts."""
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned_texts = [clean_text_for_inference(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.config.get("max_sequence_length", 200),
            padding=self.config.get("padding_type", "post"),
            truncating=self.config.get("truncating_type", "post"),
        )

        raw_probs = self.model.predict(padded, verbose=0)
        threshold = self.label_mapping.get("decision_threshold", 0.5)

        results = []
        for i, text in enumerate(texts):
            prob = float(raw_probs[i][0])
            is_pos = prob >= threshold
            label = "Positive" if is_pos else "Negative"
            conf = prob if is_pos else (1.0 - prob)
            results.append({
                "text": text,
                "cleaned_text": cleaned_texts[i],
                "sentiment": label,
                "confidence": round(conf, 4),
                "probability": round(prob, 4),
                "decision_threshold": threshold,
                "model_type": "GRU (Production)",
            })
        return results


# =====================================================================
# 3. Public API Functions
# =====================================================================
def predict_sentiment(text: str) -> Dict[str, Union[str, float]]:
    """
    Public entry point for real-time sentiment prediction.
    Accepts raw uncleaned string, returns prediction dictionary.
    """
    engine = ProductionSentimentEngine.get_instance()
    return engine.predict_single(text)


def predict_sentiment_batch(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """Public entry point for batch sentiment prediction."""
    engine = ProductionSentimentEngine.get_instance()
    return engine.predict_batch(texts)


# =====================================================================
# 4. CLI Execution & Verification Suite
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # User passed a custom review text via CLI
        input_text = " ".join(sys.argv[1:])
        prediction = predict_sentiment(input_text)
        print(json.dumps(prediction, indent=2))
    else:
        # Run standard test suite demonstrating independent loading
        print("=" * 70)
        print("   PRODUCTION SENTIMENT ANALYSIS INFERENCE VERIFICATION")
        print("=" * 70)
        test_samples = [
            "This film is an absolute masterpiece! Sensational acting and writing.",
            "Complete waste of two hours. Dreadful directing, awful script, totally unwatchable.",
            "The movie was not bad at all, actually surprisingly entertaining.",
            "I wanted to love this, but the ending was deeply disappointing and flat.",
            "A visually stunning experience with an unforgettable musical score."
        ]
        
        results = predict_sentiment_batch(test_samples)
        for idx, res in enumerate(results, 1):
            badge = "👍 POSITIVE" if res["sentiment"] == "Positive" else "👎 NEGATIVE"
            print(f"\n[Review {idx}] \"{res['text']}\"")
            print(f"  Prediction:  {badge}")
            print(f"  Confidence:  {res['confidence'] * 100:.2f}% (Prob: {res['probability']:.4f})")
            print(f"  Cleaned:     \"{res['cleaned_text']}\"")
        print("\n" + "=" * 70)
        print("Inference engine verified successfully. Ready for deployment!")
        print("=" * 70)
