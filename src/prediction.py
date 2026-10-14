"""
src/prediction.py
Re-exports prediction engine and predict_sentiment for backwards-compatibility.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction import (
    ProductionSentimentEngine,
    clean_text_for_inference,
    predict_sentiment,
    predict_sentiment_batch,
)

__all__ = [
    "ProductionSentimentEngine",
    "clean_text_for_inference",
    "predict_sentiment",
    "predict_sentiment_batch",
]
