"""
Text preprocessing and tokenization pipeline.
Performs regex cleaning, lowercase normalization, token-to-index mapping,
and sequence padding for recurrent neural network consumption.
"""

import pickle
import re
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    OOV_TOKEN,
    PADDING_TYPE,
    TOKENIZER_PATH,
    TRUNCATING_TYPE,
)


def clean_text(text: str) -> str:
    """
    Cleans raw review text:
    1. Removes HTML tags (common in scraped reviews like IMDB).
    2. Strips URL hyperlinks.
    3. Retains alphanumeric characters and basic punctuation.
    4. Converts characters to lowercase and collapses redundant whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Keep alphanumeric words and common sentence punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def fit_tokenizer(texts: List[str], max_words: int = MAX_VOCAB_SIZE) -> Tokenizer:
    """
    Fits Keras Tokenizer on the training vocabulary.
    Assigns numerical index to the most frequent words up to `max_words`.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def transform_texts(
    tokenizer: Tokenizer,
    texts: List[str],
    max_len: int = MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Converts list of text strings into uniform padded sequence matrices.
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding=PADDING_TYPE,
        truncating=TRUNCATING_TYPE,
    )
    return padded


def save_tokenizer(tokenizer: Tokenizer, path: Path = TOKENIZER_PATH) -> None:
    """Serializes fitted tokenizer to disk using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Tokenizer:
    """Deserializes saved tokenizer from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run training first.")
    with open(path, "rb") as f:
        return pickle.load(f)
