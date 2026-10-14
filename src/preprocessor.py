"""
Text preprocessing and label encoding pipeline for RNN Sentiment Analysis.
Handles HTML removal, lowercasing, punctuation normalization, whitespace collapsing,
and binary label conversion.
"""

import pickle
import re
from pathlib import Path
from typing import List, Sequence, Union
import numpy as np
import pandas as pd

from src.config import (
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    OOV_TOKEN,
    PADDING_TYPE,
    TOKENIZER_PATH,
    TRUNCATING_TYPE,
)

LABEL_MAPPING = {
    "negative": 0,
    "positive": 1,
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def clean_text(text: str) -> str:
    """
    Cleans and standardizes raw review text without aggressively destroying
    syntax or negation words needed by recurrent neural networks.

    Pipeline Steps:
    1. Input Validation: Fallback for non-string / NaN values.
    2. HTML Tag & Entity Removal: Strips <br />, <p>, &amp;, &quot;, etc.
    3. Hyperlink Removal: Removes http:// and https:// URLs.
    4. Lowercasing: Normalizes all characters to lowercase.
    5. Punctuation Handling: Isolates sentence-ending punctuation (.,!?)
       with spaces so 'film!' becomes 'film !', preventing vocabulary explosion.
    6. Special Character Filtering: Retains letters, numbers, apostrophes (for contractions),
       and basic punctuation; discards non-printable/control symbols.
    7. Whitespace Collapsing: Compresses tabs, newlines, and multiple spaces into a single space.
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove HTML tags (<br />, <b>, <a href=...>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Decode common HTML entities (&amp;, &quot;, &#39;, etc.)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Lowercase normalization
    text = text.lower()

    # 5. Isolate common punctuation with whitespace so words don't bind to punctuation
    # Example: "amazing!" -> "amazing !"
    text = re.sub(r"([.,!?])", r" \1 ", text)

    # 6. Keep only alphanumeric characters, spaces, apostrophes (contractions like don't), and punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\s]", " ", text)

    # 7. Collapse redundant whitespaces, tabs, and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


def encode_labels(labels: Union[Sequence[str], pd.Series]) -> np.ndarray:
    """
    Converts string categorical labels to binary integers:
      'negative' -> 0
      'positive' -> 1
    Raises ValueError if an unexpected label is encountered.
    """
    encoded = []
    for label in labels:
        normalized = str(label).strip().lower()
        if normalized not in LABEL_MAPPING:
            raise ValueError(
                f"Unknown sentiment label '{label}'. Expected 'positive' or 'negative'."
            )
        encoded.append(LABEL_MAPPING[normalized])
    return np.array(encoded, dtype=np.int32)


def decode_labels(ids: Union[Sequence[int], np.ndarray]) -> List[str]:
    """
    Converts binary integer predictions back to readable labels:
      0 -> 'negative'
      1 -> 'positive'
    """
    return [REVERSE_LABEL_MAPPING.get(int(i), "unknown") for i in ids]


def fit_tokenizer(texts: List[str], max_words: int = MAX_VOCAB_SIZE):
    """
    Fits Keras Tokenizer on cleaned training texts.
    Assigns numerical integer indices to the most frequent words up to `max_words`.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=max_words, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def transform_texts(
    tokenizer,
    texts: List[str],
    max_len: int = MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """
    Cleans texts, maps tokens to integer IDs, and pads/truncates to a uniform sequence length.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding=PADDING_TYPE,
        truncating=TRUNCATING_TYPE,
    )
    return padded


def save_tokenizer(tokenizer, path: Path = TOKENIZER_PATH) -> None:
    """Serializes fitted tokenizer to disk using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: Path = TOKENIZER_PATH):
    """Deserializes saved tokenizer from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}. Run training first.")
    with open(path, "rb") as f:
        return pickle.load(f)
