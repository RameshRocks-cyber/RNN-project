"""
Unit tests for text preprocessing, cleaning, label encoding, and data splitting.
"""

import pandas as pd
import numpy as np
from src.preprocessor import clean_text, encode_labels, decode_labels
from src.data_loader import preprocess_dataset, split_data


def test_clean_text_removes_html():
    raw_html = "<br /><p>The movie was <b>fantastic</b>! Definitely 10/10.</p>"
    cleaned = clean_text(raw_html)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "fantastic" in cleaned


def test_clean_text_lowercasing_and_urls():
    raw = "Great film! Check reviews at https://example.com/movie NOW."
    cleaned = clean_text(raw)
    assert "https" not in cleaned
    assert "now" in cleaned
    assert "NOW" not in cleaned


def test_clean_text_punctuation_isolation():
    raw = "Loved it! Absolutely stunning, brilliant."
    cleaned = clean_text(raw)
    # Check that punctuation is surrounded by spaces so tokens remain clean
    assert " !" in cleaned
    assert " ," in cleaned
    assert " ." in cleaned


def test_clean_text_removes_unnecessary_characters():
    raw = "Movie was amazing!!! #BestMovie @director $100 &*()%"
    cleaned = clean_text(raw)
    assert "#" not in cleaned
    assert "@" not in cleaned
    assert "$" not in cleaned
    assert "%" not in cleaned


def test_clean_text_handles_extra_spaces_and_newlines():
    raw = "  This   review   has   too\n\nmany   spaces\t\tand tabs.  "
    cleaned = clean_text(raw)
    assert "  " not in cleaned
    assert "\n" not in cleaned
    assert "\t" not in cleaned
    assert cleaned == "this review has too many spaces and tabs ."


def test_encode_and_decode_labels():
    labels = ["positive", "negative", "positive", "negative"]
    encoded = encode_labels(labels)
    assert np.array_equal(encoded, np.array([1, 0, 1, 0]))

    decoded = decode_labels(encoded)
    assert decoded == ["positive", "negative", "positive", "negative"]


def test_split_data_ratios_and_stratification():
    # Create synthetic dataset with 100 rows (50 positive, 50 negative)
    df = pd.DataFrame(
        {
            "clean_review": [f"Review sample number {i}" for i in range(100)],
            "label": [0] * 50 + [1] * 50,
        }
    )
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    assert len(X_train) == 80
    assert len(X_val) == 10
    assert len(X_test) == 10

    # Verify stratified balance in splits
    assert y_train.value_counts()[0] == 40
    assert y_train.value_counts()[1] == 40
    assert y_val.value_counts()[0] == 5
    assert y_val.value_counts()[1] == 5
    assert y_test.value_counts()[0] == 5
    assert y_test.value_counts()[1] == 5
