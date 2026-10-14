"""
Tests for text cleaning, tokenization, and padding procedures.
"""

from src.config import MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE
from src.preprocessor import clean_text, fit_tokenizer, transform_texts


def test_clean_text_removes_html():
    raw_html = "<br /><p>The movie was <b>fantastic</b>!</p>"
    cleaned = clean_text(raw_html)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "fantastic" in cleaned


def test_clean_text_lowercasing_and_urls():
    raw_url = "Great film! Check reviews at https://example.com/movie NOW."
    cleaned = clean_text(raw_url)
    assert "https" not in cleaned
    assert "now" in cleaned
    assert "NOW" not in cleaned


def test_transform_texts_output_shape():
    sample_texts = [
        "A truly remarkable performance by the entire cast.",
        "Dreadful plot and poor acting.",
    ]
    tokenizer = fit_tokenizer(sample_texts, max_words=MAX_VOCAB_SIZE)
    padded = transform_texts(tokenizer, sample_texts, max_len=MAX_SEQUENCE_LENGTH)

    assert padded.shape == (2, MAX_SEQUENCE_LENGTH)
