"""
test_prediction.py
Unit tests verifying the production prediction engine and standalone inference.
"""

from prediction import clean_text_for_inference, predict_sentiment, predict_sentiment_batch


def test_clean_text_for_inference():
    raw = "<br />Amazing movie! Check http://example.com/movie &amp; enjoy.   "
    cleaned = clean_text_for_inference(raw)
    assert "<br />" not in cleaned
    assert "http" not in cleaned
    assert "!" in cleaned
    assert "amazing" in cleaned


def test_predict_sentiment_positive():
    res = predict_sentiment("An incredible masterpiece! I loved every single second of it.")
    assert isinstance(res, dict)
    assert res["sentiment"] == "Positive"
    assert res["confidence"] > 0.5
    assert 0.0 <= res["probability"] <= 1.0
    assert res["decision_threshold"] == 0.5


def test_predict_sentiment_negative():
    res = predict_sentiment("A terrible, boring, and utterly unwatchable disaster.")
    assert isinstance(res, dict)
    assert res["sentiment"] == "Negative"
    assert res["confidence"] > 0.5
    assert res["probability"] < 0.5


def test_predict_sentiment_batch():
    texts = [
        "A wonderful and heartwarming story.",
        "Horrible acting, awful dialogue, and dreadful pacing."
    ]
    results = predict_sentiment_batch(texts)
    assert len(results) == 2
    assert results[0]["sentiment"] == "Positive"
    assert results[1]["sentiment"] == "Negative"
