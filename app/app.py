"""
Interactive Streamlit Web Application for RNN & LSTM Sentiment Analysis.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so src imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LSTM_MODEL_PATH, MODEL_PATH, TOKENIZER_PATH
from src.predict import SentimentPredictor

st.set_page_config(
    page_title="RNN & LSTM Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    .badge-positive {
        color: #15803d;
        background-color: #dcfce7;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-negative {
        color: #b91c1c;
        background-color: #fee2e2;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-model {
        color: #4338ca;
        background-color: #e0e7ff;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Sentiment Analysis: SimpleRNN vs LSTM")
st.caption("Natural Language Processing Benchmark with TensorFlow/Keras & IMDB Reviews")

# Sidebar - Architecture Selection & Benchmarks
with st.sidebar:
    st.header("⚙️ Model Selector")
    model_choice = st.radio(
        "Select Architecture to Run Inference:",
        ["LSTM (Recommended)", "SimpleRNN (Baseline)"],
        index=0,
        help="Compare how LSTM solves SimpleRNN's vanishing gradient problem.",
    )
    selected_type = "lstm" if "LSTM" in model_choice else "simple_rnn"

    st.divider()
    st.header("📊 Benchmark Comparison")
    st.markdown(
        """
        | Metric | SimpleRNN | LSTM |
        | :--- | :---: | :---: |
        | **Test Accuracy** | `49.97%` | **`87.58%`** |
        | **ROC-AUC** | `0.4918` | **`0.9431`** |
        | **F1-Score** | `0.4176` | **`0.8758`** |
        | **Parameters** | 650,369 | 675,137 |
        """
    )
    st.divider()
    st.caption("IMDB 50K Dataset • Max Length: 200 • Vocab: 10,000")

st.subheader("Enter a Customer / Movie Review")

sample_reviews = {
    "Select a preset...": "",
    "Positive: Highly Praised": "This movie was an absolute masterpiece! Incredible performances and a thrilling plot.",
    "Negative: Deeply Disappointed": "Total waste of time and money. Terrible directing, awful dialogue, and boring characters.",
    "Negation: 'Not bad at all'": "The movie was not bad at all, actually quite enjoyable and entertaining.",
    "Negation: 'Definitely not good'": "This was definitely not good, completely failed to meet any expectations.",
    "Contrast / Nuance": "The cinematography and music were breathtaking, but the terrible script ruined everything.",
}

selected_sample = st.selectbox("Quick Test with Presets:", list(sample_reviews.keys()))
default_text = sample_reviews[selected_sample] if selected_sample != "Select a preset..." else ""

user_input = st.text_area(
    "Review Text:",
    value=default_text,
    placeholder="Type or paste a movie/product review here...",
    height=130,
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("Classify Review", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.error("Please enter a review before clicking Classify.")
    else:
        with st.spinner(f"Running inference through {selected_type.upper()}..."):
            predictor = SentimentPredictor(model_type=selected_type)
            result = predictor.predict(user_input)

        label = result["label"]
        confidence = result["confidence"]
        raw_score = result["raw_score"]

        st.markdown("---")
        st.subheader("Classification Result")
        st.markdown(f'<span class="badge-model">Active Model: {selected_type.upper()}</span>', unsafe_allow_html=True)
        st.write("")

        if label == "Positive":
            st.markdown(f'<span class="badge-positive">👍 POSITIVE ({confidence*100:.2f}%)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-negative">👎 NEGATIVE ({confidence*100:.2f}%)</span>', unsafe_allow_html=True)

        st.progress(float(confidence))

        with st.expander("Technical Model Output Details"):
            st.write(f"- **Active Model:** `{selected_type.upper()}`")
            st.write(f"- **Raw Sigmoid Output:** `{raw_score:.4f}`")
            st.write(f"- **Decision Threshold:** `>= 0.5` -> Positive | `< 0.5` -> Negative")
            st.write(f"- **Confidence:** `{confidence*100:.2f}%`")
