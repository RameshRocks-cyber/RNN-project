"""
Interactive Streamlit Web Application for RNN Sentiment Analysis.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path so src imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, TOKENIZER_PATH

st.set_page_config(
    page_title="RNN Sentiment Analyzer",
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 RNN Sentiment Analysis")
st.caption("Deep Learning NLP with TensorFlow/Keras & Recurrent Neural Networks")

# Sidebar - Project Information & Architecture
with st.sidebar:
    st.header("📌 Project Overview")
    st.markdown(
        """
        **Architecture Pipeline:**
        1. **Raw Review Text**
        2. **Preprocessing & Tokenization**
        3. **Sequence Padding** (Max Len = 200)
        4. **Embedding Layer** (Dim = 64)
        5. **SimpleRNN Layer** (Units = 64)
        6. **Dropout & Dense Layer**
        7. **Sigmoid Output** [0 = Neg, 1 = Pos]
        """
    )
    st.divider()
    st.subheader("Model Status")
    artifacts_ready = MODEL_PATH.exists() and TOKENIZER_PATH.exists()
    if artifacts_ready:
        st.success("Trained Model & Tokenizer: Available")
    else:
        st.warning("Trained Model: Pending (Phase 2 Training)")

st.subheader("Enter a Customer / Movie Review")

# Sample Review buttons for quick testing
sample_reviews = {
    "Select a preset...": "",
    "Positive: Highly Praised": "This movie was absolutely fantastic! The acting was brilliant, the screenplay was captivating, and the soundtrack gave me chills throughout.",
    "Negative: Deeply Disappointed": "An utter waste of time and money. The plot was completely nonsensical, characters were shallow, and the pacing was painfully sluggish.",
    "Mixed / Nuanced": "The visual effects and cinematography were top-notch, but unfortunately the dialogue and narrative felt rushed and predictable.",
}

selected_sample = st.selectbox("Quick Test with Sample Reviews:", list(sample_reviews.keys()))
default_text = sample_reviews[selected_sample] if selected_sample != "Select a preset..." else ""

user_input = st.text_area(
    "Review Text:",
    value=default_text,
    placeholder="Type or paste a movie/product review here...",
    height=140,
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.error("Please enter a review before clicking Analyze.")
    elif not artifacts_ready:
        st.info(
            "Model artifacts (`simple_rnn_sentiment_model.keras` and `tokenizer.pkl`) "
            "will be generated in Phase 2 during training. "
            "Phase 1 scaffolding verified successfully!"
        )
    else:
        from src.predict import predict_sentiment

        with st.spinner("Processing text and running recurrent inference..."):
            result = predict_sentiment(user_input)

        label = result["label"]
        confidence = result["confidence"]
        raw_score = result["raw_score"]

        st.markdown("---")
        st.subheader("Prediction Result")

        if label == "Positive":
            st.markdown(f'<span class="badge-positive">👍 POSITIVE ({confidence*100:.1f}%)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-negative">👎 NEGATIVE ({confidence*100:.1f}%)</span>', unsafe_allow_html=True)

        st.progress(confidence)

        with st.expander("Technical Model Output Details"):
            st.write(f"- **Raw Sigmoid Output:** `{raw_score}`")
            st.write(f"- **Decision Threshold:** `>= 0.5` -> Positive | `< 0.5` -> Negative")
            st.write(f"- **Confidence Score:** `{confidence*100:.2f}%`")
