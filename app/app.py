"""
app.py
Production-grade Streamlit web application for RNN, LSTM, and GRU Sentiment Analysis.
Features:
- Cached resource loading (@st.cache_resource) with zero retraining on startup
- Complete NLP pipeline: Cleaning -> Tokenization -> Padding -> Model Inference -> Label Mapping
- Interactive presets and real-time custom review analysis
- Probability gauge, confidence metrics, and technical inspection expander
- Architectural benchmark dashboard and pipeline documentation
"""

import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st

# Configure root path for modular imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set page configuration
st.set_page_config(
    page_title="Deep Learning Sentiment Analyzer | RNN vs LSTM vs GRU",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suppress TensorFlow C++ informational logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paths
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
PROD_MODEL_PATH = SAVED_MODELS_DIR / "production_model.keras"
GRU_MODEL_PATH = SAVED_MODELS_DIR / "gru_sentiment_model.keras"
LSTM_MODEL_PATH = SAVED_MODELS_DIR / "lstm_sentiment_model.keras"
SIMPLERNN_MODEL_PATH = SAVED_MODELS_DIR / "simple_rnn_sentiment_model.keras"
TOKENIZER_PATH = SAVED_MODELS_DIR / "tokenizer.pkl"
PREPROCESSING_CONFIG_PATH = SAVED_MODELS_DIR / "preprocessing_config.json"
LABEL_MAPPING_PATH = SAVED_MODELS_DIR / "label_mapping.json"
BENCHMARK_CHART_PATH = SAVED_MODELS_DIR / "model_comparison_charts.png"


# =====================================================================
# 1. Custom Modern CSS Styling
# =====================================================================
st.markdown(
    """
    <style>
    /* Metric and Result Cards */
    .result-container {
        padding: 24px;
        border-radius: 14px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .result-positive {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #86efac;
    }
    .result-negative {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #fca5a5;
    }
    .result-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .title-pos { color: #166534; }
    .title-neg { color: #991b1b; }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .badge-pos { background-color: #bbf7d0; color: #14532d; }
    .badge-neg { background-color: #fecaca; color: #7f1d1d; }
    
    .stat-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 14px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* Pipeline Step Box */
    .pipeline-step {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# 2. Cached Artifact Loading (Zero Retraining)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_cached_artifacts(model_type: str = "gru") -> Tuple[object, object, dict, dict]:
    """
    Loads model, tokenizer, and configurations with caching.
    Ensures that artifacts are loaded into memory once and NEVER retrained on startup.
    """
    from tensorflow.keras.models import load_model

    # Resolve model file path
    if model_type == "gru":
        target_path = PROD_MODEL_PATH if PROD_MODEL_PATH.exists() else GRU_MODEL_PATH
    elif model_type == "lstm":
        target_path = LSTM_MODEL_PATH
    else:
        target_path = SIMPLERNN_MODEL_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found at: {target_path}")
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at: {TOKENIZER_PATH}")

    # Load Model
    model = load_model(target_path)

    # Load Tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Load Preprocessing Config
    if PREPROCESSING_CONFIG_PATH.exists():
        with open(PREPROCESSING_CONFIG_PATH, "r", encoding="utf-8") as f:
            prep_config = json.load(f)
    else:
        prep_config = {"max_sequence_length": 200, "padding_type": "post", "truncating_type": "post"}

    # Load Label Mapping
    if LABEL_MAPPING_PATH.exists():
        with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
    else:
        label_mapping = {"id2label": {"0": "Negative", "1": "Positive"}, "decision_threshold": 0.5}

    return model, tokenizer, prep_config, label_mapping


# =====================================================================
# 3. Text Preprocessing (Parity with Training)
# =====================================================================
def clean_user_text(text: str) -> str:
    """Deterministic cleaning mirroring training preprocessor."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)               # HTML tags
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)        # HTML entities
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
    text = text.lower()                                 # Lowercase
    text = re.sub(r"([.,!?])", r" \1 ", text)           # Isolate punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?\'\s]", " ", text)   # Filter non-whitelisted characters
    return re.sub(r"\s+", " ", text).strip()            # Collapse whitespace


def run_inference_pipeline(
    model, tokenizer, config: dict, label_mapping: dict, raw_text: str
) -> Dict:
    """Executes the full NLP pipeline from raw text to prediction."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # 1. Clean
    cleaned = clean_user_text(raw_text)

    # 2. Tokenize
    sequences = tokenizer.texts_to_sequences([cleaned])

    # 3. Pad
    max_len = config.get("max_sequence_length", 200)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding=config.get("padding_type", "post"),
        truncating=config.get("truncating_type", "post"),
    )

    # 4. Predict
    raw_prob = float(model.predict(padded, verbose=0)[0][0])
    threshold = label_mapping.get("decision_threshold", 0.5)

    is_positive = raw_prob >= threshold
    label = "Positive" if is_positive else "Negative"
    confidence = raw_prob if is_positive else (1.0 - raw_prob)

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "tokens": sequences[0],
        "padded_shape": padded.shape,
        "padded_vector": padded[0],
        "probability": raw_prob,
        "confidence": confidence,
        "label": label,
        "threshold": threshold,
    }


# =====================================================================
# 4. Sidebar: Architecture Selection & Empirical Benchmark
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/clapperboard.png", width=90)
    st.title("Model Configuration")
    
    model_selection = st.radio(
        "Select Architecture to Run:",
        [
            "GRU (Production Winner ⭐)",
            "LSTM (Gated Architecture)",
            "SimpleRNN (Baseline Model)",
        ],
        index=0,
        help="Switch between architectures to compare their predictions and latency.",
    )
    
    if "GRU" in model_selection:
        selected_model_type = "gru"
        model_display_name = "Gated Recurrent Unit (GRU)"
    elif "LSTM" in model_selection:
        selected_model_type = "lstm"
        model_display_name = "Long Short-Term Memory (LSTM)"
    else:
        selected_model_type = "simple_rnn"
        model_display_name = "SimpleRNN (Vanishing Gradient Baseline)"

    st.markdown("---")
    st.subheader("📊 Empirical Test Benchmark")
    st.markdown(
        """
        | Metric | SimpleRNN | LSTM | GRU |
        | :--- | :---: | :---: | :---: |
        | **Accuracy** | 49.97% | 87.58% | **88.26%** |
        | **F1-Score** | 0.4176 | 0.8758 | **0.8826** |
        | **ROC-AUC** | 0.4918 | 0.9431 | **0.9524** |
        | **Train Time**| 113.9s | 136.9s | **119.6s** |
        | **Params** | 650.4k | 675.1k | **667.1k** |
        """
    )
    st.caption("Benchmark on 50,000 IMDB Reviews (4,959 Test Set)")
    st.markdown("---")
    st.caption("Powered by TensorFlow / Keras 3 • Zero-Retraining Deployment")


# =====================================================================
# 5. Main Application Header & Tabs
# =====================================================================
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown(
    "A production NLP deep learning application trained on the **IMDB 50K Dataset**. "
    "Compare recurrent neural architectures (**SimpleRNN**, **LSTM**, and **GRU**) in real time."
)

tab_predict, tab_benchmark, tab_pipeline = st.tabs([
    "🚀 Live Sentiment Prediction",
    "📈 Architecture Benchmark Dashboard",
    "🔬 Pipeline & Tokenization Deep Dive",
])


# =====================================================================
# TAB 1: Live Sentiment Prediction
# =====================================================================
with tab_predict:
    st.subheader("1. Select a Preset or Enter Custom Review")
    
    sample_presets = {
        "— Select an example review preset —": "",
        "🌟 Overwhelmingly Positive": "This movie was an absolute masterpiece! Incredible performances, tight directing, and a thrilling plot that kept me hooked.",
        "💥 Deeply Negative": "Complete waste of time and money. Terrible directing, awful dialogue, wooden acting, and boring characters.",
        "⚖️ Subtle Negation ('not bad at all')": "The movie was not bad at all, actually quite enjoyable, heartwarming, and entertaining.",
        "⚠️ Tricky Negation ('definitely not good')": "This was definitely not good, completely failed to meet any reasonable expectations.",
        "🎭 Contrastive Conjunction ('breathtaking but script ruined')": "The cinematography and music were breathtaking, but the terrible script and dreadful ending ruined everything.",
        "🧐 Sarcastic Irony": "If you genuinely enjoy staring at wet paint drying for two hours, you will love this thrilling masterpiece.",
        "🎬 Mixed Nuance": "It started off very slow and boring, but the thrilling climax made it totally worthwhile.",
    }
    
    selected_preset_key = st.selectbox("Choose a sample preset for quick testing:", list(sample_presets.keys()))
    default_text = sample_presets[selected_preset_key]
    
    user_review = st.text_area(
        "Enter customer review or movie critique text:",
        value=default_text,
        placeholder="Type or paste your review here (e.g., 'An outstanding film with a captivating soundtrack...')...",
        height=140,
        help="Accepts raw unformatted text with punctuation, numbers, and contractions.",
    )
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        predict_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col_info:
        st.caption(f"Active Model: **{model_display_name}** | Threshold: **0.50**")
        
    # Execution & Error Handling
    if predict_button:
        # Error handling: Empty or whitespace-only input
        if not user_review or not user_review.strip():
            st.error("⚠️ **Input Error:** Please enter some review text before running the sentiment analysis.")
        elif len(user_review.strip()) < 5:
            st.warning("⚠️ **Short Input Warning:** The review is very short (< 5 characters). Predictions are most accurate with complete phrases or sentences.")
        else:
            try:
                # Load cached model without retraining
                with st.spinner(f"Running inference with {selected_model_type.upper()}..."):
                    model, tokenizer, prep_config, label_mapping = load_cached_artifacts(selected_model_type)
                    res = run_inference_pipeline(model, tokenizer, prep_config, label_mapping, user_review)

                st.markdown("---")
                
                # Display Results
                is_pos = res["label"] == "Positive"
                container_class = "result-positive" if is_pos else "result-negative"
                title_class = "title-pos" if is_pos else "title-neg"
                badge_class = "badge-pos" if is_pos else "badge-neg"
                icon = "👍" if is_pos else "👎"
                
                st.markdown(
                    f"""
                    <div class="result-container {container_class}">
                        <span class="badge-pill {badge_class}">PREDICTED SENTIMENT</span>
                        <div class="result-title {title_class}">{icon} {res['label'].upper()}</div>
                        <p style="margin: 0; color: #334155; font-size: 1.05rem;">
                            Model Confidence: <strong>{res['confidence'] * 100:.2f}%</strong> 
                            (Sigmoid Probability: <code>{res['probability']:.4f}</code>)
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Metrics Grid
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="stat-box"><div class="stat-label">Sentiment</div><div class="stat-value">{res["label"]}</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="stat-box"><div class="stat-label">Confidence</div><div class="stat-value">{res["confidence"]*100:.1f}%</div></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="stat-box"><div class="stat-label">Sigmoid Output</div><div class="stat-value">{res["probability"]:.4f}</div></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="stat-box"><div class="stat-label">Decision Threshold</div><div class="stat-value">{res["threshold"]}</div></div>', unsafe_allow_html=True)

                st.write("")
                st.write("**Probability Scale (0.0 = Strongly Negative, 1.0 = Strongly Positive):**")
                st.progress(float(res["probability"]))

                # Technical Inspection Expander
                with st.expander("🛠️ View Pipeline Debug & Tensor Inspection"):
                    st.write(f"- **Cleaned Text:** `{res['cleaned_text']}`")
                    st.write(f"- **Number of Tokens Extracted:** `{len(res['tokens'])}` words")
                    st.write(f"- **Sequence Token IDs:** `{res['tokens'][:30]}` {'...' if len(res['tokens']) > 30 else ''}")
                    st.write(f"- **Padded Input Vector Shape:** `{res['padded_shape']}` (Post-padded to 200 tokens)")
                    st.write(f"- **Active Model Architecture:** `{selected_model_type.upper()}`")
                    
            except FileNotFoundError as fnf_err:
                st.error(f"❌ **Artifact Error:** {str(fnf_err)}. Ensure models are trained and saved in `saved_models/`.")
            except Exception as exc:
                st.error(f"❌ **Unexpected Error during inference:** {str(exc)}")


# =====================================================================
# TAB 2: Architecture Benchmark Dashboard
# =====================================================================
with tab_benchmark:
    st.subheader("Architectural Comparison: SimpleRNN vs LSTM vs GRU")
    st.markdown(
        "All three models were trained on the identical **IMDB 50K Dataset** "
        "(39,665 Train, 4,958 Validation, 4,959 Test) with identical vocabulary (10,000 words) "
        "and sequence lengths (200 tokens)."
    )
    
    if BENCHMARK_CHART_PATH.exists():
        st.image(str(BENCHMARK_CHART_PATH), caption="Empirical Benchmark Dashboard (SimpleRNN vs LSTM vs GRU)", use_container_width=True)
    else:
        st.info("Run `python src/benchmark_charts.py` to generate the 4-panel benchmark dashboard.")
        
    st.markdown("---")
    st.subheader("Key Findings:")
    st.markdown(
        """
        1. **SimpleRNN Failure (Vanishing Gradient)**:
           - Stalled at random guessing accuracy (**49.97%**) with binary cross-entropy loss stuck at $\ln(2) \approx 0.6934$. Over sequences of 200 words, gradients vanished to near-zero.
        2. **LSTM Success (Additive Error Carousel)**:
           - Captured long-term context via the separate cell state ($C_t$), achieving **87.58% accuracy** and **0.9431 ROC-AUC**.
        3. **GRU Victory (Efficiency & Accuracy Champion)**:
           - Achieved the highest test accuracy (**88.26%**) and lowest validation loss (**0.2761**).
           - Trained **~12.6% faster** than LSTM (119.6s vs 136.9s) with **25% fewer recurrent parameters** (24,960 vs 33,024).
        """
    )


# =====================================================================
# TAB 3: Pipeline & Tokenization Deep Dive
# =====================================================================
with tab_pipeline:
    st.subheader("Under the Hood: End-to-End NLP Inference Pipeline")
    
    st.markdown(
        """
        <div class="pipeline-step">
            <strong>Step 1: Deterministic Text Cleaning</strong><br>
            Strips HTML tags (<code>&lt;br /&gt;</code>), lowercases letters, isolates sentence-ending punctuation (<code>film!</code> &rarr; <code>film !</code>), and collapses redundant spaces.
        </div>
        <div class="pipeline-step">
            <strong>Step 2: Vocabulary Tokenization</strong><br>
            Maps clean words to integer IDs using the 10,000-word fitted vocabulary. Words not seen during training are assigned to the Out-of-Vocabulary token <code>&lt;OOV&gt;</code> (ID: 1).
        </div>
        <div class="pipeline-step">
            <strong>Step 3: Uniform Sequence Padding</strong><br>
            Pads short reviews with trailing zeros (<code>post</code>-padding) and truncates long reviews beyond 200 words to create uniform <code>(1, 200)</code> input tensors.
        </div>
        <div class="pipeline-step">
            <strong>Step 4: Recurrent Model Forward Pass</strong><br>
            Feeds the 200-integer sequence into the Embedding layer (64-dim continuous vectors) &rarr; GRU / LSTM / SimpleRNN recurrent units &rarr; Dense layer &rarr; Sigmoid output unit.
        </div>
        <div class="pipeline-step">
            <strong>Step 5: Decision Thresholding</strong><br>
            If Sigmoid output $p \ge 0.50$, the review is classified as <strong>Positive</strong>; otherwise, it is classified as <strong>Negative</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
