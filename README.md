# 🎬 End-to-End RNN Sentiment Analysis System

An end-to-end Natural Language Processing (NLP) deep learning project that classifies user reviews (movie, product, or customer feedback) into **Positive** or **Negative** sentiment using a Recurrent Neural Network (SimpleRNN) implemented with **TensorFlow / Keras**.

---

## 🏗️ Project Architecture

```
rnn-sentiment-analysis/
├── data/
│   ├── raw/                  # Original raw review dataset (e.g. IMDB CSV)
│   └── processed/            # Cleaned, tokenized matrices or processed splits
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb     # Exploratory analysis & text cleaning
│   └── 02_model_experimentation.ipynb     # Architecture prototyping (SimpleRNN / LSTM)
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Global hyperparameters & absolute path definitions
│   ├── data_loader.py        # Dataset loading & stratified train/val/test splits
│   ├── preprocessor.py       # Regex normalization, Tokenizer, & sequence padding
│   ├── model.py              # SimpleRNN neural network architecture definition
│   ├── train.py              # Model training loop, EarlyStopping, & Checkpointing
│   ├── evaluate.py           # Evaluation metrics (Accuracy, F1, Confusion Matrix, ROC-AUC)
│   └── predict.py            # Real-time inference API for novel input text
├── saved_models/             # Persisted Keras model weights and fitted tokenizers
├── app/
│   └── app.py                # Interactive Streamlit web application
├── tests/
│   ├── test_preprocessor.py  # Unit tests for text cleaning and tensor padding
│   └── test_model.py         # Unit tests for model architecture and output shapes
├── .gitignore                # Git ignore rules for venv, cache, and large binaries
├── requirements.txt          # Python dependency specifications
└── README.md                 # Complete documentation and quickstart guide
```

---

## ⚡ Quickstart Guide

### 1. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install Project Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Register Jupyter Kernel (Optional for Notebooks)
```bash
python -m ipykernel install --user --name=rnn-env --display-name "Python (RNN Sentiment)"
```

### 4. Run the Streamlit Web Application
```bash
streamlit run app/app.py
```
