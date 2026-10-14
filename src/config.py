"""
Central configuration for hyperparameters, directories, and data constants.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Dataset settings
DATASET_FILENAME = "imdb_reviews.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_FILENAME
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# NLP / Tokenization Hyperparameters
MAX_VOCAB_SIZE = 10000       # Top N most frequent words in vocabulary
MAX_SEQUENCE_LENGTH = 200    # Fixed length for input sequences (truncation/padding)
OOV_TOKEN = "<OOV>"          # Out-Of-Vocabulary token for unseen words
PADDING_TYPE = "post"        # 'pre' or 'post' padding
TRUNCATING_TYPE = "post"     # 'pre' or 'post' truncation

# Model Architecture Hyperparameters
EMBEDDING_DIM = 64           # Dimensionality of the dense word embedding space
RNN_UNITS = 64               # Hidden state dimensions for the SimpleRNN layer
DROPOUT_RATE = 0.3           # Dropout regularization probability
DENSE_UNITS = 32             # Hidden units in intermediate dense layer
OUTPUT_UNITS = 1             # 1 unit with sigmoid activation for binary classification

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3

# Saved Artifact Names (SimpleRNN)
MODEL_PATH = SAVED_MODELS_DIR / "simple_rnn_sentiment_model.keras"
TOKENIZER_PATH = SAVED_MODELS_DIR / "tokenizer.pkl"
METRICS_PATH = SAVED_MODELS_DIR / "training_history.json"

# Saved Artifact Names (LSTM)
LSTM_MODEL_PATH = SAVED_MODELS_DIR / "lstm_sentiment_model.keras"
LSTM_METRICS_PATH = SAVED_MODELS_DIR / "lstm_training_history.json"
LSTM_PLOT_PATH = SAVED_MODELS_DIR / "lstm_learning_curves.png"
LSTM_CONFUSION_MATRIX_PATH = SAVED_MODELS_DIR / "lstm_confusion_matrix.png"

# Saved Artifact Names (GRU)
GRU_MODEL_PATH = SAVED_MODELS_DIR / "gru_sentiment_model.keras"
GRU_METRICS_PATH = SAVED_MODELS_DIR / "gru_training_history.json"
GRU_PLOT_PATH = SAVED_MODELS_DIR / "gru_learning_curves.png"
GRU_CONFUSION_MATRIX_PATH = SAVED_MODELS_DIR / "gru_confusion_matrix.png"


