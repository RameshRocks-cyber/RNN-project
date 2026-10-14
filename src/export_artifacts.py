"""
export_deployment_artifacts.py
Exports production deployment artifacts:
1. Copies the selected best model (GRU) to saved_models/production_model.keras
2. Exports saved_models/preprocessing_config.json
3. Exports saved_models/label_mapping.json
4. Exports saved_models/tokenizer.json for portable deserialization
5. Exports saved_models/model_metadata.json
"""

import json
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GRU_MODEL_PATH,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    OOV_TOKEN,
    PADDING_TYPE,
    SAVED_MODELS_DIR,
    TOKENIZER_PATH,
    TRUNCATING_TYPE,
)
from src.preprocessor import load_tokenizer

def export_artifacts():
    print("[1/5] Exporting production Keras model...")
    prod_model_path = SAVED_MODELS_DIR / "production_model.keras"
    if not GRU_MODEL_PATH.exists():
        raise FileNotFoundError(f"Selected GRU model not found at {GRU_MODEL_PATH}")
    shutil.copy2(GRU_MODEL_PATH, prod_model_path)
    print(f"  -> Successfully created production model artifact: {prod_model_path} ({prod_model_path.stat().st_size / 1024 / 1024:.2f} MB)")

    print("[2/5] Exporting preprocessing configuration...")
    preprocessing_config = {
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "oov_token": OOV_TOKEN,
        "padding_type": PADDING_TYPE,
        "truncating_type": TRUNCATING_TYPE,
        "cleaning": {
            "strip_html": True,
            "strip_urls": True,
            "lowercase": True,
            "isolate_punctuation": True,
            "preserve_apostrophes": True,
            "collapse_whitespace": True
        }
    }
    prep_config_path = SAVED_MODELS_DIR / "preprocessing_config.json"
    with open(prep_config_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_config, f, indent=4)
    print(f"  -> Preprocessing config saved: {prep_config_path}")

    print("[3/5] Exporting label mapping...")
    label_mapping = {
        "id2label": {
            "0": "Negative",
            "1": "Positive"
        },
        "label2id": {
            "Negative": 0,
            "Positive": 1
        },
        "decision_threshold": 0.5,
        "description": "Sigmoid output >= 0.5 is mapped to Positive (1), < 0.5 is mapped to Negative (0)."
    }
    label_mapping_path = SAVED_MODELS_DIR / "label_mapping.json"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)
    print(f"  -> Label mapping saved: {label_mapping_path}")

    print("[4/5] Exporting portable tokenizer JSON...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    tokenizer_json = tokenizer.to_json()
    tokenizer_json_path = SAVED_MODELS_DIR / "tokenizer.json"
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
    print(f"  -> Tokenizer JSON saved: {tokenizer_json_path}")

    print("[5/5] Exporting model metadata manifest...")
    metadata = {
        "model_name": "GRU Sentiment Classifier",
        "architecture_type": "Gated Recurrent Unit (GRU)",
        "framework": "TensorFlow / Keras 3",
        "serialization_format": "Native Keras v3 (.keras zip archive)",
        "training_dataset": "IMDB 50,000 Movie Reviews (Raw Text)",
        "dataset_split": {
            "train": 39665,
            "validation": 4958,
            "test": 4959
        },
        "performance_metrics": {
            "test_accuracy": 0.8826,
            "test_precision_positive": 0.8919,
            "test_recall_positive": 0.8718,
            "test_f1_positive": 0.8818,
            "test_macro_f1": 0.8826,
            "roc_auc": 0.9524,
            "validation_loss": 0.2761,
            "validation_accuracy": 0.8836
        },
        "computational_profile": {
            "total_parameters": 667073,
            "recurrent_parameters": 24960,
            "training_time_seconds": 119.61
        }
    }
    metadata_path = SAVED_MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"  -> Model metadata saved: {metadata_path}")
    print("\n[COMPLETE] All production deployment artifacts exported successfully.")

if __name__ == "__main__":
    export_artifacts()
