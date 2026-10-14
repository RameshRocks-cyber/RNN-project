"""
Data loading and dataset splitting module.
Handles importing raw datasets (e.g., IMDB reviews) and creating train/validation/test splits.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, RAW_DATA_PATH, TRAIN_SPLIT, VAL_SPLIT


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw review data from a CSV file.
    Expected schema:
      - 'review' (str): Raw text of the user review.
      - 'sentiment' (str or int): 'positive'/'negative' or 1/0.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}. "
            f"Please place your dataset CSV into the data/raw directory."
        )
    return pd.read_csv(file_path)


def split_data(
    df: pd.DataFrame,
    text_col: str = "review",
    label_col: str = "sentiment",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits DataFrame into train (80%), validation (10%), and test (10%) splits.
    Stratification ensures balanced class distribution across all sets.
    """
    X = df[text_col]
    y = df[label_col]

    # First split: Train vs Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1.0 - TRAIN_SPLIT), random_state=RANDOM_SEED, stratify=y
    )

    # Second split: Validation vs Test (50% each of the remaining 20% -> 10% each)
    val_ratio_of_temp = VAL_SPLIT / (VAL_SPLIT + 0.1)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio_of_temp), random_state=RANDOM_SEED, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
