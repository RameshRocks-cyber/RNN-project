"""
Data loading, cleaning, deduplication, and dataset splitting module.
Handles raw IMDB reviews, applies cleaning, converts labels, and generates stratified splits.
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, RAW_DATA_PATH, TRAIN_SPLIT, VAL_SPLIT
from src.preprocessor import clean_text, encode_labels


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw review data from a CSV file.
    Expected schema:
      - 'review' (str): Raw text of the user review.
      - 'sentiment' (str): 'positive' or 'negative'.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}. "
            f"Please download the dataset into data/raw/ before proceeding."
        )
    return pd.read_csv(file_path)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs full dataset-level preprocessing:
    1. Removes duplicate reviews (prevents train/test data leakage).
    2. Drops rows with missing review or sentiment values.
    3. Cleans raw text into 'clean_review'.
    4. Converts categorical sentiment strings to integer 'label' (0 or 1).
    """
    initial_rows = len(df)
    
    # 1. Drop duplicates by review content
    df = df.drop_duplicates(subset=["review"]).copy()
    dropped_dups = initial_rows - len(df)
    if dropped_dups > 0:
        print(f"[Data Cleaning] Removed {dropped_dups} duplicate review(s).")

    # 2. Drop any null entries
    df = df.dropna(subset=["review", "sentiment"]).copy()

    # 3. Apply text cleaning
    print(f"[Data Cleaning] Cleaning {len(df)} text reviews...")
    df["clean_review"] = df["review"].apply(clean_text)

    # 4. Filter out any reviews that became empty strings after cleaning
    df = df[df["clean_review"].str.len() > 0].copy()

    # 5. Convert labels: positive -> 1, negative -> 0
    df["label"] = encode_labels(df["sentiment"])

    return df


def split_data(
    df: pd.DataFrame,
    text_col: str = "clean_review",
    label_col: str = "label",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits DataFrame into train (80%), validation (10%), and test (10%) splits.
    Stratification ensures equal class balance across all subsets.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[text_col]
    y = df[label_col]

    # First split: Train (80%) vs Temp (20% for Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - TRAIN_SPLIT),
        random_state=RANDOM_SEED,
        stratify=y,
    )

    # Second split: Split remaining 20% into half (10% Val, 10% Test)
    val_ratio_of_temp = VAL_SPLIT / (VAL_SPLIT + (1.0 - TRAIN_SPLIT - VAL_SPLIT))
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio_of_temp),
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
