# data preprocessing
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from capstone_src.logger import logging
from capstone_src import constants

# Download NLTK datasets quietly if needed
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Pre-instantiate stop words set and lemmatizer ONCE at module load for speed
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def lower_case(text: str) -> str:
    """Converts text to lowercase."""
    return text.lower() if isinstance(text, str) else ""


def remove_html_tags(text: str) -> str:
    """Strips HTML tags like <br /> across single and multiline strings."""
    return re.sub(r'<[^>]+>', ' ', text)


def removing_urls(text: str) -> str:
    """Removes http/https URLs and www domain links."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def removing_punctuations(text: str) -> str:
    """Removes punctuation characters, replacing them with whitespace."""
    text = text.replace('؛', '')
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)


def remove_stop_words(text: str) -> str:
    """Filters out English stop words."""
    return " ".join([word for word in text.split() if word not in STOP_WORDS])


def lemmatization(text: str) -> str:
    """Lemmatizes each word in the text to its dictionary base form."""
    return " ".join([LEMMATIZER.lemmatize(word) for word in text.split()])


def preprocess_text(text: str) -> str:
    """
    Applies the full text normalization pipeline while retaining numbers
    for sentiment rating signals (e.g., 10/10, 1 star).
    """
    text = lower_case(text)
    text = remove_html_tags(text)
    text = removing_urls(text)
    text = removing_punctuations(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text


def preprocess_dataframe(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if col is None:
        col = constants.TEXT_COLUMN_NAME

    logging.info("Starting text preprocessing on column: '%s'", col)
    df[col] = df[col].apply(preprocess_text)

    # Drop rows with NaN or empty values after cleaning
    df = df.dropna(subset=[col])
    logging.info("Text preprocessing completed for %d records.", len(df))
    return df


def main():
    try:
        raw_dir = constants.RAW_DATA_DIR
        train_raw_path = os.path.join(raw_dir, constants.TRAIN_FILE_NAME)
        test_raw_path = os.path.join(raw_dir, constants.TEST_FILE_NAME)

        logging.info("Loading raw datasets from %s...", raw_dir)
        train_data = pd.read_csv(train_raw_path)
        test_data = pd.read_csv(test_raw_path)
        logging.info("Raw datasets loaded successfully (%d train, %d test records).", len(train_data), len(test_data))

        # Apply preprocessing
        train_processed = preprocess_dataframe(train_data, col=constants.TEXT_COLUMN_NAME)
        test_processed = preprocess_dataframe(test_data, col=constants.TEXT_COLUMN_NAME)

        # Store preprocessed data in data/interim
        interim_dir = constants.INTERIM_DATA_DIR
        os.makedirs(interim_dir, exist_ok=True)

        train_out_path = os.path.join(interim_dir, constants.TRAIN_PROCESSED_FILE_NAME)
        test_out_path = os.path.join(interim_dir, constants.TEST_PROCESSED_FILE_NAME)

        train_processed.to_csv(train_out_path, index=False)
        test_processed.to_csv(test_out_path, index=False)

        logging.info("Processed train data saved to %s", train_out_path)
        logging.info("Processed test data saved to %s", test_out_path)
        logging.info("Data Preprocessing stage execution completed successfully.")
    except Exception as e:
        logging.error("Failed to complete the data preprocessing process: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()