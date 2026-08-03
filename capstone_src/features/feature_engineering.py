# feature engineering
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from capstone_src.logger import logging
from capstone_src import constants


def load_params(params_path: str = 'params.yaml') -> dict:
    """
    Load parameters from a YAML file.
    Falls back to default constants if params.yaml is missing or invalid.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.warning("⚠️ '%s' not found. Falling back to default constants.", params_path)
        return {
            'feature_engineering': {
                'max_features': constants.DEFAULT_MAX_FEATURES,
                'ngram_range': list(constants.DEFAULT_NGRAM_RANGE),
                'sublinear_tf': constants.DEFAULT_SUBLINEAR_TF
            }
        }
    except yaml.YAMLError as e:
        logging.error('YAML error while reading %s: %s', params_path, e)
        raise
    except Exception as e:
        logging.error('Unexpected error while reading parameters: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
    ngram_range: list,
    sublinear_tf: bool
) -> tuple:
    """Apply TF-IDF Vectorizer to the text data using winning parameters."""
    try:
        logging.info(
            "Applying TF-IDF Vectorizer (max_features=%d, ngram_range=%s, sublinear_tf=%s)...",
            max_features, ngram_range, sublinear_tf
        )
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            sublinear_tf=sublinear_tf
        )

        text_col = constants.TEXT_COLUMN_NAME
        target_col = constants.TARGET_COLUMN_NAME

        X_train = train_data[text_col].astype(str).values
        y_train = train_data[target_col].values
        X_test = test_data[text_col].astype(str).values
        y_test = test_data[target_col].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        # Save fitted vectorizer artifact
        models_dir = constants.MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        vectorizer_path = os.path.join(models_dir, constants.VECTORIZER_FILE_NAME)

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        logging.info('TF-IDF Vectorizer applied successfully. Saved vectorizer artifact to %s', vectorizer_path)
        return train_df, test_df
    except Exception as e:
        logging.error('Error during TF-IDF vectorization: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        fe_params = params.get('feature_engineering', {})

        max_features = fe_params.get('max_features', constants.DEFAULT_MAX_FEATURES)
        ngram_range = fe_params.get('ngram_range', list(constants.DEFAULT_NGRAM_RANGE))
        sublinear_tf = fe_params.get('sublinear_tf', constants.DEFAULT_SUBLINEAR_TF)

        interim_dir = constants.INTERIM_DATA_DIR
        train_interim_path = os.path.join(interim_dir, constants.TRAIN_PROCESSED_FILE_NAME)
        test_interim_path = os.path.join(interim_dir, constants.TEST_PROCESSED_FILE_NAME)

        train_data = load_data(train_interim_path)
        test_data = load_data(test_interim_path)

        train_df, test_df = apply_tfidf(train_data, test_data, max_features, ngram_range, sublinear_tf)

        processed_dir = constants.PROCESSED_DATA_DIR
        train_out_path = os.path.join(processed_dir, constants.TRAIN_FEATURES_FILE_NAME)
        test_out_path = os.path.join(processed_dir, constants.TEST_FEATURES_FILE_NAME)

        save_data(train_df, train_out_path)
        save_data(test_df, test_out_path)

        logging.info("Feature Engineering stage execution completed successfully.")
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()