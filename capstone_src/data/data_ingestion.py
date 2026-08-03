import os
import sys
from pathlib import Path

# Add project root directory to sys.path so capstone_src can be imported when executing directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)

from capstone_src.logger import logging
from capstone_src.connections import s3_connection
from capstone_src import constants


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.warning("⚠️ '%s' not found. Falling back to default constants.", params_path)
        return {
            'data_ingestion': {
                'test_size': constants.DEFAULT_TEST_SIZE,
                'random_state': constants.DEFAULT_RANDOM_STATE
            }
        }
    except yaml.YAMLError as e:
        logging.error('YAML parsing error in %s: %s', params_path, e)
        raise
    except Exception as e:
        logging.error('Unexpected error while reading parameters: %s', e)
        raise


def load_data(data_path: str = None, use_s3: bool = None) -> pd.DataFrame:
    """
    Load data from S3 bucket cleanly with automatic fallback to local dataset.
    """
    if use_s3 is None:
        use_s3 = constants.USE_S3

    fallback_path = data_path or constants.FALLBACK_DATA_PATH

    if use_s3:
        logging.info("Attempting data ingestion from S3 bucket...")
        try:
            s3 = s3_connection.s3_operations()
            df = s3.fetch_file_from_s3()
            if df is not None and not df.empty:
                logging.info("✅ Successfully ingested data from S3 bucket.")
                return df
            else:
                logging.warning("⚠️ S3 fetch returned empty/None. Initiating fallback to local dataset...")
        except Exception as e:
            logging.warning("⚠️ S3 Ingestion failed (%s). Initiating fallback to local dataset...", e)

    # Local Fallback
    logging.info("Loading data from local path: %s", fallback_path)
    try:
        df = pd.read_csv(fallback_path)
        logging.info("✅ Data loaded successfully from local path with %d records.", len(df))
        return df
    except Exception as e:
        logging.error("Unexpected error occurred while loading data from %s: %s", fallback_path, e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw dataframe (filters sentiments and encodes target values)."""
    try:
        logging.info("Starting data preprocessing...")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info("Data preprocessing completed successfully.")
        return final_df
    except KeyError as e:
        logging.error("Missing required column in dataframe: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str = None) -> None:
    """Save train and test data split into the target directory."""
    try:
        if data_path is None:
            data_path = constants.DATA_DIR

        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_file_path = os.path.join(raw_data_path, constants.TRAIN_FILE_NAME)
        test_file_path = os.path.join(raw_data_path, constants.TEST_FILE_NAME)

        train_data.to_csv(train_file_path, index=False)
        test_data.to_csv(test_file_path, index=False)
        logging.info("Train data saved to %s", train_file_path)
        logging.info("Test data saved to %s", test_file_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving datasets: %s", e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        ingestion_params = params.get('data_ingestion', {})
        test_size = ingestion_params.get('test_size', constants.DEFAULT_TEST_SIZE)
        random_state = ingestion_params.get('random_state', constants.DEFAULT_RANDOM_STATE)

        df = load_data()
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=random_state
        )
        save_data(train_data, test_data, data_path=constants.DATA_DIR)
        logging.info("Data Ingestion pipeline execution completed successfully.")
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()