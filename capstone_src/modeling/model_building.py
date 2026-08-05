# model building
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
from scipy import sparse
from sklearn.svm import LinearSVC

from capstone_src.logger import logging
from capstone_src import constants


def load_params(params_path: str) -> dict:
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
            'model_building': {
                'C': constants.DEFAULT_C,
                'max_iter': constants.DEFAULT_MAX_ITER
            }
        }
    except yaml.YAMLError as e:
        logging.error('YAML error while reading %s: %s', params_path, e)
        raise
    except Exception as e:
        logging.error('Unexpected error while reading parameters: %s', e)
        raise


def load_processed_data(features_path: str, labels_path: str):
    """Loads sparse TF-IDF matrix (.npz) and sentiment labels (.npy)."""
    try:
        X_train = sparse.load_npz(features_path)
        y_train = np.load(labels_path)
        logging.info('Sparse train features loaded from %s (shape: %s)', features_path, X_train.shape)
        logging.info('Train labels loaded from %s (shape: %s)', labels_path, y_train.shape)
        return X_train, y_train
    except Exception as e:
        logging.error('Error loading processed sparse train features/labels: %s', e)
        raise


def train_model(X_train, y_train: np.ndarray, C: float, max_iter: int) -> LinearSVC:
    """Train the winning LinearSVC model using specified C and max_iter parameters."""
    try:
        logging.info("Training LinearSVC model (C=%.4f, max_iter=%d)...", C, max_iter)
        clf = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=constants.DEFAULT_RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        logging.info("LinearSVC model training completed successfully.")
        return clf
    except Exception as e:
        logging.error("Error during LinearSVC model training: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model artifact to a pickle file safely."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info("Trained model artifact saved successfully to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the model artifact: %s", e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        mb_params = params.get('model_building', {})
        C = mb_params.get('C', constants.DEFAULT_C)
        max_iter = mb_params.get('max_iter', constants.DEFAULT_MAX_ITER)

        processed_dir = constants.PROCESSED_DATA_DIR
        train_features_path = os.path.join(processed_dir, constants.TRAIN_FEATURES_FILE_NAME)
        train_labels_path = os.path.join(processed_dir, constants.TRAIN_LABELS_FILE_NAME)

        X_train, y_train = load_processed_data(train_features_path, train_labels_path)

        clf = train_model(X_train, y_train, C=C, max_iter=max_iter)

        models_dir = constants.MODELS_DIR
        model_out_path = os.path.join(models_dir, constants.MODEL_FILE_NAME)

        save_model(clf, model_out_path)
        logging.info("Model Building stage execution completed successfully.")
    except Exception as e:
        logging.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()