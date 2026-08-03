# model evaluation
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import mlflow
import mlflow.sklearn

from capstone_src.logger import logging
from capstone_src import constants


def setup_mlflow_tracking():
    """
    Configures MLflow tracking URI.
    Uses remote DagsHub URI if environment variables are available,
    otherwise falls back to local tracking directory.
    """
    # Allow local file-store tracking in newer MLflow versions
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    dagshub_token = os.getenv(constants.DAGSHUB_TOKEN_ENV) or os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", constants.DAGSHUB_REPO_OWNER)
    repo_name = os.getenv("DAGSHUB_REPO_NAME", constants.DAGSHUB_REPO_NAME)

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        logging.info("MLflow configured with remote DagsHub URI: %s", tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        logging.info("MLflow configured with custom URI: %s", os.getenv("MLFLOW_TRACKING_URI"))
    else:
        local_mlruns = os.path.join(ROOT_DIR, "mlruns")
        mlflow.set_tracking_uri(f"file:///{local_mlruns.replace(os.sep, '/')}")
        logging.info("MLflow configured with local directory tracking: %s", local_mlruns)


def load_model(file_path: str):
    """Load the trained model artifact from a pickle file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded successfully from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('Model file not found at: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load evaluation feature data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded successfully from %s with %d records.', file_path, len(df))
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file from %s: %s', file_path, e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading data from %s: %s', file_path, e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model and return accuracy, precision, recall, f1_score, and AUC metrics.
    Handles decision_function for LinearSVC and predict_proba for probabilistic models.
    """
    try:
        y_pred = clf.predict(X_test)

        # Get continuous prediction scores for ROC-AUC
        if hasattr(clf, "predict_proba"):
            y_scores = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_scores = clf.decision_function(X_test)
        else:
            y_scores = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_scores)

        metrics_dict = {
            'accuracy': float(round(accuracy, 4)),
            'precision': float(round(precision, 4)),
            'recall': float(round(recall, 4)),
            'f1_score': float(round(f1, 4)),
            'auc': float(round(auc, 4))
        }
        logging.info('Model evaluation metrics calculated successfully: %s', metrics_dict)
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved successfully to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and artifact path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info('Model info saved successfully to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    try:
        setup_mlflow_tracking()
        mlflow.set_experiment(constants.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run() as run:
            models_dir = constants.MODELS_DIR
            model_path = os.path.join(models_dir, constants.MODEL_FILE_NAME)
            clf = load_model(model_path)

            processed_dir = constants.PROCESSED_DATA_DIR
            test_features_path = os.path.join(processed_dir, constants.TEST_FEATURES_FILE_NAME)
            test_data = load_data(test_features_path)

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)

            reports_dir = constants.REPORTS_DIR
            metrics_path = os.path.join(reports_dir, constants.METRICS_FILE_NAME)
            save_metrics(metrics, metrics_path)

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model hyperparameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log model to MLflow
            mlflow.sklearn.log_model(clf, name="model")

            # Save model experiment info for registration stage
            experiment_info_path = os.path.join(reports_dir, constants.EXPERIMENT_INFO_FILE_NAME)
            save_model_info(run.info.run_id, "model", experiment_info_path)

            # Log metrics JSON as an artifact in MLflow
            mlflow.log_artifact(metrics_path)

            logging.info("Model Evaluation stage execution completed successfully.")
    except Exception as e:
        logging.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()