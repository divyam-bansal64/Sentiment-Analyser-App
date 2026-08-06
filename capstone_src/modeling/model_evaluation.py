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
from scipy import sparse
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
    dagshub_token = os.getenv(constants.DAGSHUB_TOKEN_ENV) or os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", constants.DAGSHUB_REPO_OWNER)
    repo_name = os.getenv("DAGSHUB_REPO_NAME", constants.DAGSHUB_REPO_NAME)

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        try:
            import dagshub
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logging.info("MLflow & DagsHub remote artifact storage configured.")
        except ImportError:
            tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            logging.info("dagshub package not installed; using tracking URI only: %s", tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        logging.info("MLflow configured with custom URI: %s", os.getenv("MLFLOW_TRACKING_URI"))
    else:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
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


def load_processed_test_data(features_path: str, labels_path: str):
    """Loads sparse TF-IDF test matrix (.npz) and test sentiment labels (.npy)."""
    try:
        X_test = sparse.load_npz(features_path)
        y_test = np.load(labels_path)
        logging.info('Sparse test features loaded from %s (shape: %s)', features_path, X_test.shape)
        logging.info('Test labels loaded from %s (shape: %s)', labels_path, y_test.shape)
        return X_test, y_test
    except Exception as e:
        logging.error('Error loading processed sparse test features/labels: %s', e)
        raise


def evaluate_model(clf, X_test, y_test: np.ndarray) -> dict:
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
            test_labels_path = os.path.join(processed_dir, constants.TEST_LABELS_FILE_NAME)

            X_test, y_test = load_processed_test_data(test_features_path, test_labels_path)

            metrics = evaluate_model(clf, X_test, y_test)

            reports_dir = constants.REPORTS_DIR
            metrics_path = os.path.join(reports_dir, constants.METRICS_FILE_NAME)
            save_metrics(metrics, metrics_path)

            # Log all evaluation metrics (accuracy, precision, recall, f1_score, auc) in one batch payload
            mlflow.log_metrics(metrics)
            logging.info("Batch logged evaluation metrics to MLflow: %s", metrics)

            # Log model hyperparameters cleanly to MLflow
            if hasattr(clf, 'get_params'):
                raw_params = clf.get_params()
                clean_params = {k: str(v) for k, v in raw_params.items() if v is not None}
                # Log top hyperparameters
                mlflow.log_params(clean_params)
                logging.info("Batch logged hyperparameters to MLflow.")

            # Log model pickle directly as single-file artifact to DagsHub
            mlflow.log_artifact(model_path, artifact_path="model")

            # Save model experiment info for registration stage
            experiment_info_path = os.path.join(reports_dir, constants.EXPERIMENT_INFO_FILE_NAME)
            save_model_info(run.info.run_id, "model", experiment_info_path)

            logging.info("Model Evaluation stage execution completed successfully.")
    except Exception as e:
        logging.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()