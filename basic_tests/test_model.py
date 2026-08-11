# load test + signature test + performance test

import os
import sys
from pathlib import Path

# Add project root directory to sys.path (prevents ModuleNotFoundError)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import unittest
import mlflow
import pickle
import numpy as np
import scipy.sparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Match constants from capstone_src (hardcoded to avoid ModuleNotFoundError in CI)
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "sentiment_classifier_model")
DAGSHUB_TOKEN_ENV = "CAPSTONE_TEST"
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "divyam-bansal64")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "mlops_capstone_project")


def load_model_from_registry(client, model_name, stage):
    """
    Download model.pkl artifact from MLflow registry for a given stage.
    Uses mlflow.artifacts.download_artifacts() + pickle.load() because
    models are logged as raw artifacts (not mlflow.sklearn.log_model).
    """
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        return None, None
    mv = sorted(versions, key=lambda v: int(v.version))[-1]
    artifact_uri = f"{mv.source}/model.pkl"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    with open(local_path, 'rb') as f:
        model = pickle.load(f)
    return model, mv.version


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv(DAGSHUB_TOKEN_ENV) or os.getenv("DAGSHUB_TOKEN")
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER", DAGSHUB_REPO_OWNER)
        repo_name = os.getenv("DAGSHUB_REPO_NAME", DAGSHUB_REPO_NAME)

        if not dagshub_token:
            raise EnvironmentError(f"{DAGSHUB_TOKEN_ENV} environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Configure MLflow tracking (dagshub.init preferred, bare URI fallback)
        try:
            import dagshub
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        except ImportError:
            mlflow.set_tracking_uri(f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow')

        cls.client = mlflow.MlflowClient()
        cls.new_model_name = REGISTERED_MODEL_NAME

        # Load the new candidate model from MLflow registry (Staging) via artifact download
        cls.new_model, cls.new_model_version = load_model_from_registry(
            cls.client, cls.new_model_name, "Staging"
        )

        # Load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout processed test features and labels
        cls.X_holdout = scipy.sparse.load_npz('data/processed/test_tfidf.npz')
        cls.y_holdout = np.load('data/processed/test_labels.npy')

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_data)

        # Verify output length
        self.assertEqual(len(prediction), 1)

    def test_model_performance(self):
        # Predict using the new candidate model on holdout test features
        y_pred_new = self.new_model.predict(self.X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(self.y_holdout, y_pred_new)
        precision_new = precision_score(self.y_holdout, y_pred_new, zero_division=0)
        recall_new = recall_score(self.y_holdout, y_pred_new, zero_division=0)
        f1_new = f1_score(self.y_holdout, y_pred_new, zero_division=0)

        # If a Production model currently exists in MLflow, evaluate candidate against Production (Champion vs Challenger)
        prod_model, prod_version = load_model_from_registry(
            self.client, self.new_model_name, "Production"
        )
        if prod_model:
            y_pred_prod = prod_model.predict(self.X_holdout)

            expected_accuracy = accuracy_score(self.y_holdout, y_pred_prod)
            expected_precision = precision_score(self.y_holdout, y_pred_prod, zero_division=0)
            expected_recall = recall_score(self.y_holdout, y_pred_prod, zero_division=0)
            expected_f1 = f1_score(self.y_holdout, y_pred_prod, zero_division=0)
            print(f"Comparing Candidate (Staging v{self.new_model_version}) against Production (v{prod_version}): F1 Candidate={f1_new:.4f} vs Prod={expected_f1:.4f}")
        else:
            # Baseline threshold fallback if no Production model exists yet
            expected_accuracy = 0.40
            expected_precision = 0.40
            expected_recall = 0.40
            expected_f1 = 0.40
            print(f"No Production model found. Evaluating Candidate (v{self.new_model_version}) against baseline threshold (0.40). Candidate F1={f1_new:.4f}")

        # Assert that candidate model meets or beats expectations
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy ({accuracy_new:.4f}) should be >= {expected_accuracy:.4f}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision ({precision_new:.4f}) should be >= {expected_precision:.4f}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall ({recall_new:.4f}) should be >= {expected_recall:.4f}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score ({f1_new:.4f}) should be >= {expected_f1:.4f}')

if __name__ == "__main__":
    unittest.main()