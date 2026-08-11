# promote model

import os
import sys
from pathlib import Path

# Add project root directory to sys.path (prevents ModuleNotFoundError)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mlflow

# Match constants from capstone_src (hardcoded to avoid ModuleNotFoundError in CI)
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "sentiment_classifier_model")
DAGSHUB_TOKEN_ENV = "CAPSTONE_TEST"
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "divyam-bansal64")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "mlops_capstone_project")

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv(DAGSHUB_TOKEN_ENV) or os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", DAGSHUB_REPO_OWNER)
    repo_name = os.getenv("DAGSHUB_REPO_NAME", DAGSHUB_REPO_NAME)

    if not dagshub_token:
        raise EnvironmentError(f"{DAGSHUB_TOKEN_ENV} environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

    # Configure MLflow tracking (dagshub.init preferred, bare URI fallback)
    try:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    except ImportError:
        mlflow.set_tracking_uri(f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = REGISTERED_MODEL_NAME
    # Get the latest version in staging (sorted numerically)
    latest_versions_staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_versions_staging:
        print(f"No staging model version found for '{model_name}'.")
        return
        
    latest_staging_obj = sorted(latest_versions_staging, key=lambda v: int(v.version))[-1]
    latest_version_staging = latest_staging_obj.version

    # Promote the new model to production atomically (automatically archives old production versions)
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production",
        archive_existing_versions=True
    )
    if os.getenv("GITHUB_ACTIONS"):
        run_num = os.getenv("GITHUB_RUN_NUMBER", "")
        try:
            client.set_model_version_tag(model_name, latest_version_staging, "promoted_by_cicd", f"Run #{run_num}")
        except Exception as e:
            print(f"Could not set promotion tag: {e}")
    print(f"Model version {latest_version_staging} of '{model_name}' successfully promoted to Production (previous versions archived).")

if __name__ == "__main__":
    promote_model()