# register model
import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import warnings
import mlflow
from mlflow.tracking import MlflowClient

from capstone_src.logger import logging
from capstone_src import constants

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def setup_mlflow_tracking():
    """
    Configures MLflow tracking URI.
    1. Prefer environment variables if explicitly set.
    2. Try dagshub.init() using cached local DagsHub credentials.
    3. Fall back to local directory tracking (mlruns/) if offline.
    """
    dagshub_token = os.getenv(constants.DAGSHUB_TOKEN_ENV) or os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", constants.DAGSHUB_REPO_OWNER)
    repo_name = os.getenv("DAGSHUB_REPO_NAME", constants.DAGSHUB_REPO_NAME)

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token
        try:
            import dagshub
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logging.info("MLflow & DagsHub remote artifact storage configured via token.")
        except ImportError:
            tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            logging.info("dagshub package not installed; using tracking URI only: %s", tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        logging.info("MLflow configured with custom URI: %s", os.getenv("MLFLOW_TRACKING_URI"))
    else:
        try:
            import dagshub
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            logging.info("MLflow configured via dagshub.init() using cached credentials.")
        except Exception as e:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            local_mlruns = os.path.join(ROOT_DIR, "mlruns")
            mlflow.set_tracking_uri(f"file:///{local_mlruns.replace(os.sep, '/')}")
            logging.info("MLflow configured with local directory tracking fallback (%s): %s", e, local_mlruns)


def load_model_info(file_path: str) -> dict:
    """Load the model run info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded successfully from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('Model info JSON file not found at: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """
    Registers the logged model artifact directly to the MLflow Model Registry
    and transitions its stage to 'Staging'.
    """
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']

        client = MlflowClient()

        # Ensure registered model entry exists
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass  # Model entry already exists

        # Fetch run and construct remote MLflow artifact URI (prevents local C: drive leak)
        run = client.get_run(run_id)
        raw_uri = run.info.artifact_uri.replace("\\", "/")
        if raw_uri.startswith(("c:", "C:", "file:")) or not raw_uri.startswith("mlflow-artifacts:"):
            source_uri = f"mlflow-artifacts:/{run_id}/artifacts/{model_path}"
        else:
            source_uri = f"{raw_uri}/{model_path}"

        logging.info("Registering model '%s' from source URI: %s...", model_name, source_uri)

        model_version = client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=run_id
        )
        logging.info("Registered model '%s' version %s successfully.", model_name, model_version.version)

        # Set CI/CD tags on the registered model version if running in GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            run_num = os.getenv("GITHUB_RUN_NUMBER", "")
            ci_tags = {
                "cicd_run": f"Run #{run_num}" if run_num else "CI/CD Run",
                "cicd_run_number": run_num,
                "ci.platform": "GitHub Actions",
                "ci.run_id": os.getenv("GITHUB_RUN_ID", ""),
                "ci.commit_sha": os.getenv("GITHUB_SHA", ""),
                "ci.branch": os.getenv("GITHUB_REF_NAME", ""),
                "ci.actor": os.getenv("GITHUB_ACTOR", ""),
            }
            for tag_key, tag_val in ci_tags.items():
                try:
                    client.set_model_version_tag(model_name, model_version.version, tag_key, tag_val)
                except Exception as tag_err:
                    logging.warning("Could not set model version tag %s: %s", tag_key, tag_err)

        # Transition to Staging stage (with alias fallback)
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=True
            )
            logging.info("Model '%s' version %s transitioned to stage 'Staging'.", model_name, model_version.version)
        except Exception as stage_err:
            logging.warning("Could not transition stage directly (%s). Setting alias 'Staging'...", stage_err)
            try:
                client.set_registered_model_alias(name=model_name, alias="Staging", version=model_version.version)
                logging.info("Model '%s' version %s assigned alias 'Staging'.", model_name, model_version.version)
            except Exception as alias_err:
                logging.error("Failed to set model alias: %s", alias_err)

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        setup_mlflow_tracking()

        reports_dir = constants.REPORTS_DIR
        model_info_path = os.path.join(reports_dir, constants.EXPERIMENT_INFO_FILE_NAME)
        model_info = load_model_info(model_info_path)

        model_name = constants.REGISTERED_MODEL_NAME
        register_model(model_name, model_info)

        logging.info("Model Registration stage execution completed successfully.")
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()