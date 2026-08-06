import os
import pickle
import mlflow

# Match constants from capstone_src
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "sentiment_classifier_model")
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "divyam-bansal64")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "mlops_capstone_project")


def load_vectorizer(vectorizer_path="models/vectorizer.pkl"):
    """
    Load the saved TF-IDF vectorizer artifact.
    Checks relative paths for both app root and subfolder execution.
    """
    candidate_paths = [
        vectorizer_path,
        os.path.join("..", vectorizer_path),
        os.path.join(os.path.dirname(__file__), "..", vectorizer_path)
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    vectorizer = pickle.load(f)
                print(f"✅ Vectorizer loaded successfully from {path}")
                return vectorizer
            except Exception as e:
                print(f"⚠️ Error loading vectorizer from {path}: {e}")

    raise FileNotFoundError(f"❌ Vectorizer file not found in any candidate locations: {candidate_paths}")


def setup_mlflow():
    """Configure MLflow tracking URI and DagsHub remote artifact repository."""
    dagshub_token = os.getenv("CAPSTONE_TEST") or os.getenv("DAGSHUB_TOKEN")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", DAGSHUB_REPO_OWNER)
    repo_name = os.getenv("DAGSHUB_REPO_NAME", DAGSHUB_REPO_NAME)

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        try:
            import dagshub
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            print("✅ MLflow & DagsHub remote artifact repository configured via dagshub.init().")
        except ImportError:
            tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            print(f"✅ MLflow Tracking URI set to: {tracking_uri}")
    else:
        print("⚠️ Warning: CAPSTONE_TEST environment variable is not set. Using fallback tracking URI.")
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)


def load_registered_model(model_name=REGISTERED_MODEL_NAME):
    """
    Fetch and load latest model version from MLflow Model Registry.
    Downloads model.pkl artifact via mlflow.artifacts.download_artifacts().
    Tries Production stage -> Staging stage -> Latest Version -> Local Fallback.
    """
    setup_mlflow()
    client = mlflow.MlflowClient()

    # Priority 1: Check Production and Staging stages
    stages_to_try = ["Production", "Staging"]
    for stage in stages_to_try:
        try:
            print(f"Attempting to load model from MLflow stage: '{stage}'...")
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                print(f"ℹ️ No model version found in stage '{stage}'. Trying next...")
                continue

            # Target the highest version number in this stage
            mv = sorted(versions, key=lambda v: int(v.version))[-1]
            artifact_uri = f"{mv.source}/model.pkl"
            print(f"  Downloading artifact from: {artifact_uri}")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
            with open(local_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded model successfully from MLflow stage: '{stage}' (version {mv.version}).")
            return model
        except Exception as e:
            print(f"ℹ️ Failed to load from stage '{stage}' ({e}). Trying next...")

    # Priority 2: Try fetching any latest version from registry
    try:
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            mv = sorted(latest_versions, key=lambda v: int(v.version))[-1]
            artifact_uri = f"{mv.source}/model.pkl"
            print(f"  Downloading artifact from: {artifact_uri}")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
            with open(local_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded model version {mv.version} from MLflow.")
            return model
    except Exception as inner_e:
        print(f"⚠️ MLflow registry lookup failed: {inner_e}")

    # Priority 3: Local fallback
    local_model_paths = ["models/model.pkl", "../models/model.pkl"]
    for path in local_model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Loaded local model fallback from {path}")
            return model

    raise RuntimeError("❌ Could not load model from MLflow Registry or local fallback.")

