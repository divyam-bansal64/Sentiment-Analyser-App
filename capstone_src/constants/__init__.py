import os

# AWS S3 Bucket Configuration (Replace with actual bucket details later)
BUCKET_NAME = "capstone-proj-dataset-imdb"
S3_FILE_KEY = "IMDB Dataset.csv"
AWS_REGION_NAME = "us-east-1"

# Environment Variable Key Names
AWS_ACCESS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV = "AWS_SECRET_ACCESS_KEY"

# Toggle to prefer S3 over local/URL data source
USE_S3 = False

# Fallback Local Dataset File Path (used when S3 is unavailable or USE_S3 is False)
FALLBACK_DATA_PATH = os.path.join("notebooks", "IMDB Dataset.csv")
FALLBACK_DATA_URL = FALLBACK_DATA_PATH  # Alias for backward compatibility

# Data Ingestion Defaults (used if params.yaml is not yet generated)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Data Directories & File Names
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
