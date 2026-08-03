import os

# AWS S3 Bucket Configuration
BUCKET_NAME = "capstone-proj-dataset-imdb"
S3_FILE_KEY = "IMDB Dataset.csv"
AWS_REGION_NAME = "us-east-1"

# Environment Variable Key Names
AWS_ACCESS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV = "AWS_SECRET_ACCESS_KEY"

# Toggle to prefer S3 over local/URL data source
USE_S3 = False

# Fallback Local Dataset File Path (used when S3 is unavailable or USE_S3 is False)
FALLBACK_DATA_PATH = os.path.join("notebooks", "data.csv")
FALLBACK_DATA_URL = FALLBACK_DATA_PATH  # Alias for backward compatibility

# Data Ingestion Defaults (used if params.yaml is not yet generated)
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Feature Engineering Defaults (used if params.yaml is not yet generated)
DEFAULT_MAX_FEATURES = 15000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_SUBLINEAR_TF = True

# Data Directories & Path Configurations
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Dataset Columns
TEXT_COLUMN_NAME = "review"
TARGET_COLUMN_NAME = "sentiment"

# Dataset File Names
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRAIN_PROCESSED_FILE_NAME = "train_processed.csv"
TEST_PROCESSED_FILE_NAME = "test_processed.csv"
TRAIN_FEATURES_FILE_NAME = "train_tfidf.csv"
TEST_FEATURES_FILE_NAME = "test_tfidf.csv"

# Models & Vectorizer Directories & File Names
MODELS_DIR = "models"
VECTORIZER_FILE_NAME = "vectorizer.pkl"
MODEL_FILE_NAME = "model.pkl"
