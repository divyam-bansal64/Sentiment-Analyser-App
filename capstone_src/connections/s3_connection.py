import os
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import boto3
import pandas as pd
from io import StringIO
from capstone_src.logger import logging
from capstone_src import constants


class s3_operations:
    def __init__(self):
        """
        Initialize S3 operations. Automatically fetches bucket name, region, 
        and AWS credentials from constants and environment variables.
        """
        self.bucket_name = constants.BUCKET_NAME
        self.region_name = constants.AWS_REGION_NAME

        # Fetch credentials securely from environment variables
        self.aws_access_key = os.getenv(constants.AWS_ACCESS_KEY_ID_ENV)
        self.aws_secret_key = os.getenv(constants.AWS_SECRET_ACCESS_KEY_ENV)

        # Guard Clause 1: If credentials are missing, log warning and exit early
        if not self.aws_access_key or not self.aws_secret_key:
            logging.warning("⚠️ AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) not found in environment variables.")
            self.s3_client = None
            return

        # Attempt to create S3 client (Flat, no nested else!)
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region_name
            )
            logging.info(f"S3 Connection initialized for bucket: '{self.bucket_name}' in region: '{self.region_name}'")
        except Exception as e:
            logging.error(f"❌ Failed to initialize S3 Client: {e}")
            self.s3_client = None

    def fetch_file_from_s3(self, file_key: str = None) -> pd.DataFrame:
        """
        Fetches a CSV file from S3 bucket and returns it as a Pandas DataFrame.
        Defaults to constants.S3_FILE_KEY if file_key is not specified.
        """
        file_key = file_key or constants.S3_FILE_KEY

        # Guard Clause 2: If S3 client failed to initialize, exit early
        if self.s3_client is None:
            logging.error("❌ S3 Client is not initialized. Unable to fetch file from S3.")
            return None

        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched '{file_key}' from S3 with {len(df)} records.")
            return df
        except Exception as e:
            logging.error(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            return None