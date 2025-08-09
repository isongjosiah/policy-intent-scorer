import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Represents the application settings.

    Attributes:
        S3_BUCKET (str): The name of the S3 bucket.
        REDIS_URL (str): The URL for the Redis instance.
        CONGRESS_API_KEY (str): The API key for the Congress API.
        MARKET_SECTORS (list[str]): A list of market sectors.
        INPUT_FILE (str): The input file path.
        OUTPUT_FILE (str): The output file path.
        S3_MODEL_BUCKET (str): The S3 bucket for storing models.
        LOCAL_DATA_PATH (str): The local path for data.
        LOCAL_MODEL_PATH (str): The local path for models.
        MODEL_FILE_NAME (str): The name of the model file.
        OFFLINE_MODE (bool): Whether to run in offline mode.
    """

    def __init__(self):
        self.S3_BUCKET = os.environ.get("S3_BUCKET")
        self.REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")
        self.MARKET_SECTORS = os.environ.get("MARKET_SECTORS", "SPY,XLI,XLE,XLF").split(
            ","
        )
        self.INPUT_FILE = os.environ.get(
            "INPUT_FILE", "../data/raw_archive_latest.parquet"
        )
        self.OUTPUT_FILE = os.environ.get(
            "OUTPUT_FILE", "../data/processed_archive_latest.parquet"
        )
        self.S3_MODEL_BUCKET = os.environ.get("S3_MODEL_BUCKET")
        self.LOCAL_DATA_PATH = os.environ.get(
            "LOCAL_DATA_PATH", "../data/raw_archive_latest.parquet"
        )
        self.LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "../model/")
        self.MODEL_FILE_NAME = os.environ.get(
            "MODEL_FILE_NAME",
            "text_classifier_model.pkl",
        )
        self.OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "false").lower() in (
            "true",
            "1",
            "t",
        )
        self.LOOKBACK_DAYS: int = int(os.environ.get("LOOKBACK_DAYS", "30"))


settings = Settings()

