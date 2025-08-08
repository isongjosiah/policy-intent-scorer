from dataclasses import dataclass
from io import BytesIO

import boto3
import pandas as pd

from config.config import settings


@dataclass
class S3Config:
    bucket: str
    folder: str


class S3Service:
    def __init__(self, config: S3Config):
        self.s3 = boto3.client("s3")
        self.bucket = config.bucket
        self.folder = config.folder

    def _get_s3_key(self, file_name: str) -> str:
        return f"{self.folder}/{file_name}"

    def save_df_to_s3(self, df: pd.DataFrame, file_name: str) -> None:
        key = self._get_s3_key(file_name)
        with BytesIO() as buffer:
            df.to_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())

    def load_df_from_s3(self, file_name: str) -> pd.DataFrame:
        key = self._get_s3_key(file_name)
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        with BytesIO(response["Body"].read()) as buffer:
            return pd.read_parquet(buffer)


def get_s3_config(folder: str) -> S3Config:
    return S3Config(bucket=settings.S3_BUCKET, folder=folder)
