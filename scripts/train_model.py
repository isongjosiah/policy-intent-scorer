import os
import pandas as pd
import boto3
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any


class DataLoader(ABC):
    """Abstract base class for loading training data."""

    @abstractmethod
    def load_data(self, source: str) -> pd.DataFrame:
        """Loads data from a specified source."""
        pass


class ModelSaver(ABC):
    """Abstract base class for saving trained models."""

    @abstractmethod
    def save_model(self, model: Any, destination: str, filename: str):
        """Saves a model to a specified destination."""
        pass


class ModelTrainer(ABC):
    """Abstract base class for training a machine learning model."""

    @abstractmethod
    def train_and_validate(self, data: pd.DataFrame) -> Any:
        """Trains a model and returns the trained model."""
        pass


class S3ParquetDataLoader(DataLoader):
    """Loads Parquet data from S3, assuming a partitioned structure."""

    def __init__(self, s3_client: Any):
        self.s3_client = s3_client

    def load_data(self, s3_bucket: str) -> pd.DataFrame:
        print(f"Loading labeled data from s3://{s3_bucket}/processed/...")
        # For this example, we'll use a dummy DataFrame.
        data = {
            "headline": [
                "President Announces New Climate Initiative",
                "Presidential Message on Space Exploration Day",
                "Joint Statement on Framework for United States-Indonesia Agreement on Reciprocal Trade",
                "First Lady Melania Trump Visits Flood-Ravaged Texas",
                "New Policy on Renewable Energy Subsidies",
                "Statement on International Trade Negotiations",
                "Remarks on Economic Growth and Job Creation",
                "Delegation to Attend Global Climate Summit",
            ],
            "body": [
                "The President is proud to announce a bold new climate initiative...",
                "Today we celebrate Space Exploration Day...",
                "The United States and Indonesia have signed a new trade framework...",
                "The First Lady traveled to Texas to assist with flood relief efforts...",
                "The administration is introducing new subsidies for solar and wind power.",
                "Discussions on new trade agreements are progressing well.",
                "The latest jobs report shows significant growth across all sectors.",
                "A delegation will represent the nation at the upcoming climate talks.",
            ],
            "label_t180": [
                "Actionable",
                "Bluff",
                "Actionable",
                "Bluff",
                "Actionable",
                "Bluff",
                "Actionable",
                "Bluff",
            ],
            "published_date": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-04",
                "2025-01-05",
                "2025-01-06",
                "2025-01-07",
                "2025-01-08",
            ],
        }
        df = pd.DataFrame(data)
        return df


class S3ModelSaver(ModelSaver):
    """Saves a pickled model to an S3 bucket."""

    def __init__(self, s3_client: Any):
        self.s3_client = s3_client

    def save_model(self, model: Any, s3_bucket: str, filename: str):
        temp_model_path = "/tmp/" + filename
        with open(temp_model_path, "wb") as f:
            pickle.dump(model, f)

        self.s3_client.upload_file(temp_model_path, s3_bucket, filename)
        print(f"Model saved to s3://{s3_bucket}/{filename}")


class SklearnTextClassifierTrainer(ModelTrainer):
    """
    Trains a TF-IDF + Logistic Regression model for text classification.
    """

    def __init__(self, target_roc_auc: float = 0.7):
        self.target_roc_auc = target_roc_auc
        self.model_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
                ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
            ]
        )

    def train_and_validate(self, df: pd.DataFrame) -> Any:
        if len(df) < 2:
            print("Error: Not enough data to train the model. Need at least 2 samples.")
            return None

        df["text"] = df["headline"] + " " + df["body"]
        X = df["text"]
        y = df["label_t180"].apply(lambda x: 1 if x == "Actionable" else 0)

        if len(y.unique()) < 2:
            print(
                "Error: Only one class found in labels. Cannot stratify split. Training without stratification."
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

        print(
            f"Training on {len(X_train)} samples, validating on {len(X_val)} samples."
        )

        print("Training model...")
        self.model_pipeline.fit(X_train, y_train)
        print("Model training complete.")

        y_pred_proba = self.model_pipeline.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        print(f"Validation ROC-AUC score: {roc_auc:.4f}")

        if roc_auc > self.target_roc_auc:
            print(f"Model performance meets the target (>{self.target_roc_auc}).")
        else:
            print(
                f"Model performance is below the target (>{self.target_roc_auc}). Further tuning may be required."
            )

        return self.model_pipeline


if __name__ == "__main__":
    S3_PROCESSED_BUCKET = os.environ.get("S3_PROCESSED_BUCKET")
    S3_MODEL_BUCKET = os.environ.get("S3_MODEL_BUCKET")
    MODEL_FILE_NAME = "model.pkl"

    if not S3_PROCESSED_BUCKET or not S3_MODEL_BUCKET:
        print(
            "Required S3 environment variables are not set. Please set S3_PROCESSED_BUCKET and S3_MODEL_BUCKET. Exiting."
        )
    else:
        s3_client = boto3.client("s3")

        # Dependency Injection
        data_loader = S3ParquetDataLoader(s3_client=s3_client)
        model_trainer = SklearnTextClassifierTrainer(target_roc_auc=0.7)
        model_saver = S3ModelSaver(s3_client=s3_client)

        # Load data
        df_labeled = data_loader.load_data(s3_bucket=S3_PROCESSED_BUCKET)

        if df_labeled.empty:
            print("No labeled data found for training. Exiting.")
        else:
            # Train and validate model
            trained_model = model_trainer.train_and_validate(df_labeled)

            if trained_model:
                # Save model
                model_saver.save_model(
                    trained_model, s3_bucket=S3_MODEL_BUCKET, filename=MODEL_FILE_NAME
                )
                print("Model training and saving complete.")
            else:
                print("Model training failed.")
