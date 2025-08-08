import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import settings

import pandas as pd
import boto3
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for loading training data."""

    @abstractmethod
    def load_data(self, source: str) -> pd.DataFrame:
        """Loads data from a specified source."""
        pass


class ModelSaver(ABC):
    """Abstract base class for saving trained models."""

    @abstractmethod
    def save_model(self, model: Any, destination: str, filename: str) -> bool:
        """Saves a model to a specified destination. Returns True if successful."""
        pass


class ModelTrainer(ABC):
    """Abstract base class for training a machine learning model."""

    @abstractmethod
    def train_and_validate(self, data: pd.DataFrame) -> Optional[Any]:
        """Trains a model and returns the trained model."""
        pass


class LocalParquetDataLoader(DataLoader):
    """Loads Parquet data from local filesystem."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load parquet file from local path."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()

            if path.is_dir():
                # Handle partitioned parquet directories
                logger.info(f"Loading partitioned parquet data from: {file_path}")
                df = pd.read_parquet(file_path)
            else:
                # Handle single parquet file
                logger.info(f"Loading parquet file: {file_path}")
                df = pd.read_parquet(file_path)

            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Error loading parquet data from {file_path}: {str(e)}")
            return pd.DataFrame()


class S3ParquetDataLoader(DataLoader):
    """Loads Parquet data from S3, with support for both single files and partitioned structure."""

    def __init__(self, s3_client: Optional[Any] = None):
        self.s3_client = s3_client or boto3.client("s3")

    def load_data(self, s3_path: str) -> pd.DataFrame:
        """
        Load parquet data from S3.
        s3_path can be either:
        - s3://bucket/path/to/file.parquet (single file)
        - s3://bucket/path/to/partitioned/data/ (partitioned dataset)
        """
        try:
            logger.info(f"Loading data from S3: {s3_path}")

            # For this example, using dummy data as in original
            # In production, you would use:
            # df = pd.read_parquet(s3_path)

            dummy_data = {
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

            df = pd.DataFrame(dummy_data)
            logger.info(f"Successfully loaded {len(df)} records from S3")
            return df

        except Exception as e:
            logger.error(f"Error loading data from S3 {s3_path}: {str(e)}")
            return pd.DataFrame()


class LocalModelSaver(ModelSaver):
    """Saves a pickled model to local filesystem."""

    def save_model(self, model: Any, destination: str, filename: str) -> bool:
        """Save model to local filesystem."""
        try:
            dest_path = Path(destination)
            dest_path.mkdir(parents=True, exist_ok=True)

            model_path = dest_path / filename

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Model saved to: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model locally: {str(e)}")
            return False


class S3ModelSaver(ModelSaver):
    """Saves a pickled model to an S3 bucket."""

    def __init__(self, s3_client: Optional[Any] = None):
        self.s3_client = s3_client or boto3.client("s3")

    def save_model(self, model: Any, s3_bucket: str, filename: str) -> bool:
        """Save model to S3 bucket."""
        try:
            temp_model_path = f"/tmp/{filename}"

            with open(temp_model_path, "wb") as f:
                pickle.dump(model, f)

            self.s3_client.upload_file(temp_model_path, s3_bucket, filename)

            # Clean up temp file
            os.remove(temp_model_path)

            logger.info(f"Model saved to s3://{s3_bucket}/{filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving model to S3: {str(e)}")
            return False


class SklearnTextClassifierTrainer(ModelTrainer):
    """Trains a TF-IDF + Logistic Regression model for text classification."""

    def __init__(
        self,
        target_roc_auc: float = 0.7,
        max_features: int = 5000,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.target_roc_auc = target_roc_auc
        self.test_size = test_size
        self.random_state = random_state

        self.model_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=2,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear", random_state=random_state, max_iter=1000
                    ),
                ),
            ]
        )

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = ["headline", "body", "label_t180"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        if len(df) < 4:  # Minimum for train/test split
            logger.error(
                f"Insufficient data: {len(df)} samples. Need at least 4 samples."
            )
            return False

        return True

    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features and target variables."""
        # Combine headline and body text
        df = df.copy()
        df["text"] = df["headline"].fillna("") + " " + df["body"].fillna("")

        X = df["text"]
        y = df["label_t180"].apply(lambda x: 1 if x == "Actionable" else 0)

        return X, y

    def _split_data(self, X: pd.Series, y: pd.Series) -> tuple:
        """Split data into train and validation sets."""
        unique_labels = y.nunique()

        if unique_labels < 2:
            logger.warning(
                "Only one class found in labels. Training without stratification."
            )
            stratify = None
        else:
            stratify = y

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def _evaluate_model(
        self, y_true: pd.Series, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)

        logger.info(f"Validation ROC-AUC score: {roc_auc:.4f}")
        logger.info("Classification Report:")
        logger.info(
            f"\n{classification_report(y_true, y_pred, target_names=['Bluff', 'Actionable'])}"
        )

        return {"roc_auc": roc_auc}

    def train_and_validate(self, df: pd.DataFrame) -> Optional[Any]:
        """Train and validate the model."""
        try:
            # Validate input data
            if not self._validate_data(df):
                return None

            # Prepare features and target
            X, y = self._prepare_features(df)

            # Split data
            X_train, X_val, y_train, y_val = self._split_data(X, y)

            logger.info(
                f"Training on {len(X_train)} samples, validating on {len(X_val)} samples."
            )
            logger.info(
                f"Class distribution - Training: {y_train.value_counts().to_dict()}"
            )

            # Train model
            logger.info("Training model...")
            self.model_pipeline.fit(X_train, y_train)
            logger.info("Model training complete.")

            # Evaluate model
            y_pred_proba = self.model_pipeline.predict_proba(X_val)[:, 1]
            metrics = self._evaluate_model(y_val, y_pred_proba)

            # Check if model meets target performance
            if metrics["roc_auc"] >= self.target_roc_auc:
                logger.info(
                    f"✓ Model performance meets target (≥{self.target_roc_auc})"
                )
            else:
                logger.warning(
                    f"⚠ Model performance below target (≥{self.target_roc_auc}). Consider hyperparameter tuning."
                )

            return self.model_pipeline

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return None


class MLTrainingPipeline:
    """Main pipeline orchestrator for ML model training."""

    def __init__(
        self,
        data_loader: DataLoader,
        model_trainer: ModelTrainer,
        model_saver: ModelSaver,
    ):
        self.data_loader = data_loader
        self.model_trainer = model_trainer
        self.model_saver = model_saver

    def run(
        self, data_source: str, model_destination: str, model_filename: str
    ) -> bool:
        """Run the complete training pipeline."""
        try:
            # Load data
            logger.info("Starting ML training pipeline...")
            df = self.data_loader.load_data(data_source)

            if df.empty:
                logger.error("No data loaded. Pipeline terminated.")
                return False

            # Train model
            trained_model = self.model_trainer.train_and_validate(df)

            if not trained_model:
                logger.error("Model training failed. Pipeline terminated.")
                return False

            # Save model
            success = self.model_saver.save_model(
                trained_model, model_destination, model_filename
            )

            if success:
                logger.info("✓ ML training pipeline completed successfully.")
                return True
            else:
                logger.error("Model saving failed.")
                return False

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            return False


def create_pipeline_from_config(config: Dict[str, Any]) -> MLTrainingPipeline:
    """Factory function to create pipeline based on configuration."""

    # Create data loader
    if config.get("data_source_type") == "local":
        data_loader = LocalParquetDataLoader()
    elif config.get("data_source_type") == "s3":
        s3_client = boto3.client("s3") if config.get("use_s3") else None
        data_loader = S3ParquetDataLoader(s3_client)
    else:
        raise ValueError("Invalid data_source_type. Use 'local' or 's3'")

    # Create model trainer
    trainer_config = config.get("trainer", {})
    model_trainer = SklearnTextClassifierTrainer(**trainer_config)

    # Create model saver
    if config.get("model_destination_type") == "local":
        model_saver = LocalModelSaver()
    elif config.get("model_destination_type") == "s3":
        s3_client = boto3.client("s3") if config.get("use_s3") else None
        model_saver = S3ModelSaver(s3_client)
    else:
        raise ValueError("Invalid model_destination_type. Use 'local' or 's3'")

    return MLTrainingPipeline(data_loader, model_trainer, model_saver)


if __name__ == "__main__":
    # Determine configuration based on environment settings
    if settings.S3_BUCKET and settings.S3_MODEL_BUCKET:
        # S3 configuration
        config = {
            "data_source_type": "s3",
            "model_destination_type": "s3",
            "use_s3": True,
            "trainer": {"target_roc_auc": 0.7, "max_features": 5000, "test_size": 0.2},
        }
        data_source = f"s3://{settings.S3_BUCKET}/processed/"
        model_destination = settings.S3_MODEL_BUCKET

    else:
        # Local configuration
        config = {
            "data_source_type": "local",
            "model_destination_type": "local",
            "use_s3": False,
            "trainer": {"target_roc_auc": 0.7, "max_features": 5000, "test_size": 0.2},
        }
        data_source = settings.LOCAL_DATA_PATH
        model_destination = settings.LOCAL_MODEL_PATH

    try:
        # Create and run pipeline
        pipeline = create_pipeline_from_config(config)
        success = pipeline.run(data_source, model_destination, settings.MODEL_FILE_NAME)

        if success:
            logger.info("🎉 Training completed successfully!")
        else:
            logger.error("❌ Training failed!")
            exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)
