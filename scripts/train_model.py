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
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

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
            print(df.columns)
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


class TextFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for advanced text feature engineering."""

    def __init__(self):
        self.feature_names_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract additional text features."""
        features = []
        feature_names = []

        for text in X:
            text_str = str(text)

            # Length features
            char_count = len(text_str)
            word_count = len(text_str.split())
            sentence_count = len(text_str.split("."))

            # Punctuation features
            exclamation_count = text_str.count("!")
            question_count = text_str.count("?")

            # Uppercase features
            upper_ratio = sum(1 for c in text_str if c.isupper()) / max(
                len(text_str), 1
            )

            # Financial/action keywords
            financial_keywords = [
                "billion",
                "million",
                "budget",
                "fund",
                "tax",
                "spending",
            ]
            action_keywords = [
                "will",
                "shall",
                "must",
                "require",
                "implement",
                "establish",
            ]
            urgent_keywords = ["immediately", "urgent", "crisis", "emergency"]

            financial_count = sum(
                1 for word in financial_keywords if word.lower() in text_str.lower()
            )
            action_count = sum(
                1 for word in action_keywords if word.lower() in text_str.lower()
            )
            urgent_count = sum(
                1 for word in urgent_keywords if word.lower() in text_str.lower()
            )

            # Combine features
            row_features = [
                char_count,
                word_count,
                sentence_count,
                exclamation_count,
                question_count,
                upper_ratio,
                financial_count,
                action_count,
                urgent_count,
            ]

            features.append(row_features)

        self.feature_names_ = [
            "char_count",
            "word_count",
            "sentence_count",
            "exclamation_count",
            "question_count",
            "upper_ratio",
            "financial_count",
            "action_count",
            "urgent_count",
        ]

        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class ModelTrainer(ABC):
    @abstractmethod
    def train_and_validate(self, df: pd.DataFrame) -> Optional[Any]:
        pass


class ImprovedSklearnTextClassifierTrainer(ModelTrainer):
    """Enhanced trainer with cross-validation, hyperparameter tuning, and feature engineering."""

    def __init__(
        self,
        target_roc_auc: float = 0.7,
        max_features: int = 10000,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        use_hyperparameter_tuning: bool = True,
        use_feature_selection: bool = True,
        use_ensemble: bool = False,
    ):
        self.target_roc_auc = target_roc_auc
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_feature_selection = use_feature_selection
        self.use_ensemble = use_ensemble
        self.max_features = max_features

        self.best_model = None
        self.best_score = 0
        self.feature_importance = None

    def _create_base_pipeline(self, classifier_name: str = "logistic") -> Pipeline:
        """Create a base pipeline with advanced feature engineering."""

        steps = []

        # Text preprocessing and vectorization
        if self.use_feature_selection:
            steps.extend(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=self.max_features,
                            stop_words="english",
                            ngram_range=(1, 3),  # Include trigrams
                            min_df=2,
                            max_df=0.95,  # Remove very common words
                            sublinear_tf=True,  # Use sublinear tf scaling
                        ),
                    ),
                    (
                        "feature_selection",
                        SelectKBest(chi2, k=min(5000, self.max_features)),
                    ),
                ]
            )
        else:
            steps.append(
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words="english",
                        ngram_range=(1, 3),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                    ),
                )
            )

        # Classifier
        if classifier_name == "logistic":
            classifier = LogisticRegression(
                solver="liblinear",
                random_state=self.random_state,
                max_iter=2000,
                class_weight="balanced",  # Handle class imbalance
            )
        elif classifier_name == "svm":
            classifier = SVC(
                probability=True,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif classifier_name == "rf":
            classifier = RandomForestClassifier(
                random_state=self.random_state,
                class_weight="balanced",
                n_estimators=100,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        steps.append(("classifier", classifier))

        return Pipeline(steps)

    def _get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Define hyperparameter grids for different models."""
        return {
            "logistic": {
                "tfidf__max_features": [5000, 10000, 15000],
                "tfidf__ngram_range": [(1, 2), (1, 3)],
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
            },
            "svm": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 2), (1, 3)],
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
            },
            "rf": {
                "tfidf__max_features": [5000, 10000],
                "tfidf__ngram_range": [(1, 2), (1, 3)],
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [10, 20, None],
            },
        }

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = ["headline", "body", "label_t180"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        if len(df) < 10:  # Minimum for cross-validation
            logger.error(
                f"Insufficient data: {len(df)} samples. Need at least 10 samples for cross-validation."
            )
            return False

        return True

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Prepare features and target variables."""
        df = df.copy()

        # Combine headline and body with better formatting
        df["text"] = (
            df["headline"].fillna("") + ". " + df["body"].fillna("")
        ).str.strip()

        # Remove extra whitespace and clean text
        df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

        X = df["text"]
        y = df["label_t180"].apply(lambda x: 1 if x == "Actionable" else 0)

        return X, y

    def _perform_cross_validation(
        self, pipeline: Pipeline, X: pd.Series, y: pd.Series
    ) -> Dict[str, float]:
        """Perform cross-validation and return metrics."""
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        # ROC AUC scores
        roc_scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )

        # F1 scores
        f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)

        return {
            "roc_auc_mean": roc_scores.mean(),
            "roc_auc_std": roc_scores.std(),
            "f1_mean": f1_scores.mean(),
            "f1_std": f1_scores.std(),
        }

    def _hyperparameter_tuning(
        self,
        pipeline: Pipeline,
        X_train: pd.Series,
        y_train: pd.Series,
        classifier_name: str,
    ) -> Pipeline:
        """Perform hyperparameter tuning using GridSearchCV."""
        param_grids = self._get_hyperparameter_grids()
        param_grid = param_grids.get(classifier_name, {})

        if not param_grid:
            logger.warning(f"No hyperparameter grid found for {classifier_name}")
            return pipeline

        logger.info(f"Starting hyperparameter tuning for {classifier_name}...")

        cv = StratifiedKFold(
            n_splits=min(3, self.cv_folds), shuffle=True, random_state=self.random_state
        )

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _evaluate_model(
        self, pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"\n=== {model_name} Results ===")
        logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        logger.info(
            f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Bluff', 'Actionable'])}"
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        return {"roc_auc": roc_auc, "f1_score": f1, "model_name": model_name}

    def _extract_feature_importance(
        self, pipeline: Pipeline, model_name: str
    ) -> Optional[pd.DataFrame]:
        """Extract and log feature importance."""
        try:
            if hasattr(pipeline.named_steps["classifier"], "coef_"):
                # For linear models
                feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
                if self.use_feature_selection:
                    # Get selected features
                    selected_features = pipeline.named_steps[
                        "feature_selection"
                    ].get_support()
                    feature_names = feature_names[selected_features]

                coefficients = pipeline.named_steps["classifier"].coef_[0]

                feature_importance = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": np.abs(coefficients),
                        "coefficient": coefficients,
                    }
                ).sort_values("importance", ascending=False)

                logger.info(f"\nTop 10 Most Important Features for {model_name}:")
                logger.info(feature_importance.head(10).to_string(index=False))

                return feature_importance

            elif hasattr(pipeline.named_steps["classifier"], "feature_importances_"):
                # For tree-based models
                feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
                if self.use_feature_selection:
                    selected_features = pipeline.named_steps[
                        "feature_selection"
                    ].get_support()
                    feature_names = feature_names[selected_features]

                importances = pipeline.named_steps["classifier"].feature_importances_

                feature_importance = pd.DataFrame(
                    {"feature": feature_names, "importance": importances}
                ).sort_values("importance", ascending=False)

                logger.info(f"\nTop 10 Most Important Features for {model_name}:")
                logger.info(feature_importance.head(10).to_string(index=False))

                return feature_importance

        except Exception as e:
            logger.warning(
                f"Could not extract feature importance for {model_name}: {e}"
            )

        return None

    def train_and_validate(self, df: pd.DataFrame) -> Optional[Any]:
        """Train and validate multiple models with cross-validation."""
        try:
            # Validate input data
            if not self._validate_data(df):
                return None

            # Prepare features and target
            X, y = self._prepare_features(df)

            # Check class balance
            class_counts = y.value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")

            if len(class_counts) < 2:
                logger.error(
                    "Only one class found in the data. Cannot train classifier."
                )
                return None

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

            logger.info(
                f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
            )

            # Models to try
            models_to_try = (
                ["logistic", "svm"]
                if not self.use_ensemble
                else ["logistic", "svm", "rf"]
            )

            best_score = 0
            best_model = None
            best_model_name = ""

            for model_name in models_to_try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {model_name.upper()} model")
                logger.info(f"{'='*50}")

                # Create pipeline
                pipeline = self._create_base_pipeline(model_name)

                # Hyperparameter tuning
                if self.use_hyperparameter_tuning:
                    pipeline = self._hyperparameter_tuning(
                        pipeline, X_train, y_train, model_name
                    )
                else:
                    pipeline.fit(X_train, y_train)

                # Cross-validation on training data
                cv_results = self._perform_cross_validation(pipeline, X_train, y_train)
                logger.info(
                    f"Cross-validation ROC-AUC: {cv_results['roc_auc_mean']:.4f} (+/- {cv_results['roc_auc_std']*2:.4f})"
                )
                logger.info(
                    f"Cross-validation F1: {cv_results['f1_mean']:.4f} (+/- {cv_results['f1_std']*2:.4f})"
                )

                # Evaluate on test set
                test_results = self._evaluate_model(
                    pipeline, X_test, y_test, model_name
                )

                # Extract feature importance
                feature_importance = self._extract_feature_importance(
                    pipeline, model_name
                )

                # Keep best model
                if test_results["roc_auc"] > best_score:
                    best_score = test_results["roc_auc"]
                    best_model = pipeline
                    best_model_name = model_name
                    self.feature_importance = feature_importance

            # Final evaluation
            logger.info(f"\n{'='*50}")
            logger.info(
                f"BEST MODEL: {best_model_name.upper()} (ROC-AUC: {best_score:.4f})"
            )
            logger.info(f"{'='*50}")

            if best_score >= self.target_roc_auc:
                logger.info(
                    f"✓ Best model performance meets target (≥{self.target_roc_auc})"
                )
            else:
                logger.warning(
                    f"⚠ Best model performance below target (≥{self.target_roc_auc})"
                )
                logger.info("Consider:")
                logger.info("- Collecting more training data")
                logger.info("- Feature engineering")
                logger.info("- Different algorithms")
                logger.info("- Adjusting class weights")

            self.best_model = best_model
            self.best_score = best_score

            return best_model

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return None

    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on new texts."""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train_and_validate first.")

        return self.best_model.predict_proba(texts)[:, 1]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the best model."""
        return {
            "best_score": self.best_score,
            "target_score": self.target_roc_auc,
            "model_type": (
                type(self.best_model.named_steps["classifier"]).__name__
                if self.best_model
                else None
            ),
            "feature_count": (
                len(self.feature_importance)
                if self.feature_importance is not None
                else None
            ),
            "hyperparameter_tuning": self.use_hyperparameter_tuning,
            "feature_selection": self.use_feature_selection,
        }


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
    model_trainer = ImprovedSklearnTextClassifierTrainer(**trainer_config)

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
        data_source = settings.OUTPUT_FILE
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
