import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_model import (
    ImprovedSklearnTextClassifierTrainer,
    LocalParquetDataLoader,
    S3ParquetDataLoader,
    LocalModelSaver,
    S3ModelSaver,
    SklearnTextClassifierTrainer,
    MLTrainingPipeline,
    create_pipeline_from_config,
)


class TestDataLoaders(unittest.TestCase):
    def test_local_parquet_data_loader_success(self):
        # Create a dummy parquet file
        data = pd.DataFrame(
            {
                "headline": ["a", "b"],
                "body": ["c", "d"],
                "label_t180": ["Actionable", "Bluff"],
            }
        )
        os.makedirs("/tmp/data", exist_ok=True)
        data.to_parquet("/tmp/data/test.parquet")

        loader = LocalParquetDataLoader()
        df = loader.load_data("/tmp/data/test.parquet")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_local_parquet_data_loader_file_not_found(self):
        loader = LocalParquetDataLoader()
        df = loader.load_data("non_existent_file.parquet")
        self.assertTrue(df.empty)

    def test_s3_parquet_data_loader(self):
        mock_s3_client = MagicMock()
        loader = S3ParquetDataLoader(s3_client=mock_s3_client)
        df = loader.load_data("s3://bucket/key")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)


class TestModelSavers(unittest.TestCase):
    def test_local_model_saver(self):
        saver = LocalModelSaver()
        model = "dummy_model"
        os.makedirs("/tmp/models", exist_ok=True)
        self.assertTrue(saver.save_model(model, "/tmp/models", "model.pkl"))

    def test_s3_model_saver(self):
        mock_s3_client = MagicMock()
        saver = S3ModelSaver(s3_client=mock_s3_client)
        model = "dummy_model"
        self.assertTrue(saver.save_model(model, "bucket", "model.pkl"))
        mock_s3_client.upload_file.assert_called_once()


class TestSklearnTextClassifierTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ImprovedSklearnTextClassifierTrainer()
        self.data = pd.DataFrame(
            {
                "headline": [f"headline_{i}" for i in range(10)],
                "body": [f"body_{i}" for i in range(10)],
                "label_t180": ["Actionable", "Bluff"] * 5,
            }
        )

    def test_train_and_validate_success(self):
        model = self.trainer.train_and_validate(self.data)
        self.assertIsNotNone(model)

    def test_train_and_validate_insufficient_data(self):
        small_data = self.data.iloc[:3]
        model = self.trainer.train_and_validate(small_data)
        self.assertIsNone(model)

    def test_train_and_validate_missing_columns(self):
        invalid_data = self.data.drop(columns=["headline"])
        model = self.trainer.train_and_validate(invalid_data)
        self.assertIsNone(model)


class TestMLTrainingPipeline(unittest.TestCase):
    def test_run_pipeline_success(self):
        mock_data_loader = MagicMock()
        mock_data_loader.load_data.return_value = pd.DataFrame(
            {
                "headline": ["a", "b", "c", "d"],
                "body": ["e", "f", "g", "h"],
                "label_t180": ["Actionable", "Bluff", "Actionable", "Bluff"],
            }
        )
        mock_model_trainer = MagicMock()
        mock_model_trainer.train_and_validate.return_value = "trained_model"
        mock_model_saver = MagicMock()
        mock_model_saver.save_model.return_value = True

        pipeline = MLTrainingPipeline(
            mock_data_loader, mock_model_trainer, mock_model_saver
        )
        self.assertTrue(pipeline.run("source", "dest", "file"))

    def test_run_pipeline_data_load_fails(self):
        mock_data_loader = MagicMock()
        mock_data_loader.load_data.return_value = pd.DataFrame()
        pipeline = MLTrainingPipeline(mock_data_loader, MagicMock(), MagicMock())
        self.assertFalse(pipeline.run("source", "dest", "file"))

    def test_run_pipeline_training_fails(self):
        mock_data_loader = MagicMock()
        mock_data_loader.load_data.return_value = pd.DataFrame(
            {
                "headline": ["a", "b", "c", "d"],
                "body": ["e", "f", "g", "h"],
                "label_t180": ["Actionable", "Bluff", "Actionable", "Bluff"],
            }
        )
        mock_model_trainer = MagicMock()
        mock_model_trainer.train_and_validate.return_value = None
        pipeline = MLTrainingPipeline(mock_data_loader, mock_model_trainer, MagicMock())
        self.assertFalse(pipeline.run("source", "dest", "file"))


class TestConfigFactory(unittest.TestCase):
    @patch("train_model.boto3")
    def test_create_pipeline_from_config_s3(self, mock_boto3):
        config = {
            "data_source_type": "s3",
            "model_destination_type": "s3",
            "use_s3": True,
            "trainer": {"target_roc_auc": 0.8},
        }
        pipeline = create_pipeline_from_config(config)
        self.assertIsInstance(pipeline.data_loader, S3ParquetDataLoader)
        self.assertIsInstance(pipeline.model_saver, S3ModelSaver)
        self.assertIsInstance(
            pipeline.model_trainer, ImprovedSklearnTextClassifierTrainer
        )
        self.assertEqual(pipeline.model_trainer.target_roc_auc, 0.8)

    def test_create_pipeline_from_config_local(self):
        config = {
            "data_source_type": "local",
            "model_destination_type": "local",
            "use_s3": False,
        }
        pipeline = create_pipeline_from_config(config)
        self.assertIsInstance(pipeline.data_loader, LocalParquetDataLoader)
        self.assertIsInstance(pipeline.model_saver, LocalModelSaver)
        self.assertIsInstance(
            pipeline.model_trainer, ImprovedSklearnTextClassifierTrainer
        )

    def test_create_pipeline_from_config_invalid(self):
        with self.assertRaises(ValueError):
            create_pipeline_from_config({"data_source_type": "invalid"})
        with self.assertRaises(ValueError):
            create_pipeline_from_config(
                {"data_source_type": "local", "model_destination_type": "invalid"}
            )


if __name__ == "__main__":
    unittest.main()
