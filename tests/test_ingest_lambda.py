import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from scripts.ingest_lambda import (
    IngestionService,
    WhiteHouseHTMLScraper,
    WhiteHouseRssScrapper,
    handler,
)


class TestIngestLambda(unittest.TestCase):
    @patch("scripts.ingest_lambda.WhiteHouseHTMLScraper")
    @patch("scripts.ingest_lambda.WhiteHouseRssScrapper")
    def test_handler_with_primary_source(self, mock_rss_scraper, mock_html_scraper):
        # Arrange
        mock_html_scraper.return_value.get_data.return_value = [
            {
                "headline": "Test Headline",
                "body": "Test Body",
                "date": datetime(2025, 1, 1),
            }
        ]
        mock_rss_scraper.return_value.get_data.return_value = []

        # Act
        result = handler(None, None)

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["headline"].iloc[0], "Test Headline")

    @patch("scripts.ingest_lambda.WhiteHouseHTMLScraper")
    @patch("scripts.ingest_lambda.WhiteHouseRssScrapper")
    def test_handler_with_fallback_source(self, mock_rss_scraper, mock_html_scraper):
        # Arrange
        mock_html_scraper.return_value.get_data.return_value = []
        mock_rss_scraper.return_value.get_data.return_value = [
            {
                "headline": "Fallback Headline",
                "body": "Fallback Body",
                "date": datetime(2025, 1, 1),
            }
        ]

        # Act
        result = handler(None, None)

        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["headline"].iloc[0], "Fallback Headline")

    @patch("scripts.ingest_lambda.WhiteHouseHTMLScraper")
    @patch("scripts.ingest_lambda.WhiteHouseRssScrapper")
    def test_handler_with_no_data(self, mock_rss_scraper, mock_html_scraper):
        # Arrange
        mock_html_scraper.return_value.get_data.return_value = []
        mock_rss_scraper.return_value.get_data.return_value = []

        # Act
        result = handler(None, None)

        # Assert
        self.assertIsNone(result)

    def test_ingestion_service_with_primary_source(self):
        # Arrange
        primary_source = MagicMock()
        fallback_source = MagicMock()
        primary_source.get_data.return_value = [{"key": "primary"}]
        fallback_source.get_data.return_value = [{"key": "fallback"}]
        service = IngestionService(primary_source, fallback_source)

        # Act
        result = service.get_all_releases()

        # Assert
        self.assertEqual(result, [{"key": "primary"}])
        primary_source.get_data.assert_called_once()
        fallback_source.get_data.assert_not_called()

    def test_ingestion_service_with_fallback_source(self):
        # Arrange
        primary_source = MagicMock()
        fallback_source = MagicMock()
        primary_source.get_data.return_value = []
        fallback_source.get_data.return_value = [{"key": "fallback"}]
        service = IngestionService(primary_source, fallback_source)

        # Act
        result = service.get_all_releases()

        # Assert
        self.assertEqual(result, [{"key": "fallback"}])
        primary_source.get_data.assert_called_once()
        fallback_source.get_data.assert_called_once()


if __name__ == "__main__":
    unittest.main()
