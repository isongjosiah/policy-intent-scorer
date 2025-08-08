import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
from datetime import datetime
import asyncio

from scripts.archive_puller import ArchiveDataPuller


class TestArchivePuller(unittest.TestCase):
    @patch("scripts.archive_puller.WhiteHouseHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.BidenArchiveHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.ObamaArchiveHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.BushArchiveHTMLScraper", new_callable=AsyncMock)
    def test_pull_archive_with_successful_scrape(
        self, mock_bush, mock_obama, mock_biden, mock_whitehouse
    ):
        # Arrange
        mock_whitehouse.return_value.get_data.return_value = [
            {"headline": "WH Headline", "date": datetime(2025, 1, 1)}
        ]
        mock_biden.return_value.get_data.return_value = [
            {"headline": "Biden Headline", "date": datetime(2025, 1, 2)}
        ]
        mock_obama.return_value.get_data.return_value = [
            {"headline": "Obama Headline", "date": datetime(2025, 1, 3)}
        ]
        mock_bush.return_value.get_data.return_value = [
            {"headline": "Bush Headline", "date": datetime(2025, 1, 4)}
        ]

        puller = ArchiveDataPuller()

        # Act
        result_df = asyncio.run(puller.pull_archive())

        # Assert
        self.assertEqual(len(result_df), 4)
        self.assertIn("WH Headline", result_df["headline"].values)

    @patch("scripts.archive_puller.WhiteHouseHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.BidenArchiveHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.ObamaArchiveHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.BushArchiveHTMLScraper", new_callable=AsyncMock)
    def test_pull_archive_with_one_failed_source(
        self, mock_bush, mock_obama, mock_biden, mock_whitehouse
    ):
        # Arrange
        mock_whitehouse.return_value.get_data.return_value = [
            {"headline": "WH Headline", "date": datetime(2025, 1, 1)}
        ]
        mock_biden.return_value.get_data.side_effect = Exception("Scrape failed")
        mock_obama.return_value.get_data.return_value = []
        mock_bush.return_value.get_data.return_value = [
            {"headline": "Bush Headline", "date": datetime(2025, 1, 4)}
        ]

        puller = ArchiveDataPuller()

        # Act
        result_df = asyncio.run(puller.pull_archive())

        # Assert
        self.assertEqual(len(result_df), 2)
        self.assertIn("WH Headline", result_df["headline"].values)
        self.assertNotIn("Biden Headline", result_df["headline"].values)

    @patch("scripts.archive_puller.WhiteHouseHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.BidenArchiveHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.ObamaArchiveHTMLScraper", new_callable=AsyncMock)
    @patch("scripts.archive_puller.BushArchiveHTMLScraper", new_callable=AsyncMock)
    def test_pull_archive_with_no_data(
        self, mock_bush, mock_obama, mock_biden, mock_whitehouse
    ):
        # Arrange
        mock_whitehouse.return_value.get_data.return_value = []
        mock_biden.return_value.get_data.return_value = []
        mock_obama.return_value.get_data.return_value = []
        mock_bush.return_value.get_data.return_value = []

        puller = ArchiveDataPuller()

        # Act
        result_df = asyncio.run(puller.pull_archive())

        # Assert
        self.assertTrue(result_df.empty)


if __name__ == "__main__":
    unittest.main()
