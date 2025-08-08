import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd

from scripts.labeler import (
    KeywordExtractor,
    CongressAPIClient,
    MarketDataClient,
    CongressOutcomeChecker,
    MarketOutcomeChecker,
    PressReleaseLabelingService,
    LabelingResult,
)


class TestKeywordExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = KeywordExtractor()
        self.text = "The President announced a new policy on climate change. This policy will impact the economy."

    def test_preprocess_text(self):
        words = self.extractor.preprocess_text(self.text)
        self.assertNotIn("the", words)
        self.assertIn("president", words)
        self.assertIn("policy", words)

    def test_extract_by_frequency(self):
        keywords = self.extractor.extract_by_frequency(self.text, top_k=2)
        self.assertEqual(len(keywords), 2)
        self.assertEqual(keywords[0][0], "policy")


class TestPressReleaseLabelingService(unittest.TestCase):
    def setUp(self):
        self.congress_checker = MagicMock()
        self.market_checker = MagicMock()
        self.service = PressReleaseLabelingService(
            outcome_checkers=[self.congress_checker, self.market_checker]
        )
        self.headline = "Test Headline"
        self.body = "Test Body"
        self.pub_date = datetime(2025, 1, 1)

    def test_label_press_release_actionable(self):
        self.congress_checker.check_outcome.return_value = (True, "some_url", 30)
        self.market_checker.check_outcome.return_value = (False, None, None)

        result = self.service.label_press_release(
            self.headline, self.body, self.pub_date
        )

        self.assertEqual(result.label_t180, "Actionable")
        self.assertEqual(result.match_type, "Law Passed")

    def test_label_press_release_bluff(self):
        self.congress_checker.check_outcome.return_value = (False, None, None)
        self.market_checker.check_outcome.return_value = (False, None, None)

        result = self.service.label_press_release(
            self.headline, self.body, self.pub_date
        )

        self.assertEqual(result.label_t180, "Bluff")


class TestCongressOutcomeChecker(unittest.TestCase):
    def setUp(self):
        self.api_client = MagicMock(spec=CongressAPIClient)
        self.keyword_extractor = MagicMock(spec=KeywordExtractor)
        self.checker = CongressOutcomeChecker(self.api_client, self.keyword_extractor)
        self.text = "Some text about a new bill."
        self.pub_date = datetime(2025, 1, 1)

    def test_check_outcome_law_passed(self):
        self.keyword_extractor.extract_by_tfidf.return_value = [("bill", 1.0)]
        self.api_client.search_bills.return_value = [
            MagicMock(
                status="Became Law",
                latest_action_date=self.pub_date + timedelta(days=30),
                url="some_url",
            )
        ]

        found, url, days = self.checker.check_outcome(self.text, self.pub_date)

        self.assertTrue(found)
        self.assertEqual(url, "some_url")
        self.assertEqual(days, 30)

    def test_check_outcome_no_law(self):
        self.keyword_extractor.extract_by_tfidf.return_value = [("bill", 1.0)]
        self.api_client.search_bills.return_value = []

        found, _, _ = self.checker.check_outcome(self.text, self.pub_date)

        self.assertFalse(found)


class TestMarketOutcomeChecker(unittest.TestCase):
    def setUp(self):
        self.market_client = MagicMock(spec=MarketDataClient)
        self.checker = MarketOutcomeChecker(self.market_client, sectors=["SPY"])
        self.text = "Some text about the market."
        self.pub_date = datetime(2025, 1, 1)

    def test_check_outcome_market_moved(self):
        self.market_client.calculate_volatility_spike.return_value = (True, 3.0)

        found, info, days = self.checker.check_outcome(self.text, self.pub_date)

        self.assertTrue(found)
        self.assertIn("Market volatility spike", info)

    def test_check_outcome_no_market_move(self):
        self.market_client.calculate_volatility_spike.return_value = (False, 1.0)

        found, _, _ = self.checker.check_outcome(self.text, self.pub_date)

        self.assertFalse(found)


if __name__ == "__main__":
    unittest.main()
