import unittest
from unittest.mock import patch, MagicMock
import requests
from scripts.ingest_lambda import (
    WhiteHouseHtmlScraper,
    WhiteHouseRssScraper,
    IngestionService,
    DataSource,
)


class TestWhiteHouseScrapers(unittest.TestCase):

    @patch("scripts.ingest_lambda.requests.get")
    def test_html_scraper_success(self, mock_requests_get):
        """
        Tests successful scraping from the WhiteHouseHtmlScraper.
        Mocks requests.get for both the main page and individual article bodies.
        """
        # Mock response for the main page (list of articles)
        mock_main_page_response = MagicMock()
        mock_main_page_response.raise_for_status.return_value = None
        mock_main_page_response.text = """
            <html><body>
                <article class="news-item">
                    <h2>Headline One</h2>
                    <a href="https://www.whitehouse.gov/briefing-room/statements-and-releases/release-1"></a>
                    <time datetime="2025-08-01T10:00:00Z"></time>
                </article>
                <article class="news-item">
                    <h2>Headline Two</h2>
                    <a href="https://www.whitehouse.gov/briefing-room/statements-and-releases/release-2"></a>
                    <time datetime="2025-08-02T11:00:00Z"></time>
                </article>
            </body></html>
        """

        # Mock responses for individual article bodies
        mock_body_response_1 = MagicMock()
        mock_body_response_1.raise_for_status.return_value = None
        mock_body_response_1.text = """
            <html><body><div class="body-content">Body of Release One.</div></body></html>
        """

        mock_body_response_2 = MagicMock()
        mock_body_response_2.raise_for_status.return_value = None
        mock_body_response_2.text = """
            <html><body><div class="body-content">Body of Release Two.</div></body></html>
        """

        # Configure mock_requests_get to return different responses for sequential calls
        mock_requests_get.side_effect = [
            mock_main_page_response,
            mock_body_response_1,
            mock_body_response_2,
        ]

        scraper = WhiteHouseHtmlScraper(
            "https://www.whitehouse.gov/briefing-room/statements-and-releases/"
        )
        releases = scraper.get_releases()

        self.assertEqual(len(releases), 2)
        self.assertEqual(releases[0]["headline"], "Headline One")
        self.assertEqual(releases[0]["published_date"], "2025-08-01")
        self.assertIn("Body of Release One", releases[0]["body"])
        self.assertEqual(releases[1]["headline"], "Headline Two")
        self.assertEqual(releases[1]["published_date"], "2025-08-02")
        self.assertIn("Body of Release Two", releases[1]["body"])
        self.assertEqual(releases[0]["source"], "whitehouse")
        self.assertEqual(releases[1]["source"], "whitehouse")

    @patch("scripts.ingest_lambda.requests.get")
    def test_html_scraper_no_articles(self, mock_requests_get):
        """
        Tests HTML scraper when no articles are found on the page.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<html><body><div class='no-news'></div></body></html>"
        mock_requests_get.return_value = mock_response

        scraper = WhiteHouseHtmlScraper(
            "https://www.whitehouse.gov/briefing-room/statements-and-releases/"
        )
        releases = scraper.get_releases()
        self.assertEqual(len(releases), 0)

    @patch("scripts.ingest_lambda.requests.get")
    def test_html_scraper_request_exception(self, mock_requests_get):
        """
        Tests HTML scraper when a request exception occurs.
        """
        mock_requests_get.side_effect = requests.exceptions.RequestException(
            "Network error"
        )
        scraper = WhiteHouseHtmlScraper(
            "https://www.whitehouse.gov/briefing-room/statements-and-releases/"
        )
        releases = scraper.get_releases()
        self.assertEqual(len(releases), 0)

    @patch("scripts.ingest_lambda.feedparser.parse")
    @patch(
        "scripts.ingest_lambda.requests.get"
    )  # Mock requests.get for fetching RSS entry bodies
    def test_rss_scraper_success(self, mock_requests_get, mock_feedparser_parse):
        """
        Tests successful scraping from the WhiteHouseRssScraper.
        Mocks feedparser.parse and requests.get for individual RSS entry bodies.
        """
        # Mock an RSS entry
        mock_entry_1 = MagicMock()
        mock_entry_1.title = "RSS Headline One"
        mock_entry_1.link = "http://example.com/rss-release-1"
        mock_entry_1.published = "Thu, 01 Aug 2025 10:00:00 +0000"

        mock_entry_2 = MagicMock()
        mock_entry_2.title = "RSS Headline Two"
        mock_entry_2.link = "http://example.com/rss-release-2"
        mock_entry_2.published = "Fri, 02 Aug 2025 11:00:00 +0000"

        # Mock the feedparser.parse result
        mock_feedparser_parse.return_value.entries = [mock_entry_1, mock_entry_2]

        # Mock responses for fetching bodies from RSS links
        mock_body_response_1 = MagicMock()
        mock_body_response_1.raise_for_status.return_value = None
        mock_body_response_1.text = """
            <html><body><div class="body-content">Body from RSS Release One.</div></body></html>
        """

        mock_body_response_2 = MagicMock()
        mock_body_response_2.raise_for_status.return_value = None
        mock_body_response_2.text = """
            <html><body><div class="body-content">Body from RSS Release Two.</div></body></html>
        """
        mock_requests_get.side_effect = [mock_body_response_1, mock_body_response_2]

        scraper = WhiteHouseRssScraper("https://www.whitehouse.gov/briefing-room/feed/")
        releases = scraper.get_releases()

        self.assertEqual(len(releases), 2)
        self.assertEqual(releases[0]["headline"], "RSS Headline One")
        self.assertEqual(releases[0]["published_date"], "2025-08-01")
        self.assertIn("Body from RSS Release One", releases[0]["body"])
        self.assertEqual(releases[1]["headline"], "RSS Headline Two")
        self.assertEqual(releases[1]["published_date"], "2025-08-02")
        self.assertIn("Body from RSS Release Two", releases[1]["body"])
        self.assertEqual(releases[0]["source"], "whitehouse_rss")
        self.assertEqual(releases[1]["source"], "whitehouse_rss")

    @patch("scripts.ingest_lambda.feedparser.parse")
    def test_rss_scraper_parsing_error(self, mock_feedparser_parse):
        """
        Tests RSS scraper when a parsing error occurs.
        """
        mock_feedparser_parse.side_effect = Exception("Parsing error")
        scraper = WhiteHouseRssScraper("https://www.whitehouse.gov/briefing-room/feed/")
        releases = scraper.get_releases()
        self.assertEqual(len(releases), 0)


class TestIngestionService(unittest.TestCase):

    def test_ingestion_service_primary_success(self):
        """
        Tests IngestionService when the primary source is successful.
        """
        mock_primary_source = MagicMock(spec=DataSource)
        mock_primary_source.get_releases.return_value = [
            {"headline": "Primary Release"}
        ]

        mock_fallback_source = MagicMock(spec=DataSource)

        service = IngestionService(
            primary_source=mock_primary_source, fallback_source=mock_fallback_source
        )
        releases = service.get_all_releases()

        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0]["headline"], "Primary Release")
        mock_primary_source.get_releases.assert_called_once()
        mock_fallback_source.get_releases.assert_not_called()

    def test_ingestion_service_fallback_success(self):
        """
        Tests IngestionService when the primary source fails and fallback succeeds.
        """
        mock_primary_source = MagicMock(spec=DataSource)
        mock_primary_source.get_releases.return_value = []  # Simulate primary failure

        mock_fallback_source = MagicMock(spec=DataSource)
        mock_fallback_source.get_releases.return_value = [
            {"headline": "Fallback Release"}
        ]

        service = IngestionService(
            primary_source=mock_primary_source, fallback_source=mock_fallback_source
        )
        releases = service.get_all_releases()

        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0]["headline"], "Fallback Release")
        mock_primary_source.get_releases.assert_called_once()
        mock_fallback_source.get_releases.assert_called_once()

    def test_ingestion_service_all_fail(self):
        """
        Tests IngestionService when both primary and fallback sources fail.
        """
        mock_primary_source = MagicMock(spec=DataSource)
        mock_primary_source.get_releases.return_value = []

        mock_fallback_source = MagicMock(spec=DataSource)
        mock_fallback_source.get_releases.return_value = []

        service = IngestionService(
            primary_source=mock_primary_source, fallback_source=mock_fallback_source
        )
        releases = service.get_all_releases()

        self.assertEqual(len(releases), 0)
        mock_primary_source.get_releases.assert_called_once()
        mock_fallback_source.get_releases.assert_called_once()


if __name__ == "__main__":
    unittest.main()
