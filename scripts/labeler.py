from config.config import settings
import requests
import yfinance as yf
import pandas as pd
import boto3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import re
import math
import logging
from collections import Counter, defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """Data class for labeling results."""

    headline: str
    body: str
    published_date: str
    label_t180: str
    match_type: str
    match_url: Optional[str]
    days_to_outcome: Optional[int]
    quality_checked: bool = False


@dataclass
class TextStats:
    """Data class for text statistics."""

    total_words: int
    unique_words: int
    processed_words: int
    sentences: int
    characters: int
    avg_word_length: float


class KeywordExtractor:
    """Advanced NLP keyword extraction with multiple algorithms."""

    def __init__(self, custom_stop_words: Optional[List[str]] = None):
        self.stop_words = self._get_stop_words()
        if custom_stop_words:
            self.stop_words.update(custom_stop_words)

    def _get_stop_words(self) -> set:
        """Get comprehensive set of English stop words."""
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "been",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "have",
            "had",
            "this",
            "they",
            "i",
            "you",
            "we",
            "us",
            "our",
            "your",
            "their",
            "them",
            "his",
            "her",
            "but",
            "not",
            "or",
            "can",
            "could",
            "should",
            "would",
            "may",
            "might",
            "shall",
            "do",
            "does",
            "did",
            "am",
            "were",
            "been",
            "being",
            "having",
            "get",
            "got",
            "go",
            "going",
            "went",
            "gone",
            "said",
            "say",
            "says",
            "new",
            "also",
            "one",
            "two",
            "first",
            "last",
            "now",
            "today",
            "time",
            "way",
            "make",
            "made",
        }

    def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        if not text or not isinstance(text, str):
            return []

        text = text.lower().strip()
        # Remove punctuation but keep hyphens in compound words
        text = re.sub(r"[^\w\s-]", " ", text)
        # Split and filter words
        words = [
            word
            for word in text.split()
            if (
                word not in self.stop_words
                and len(word) > 2
                and word.isalpha()
                and not word.isdigit()
            )
        ]
        return words

    def extract_by_frequency(self, text: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """Extract keywords based on word frequency."""
        words = self.preprocess_text(text)
        if not words:
            return []
        return Counter(words).most_common(top_k)

    def extract_by_tfidf(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF scoring."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]

        if len(sentences) < 2:
            # Fallback to frequency-based for short texts
            freq_results = self.extract_by_frequency(text, top_k)
            return [(word, float(freq)) for word, freq in freq_results]

        # Process each sentence
        sentence_words = [self.preprocess_text(sentence) for sentence in sentences]
        all_words = set(word for words in sentence_words for word in words)

        if not all_words:
            return []

        word_tfidf = {}
        total_sentences = len(sentence_words)

        for word in all_words:
            # Calculate TF (term frequency)
            total_tf = sum(words.count(word) for words in sentence_words)
            total_words = sum(len(words) for words in sentence_words)
            tf = total_tf / max(total_words, 1)

            # Calculate IDF (inverse document frequency)
            doc_count = sum(1 for words in sentence_words if word in words)
            idf = math.log(total_sentences / max(doc_count, 1))

            word_tfidf[word] = tf * idf

        return sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def extract_ngrams(
        self, text: str, n: int = 2, top_k: int = 10
    ) -> List[Tuple[str, int]]:
        """Extract n-gram phrases as keywords."""
        words = self.preprocess_text(text)

        if len(words) < n:
            return []

        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        return Counter(ngrams).most_common(top_k)

    def extract_compound_keywords(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract compound keywords by combining adjacent important words."""
        words = self.preprocess_text(text)
        if len(words) < 2:
            return []

        # Get TF-IDF scores for individual words
        tfidf_scores = dict(self.extract_by_tfidf(text, len(set(words))))

        compound_scores = {}
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            score1 = tfidf_scores.get(word1, 0)
            score2 = tfidf_scores.get(word2, 0)

            if score1 > 0 and score2 > 0:
                compound = f"{word1} {word2}"
                compound_scores[compound] = (score1 + score2) / 2

        return sorted(compound_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def extract_all_methods(self, text: str, top_k: int = 10) -> Dict[str, List]:
        """Extract keywords using all available methods."""
        if not text or not isinstance(text, str):
            return {
                method: []
                for method in ["frequency", "tfidf", "bigrams", "trigrams", "compound"]
            }

        return {
            "frequency": self.extract_by_frequency(text, top_k),
            "tfidf": self.extract_by_tfidf(text, top_k),
            "bigrams": self.extract_ngrams(text, n=2, top_k=top_k),
            "trigrams": self.extract_ngrams(text, n=3, top_k=top_k),
            "compound": self.extract_compound_keywords(text, top_k),
        }

    def get_text_statistics(self, text: str) -> TextStats:
        """Get basic text statistics."""
        if not text or not isinstance(text, str):
            return TextStats(0, 0, 0, 0, 0, 0.0)

        words = self.preprocess_text(text)
        raw_words = text.split()
        sentences = [s for s in re.split(r"[.!?]+", text.strip()) if s.strip()]

        return TextStats(
            total_words=len(raw_words),
            unique_words=len(set(words)),
            processed_words=len(words),
            sentences=len(sentences),
            characters=len(text),
            avg_word_length=sum(len(word) for word in words) / max(len(words), 1),
        )


class CongressAPIClient:
    """Client for Congress.gov API interactions."""

    def __init__(self, api_key: str, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://api.congress.gov/v3"

    def search_bills(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for bills using the Congress.gov API."""
        try:
            url = f"{self.base_url}/bill/search"
            params = {
                "query": query,
                "api_key": self.api_key,
                "limit": limit,
                "format": "json",
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            return data.get("bills", [])

        except requests.RequestException as e:
            logger.warning(f"Error searching Congress API for '{query}': {e}")
            return []
        except (ValueError, KeyError) as e:
            logger.warning(f"Error parsing Congress API response for '{query}': {e}")
            return []


class MarketDataClient:
    """Client for market data interactions using yfinance."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_stock_data(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data for a given ticker and date range."""
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )

            if df.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None

            return df

        except Exception as e:
            logger.warning(f"Error fetching market data for {ticker}: {e}")
            return None

    def calculate_volatility_spike(
        self,
        ticker: str,
        pub_date: datetime,
        lookback_days: int = 30,
        threshold: float = 2.0,
    ) -> Tuple[bool, float]:
        """Calculate if there was a significant volatility spike."""
        # Skip weekends
        if pub_date.weekday() > 4:  # 5=Saturday, 6=Sunday
            return False, 0.0

        start_date = pub_date - timedelta(days=lookback_days + 5)  # Buffer for weekends
        end_date = pub_date + timedelta(days=2)

        df = self.get_stock_data(ticker, start_date, end_date)
        if df is None or len(df) < lookback_days:
            return False, 0.0

        try:
            # Calculate daily returns
            daily_returns = df["Close"].pct_change().dropna()

            if len(daily_returns) < lookback_days:
                return False, 0.0

            # Get the return on the announcement day
            pub_date_str = pub_date.strftime("%Y-%m-%d")
            if pub_date_str not in daily_returns.index:
                # Find the next available trading day
                available_dates = daily_returns.index[
                    daily_returns.index >= pub_date_str
                ]
                if len(available_dates) == 0:
                    return False, 0.0
                pub_date_str = available_dates[0]

            day_return = daily_returns.loc[pub_date_str]

            # Calculate rolling standard deviation (excluding the announcement day)
            historical_returns = daily_returns.loc[:pub_date_str].iloc[
                :-1
            ]  # Exclude announcement day
            if len(historical_returns) < lookback_days:
                return False, 0.0

            rolling_std = historical_returns.tail(lookback_days).std()

            if rolling_std == 0:
                return False, 0.0

            # Calculate z-score
            z_score = abs(day_return) / rolling_std

            return z_score >= threshold, z_score

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating volatility for {ticker}: {e}")
            return False, 0.0


class OutcomeChecker:
    """Abstract base class for outcome checking."""

    @abstractmethod
    def check_outcome(
        self, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check for outcome. Returns (found, url/info, days_to_outcome)."""
        pass


class CongressOutcomeChecker(OutcomeChecker):
    """Checks for legislative outcomes using Congress.gov API."""

    def __init__(
        self, congress_client: CongressAPIClient, keyword_extractor: KeywordExtractor
    ):
        self.congress_client = congress_client
        self.keyword_extractor = keyword_extractor

    def _extract_policy_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract relevant policy keywords from text."""
        # Use TF-IDF to get the most important terms
        keywords = self.keyword_extractor.extract_by_tfidf(text, max_keywords * 2)

        # Filter for policy-relevant terms (simple heuristic)
        policy_terms = []
        for word, score in keywords:
            if (
                len(word) > 3
                and word not in ["president", "administration", "government", "federal"]
                and not word.isdigit()
            ):
                policy_terms.append(word)

        return policy_terms[:max_keywords]

    def check_outcome(
        self, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check if related legislation became law within 180 days."""
        search_terms = self._extract_policy_keywords(text)

        for term in search_terms:
            bills = self.congress_client.search_bills(term)

            for bill in bills:
                if bill.get("status") == "Became Law":
                    law_date_str = bill.get("signed_date")
                    if law_date_str:
                        try:
                            law_date = datetime.fromisoformat(
                                law_date_str.replace("Z", "+00:00")
                            )
                            days_diff = (law_date - pub_date).days

                            if 0 < days_diff <= 180:
                                return True, bill.get("url"), days_diff
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Error parsing law date {law_date_str}: {e}"
                            )
                            continue

        return False, None, None


class MarketOutcomeChecker(OutcomeChecker):
    """Checks for market volatility outcomes."""

    def __init__(self, market_client: MarketDataClient, sectors: List[str]):
        self.market_client = market_client
        self.sectors = sectors

    def check_outcome(
        self, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check for significant market volatility spike."""
        for ticker in self.sectors:
            is_spike, z_score = self.market_client.calculate_volatility_spike(
                ticker, pub_date
            )

            if is_spike:
                info = f"Market volatility spike for {ticker} (Z-score: {z_score:.2f})"
                logger.info(f"Market spike detected: {info}")
                return True, info, 1  # Market reaction is typically immediate

        return False, None, None


class PressReleaseLabelingService:
    """Main service for labeling press releases."""

    def __init__(self, outcome_checkers: List[OutcomeChecker]):
        self.outcome_checkers = outcome_checkers

    def label_press_release(
        self, headline: str, body: str, pub_date: datetime
    ) -> LabelingResult:
        """Label a press release as Actionable or Bluff based on outcomes."""
        label = "Bluff"
        match_type = "None"
        match_url = None
        days_to_outcome = None

        # Combine headline and body for analysis
        full_text = f"{headline} {body}"

        # Check each outcome type
        for checker in self.outcome_checkers:
            try:
                found, url, days = checker.check_outcome(full_text, pub_date)
                if found:
                    label = "Actionable"
                    if isinstance(checker, CongressOutcomeChecker):
                        match_type = "Law Passed"
                    elif isinstance(checker, MarketOutcomeChecker):
                        match_type = "Market Moved"
                    else:
                        match_type = "Outcome Found"

                    match_url = url
                    days_to_outcome = days
                    break  # Use first found outcome

            except Exception as e:
                logger.warning(
                    f"Error checking outcome with {type(checker).__name__}: {e}"
                )
                continue

        return LabelingResult(
            headline=headline,
            body=body,
            published_date=pub_date.strftime("%Y-%m-%d"),
            label_t180=label,
            match_type=match_type,
            match_url=match_url,
            days_to_outcome=days_to_outcome,
            quality_checked=False,
        )


class DataProcessor:
    """Handles data loading and saving operations."""

    def __init__(self, s3_client: Optional[Any] = None):
        self.s3_client = s3_client

    def load_from_s3(self, bucket: str, key: str) -> pd.DataFrame:
        """Load data from S3."""
        try:
            s3_path = f"s3://{bucket}/{key}"
            df = pd.read_parquet(s3_path)
            logger.info(f"Loaded {len(df)} records from {s3_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return pd.DataFrame()

    def save_to_s3(self, df: pd.DataFrame, bucket: str, key: str) -> bool:
        """Save dataframe to S3."""
        try:
            s3_path = f"s3://{bucket}/{key}"
            df.to_parquet(s3_path, index=False)
            logger.info(f"Saved {len(df)} records to {s3_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving to S3: {e}")
            return False

    def load_from_local(self, file_path: str) -> pd.DataFrame:
        """Load data from local file."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()

            if file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading from local file: {e}")
            return pd.DataFrame()

    def save_to_local(self, df: pd.DataFrame, file_path: str) -> bool:
        """Save dataframe to local file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.endswith(".parquet"):
                df.to_parquet(file_path, index=False)
            elif file_path.endswith(".csv"):
                df.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Saved {len(df)} records to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving to local file: {e}")
            return False


def create_labeling_service(config: Dict[str, Any]) -> PressReleaseLabelingService:
    """Factory function to create labeling service from configuration."""
    # Initialize keyword extractor
    keyword_extractor = KeywordExtractor()

    # Initialize outcome checkers
    outcome_checkers = []

    # Congress checker
    if config.get("congress_api_key"):
        congress_client = CongressAPIClient(config["congress_api_key"])
        congress_checker = CongressOutcomeChecker(congress_client, keyword_extractor)
        outcome_checkers.append(congress_checker)

    # Market checker
    if config.get("market_sectors"):
        market_client = MarketDataClient()
        market_checker = MarketOutcomeChecker(market_client, config["market_sectors"])
        outcome_checkers.append(market_checker)

    return PressReleaseLabelingService(outcome_checkers)


def main():
    """Main function for local execution."""
    # Configuration
    config = {
        "congress_api_key": settings.CONGRESS_API_KEY,
        "market_sectors": settings.MARKET_SECTORS,
        "input_file": settings.INPUT_FILE,
        "output_file": settings.OUTPUT_FILE,
    }

    # Validate required configuration
    if not config["congress_api_key"]:
        logger.error("CONGRESS_API_KEY environment variable is required")
        return

    # Initialize services
    labeling_service = create_labeling_service(config)
    data_processor = DataProcessor()

    # Load data
    df_raw = data_processor.load_from_local(config["input_file"])
    if df_raw.empty:
        logger.warning("No input data found. Creating sample data for testing.")
        # Create sample data for testing
        sample_data = [
            {
                "headline": "President Announces New Climate Initiative",
                "body": "The President is proud to announce a bold new climate initiative...",
                "published_date": "2025-01-01",
            }
        ]
        df_raw = pd.DataFrame(sample_data)

    # Process data
    labeled_results = []
    for _, row in df_raw.iterrows():
        try:
            pub_date = datetime.fromisoformat(row["published_date"])
            result = labeling_service.label_press_release(
                row["headline"], row["body"], pub_date
            )
            labeled_results.append(result.__dict__)
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            continue

    # Save results
    if labeled_results:
        df_labeled = pd.DataFrame(labeled_results)
        success = data_processor.save_to_local(df_labeled, config["output_file"])

        if success:
            logger.info(
                f"✓ Successfully processed {len(labeled_results)} press releases"
            )
        else:
            logger.error("❌ Failed to save results")
    else:
        logger.warning("No results to save")


def handler(event, context):
    """AWS Lambda entry point."""
    try:
        # Environment configuration
        config = {
            "congress_api_key": os.environ.get("CONGRESS_API_KEY"),
            "market_sectors": os.environ.get("MARKET_SECTORS", "SPY,XLI,XLE,XLF").split(
                ","
            ),
            "s3_raw_bucket": os.environ.get("S3_RAW_BUCKET"),
            "s3_processed_bucket": os.environ.get("S3_PROCESSED_BUCKET"),
        }

        # Validate configuration
        required_vars = ["congress_api_key", "s3_raw_bucket", "s3_processed_bucket"]
        missing_vars = [var for var in required_vars if not config.get(var)]

        if missing_vars:
            error_msg = f"Missing required environment variables: {missing_vars}"
            logger.error(error_msg)
            return {"statusCode": 500, "body": error_msg}

        # Initialize services
        labeling_service = create_labeling_service(config)
        data_processor = DataProcessor()

        # Process data
        logger.info("Starting daily labeling job...")

        # In a real implementation, you would load actual data from S3
        # df_raw = data_processor.load_from_s3(config["s3_raw_bucket"], "raw_data.parquet")

        # For this example, using sample data
        sample_data = []  # Would be populated with actual data
        df_raw = pd.DataFrame(sample_data)

        if df_raw.empty:
            logger.warning("No raw data found to process")
            return {"statusCode": 200, "body": "No data to process"}

        # Process each press release
        labeled_results = []
        for _, row in df_raw.iterrows():
            try:
                pub_date = datetime.fromisoformat(row["published_date"])
                result = labeling_service.label_press_release(
                    row["headline"], row["body"], pub_date
                )
                labeled_results.append(result.__dict__)
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue

        # Save results
        if labeled_results:
            df_labeled = pd.DataFrame(labeled_results)
            success = data_processor.save_to_s3(
                df_labeled,
                config["s3_processed_bucket"],
                f"processed_data_{datetime.now().strftime('%Y%m%d')}.parquet",
            )

            if success:
                message = f"Successfully labeled {len(labeled_results)} press releases"
                logger.info(message)
                return {"statusCode": 200, "body": message}
            else:
                return {"statusCode": 500, "body": "Failed to save results"}
        else:
            return {"statusCode": 200, "body": "No results to save"}

    except Exception as e:
        error_msg = f"Lambda execution failed: {str(e)}"
        logger.error(error_msg)
        return {"statusCode": 500, "body": error_msg}


if __name__ == "__main__":
    main()
