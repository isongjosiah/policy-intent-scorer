import sys
import os
import asyncio
import aiohttp
from typing import List, Dict, Optional, Tuple, Any

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import settings
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import re
import math
import logging
from collections import Counter
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


@dataclass
class Bill:
    """Data class for a bill from the Congress.gov API."""

    title: str
    url: str
    latest_action_date: Optional[datetime] = None
    status: Optional[str] = None


class AsyncKeywordExtractor:
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

    async def preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text asynchronously."""
        if not text or not isinstance(text, str):
            return []

        # This could be CPU-intensive for very large texts, so we make it async
        await asyncio.sleep(0)  # Yield control

        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", " ", text)
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

    async def extract_by_frequency(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, int]]:
        """Extract keywords based on word frequency."""
        words = await self.preprocess_text(text)
        if not words:
            return []
        return Counter(words).most_common(top_k)

    async def extract_by_tfidf(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF scoring."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", text.strip()) if s.strip()]

        if len(sentences) < 2:
            freq_results = await self.extract_by_frequency(text, top_k)
            return [(word, float(freq)) for word, freq in freq_results]

        # Process each sentence asynchronously
        sentence_words_tasks = [
            self.preprocess_text(sentence) for sentence in sentences
        ]
        sentence_words = await asyncio.gather(*sentence_words_tasks)

        all_words = set(word for words in sentence_words for word in words)

        if not all_words:
            return []

        await asyncio.sleep(0)  # Yield control for CPU-intensive operation

        word_tfidf = {}
        total_sentences = len(sentence_words)

        for word in all_words:
            total_tf = sum(words.count(word) for words in sentence_words)
            total_words = sum(len(words) for words in sentence_words)
            tf = total_tf / max(total_words, 1)

            doc_count = sum(1 for words in sentence_words if word in words)
            idf = math.log(total_sentences / max(doc_count, 1))

            word_tfidf[word] = tf * idf

        return sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    async def extract_compound_keywords(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Extract compound keywords by combining adjacent important words."""
        words = await self.preprocess_text(text)
        if len(words) < 2:
            return []

        tfidf_scores = dict(await self.extract_by_tfidf(text, len(set(words))))

        compound_scores = {}
        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]
            score1 = tfidf_scores.get(word1, 0)
            score2 = tfidf_scores.get(word2, 0)

            if score1 > 0 and score2 > 0:
                compound = f"{word1} {word2}"
                compound_scores[compound] = (score1 + score2) / 2

        return sorted(compound_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    async def get_text_statistics(self, text: str) -> TextStats:
        """Get basic text statistics."""
        if not text or not isinstance(text, str):
            return TextStats(0, 0, 0, 0, 0, 0.0)

        words = await self.preprocess_text(text)
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


class AsyncCongressAPIClient:
    """Async client for Congress.gov API interactions."""

    def __init__(self, api_key: str, timeout: int = 10, max_concurrent: int = 5):
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.base_url = "https://api.congress.gov/v3"
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def search_bills(
        self,
        session: aiohttp.ClientSession,
        query: str,
        limit: int = 20,
        offset: int = 0,
        sort: str = "relevance",
    ) -> List[Bill]:
        """Search for bills using the Congress.gov API."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                url = f"{self.base_url}/bill"
                params = {
                    "query": query,
                    "api_key": self.api_key,
                    "limit": limit,
                    "offset": offset,
                    "sort": sort,
                    "format": "json",
                }

                async with session.get(
                    url, params=params, timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    bills = []
                    for item in data.get("bills", []):
                        bills.append(
                            Bill(
                                title=item.get("title"),
                                url=item.get("url"),
                                latest_action_date=self._to_datetime(
                                    item.get("latestAction", {}).get("actionDate")
                                ),
                                status=item.get("status"),
                            )
                        )
                    return bills

            except aiohttp.ClientError as e:
                logger.error(f"HTTP error searching Congress API for '{query}': {e}")
                return []
            except (ValueError, KeyError) as e:
                logger.error(f"Error parsing Congress API response for '{query}': {e}")
                return []

    def _to_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


class AsyncMarketDataClient:
    """Async client for market data interactions."""

    def __init__(self, cache_dir: Optional[str] = None, max_concurrent: int = 3):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def get_stock_data(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data for a given ticker and date range."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Run yfinance download in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        ticker,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                    ),
                )

                if df.empty:
                    logger.warning(f"No data found for ticker {ticker}")
                    return None

                return df

            except Exception as e:
                logger.warning(f"Error fetching market data for {ticker}: {e}")
                return None

    async def calculate_volatility_spike(
        self,
        ticker: str,
        pub_date: datetime,
        lookback_days: int = 30,
        threshold: float = 2.0,
    ) -> Tuple[bool, float]:
        """Calculate if there was a significant volatility spike."""
        if pub_date.weekday() > 4:  # Skip weekends
            return False, 0.0

        start_date = pub_date - timedelta(days=lookback_days + 5)
        end_date = pub_date + timedelta(days=2)

        df = await self.get_stock_data(ticker, start_date, end_date)
        if df is None or len(df) < lookback_days:
            return False, 0.0

        try:
            daily_returns = df["Close"].pct_change().dropna()

            if len(daily_returns) < lookback_days:
                return False, 0.0

            pub_date_str = pub_date.strftime("%Y-%m-%d")
            if pub_date_str not in daily_returns.index:
                available_dates = daily_returns.index[
                    daily_returns.index >= pub_date_str
                ]
                if len(available_dates) == 0:
                    return False, 0.0
                pub_date_str = available_dates[0]

            day_return = daily_returns.loc[pub_date_str]
            historical_returns = daily_returns.loc[:pub_date_str].iloc[:-1]

            if len(historical_returns) < lookback_days:
                return False, 0.0

            rolling_std = historical_returns.tail(lookback_days).std()

            if rolling_std == 0:
                return False, 0.0

            z_score = abs(day_return) / rolling_std
            return z_score >= threshold, z_score

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating volatility for {ticker}: {e}")
            return False, 0.0


class AsyncOutcomeChecker(ABC):
    """Abstract base class for async outcome checking."""

    @abstractmethod
    async def check_outcome(
        self, session: aiohttp.ClientSession, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check for outcome. Returns (found, url/info, days_to_outcome)."""
        pass


class AsyncCongressOutcomeChecker(AsyncOutcomeChecker):
    """Checks for legislative outcomes using Congress.gov API."""

    def __init__(
        self,
        congress_client: AsyncCongressAPIClient,
        keyword_extractor: AsyncKeywordExtractor,
    ):
        self.congress_client = congress_client
        self.keyword_extractor = keyword_extractor

    async def _extract_policy_keywords(
        self, text: str, max_keywords: int = 5
    ) -> List[str]:
        """Extract relevant policy keywords from text."""
        keywords = await self.keyword_extractor.extract_by_tfidf(text, max_keywords * 2)

        policy_terms = []
        for word, score in keywords:
            if (
                len(word) > 3
                and word not in ["president", "administration", "government", "federal"]
                and not word.isdigit()
            ):
                policy_terms.append(word)

        return policy_terms[:max_keywords]

    async def check_outcome(
        self, session: aiohttp.ClientSession, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check if related legislation became law within 180 days."""
        search_terms = await self._extract_policy_keywords(text)

        # Search for bills concurrently
        bill_search_tasks = [
            self.congress_client.search_bills(session, term) for term in search_terms
        ]

        all_bills_results = await asyncio.gather(
            *bill_search_tasks, return_exceptions=True
        )

        for bills_result in all_bills_results:
            if isinstance(bills_result, Exception):
                logger.warning(f"Error in bill search: {bills_result}")
                continue

            for bill in bills_result:
                if bill.status == "Became Law" and bill.latest_action_date:
                    days_diff = (bill.latest_action_date - pub_date).days
                    if 0 < days_diff <= 180:
                        return True, bill.url, days_diff

        return False, None, None


class AsyncMarketOutcomeChecker(AsyncOutcomeChecker):
    """Checks for market volatility outcomes."""

    def __init__(self, market_client: AsyncMarketDataClient, sectors: List[str]):
        self.market_client = market_client
        self.sectors = sectors

    async def check_outcome(
        self, session: aiohttp.ClientSession, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check for significant market volatility spike."""
        # Check all sectors concurrently
        volatility_tasks = [
            self.market_client.calculate_volatility_spike(ticker, pub_date)
            for ticker in self.sectors
        ]

        results = await asyncio.gather(*volatility_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Error checking volatility for {self.sectors[i]}: {result}"
                )
                continue

            is_spike, z_score = result
            if is_spike:
                ticker = self.sectors[i]
                info = f"Market volatility spike for {ticker} (Z-score: {z_score:.2f})"
                logger.info(f"Market spike detected: {info}")
                return True, info, 1

        return False, None, None


class AsyncPressReleaseLabelingService:
    """Main async service for labeling press releases."""

    def __init__(
        self, outcome_checkers: List[AsyncOutcomeChecker], max_concurrent: int = 10
    ):
        self.outcome_checkers = outcome_checkers
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def label_press_release(
        self,
        session: aiohttp.ClientSession,
        headline: str,
        body: str,
        pub_date: datetime,
    ) -> LabelingResult:
        """Label a press release as Actionable or Bluff based on outcomes."""
        async with self.semaphore:
            label = "Bluff"
            match_type = "None"
            match_url = None
            days_to_outcome = None

            full_text = f"{headline} {body}"

            # Check each outcome type concurrently
            outcome_tasks = [
                checker.check_outcome(session, full_text, pub_date)
                for checker in self.outcome_checkers
            ]

            results = await asyncio.gather(*outcome_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Error checking outcome with {type(self.outcome_checkers[i]).__name__}: {result}"
                    )
                    continue

                found, url, days = result
                if found:
                    label = "Actionable"
                    checker = self.outcome_checkers[i]
                    if isinstance(checker, AsyncCongressOutcomeChecker):
                        match_type = "Law Passed"
                    elif isinstance(checker, AsyncMarketOutcomeChecker):
                        match_type = "Market Moved"
                    else:
                        match_type = "Outcome Found"

                    match_url = url
                    days_to_outcome = days
                    break  # Use first found outcome

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

    async def process_batch(
        self, data_batch: List[Dict[str, Any]], batch_size: int = 50
    ) -> List[LabelingResult]:
        """Process a batch of press releases concurrently."""
        results = []

        async with aiohttp.ClientSession() as session:
            # Process in smaller chunks to avoid overwhelming APIs
            for i in range(0, len(data_batch), batch_size):
                chunk = data_batch[i : i + batch_size]

                tasks = []
                for row in chunk:
                    try:
                        if isinstance(row["date"], str):
                            pub_date = datetime.fromisoformat(row["date"])
                        else:
                            pub_date = row["date"]

                        task = self.label_press_release(
                            session, row["headline"], row["content"], pub_date
                        )
                        tasks.append(task)
                    except Exception as e:
                        logger.error(f"Error preparing task for row: {e}")
                        continue

                # Process chunk concurrently
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in chunk_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing press release: {result}")
                        continue
                    results.append(result)

                logger.info(
                    f"Processed {len(results)} / {len(data_batch)} press releases"
                )

                # Small delay between chunks to be respectful to APIs
                if i + batch_size < len(data_batch):
                    await asyncio.sleep(1)

        return results


class AsyncDataProcessor:
    """Handles async data loading and saving operations."""

    def __init__(self, s3_client: Optional[Any] = None):
        self.s3_client = s3_client

    async def load_from_local(self, file_path: str) -> pd.DataFrame:
        """Load data from local file asynchronously."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return pd.DataFrame()

            # Run pandas operations in thread pool
            loop = asyncio.get_event_loop()

            if file_path.endswith(".parquet"):
                df = await loop.run_in_executor(None, pd.read_parquet, file_path)
            elif file_path.endswith(".csv"):
                df = await loop.run_in_executor(None, pd.read_csv, file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading from local file: {e}")
            return pd.DataFrame()

    async def save_to_local(self, df: pd.DataFrame, file_path: str) -> bool:
        """Save dataframe to local file asynchronously."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Run pandas operations in thread pool
            loop = asyncio.get_event_loop()

            if file_path.endswith(".parquet"):
                await loop.run_in_executor(
                    None, lambda: df.to_parquet(file_path, index=False)
                )
            elif file_path.endswith(".csv"):
                await loop.run_in_executor(
                    None, lambda: df.to_csv(file_path, index=False)
                )
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Saved {len(df)} records to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving to local file: {e}")
            return False


def create_async_labeling_service(
    config: Dict[str, Any],
) -> AsyncPressReleaseLabelingService:
    """Factory function to create async labeling service from configuration."""
    keyword_extractor = AsyncKeywordExtractor()
    outcome_checkers = []

    # Congress checker
    if config.get("congress_api_key"):
        congress_client = AsyncCongressAPIClient(config["congress_api_key"])
        congress_checker = AsyncCongressOutcomeChecker(
            congress_client, keyword_extractor
        )
        outcome_checkers.append(congress_checker)

    # Market checker
    if config.get("market_sectors"):
        market_client = AsyncMarketDataClient()
        market_checker = AsyncMarketOutcomeChecker(
            market_client, config["market_sectors"]
        )
        outcome_checkers.append(market_checker)

    return AsyncPressReleaseLabelingService(outcome_checkers)


async def main():
    """Main async function for local execution."""
    config = {
        "congress_api_key": settings.CONGRESS_API_KEY,
        "market_sectors": settings.MARKET_SECTORS,
        "input_file": settings.INPUT_FILE,
        "output_file": settings.OUTPUT_FILE,
    }

    if not config["congress_api_key"]:
        logger.error("CONGRESS_API_KEY environment variable is required")
        return

    # Initialize services
    labeling_service = create_async_labeling_service(config)
    data_processor = AsyncDataProcessor()

    # Load data
    df_raw = await data_processor.load_from_local(config["input_file"])
    if df_raw.empty:
        logger.warning("No input data found. Creating sample data for testing.")
        sample_data = [
            {
                "headline": "President Announces New Climate Initiative",
                "content": "The President is proud to announce a bold new climate initiative...",
                "date": "2025-01-01",
            }
        ]
        df_raw = pd.DataFrame(sample_data)

    # Convert dataframe to list of dicts for processing
    data_batch = df_raw.to_dict("records")

    # Process data asynchronously
    logger.info(f"Starting async processing of {len(data_batch)} press releases...")
    start_time = datetime.now()

    labeled_results = await labeling_service.process_batch(data_batch, batch_size=10)

    end_time = datetime.now()
    logger.info(
        f"Completed processing in {(end_time - start_time).total_seconds():.2f} seconds"
    )

    # Save results
    if labeled_results:
        results_dicts = [result.__dict__ for result in labeled_results]
        df_labeled = pd.DataFrame(results_dicts)
        success = await data_processor.save_to_local(df_labeled, config["output_file"])

        if success:
            logger.info(
                f"✓ Successfully processed {len(labeled_results)} press releases"
            )
        else:
            logger.error("❌ Failed to save results")
    else:
        logger.warning("No results to save")


async def async_handler(event, context):
    """AWS Lambda async entry point."""
    try:
        config = {
            "congress_api_key": settings.CONGRESS_API_KEY,
            "market_sectors": settings.MARKET_SECTORS,
            "s3_raw_bucket": settings.S3_BUCKET,
            "s3_processed_bucket": settings.S3_BUCKET,
        }

        required_vars = ["congress_api_key", "s3_raw_bucket", "s3_processed_bucket"]
        missing_vars = [var for var in required_vars if not config.get(var)]

        if missing_vars:
            error_msg = f"Missing required environment variables: {missing_vars}"
            logger.error(error_msg)
            return {"statusCode": 500, "body": error_msg}

        # Initialize services
        labeling_service = create_async_labeling_service(config)
        data_processor = AsyncDataProcessor()

        logger.info("Starting async daily labeling job...")

        # In production, load actual data from S3
        sample_data = []  # Would be populated with actual data

        if not sample_data:
            logger.warning("No raw data found to process")
            return {"statusCode": 200, "body": "No data to process"}

        # Process data asynchronously
        labeled_results = await labeling_service.process_batch(sample_data)

        if labeled_results:
            results_dicts = [result.__dict__ for result in labeled_results]
            df_labeled = pd.DataFrame(results_dicts)

            # Save to S3 (would need async S3 implementation)
            message = f"Successfully labeled {len(labeled_results)} press releases"
            logger.info(message)
            return {"statusCode": 200, "body": message}
        else:
            return {"statusCode": 200, "body": "No results to save"}

    except Exception as e:
        error_msg = f"Lambda execution failed: {str(e)}"
        logger.error(error_msg)
        return {"statusCode": 500, "body": error_msg}


def handler(event, context):
    """AWS Lambda entry point that wraps async handler."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_handler(event, context))
    finally:
        loop.close()


if __name__ == "__main__":
    asyncio.run(main())
