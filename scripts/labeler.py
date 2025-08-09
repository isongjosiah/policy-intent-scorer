import sys
import os
import asyncio
import aiohttp
import pickle
from typing import List, Dict, Optional, Tuple, Any


# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.config import settings
import yfinance as yf
import pandas as pd
from datetime import timezone, datetime, timedelta
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

    def _split_sentences_with_abbreviations(self, text: str) -> List[str]:
        """
        Splits text into sentences, but first protects abbreviations to prevent
        them from being split incorrectly.
        """
        # 1. Define a list of abbreviations to protect.
        # Add more as needed, e.g., 'etc.', 'e.g.', 'vs.'
        abbreviations = ["H.R.", "S.", "A.M.", "P.M.", "Dr.", "Mr.", "Mrs.", "Ms."]

        # 2. Create a regex pattern to find these abbreviations.
        # We use a word boundary (\b) to ensure we match the full word.
        abbr_pattern = "|".join(
            map(re.escape, sorted(abbreviations, key=len, reverse=True))
        )

        # 3. Create a unique placeholder to temporarily replace the abbreviations.
        placeholder = "__ABBR_PLACEHOLDER__"

        # 4. Temporarily replace the abbreviations with the placeholder.
        # This prevents the primary split from breaking up H.R. etc.
        text_with_placeholders = re.sub(
            r"\b(" + abbr_pattern + r")",
            lambda m: m.group(0).replace(".", placeholder),
            text,
        )

        # 5. Split the text into sentences using the standard punctuation.
        sentences = re.split(r"[.!?]+", text_with_placeholders)

        # 6. Replace the placeholder back with a period in the resulting sentences.
        cleaned_sentences = [
            s.replace(placeholder, ".") for s in sentences if s.strip()
        ]

        return cleaned_sentences

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
        sentences = self._split_sentences_with_abbreviations(text)

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
        bill_type: str,
        bill_number: str,
        congress: str,
        limit: int = 20,
        offset: int = 0,
        sort: str = "relevance",
    ) -> Optional[Bill]:
        """Search for bills using the Congress.gov API."""
        async with self.semaphore:  # Limit concurrent requests
            bill: Optional[Bill] = None
            try:
                url = f"{self.base_url}/bill/{congress}/{bill_type}/{bill_number}"
                params = {
                    "query": "",
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
                    bill_data = data.get("bill")

                    bill = Bill(
                        title=bill_data.get("title"),
                        url="",
                        latest_action_date=self._to_datetime(
                            bill_data.get("latestAction", {}).get("actionDate", "")
                        ),
                        status=(
                            "Became Law"
                            if len(bill_data.get("laws", [])) > 0
                            else "Not Law"
                        ),
                    )
                    return bill

            except aiohttp.ClientError as e:
                logger.error(
                    f"HTTP error searching Congress API for '{bill_type}/{bill_number}': {e}"
                )
                return bill
            except (ValueError, KeyError) as e:
                logger.error(
                    f"Error parsing Congress API response for '{bill_type}/{bill_number}': {e}"
                )
                return bill

    def _to_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(
                tzinfo=timezone(timedelta(hours=-4))
            )
        except (ValueError, TypeError):
            return None


class AsyncCachedMarketDataClient:
    """Async client for market data interactions with local caching."""

    def __init__(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        cache_dir: Optional[str] = None,
        max_concurrent: int = 3,
        force_refresh: bool = False,
    ):
        """
        Initialize client and download all required data upfront.

        Args:
            tickers: List of stock tickers to download
            start_date: Earliest date needed for analysis
            end_date: Latest date needed for analysis
            cache_dir: Directory to store cached data
            max_concurrent: Max concurrent requests (unused now, kept for compatibility)
            force_refresh: Force re-download even if cache exists
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./market_data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_count = 0

        # Cache file path
        self.cache_file = (
            self.cache_dir
            / f"market_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
        )

        # Initialize data storage
        self.data_cache = {}

        # Load or download data
        asyncio.create_task(self._initialize_data(force_refresh))

    async def _initialize_data(self, force_refresh: bool = False):
        """Download and cache all required market data."""
        if not force_refresh and self.cache_file.exists():
            logger.info(f"Loading cached data from {self.cache_file}")
            try:
                with open(self.cache_file, "rb") as f:
                    self.data_cache = pickle.load(f)
                logger.info(
                    f"Loaded data for {len(self.data_cache)} tickers from cache"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Will re-download data.")

        logger.info(
            f"Downloading market data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}"
        )

        # Download data in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(self.tickers), batch_size):
            batch_tickers = self.tickers[i : i + batch_size]
            logger.info(f"Downloading batch {i//batch_size + 1}: {batch_tickers}")

            try:
                # Run yfinance download in thread pool
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        batch_tickers,
                        start=self.start_date.strftime("%Y-%m-%d"),
                        end=self.end_date.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                        group_by="ticker" if len(batch_tickers) > 1 else None,
                    ),
                )

                if not df.empty:
                    # Store data for each ticker
                    if len(batch_tickers) == 1:
                        # Single ticker case
                        ticker = batch_tickers[0]
                        self.data_cache[ticker] = df
                    else:
                        # Multiple tickers case
                        for ticker in batch_tickers:
                            if ticker in df.columns.get_level_values(0):
                                self.data_cache[ticker] = df[ticker]
                            else:
                                logger.warning(f"No data found for ticker {ticker}")
                else:
                    logger.warning(f"No data returned for batch: {batch_tickers}")

                # Add delay between batches to be respectful to the API
                if i + batch_size < len(self.tickers):
                    await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error downloading batch {batch_tickers}: {e}")

        # Save cache to disk
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.data_cache, f)
            logger.info(f"Cached data saved to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

        logger.info(
            f"Data initialization complete. Cached data for {len(self.data_cache)} tickers"
        )

    async def get_stock_data(
        self, tickers: List[str], start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data from local cache for given tickers and date range."""

        # Wait for initialization to complete if still in progress
        while not hasattr(self, "data_cache") or not self.data_cache:
            await asyncio.sleep(0.1)

        try:
            # If single ticker, return that ticker's data
            if len(tickers) == 1:
                ticker = tickers[0]
                if ticker not in self.data_cache:
                    logger.warning(f"No cached data found for ticker {ticker}")
                    return None

                df = self.data_cache[ticker].copy()
                print("df is")
                print(df)

                # Filter by date range
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                # Filter the dataframe by date range
                mask = (df.index >= start_str) & (df.index <= end_str)
                filtered_df = df.loc[mask]
                print("df is")
                print(filtered_df)

                if filtered_df.empty:
                    logger.warning(
                        f"No data found for ticker {ticker} in date range {start_str} to {end_str}"
                    )
                    return None

                return filtered_df

            else:
                # Multiple tickers - combine data
                combined_data = {}
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                for ticker in tickers:
                    if ticker not in self.data_cache:
                        logger.warning(f"No cached data found for ticker {ticker}")
                        continue

                    df = self.data_cache[ticker].copy()

                    # Filter by date range
                    mask = (df.index >= start_str) & (df.index <= end_str)
                    filtered_df = df.loc[mask]

                    if not filtered_df.empty:
                        combined_data[ticker] = filtered_df

                if not combined_data:
                    logger.warning(f"No data found for any tickers in date range")
                    return None

                # Create multi-level column structure similar to yfinance
                result = pd.concat(combined_data, axis=1, names=["Ticker", "Price"])

                # Reorder columns to match yfinance format (Price, Ticker)
                if len(tickers) > 1:
                    result = result.swaplevel(0, 1, axis=1).sort_index(axis=1)

                return result

        except Exception as e:
            logger.error(f"Error retrieving cached data for {tickers}: {e}")
            return None

    async def calculate_volatility_spike(
        self,
        tickers: List[str],
        pub_date: datetime,
        lookback_days: int = settings.LOOKBACK_DAYS,
        threshold: float = 2.0,
    ) -> Tuple[bool, float, str]:
        """Calculate if there was a significant volatility spike."""

        start_date = pub_date - timedelta(days=lookback_days)
        end_date = pub_date + timedelta(days=0)

        df = await self.get_stock_data(tickers, start_date, end_date)
        if df is None:
            return False, 0.0, ""

        if df is None:
            return False, 0.0, ""

        try:
            # Handle both single and multiple ticker cases
            if len(tickers) == 1:
                close_prices = df["Close"]
            else:
                close_prices = (
                    df["Close"]
                    if "Close" in df.columns
                    else df.xs("Close", level=1, axis=1)
                )

            daily_returns = close_prices.pct_change(fill_method=None).fillna(0)

            pub_date_str = pub_date.strftime("%Y-%m-%d")
            if pub_date_str not in daily_returns.index:
                available_dates = daily_returns.index[
                    daily_returns.index >= pub_date_str
                ]
                if len(available_dates) == 0:
                    return False, 0.0, ""
                pub_date_str = available_dates[0]

            day_return = daily_returns.loc[pub_date_str]
            historical_returns = daily_returns.loc[:pub_date_str].iloc[:-1]

            rolling_std = historical_returns.tail(lookback_days).std()

            z_score = pd.Series(
                0.0, index=day_return.index if hasattr(day_return, "index") else [0]
            )

            # Create a boolean mask to find tickers where rolling_std is not 0
            non_zero_std_mask = rolling_std != 0

            # Perform the z-score calculation only where the rolling_std is not 0
            z_score.loc[non_zero_std_mask] = abs(day_return) / rolling_std

            # Check against the threshold
            is_spike = (
                any(z_score >= threshold)
                if hasattr(z_score, "__iter__")
                else z_score >= threshold
            )
            max_z = z_score.max() if hasattr(z_score, "max") else z_score
            max_ticker = (
                str(z_score.idxmax()) if hasattr(z_score, "idxmax") else tickers[0]
            )

            return is_spike, max_z, max_ticker

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating volatility for {tickers}: {e}")
            return False, 0.0, ""

    def clear_cache(self):
        """Clear the cached data file."""
        if self.cache_file.exists():
            self.cache_file.unlink()
            logger.info(f"Cleared cache file: {self.cache_file}")

        self.data_cache = {}

    def get_cache_info(self):
        """Get information about cached data."""
        if not self.data_cache:
            return "No data cached"

        info = {
            "tickers_count": len(self.data_cache),
            "tickers": list(self.data_cache.keys()),
            "date_range": f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            "cache_file": str(self.cache_file),
            "cache_file_exists": self.cache_file.exists(),
        }

        return info


class AsyncMarketDataClient:
    """Async client for market data interactions."""

    def __init__(self, cache_dir: Optional[str] = None, max_concurrent: int = 3):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_count = 0

    async def get_stock_data(
        self, tickers: List[str], start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data for a given ticker and date range."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Run yfinance download in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None,
                    lambda: yf.download(
                        tickers,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                    ),
                )
                self.request_count += 1

                if df.empty:
                    logger.warning(f"No data found for ticker {tickers}")
                    return None

                return df

            except Exception as e:
                logger.warning(f"Error fetching market data for {tickers}: {e}")
                return None
            finally:
                # Check if the counter has reached 100 and apply a longer delay
                if self.request_count % 90 == 0 and self.request_count > 0:
                    logger.info(
                        f"Made {self.request_count} requests. Pausing for a longer delay."
                    )
                    await asyncio.sleep(
                        60
                    )  # Pause for 60 seconds after every 100 requests
                else:
                    # Regular, short delay to prevent a flood of requests
                    await asyncio.sleep(1)

    async def calculate_volatility_spike(
        self,
        tickers: List[str],
        pub_date: datetime,
        lookback_days: int = settings.LOOKBACK_DAYS,
        threshold: float = 2.0,
    ) -> Tuple[bool, float, str]:
        """Calculate if there was a significant volatility spike."""
        if pub_date.weekday() > 4:  # Skip weekends
            return False, 0.0, ""

        start_date = pub_date - timedelta(days=lookback_days + 5)
        end_date = pub_date + timedelta(days=2)

        df = await self.get_stock_data(tickers, start_date, end_date)
        if df is None or len(df) < lookback_days:
            return False, 0.0, ""

        try:
            daily_returns = df["Close"].pct_change(fill_method=None).fillna(0)

            if len(daily_returns) < lookback_days:
                return False, 0.0, ""

            pub_date_str = pub_date.strftime("%Y-%m-%d")
            if pub_date_str not in daily_returns.index:
                available_dates = daily_returns.index[
                    daily_returns.index >= pub_date_str
                ]
                if len(available_dates) == 0:
                    return False, 0.0, ""
                pub_date_str = available_dates[0]

            day_return = daily_returns.loc[pub_date_str]
            historical_returns = daily_returns.loc[:pub_date_str].iloc[:-1]

            if len(historical_returns) < lookback_days:
                return False, 0.0, ""

            rolling_std = historical_returns.tail(lookback_days).std()

            z_score = pd.Series(0.0, index=day_return.index)

            # Create a boolean mask to find tickers where rolling_std is not 0
            non_zero_std_mask = rolling_std != 0

            # Perform the z-score calculation only where the rolling_std is not 0
            z_score.loc[non_zero_std_mask] = abs(day_return) / rolling_std

            # Check against the threshold
            is_spike = any(z_score >= threshold)
            return is_spike, z_score.max(), str(z_score.idxmax())

        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"Error calculating volatility for {ticker}: {e}")
            return False, 0.0, ""


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

    async def _extract_bill_numbers(self, text: str) -> List[str]:
        """Extract relevant policy keywords from text."""
        # NOTE: at this point, the thing of importance to us is the bill numbers
        bill_regex_pattern = r"(\b(?:[a-zA-Z]+\.)+)\s?(\d+\b)"
        return list(set(re.findall(bill_regex_pattern, text)))

    async def _get_congress_number_from_date(self, pub_date: datetime) -> str:
        return str(int(((pub_date.year - 1789) / 2) + 1))

    async def check_outcome(
        self, session: aiohttp.ClientSession, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check if related legislation became law within 180 days."""
        bill_numbers = await self._extract_bill_numbers(text)

        # Search for bills concurrently
        bill_search_tasks = [
            self.congress_client.search_bills(
                session,
                numbers[0].lower().replace(".", ""),
                numbers[1],
                await self._get_congress_number_from_date(pub_date),
            )
            for numbers in bill_numbers
            if numbers[0].lower().replace(".", "")
            in ["hr", "s", "hjres", "sjres", "hconres", "sconres", "hres", "sres"]
        ]

        all_bills_results = await asyncio.gather(
            *bill_search_tasks, return_exceptions=True
        )

        for bill_result in all_bills_results:
            if isinstance(bill_result, Exception):
                logger.warning(f"Error in bill search: {bill_result}")
                continue

            if bill_result is None:
                logger.warning(f"Error in bill search: {bill_result}")
                continue

            if bill_result.status == "Became Law" and bill_result.latest_action_date:
                days_diff = (bill_result.latest_action_date - pub_date).days
                if days_diff <= 180:
                    return True, bill_result.url, days_diff

        return False, None, None


class AsyncMarketOutcomeChecker(AsyncOutcomeChecker):
    """Checks for market volatility outcomes."""

    def __init__(self, market_client: AsyncCachedMarketDataClient, sectors: List[str]):
        self.market_client = market_client
        self.sectors = sectors

    async def check_outcome(
        self, session: aiohttp.ClientSession, text: str, pub_date: datetime
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """Check for significant market volatility spike."""

        # Fetch volatility for all sectors in a single asynchronous call
        try:
            is_spike, z_score, ticker = (
                await self.market_client.calculate_volatility_spike(
                    self.sectors, pub_date
                )
            )

            if is_spike:
                info = f"Market volatility spike for {ticker} (Z-score: {z_score:.2f})"
                logger.info(f"Market spike detected: {info}")
                return True, info, 1

        except Exception as e:
            logger.warning(f"Error checking volatility for all sectors: {e}")
            # In case of an error, we don't consider it a spike
            return False, None, None

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
    config: Dict[str, Any], start: datetime, end: datetime
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
        market_client = AsyncCachedMarketDataClient(
            tickers=settings.MARKET_SECTORS, start_date=start, end_date=end
        )
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
    data_processor = AsyncDataProcessor()

    # Load data
    df_raw: pd.DataFrame = await data_processor.load_from_local(config["input_file"])
    start: datetime = df_raw["date"].min() - timedelta(days=35)
    end: datetime = df_raw["date"].max() + timedelta(days=1)
    labeling_service = create_async_labeling_service(config, start=start, end=end)
    if df_raw.empty:
        logger.warning("No input data found. Creating sample data for testing.")
        sample_data = [
            {
                "content": "Office of the First Lady\t\t\t\t\n\n\t\t\t\t\tFirst Lady Melania Trump Celebrates Committee Passage of Take It Down Act in House Energy & Commerce\t\t\t\t\n\n\n\n\tThe White House\n\n\nApril 9, 2025 \n\n\n\n\nWASHINGTON, D.C. – First Lady Melania Trump celebrated committee passage of H.R. 633, the Take It Down Act, in the House of Representatives Committee on Energy and Commerce on Tuesday, April 8, 2025. The bill advanced with broad bipartisan support.\nIn an official statement shared with the House Energy and Commerce Committee and on social media, the First Lady said, “I remain dedicated to championing child well-being, ensuring that every young person can thrive and ‘Be Best.’ Thank you to the House Energy & Commerce Committee for advancing the Take It Down Act. This marks a significant step in our bipartisan efforts to safeguard our children from online threats. I urge Congress to swiftly pass this important legislation. Together, we can create a safer, brighter future for all Americans!”\nThe Take It Down Act would protect children who are targeted by the publication of non-consensual intimate imagery (NCII) online, require internet platforms—including social media platforms—to remove such imagery within 48 hours of notice from a victim, and provide justice for survivors by criminalizing the publication of, or threat to publish, NCII.Read the House Energy and Commerce Committee press release here or below.\n\nApril 8, 2025\nChairman Guthrie, First Lady Melania Trump, Chairman Bilirakis Join Advocates in Celebrating Committee Passage of TAKE IT DOWN Act\nWASHINGTON, D.C. – Today, Congressman Brett Guthrie (KY-02), Chairman of the House Committee on Energy and Commerce, along with advocates for the TAKE IT DOWN Act, issued the following statements of support after the bill was reported out of Committee by a vote of 49 to 1.\n“No man, woman, or child should be subjected to the spread of explicit AI images meant to target and harass innocent victims. I am so thankful for our outstanding advocates and legislators who have worked hard to raise awareness and build a strong coalition to support this bipartisan bill,” said Chairman Guthrie. “Today, the Committee on Energy and Commerce advanced the bill to the full House of Representatives, where I look forward to, once again, voting in favor of the TAKE IT DOWN Act, so that we can send it to the President’s desk for signature.”\n“I remain dedicated to championing child well-being, ensuring that every young person can thrive and ‘Be Best.’ Thank you to the House Energy & Commerce Committee for advancing the TAKE IT DOWN Act. This marks a significant step in our bipartisan efforts to safeguard our children from online threats,” said First Lady Melania Trump. “I urge Congress to swiftly pass this important legislation. Together, we can create a safer, brighter future for all Americans!”\n“I am glad we are one step closer to protecting victims of online sexual exploitation. Giving victims rights to flag non-consensual images and requiring social media companies to remove that content quickly is a pivotal and necessary change to the online landscape,” said Congressman Gus Bilirakis (FL-12), Chairman of the Subcommittee on Commerce, Manufacturing, and Trade. “And by ensuring that AI-generated deep-fake content is included in these protections, Congress is showing its commitment to fighting 21st Century harms that are plaguing our children and grandchildren.”\n“In February, our family mourned the loss of our loving son and brother, Elijah Heacock, after he fell victim to an extortion scheme on the internet,” said Shannon Cronister-Heacock, mother of Elijah Heacock. “We are grateful for the support of Chairman Guthrie and the House Committee on Energy and Commerce for passing the TAKE IT DOWN Act today to ensure that no parent, sibling, or loved one experiences a similar tragedy in the future. This bill honors Elijah’s life, and we are appreciative of Congress’ actions to protect children online and save lives.”\n“I was only fourteen years old when one of my classmates created deepfake, AI nudes of me and distributed them on social media. I was shocked, violated, and felt unsafe going to school. Thankfully, I was able to work with Senator Ted Cruz’s office to write the TAKE IT DOWN Act — and today is an important milestone towards that bill becoming law, so that no other girl has to go through what I went through without legal protections in place,” said Elliston Berry, survivor and advocate. “Thank you to Chairman Guthrie for prioritizing the TAKE IT DOWN Act for committee passage.”\n“At 14, for almost two years, I stood alone, advocating for AI deep fake laws to protect us after my school’s inaction and lack of accountability insulted my self-respect. This journey is dedicated to every woman and teenager who was told to stay silent and move on. It is also a testament to the courageous bipartisan leaders who stood beside me, proving that change is possible. Today, we celebrate a critical step towards the passage of the TAKE IT DOWN Act into federal law,” said Francesca Mani, AI victim turned advocate & TIME100 AI Most Influential Person. “A heartfelt thank you to Chairman Guthrie for standing with us and making swift committee passage possible. We are no longer alone.”\n“Today, we celebrate an important victory with House committee passage of the TAKE IT DOWN Act, a federal safeguard against non-consensual AI-generated intimate images,” said Dorota Mani, an educator, advocate, and mother. “This important legislation, which is now well on its way to the President’s desk, staunchly defends our women and children while preserving every American’s dignity and rights.” “Survivors—both minors and adults—deserve protection and justice. Every survivor should be able to report their abuse to law enforcement, have their abuse content removed fully and abusers should be found and held appropriately accountable. Image-based sexual abuse is sexual assault facilitated online. You cannot accidentally sexual assault someone offline and the same should be true for the online. The harms of all forms of image-based sexual abuse—including deepfake abuse—quickly follow that victim home, to school, to work and anywhere they try to exist after such a profound and public trauma,” said Andrea Powell, Co-Founder and Chief of Impact, Alecto AI. “Alecto AI supports the TAKE IT DOWN Act because we believe that in its passage, we will be getting closer to a world where young women and girls don’t have worry that being online means being targets of sexual violence. All survivors deserve protection and justice.”",
                "headline": "First Lady Melania Trump Celebrates Committee Passage of Take It Down Act in House Energy & Commerce",
                "link": "https://www.whitehouse.gov/briefings-statements/2025/04/first-lady-melania-trump-celebrates-committee-passage-of-take-it-down-act-in-house-energy-commerce/",
                "date": "2025-04-09T15:00:00-04:00",
            },
            {
                "content": "STATEMENT OF ADMINISTRATION POLICY H.R. 21 – Born-Alive Abortion Survivors Protection Act \t\t\t\t\n\n\n\n\tThe White House\n\n\nJanuary 23, 2025 \n\n\n\n\nEXECUTIVE OFFICE OF THE PRESIDENTOFFICE OF MANAGEMENT AND BUDGETWASHINGTON, D.C. 20503January 23, 2025(House)\nSTATEMENT OF ADMINISTRATION POLICY\nH.R. 21 – Born-Alive Abortion Survivors Protection Act(Rep. Ann Wagner, R-MO-2, and 151 cosponsors)\nThe Administration strongly supports H.R. 21 the Born-Alive Abortion Survivors Protection Act, and applauds the House for its efforts to protect the most vulnerable and prevent infanticide.\nCurrent law fails to provide adequate protections, including adequate requirements for the provision of medical care, for vulnerable newborns who survive an abortion attempt. If enacted, H.R. 21 would require any healthcare practitioner who is present at the time that such a child is born to exercise care to preserve the child’s life and health, and to ensure the child is immediately transported and admitted to a hospital. The bill would also require a healthcare practitioner, or hospital employee, to immediately report a violation of these requirements. H.R. 21 would establish a civil right of action for, and prevent criminal prosecution and penalties from being brought against, the mothers of such children.\nAs President Trump established through Executive Order 13952 of September 25, 2020, it is the policy of the United States to recognize the human dignity and inherent worth of every newborn or other infant child, regardless of prematurity or disability, and to ensure for each child due protection under the law.\nA baby that survives an abortion and is born alive into this world should be treated just like any other baby born alive. H.R. 21 would properly amend current law to ensure that the life of one baby is not treated as being more or less valuable than another.\nIf H.R. 21 were presented to the President in its current form, his advisors would recommend he sign it into law.",
                "headline": "Statement of Administration Policy H.R. 21 – Born-Alive Abortion Survivors Protection Act",
                "link": "https://www.whitehouse.gov/briefings-statements/2025/01/statement-of-administration-policy-h-r-21-born-alive-abortion-survivors-protection-act/",
                "date": "2025-01-23T15:54:10-05:00",
            },
            {
                "content": "H.R. 1815 Signed into Law\t\t\t\t\n\n\n\n\tThe White House\n\n\nJuly 30, 2025 \n\n\n\n\nOn Wednesday, July 30, 2025, the President signed into law: H.R. 1815, “VA Home Loan Program Reform Act”, which amends title 38, United States Code, to authorize the Secretary of Veterans Affairs to take certain actions in the case of a default on a home loan guaranteed by the Secretary, and for other purposes.",
                "headline": "H.R. 1815 Signed into Law",
                "link": "https://www.whitehouse.gov/briefings-statements/2025/07/h-r-1815-signed-into-law/",
                "date": "2025-07-30T14:48:31-04:00",
            },
            {
                "content": "H.R. 4 and H.R. 517 Signed into Law S. 1582\t\t\t\t\n\n\n\n\tThe White House\n\n\nJuly 24, 2025 \n\n\n\n\nOn Thursday, July 24, 2025, the President signed into law: H.R. 4, the “Rescissions Act of 2025,” which rescinds certain budget authority proposed to be rescinded in special messages transmitted to the Congress by the President on June 3, 2025, in accordance with section 1012(a) of the Congressional Budget and Impoundment Control Act of 1974; H.R. 517, the “Filing Relief for Natural Disasters Act,” which amends the Internal Revenue Code of 1986 to modify the rules for postponing certain deadlines by reason of disaster; and S. 1596, the “Jocelyn Nungaray National Wildlife Refuge Act,” which renames the Anahuac National Wildlife Refuge located in the State of Texas as the “Jocelyn Nungaray National Wildlife Refuge”.",
                "headline": "H.R. 4 and H.R. 517 Signed into Law S. 1582",
                "link": "https://www.whitehouse.gov/briefings-statements/2025/07/h-r-4-and-h-r-517-signed-into-law-s-1582/",
                "date": "2025-07-24T17:59:40-04:00",
            },
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
