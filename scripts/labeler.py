import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import math


class KeywordExtractor:
    """
    Advanced NLP keyword extraction with multiple algorithms
    """

    def __init__(self):
        # Common English stop words
        self.stop_words = {
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
            "have",
            "has",
            "had",
            "having",
            "get",
            "got",
            "go",
            "going",
            "went",
            "gone",
        }

    def preprocess_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text
        """
        text = text.lower()
        text = re.sub(r"[^\w\s-]", " ", text)
        words = text.split()
        filtered_words = [
            word
            for word in words
            if word not in self.stop_words and len(word) > 2 and word.isalpha()
        ]

        return filtered_words

    def extract_by_frequency(self, text: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Extract keywords based on word frequency
        """
        words = self.preprocess_text(text)
        word_freq = Counter(words)

        return word_freq.most_common(top_k)

    def extract_by_tfidf(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF scoring
        """
        sentences = re.split(r"[.!?]+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            freq_results = self.extract_by_frequency(text, top_k)
            return [(word, float(freq)) for word, freq in freq_results]

        word_tfidf = defaultdict(float)

        sentence_words = []
        all_words = set()

        for sentence in sentences:
            words = self.preprocess_text(sentence)
            sentence_words.append(words)
            all_words.update(words)

        for word in all_words:
            total_tf = sum(words.count(word) for words in sentence_words)
            total_words = sum(len(words) for words in sentence_words)
            tf = total_tf / max(total_words, 1)

            doc_count = sum(1 for words in sentence_words if word in words)
            idf = math.log(len(sentence_words) / max(doc_count, 1))

            word_tfidf[word] = tf * idf

        sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]

    def extract_by_position_weight(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords considering position weight (earlier words get higher scores)
        """
        words = self.preprocess_text(text)
        word_scores = defaultdict(float)

        total_words = len(words)

        for i, word in enumerate(words):
            position_weight = (total_words - i) / total_words
            word_scores[word] += position_weight

        word_freq = Counter(words)
        for word in word_scores:
            word_scores[word] = (word_scores[word] / word_freq[word]) * word_freq[word]

        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]

    def extract_ngrams(
        self, text: str, n: int = 2, top_k: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Extract n-gram phrases as keywords
        """
        words = self.preprocess_text(text)

        if len(words) < n:
            return []

        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)

        ngram_freq = Counter(ngrams)
        return ngram_freq.most_common(top_k)

    def extract_by_textrank(
        self, text: str, top_k: int = 10, window_size: int = 4
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using TextRank algorithm (simplified version)
        """
        words = self.preprocess_text(text)

        if len(words) < window_size:
            freq_results = self.extract_by_frequency(text, top_k)
            return [(word, float(freq)) for word, freq in freq_results]

        word_graph = defaultdict(set)
        word_list = list(set(words))

        for i in range(len(words)):
            for j in range(
                max(0, i - window_size), min(len(words), i + window_size + 1)
            ):
                if i != j:
                    word_graph[words[i]].add(words[j])

        scores = {word: 1.0 for word in word_list}

        damping = 0.85
        iterations = 30

        for _ in range(iterations):
            new_scores = {}
            for word in word_list:
                rank_sum = sum(
                    scores[neighbor] / len(word_graph[neighbor])
                    for neighbor in word_graph[word]
                    if len(word_graph[neighbor]) > 0
                )
                new_scores[word] = (1 - damping) + damping * rank_sum
            scores = new_scores

        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_k]

    def extract_compound_keywords(
        self, text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Extract compound keywords by combining adjacent important words
        """
        words = self.preprocess_text(text)

        tfidf_scores = dict(self.extract_by_tfidf(text, len(set(words))))

        compound_scores = {}

        for i in range(len(words) - 1):
            word1, word2 = words[i], words[i + 1]

            score1 = tfidf_scores.get(word1, 0)
            score2 = tfidf_scores.get(word2, 0)

            if score1 > 0 and score2 > 0:
                compound = f"{word1} {word2}"
                compound_scores[compound] = (score1 + score2) / 2

        sorted_compounds = sorted(
            compound_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_compounds[:top_k]

    def extract_all_methods(self, text: str, top_k: int = 10) -> Dict[str, List]:
        """
        Extract keywords using all available methods
        """
        results = {}

        results["frequency"] = self.extract_by_frequency(text, top_k)
        results["tfidf"] = self.extract_by_tfidf(text, top_k)
        results["position_weighted"] = self.extract_by_position_weight(text, top_k)
        results["bigrams"] = self.extract_ngrams(text, n=2, top_k=top_k)
        results["trigrams"] = self.extract_ngrams(text, n=3, top_k=top_k)
        results["textrank"] = self.extract_by_textrank(text, top_k)
        results["compound"] = self.extract_compound_keywords(text, top_k)

        return results

    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Get basic text statistics
        """
        words = self.preprocess_text(text)

        return {
            "total_words": len(text.split()),
            "unique_words": len(set(words)),
            "processed_words": len(words),
            "sentences": len(re.split(r"[.!?]+", text.strip())),
            "characters": len(text),
            "avg_word_length": sum(len(word) for word in words) / max(len(words), 1),
        }


if __name__ == "__main__":
    main()


def check_congress_outcome(
    text: str, pub_date: datetime, api_key: str
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a related bill became law on Congress.gov within 180 days.
    """
    search_terms = extract_policy_keywords(text)
    for term in search_terms:
        # The brief suggests this URL structure
        url = f"https://api.congress.gov/v3/bill/search?query={term}&api_key={api_key}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            bills = response.json().get("bills", [])
            for bill in bills:
                # The brief specifies a status of "Became Law"
                if bill.get("status") == "Became Law":
                    law_date_str = bill.get("signed_date")
                    if law_date_str:
                        law_date = datetime.fromisoformat(law_date_str)
                        # Check the 180-day window
                        if timedelta(0) < (law_date - pub_date) <= timedelta(days=180):
                            return True, bill.get("url")
        except (requests.RequestException, ValueError, TypeError) as e:
            print(f"Error checking Congress.gov for term '{term}': {e}")
            continue
    return False, None


def market_volatility_spike(
    pub_date: datetime, sectors: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Checks for a significant market volatility spike ( > 2 sigma) in a 24-hour window.
    """
    # The brief mentions skipping weekends, which is important for clean data.
    if pub_date.weekday() > 4:  # 5=Saturday, 6=Sunday
        return False, None

    start_date = pub_date - timedelta(days=30)
    end_date = pub_date + timedelta(days=2)

    for ticker in sectors:
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )
            if len(df) < 32:  # Need at least 32 days for 30-day rolling std
                continue

            # Calculate 24-hour return after announcement
            day_return = (df["Close"].iloc[-1] / df["Close"].iloc[-2]) - 1

            # Calculate 30-day rolling volatility (standard deviation of daily returns)
            rolling_std = df["Close"].pct_change().rolling(30).std().iloc[-2]

            # The brief specifies a threshold of 2.0 sigma
            z_score = abs(day_return) / rolling_std

            if z_score >= 2.0:
                print(f"Market spike detected for {ticker} with Z-score {z_score:.2f}")
                return True, f"Market moved for {ticker}"
        except Exception as e:
            print(f"Error checking market data for {ticker}: {e}")
            continue
    return False, None


def label_press_release(
    headline: str,
    body: str,
    pub_date: datetime,
    congress_api_key: str,
    market_sectors: List[str],
) -> Dict:
    """
    Applies the "Actionable vs Bluff" label based on the heuristics.
    """
    match_type = "None"
    match_url = None
    label = "Bluff"

    # Check Congress outcomes first
    is_law_passed, law_url = check_congress_outcome(
        headline + " " + body, pub_date, congress_api_key
    )
    if is_law_passed:
        label = "Actionable"
        match_type = "Law Passed"
        match_url = law_url
    else:
        # If no law, check for market volatility
        is_market_moved, market_info = market_volatility_spike(pub_date, market_sectors)
        if is_market_moved:
            label = "Actionable"
            match_type = "Market Moved"
            match_url = market_info

    # The brief mentions a "days_to_outcome" but we need to track this separately for a law.
    # For a simple implementation, we can leave it as None for now unless a law is found.

    return {
        "headline": headline,
        "body": body,
        "published_date": pub_date.strftime("%Y-%m-%d"),
        "label_t180": label,
        "match_type": match_type,
        "match_url": match_url,
        "days_to_outcome": None,  # This would be populated if a law is found.
        "quality_checked": False,
    }


def handler(event, context):
    """
    AWS Lambda entry point for the daily labeling job.
    """
    # Environment variables for configuration
    S3_RAW_BUCKET = os.environ.get("S3_RAW_BUCKET")
    S3_PROCESSED_BUCKET = os.environ.get("S3_PROCESSED_BUCKET")
    CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")
    MARKET_SECTORS = os.environ.get("MARKET_SECTORS", "SPY,XLI,XLE,XLF").split(",")

    if not all([S3_RAW_BUCKET, S3_PROCESSED_BUCKET, CONGRESS_API_KEY]):
        print("Required environment variables are not set. Exiting.")
        return {"statusCode": 500, "body": "Configuration error."}

    print("Starting daily labeling job.")

    sample_data = []
    df_raw = pd.DataFrame(sample_data)

    labeled_data = []
    for _, row in df_raw.iterrows():
        pub_date = datetime.fromisoformat(row["published_date"])
        labeled_item = label_press_release(
            row["headline"], row["body"], pub_date, CONGRESS_API_KEY, MARKET_SECTORS
        )
        labeled_data.append(labeled_item)

    df_labeled = pd.DataFrame(labeled_data)

    # Placeholder for writing data to S3 processed bucket (boto3 would be used here)
    # df_labeled.to_parquet(f"s3://{S3_PROCESSED_BUCKET}/processed_data.parquet")

    print(f"Successfully labeled {len(df_labeled)} press releases.")
    return {
        "statusCode": 200,
        "body": f"Successfully labeled {len(df_labeled)} press releases.",
    }
