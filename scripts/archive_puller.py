import asyncio
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from ingest_lambda import (
    WhiteHouseHTMLScraper,
    BidenArchiveHTMLScraper,
    ObamaArchiveHTMLScraper,
    BushArchiveHTMLScraper,
    DataSource,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    """Container for scraping results from a single source."""

    source_name: str
    url: str
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    scrape_time: Optional[datetime] = None


class ArchiveDataPuller:
    """Handles async scraping of multiple archive sources and data management."""

    def __init__(self, output_dir: str = "data", max_concurrent_sources: int = 2):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_concurrent_sources = max_concurrent_sources
        self.semaphore = asyncio.Semaphore(max_concurrent_sources)

    def _get_archive_sources(self) -> List[tuple[str, DataSource]]:
        """Define all archive sources with names."""
        return [
            (
                "current_whitehouse",
                WhiteHouseHTMLScraper(
                    "https://www.whitehouse.gov/briefings-statements/"
                ),
            ),
            (
                "biden_archive",
                BidenArchiveHTMLScraper(
                    "https://bidenwhitehouse.archives.gov/briefing-room/press-briefings/"
                ),
            ),
            (
                "obama_archive",
                ObamaArchiveHTMLScraper(
                    "https://obamawhitehouse.archives.gov/briefing-room/press-briefings"
                ),
            ),
            (
                "bush_archive",
                BushArchiveHTMLScraper(
                    "https://georgewbush-whitehouse.archives.gov/news/briefings/"
                ),
            ),
        ]

    async def pull_archive(self) -> pd.DataFrame:
        """Main method to pull all archive data asynchronously."""
        logger.info("Starting archive data pull...")
        start_time = datetime.now()

        # Get all sources
        sources = self._get_archive_sources()

        # Scrape all sources concurrently
        scraping_tasks = [
            self._scrape_single_source(source_name, data_source)
            for source_name, data_source in sources
        ]

        results = await asyncio.gather(*scraping_tasks, return_exceptions=True)

        # Process results and create combined DataFrame
        combined_df = await self._process_scraping_results(results)

        # Save final results
        await self._save_combined_data(combined_df)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            f"Archive pull completed in {duration:.2f} seconds. Total articles: {len(combined_df)}"
        )

        return combined_df

    async def _scrape_single_source(
        self, source_name: str, data_source: DataSource
    ) -> ScrapingResult:
        """Scrape a single source with proper error handling and rate limiting."""
        async with self.semaphore:  # Limit concurrent sources
            logger.info(f"Starting scrape for {source_name}")
            scrape_start = datetime.now()

            try:
                # Use the async scraper directly
                data = await data_source.get_data()
                print(f"data is {data}")

                scrape_time = datetime.now()
                duration = (scrape_time - scrape_start).total_seconds()

                if data:
                    logger.info(
                        f"Successfully scraped {len(data)} articles from {source_name} in {duration:.2f}s"
                    )
                else:
                    logger.warning(f"No data retrieved from {source_name}")

                return ScrapingResult(
                    source_name=source_name,
                    url=str(data_source),
                    data=data,
                    scrape_time=scrape_time,
                )

            except Exception as e:
                logger.error(f"Error scraping {source_name}: {str(e)}")
                return ScrapingResult(
                    source_name=source_name,
                    url=data_source.url,
                    data=[],
                    error=str(e),
                    scrape_time=datetime.now(),
                )

    async def _process_scraping_results(
        self, results: List[ScrapingResult]
    ) -> pd.DataFrame:
        """Process all scraping results and create a combined DataFrame."""
        combined_data = []
        successful_sources = 0
        total_articles = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scraping task failed with exception: {result}")
                continue

            # Save individual source data
            await self._save_source_data(result)

            if result.error:
                logger.error(f"Source {result.source_name} failed: {result.error}")
                continue

            if not result.data:
                logger.warning(f"No data from source {result.source_name}")
                continue

            # Add source metadata to each article
            for article in result.data:
                article_with_metadata = article.copy()
                article_with_metadata.update(
                    {
                        "source_name": result.source_name,
                        "source_url": result.url,
                        "scraped_at": result.scrape_time,
                    }
                )
                combined_data.append(article_with_metadata)

            successful_sources += 1
            total_articles += len(result.data)

        logger.info(
            f"Successfully processed {successful_sources} sources with {total_articles} total articles"
        )

        # Create DataFrame from combined data
        if combined_data:
            df = pd.DataFrame(combined_data)
            # Ensure consistent column order
            column_order = [
                "headline",
                "date",
                "content",
                "link",
                "source_name",
                "source_url",
                "scraped_at",
            ]
            # Only reorder columns that exist
            existing_columns = [col for col in column_order if col in df.columns]
            other_columns = [col for col in df.columns if col not in column_order]
            df = df[existing_columns + other_columns]
        else:
            df = pd.DataFrame()

        return df

    async def _save_source_data(self, result: ScrapingResult) -> None:
        """Save individual source data to JSON file."""
        try:
            filename = f"{result.source_name}_{result.scrape_time.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename

            # Prepare data for JSON serialization
            json_data = {
                "source_name": result.source_name,
                "url": result.url,
                "scraped_at": (
                    result.scrape_time.isoformat() if result.scrape_time else None
                ),
                "error": result.error,
                "article_count": len(result.data),
                "articles": [],
            }

            # Convert articles to JSON-serializable format
            for article in result.data:
                json_article = article.copy()
                # Convert datetime objects to ISO format
                if "date" in json_article and isinstance(
                    json_article["date"], datetime
                ):
                    json_article["date"] = json_article["date"].isoformat()
                json_data["articles"].append(json_article)

            # Write to file asynchronously
            await asyncio.to_thread(self._write_json_file, filepath, json_data)
            logger.debug(f"Saved {result.source_name} data to {filepath}")

        except Exception as e:
            logger.error(f"Error saving data for {result.source_name}: {e}")

    def _write_json_file(self, filepath: Path, data: dict) -> None:
        """Write JSON data to file (blocking operation)."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def _save_combined_data(self, df: pd.DataFrame) -> None:
        """Save combined DataFrame to both parquet and CSV formats."""
        if df.empty:
            logger.warning("No data to save - DataFrame is empty")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to parquet and CSV concurrently
        parquet_path = self.output_dir / f"archive_{timestamp}.parquet"
        csv_path = self.output_dir / f"archive_{timestamp}.csv"
        latest_parquet = self.output_dir / "archive_latest.parquet"
        latest_csv = self.output_dir / "archive_latest.csv"

        save_tasks = [
            asyncio.to_thread(df.to_parquet, parquet_path, index=False),
            asyncio.to_thread(df.to_csv, csv_path, index=False),
            asyncio.to_thread(df.to_parquet, latest_parquet, index=False),
            asyncio.to_thread(df.to_csv, latest_csv, index=False),
        ]

        await asyncio.gather(*save_tasks)
        logger.info(f"Saved combined data to {parquet_path} and {csv_path}")


class LegacyArchiveDataPuller(ArchiveDataPuller):
    """Legacy version that works with the existing IngestionService."""

    async def _scrape_single_source(
        self, source_name: str, data_source: DataSource
    ) -> ScrapingResult:
        """Override to use IngestionService (for backward compatibility)."""
        async with self.semaphore:
            logger.info(f"Starting scrape for {source_name}: {data_source.url}")
            scrape_start = datetime.now()

            try:
                # Use IngestionService in a thread (assuming it's synchronous)
                ingestion_service = IngestionService(
                    primary_source=data_source, fallback_source=data_source
                )

                data = await asyncio.to_thread(ingestion_service.get_all_releases)

                scrape_time = datetime.now()
                duration = (scrape_time - scrape_start).total_seconds()

                if data:
                    logger.info(
                        f"Successfully scraped {len(data)} articles from {source_name} in {duration:.2f}s"
                    )
                else:
                    logger.warning(f"No data retrieved from {source_name}")

                return ScrapingResult(
                    source_name=source_name,
                    url=data_source.url,
                    data=data,
                    scrape_time=scrape_time,
                )

            except Exception as e:
                logger.error(f"Error scraping {source_name}: {str(e)}")
                return ScrapingResult(
                    source_name=source_name,
                    url=data_source.url,
                    data=[],
                    error=str(e),
                    scrape_time=datetime.now(),
                )


# Convenience functions
async def pull_archive_async() -> pd.DataFrame:
    """Async version of the original pull_archive function."""
    puller = ArchiveDataPuller()
    return await puller.pull_archive()


async def pull_archive_with_ingestion_service() -> pd.DataFrame:
    """Version that uses IngestionService for backward compatibility."""
    puller = LegacyArchiveDataPuller()
    return await puller.pull_archive()


def pull_archive() -> pd.DataFrame:
    """Synchronous wrapper for backward compatibility."""
    return asyncio.run(pull_archive_async())


# Usage examples
async def main():
    """Example usage."""
    # Direct async usage (recommended)
    puller = ArchiveDataPuller(output_dir="data", max_concurrent_sources=2)
    df = await puller.pull_archive()
    print(f"Pulled {len(df)} total articles")

    # Or use convenience function
    # df = await pull_archive_async()


if __name__ == "__main__":
    asyncio.run(main())
