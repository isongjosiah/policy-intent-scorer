from scripts.ingest_lambda import (
    WhiteHouseHTMLScrapper,
    WhiteHouseRssScrapper,
    IngestionService,
)
import pandas as pd


def main():
    primary_url = "https://www.whitehouse.gov/briefing-room/statements-and-releases/"
    fallback_url = "https://www.whitehouse.gov/briefing-room/feed/"

    primary_scraper = WhiteHouseHTMLScrapper(primary_url)
    fallback_scraper = WhiteHouseRssScrapper(fallback_url)

    ingestion_service = IngestionService(
        primary_source=primary_scraper, fallback_source=fallback_scraper
    )

    data = ingestion_service.get_all_releases()

    if data:
        # Save data to S3 as a Parquet file (implementation not included here)
        print(f"Successfully scraped {len(data)} press releases.")
        df = pd.DataFrame(data)
        print(df.head())

        return {
            "statusCode": 200,
            "body": f"Successfully ingested {len(data)} press releases.",
        }
    else:
        return {"statusCode": 500, "body": "Failed to ingest press releases."}


if __name__ == "__main__":
    main()
