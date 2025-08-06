from abc import abstractmethod
from functools import partialmethod
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict


class DataSource:
    @abstractmethod
    def get_data(self) -> List:
        pass


class WhiteHouseHTMLScrapper(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def get_data(self) -> List:
        article_data: List[Dict] = []
        count = 0
        while self.url != "" and count <= 2:
            count += 1
            print(f"Scraping URL: {self.url}")
            try:
                response = requests.get(self.url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # identify the current page
                current_page = soup.find_all("span", class_="page-numbers current")
                if not current_page:
                    return []
                current_page = int(current_page[0].text.strip())
                print(f"current page is {current_page}")

                # keep track of the next page
                pages = soup.find_all("a", class_="page-numbers")
                if not pages:
                    return []

                for page in pages:
                    next_page = int(page.text.strip())
                    next_url = page.get("href", "")
                    self.url = next_url if next_page > current_page else ""
                    if self.url != "":
                        break

                # handle articles for the current page
                articles = soup.find_all(
                    "ul",
                    class_="wp-block-post-template is-layout-flow wp-block-post-template-is-layout-flow",
                )
                if not articles:
                    return []

                for article in articles:
                    items = article.find_all("li")
                    for item in items:
                        date = str(
                            item.find("div", class_="wp-block-post-date")
                            .find("time")
                            .get("datetime", "")
                        )
                        headline = item.find("h2", class_="wp-block-post-title")
                        link = headline.find("a")["href"]
                        headline = headline.text.strip()

                        article_response = requests.get(link, timeout=10)
                        article_response.raise_for_status()
                        soup = BeautifulSoup(article_response.text, "html.parser")
                        content_div = soup.find(
                            "div",
                            class_="entry-content wp-block-post-content has-global-padding is-layout-constrained wp-block-post-content-is-layout-constrained",
                        )
                        content = (
                            content_div.get_text()
                            .split("Briefings & Statements", 1)[-1]
                            .strip()
                        )
                        article_data.append(
                            {
                                "content": content,
                                "headline": headline,
                                "link": link,
                                "date": datetime.fromisoformat(date),
                            }
                        )

            except (requests.RequestException, IndexError, TypeError) as e:
                print(f"Error scraping primary URL: {e}")
                return []
        return article_data

    def fetch_body_from_url(self, url):
        # Implementation remains the same as before
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)
            return "Body text not found."
        except requests.RequestException as e:
            print(f"Error fetching body from {url}: {e}")
            return "Body text could not be fetched."


class BidenArchiveHTMLScrapper(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def get_data(self) -> List:
        article_data: List[Dict] = []
        count = 0
        while self.url != "" and count <= 2:
            count += 1
            print(f"Scraping URL: {self.url}")
            try:
                response = requests.get(self.url, timeout=10)
                with open("biden.text", "w") as f:
                    f.write(response.text)

                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # identify the current page
                current_page = soup.find_all("span", class_="page-numbers current")
                if not current_page:
                    return []
                current_page = int(current_page[0].text.strip().split(" ")[1])

                # keep track of the next page
                pages = soup.find_all("a", class_="page-numbers")
                if not pages:
                    return []

                for page in pages:
                    # print(f"page is {page.text.strip().split(" }")
                    next_page = int(page.text.strip().split(" ")[1])
                    next_url = page.get("href", "")
                    self.url = (
                        f"https://bidenwhitehouse.archives.gov{next_url}"
                        if next_page > current_page
                        else ""
                    )
                    if self.url != "":
                        break

                print(f"self url is {self.url}")

                # handle articles for the current page
                articles = soup.find_all(
                    "div",
                    class_=" article-wrapper col col-xs-12 col-md-8 col-lg-6 offset-lg-3",
                )
                if not articles:
                    return []

                for article in articles:
                    items = article.find_all("li")
                    for item in items:
                        date = str(
                            item.find("div", class_="wp-block-post-date")
                            .find("time")
                            .get("datetime", "")
                        )
                        headline = item.find("h2", class_="wp-block-post-title")
                        link = headline.find("a")["href"]
                        headline = headline.text.strip()

                        article_response = requests.get(link, timeout=10)
                        article_response.raise_for_status()
                        soup = BeautifulSoup(article_response.text, "html.parser")
                        content_div = soup.find(
                            "div",
                            class_="entry-content wp-block-post-content has-global-padding is-layout-constrained wp-block-post-content-is-layout-constrained",
                        )
                        content = (
                            content_div.get_text()
                            .split("Briefings & Statements", 1)[-1]
                            .strip()
                        )
                        article_data.append(
                            {
                                "content": content,
                                "headline": headline,
                                "link": link,
                                "date": datetime.fromisoformat(date),
                            }
                        )

            except (requests.RequestException, IndexError, TypeError) as e:
                print(f"Error scraping primary URL: {e}")
                return []
        return article_data

    def fetch_body_from_url(self, url):
        # Implementation remains the same as before
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)
            return "Body text not found."
        except requests.RequestException as e:
            print(f"Error fetching body from {url}: {e}")
            return "Body text could not be fetched."


class ObamaArchiveHTMLScrapper(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def get_data(self) -> List:
        article_data: List[Dict] = []
        count = 0
        while self.url != "" and count <= 2:
            count += 1
            print(f"Scraping URL: {self.url}")
            try:
                response = requests.get(self.url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # identify the current page
                current_page = soup.find_all("span", class_="page-numbers current")
                if not current_page:
                    return []
                current_page = int(current_page[0].text.strip())
                print(f"current page is {current_page}")

                # keep track of the next page
                pages = soup.find_all("a", class_="page-numbers")
                if not pages:
                    return []

                for page in pages:
                    next_page = int(page.text.strip())
                    next_url = page.get("href", "")
                    self.url = next_url if next_page > current_page else ""
                    if self.url != "":
                        break

                # handle articles for the current page
                articles = soup.find_all(
                    "ul",
                    class_="wp-block-post-template is-layout-flow wp-block-post-template-is-layout-flow",
                )
                if not articles:
                    return []

                for article in articles:
                    items = article.find_all("li")
                    for item in items:
                        date = str(
                            item.find("div", class_="wp-block-post-date")
                            .find("time")
                            .get("datetime", "")
                        )
                        headline = item.find("h2", class_="wp-block-post-title")
                        link = headline.find("a")["href"]
                        headline = headline.text.strip()

                        article_response = requests.get(link, timeout=10)
                        article_response.raise_for_status()
                        soup = BeautifulSoup(article_response.text, "html.parser")
                        content_div = soup.find(
                            "div",
                            class_="entry-content wp-block-post-content has-global-padding is-layout-constrained wp-block-post-content-is-layout-constrained",
                        )
                        content = (
                            content_div.get_text()
                            .split("Briefings & Statements", 1)[-1]
                            .strip()
                        )
                        article_data.append(
                            {
                                "content": content,
                                "headline": headline,
                                "link": link,
                                "date": datetime.fromisoformat(date),
                            }
                        )

            except (requests.RequestException, IndexError, TypeError) as e:
                print(f"Error scraping primary URL: {e}")
                return []
        return article_data

    def fetch_body_from_url(self, url):
        # Implementation remains the same as before
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)
            return "Body text not found."
        except requests.RequestException as e:
            print(f"Error fetching body from {url}: {e}")
            return "Body text could not be fetched."


class BushArchiveHTMLScrapper(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def get_data(self) -> List:
        article_data: List[Dict] = []
        count = 0
        while self.url != "" and count <= 2:
            count += 1
            print(f"Scraping URL: {self.url}")
            try:
                response = requests.get(self.url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # identify the current page
                current_page = soup.find_all("span", class_="page-numbers current")
                if not current_page:
                    return []
                current_page = int(current_page[0].text.strip())
                print(f"current page is {current_page}")

                # keep track of the next page
                pages = soup.find_all("a", class_="page-numbers")
                if not pages:
                    return []

                for page in pages:
                    next_page = int(page.text.strip())
                    next_url = page.get("href", "")
                    self.url = next_url if next_page > current_page else ""
                    if self.url != "":
                        break

                # handle articles for the current page
                articles = soup.find_all(
                    "ul",
                    class_="wp-block-post-template is-layout-flow wp-block-post-template-is-layout-flow",
                )
                if not articles:
                    return []

                for article in articles:
                    items = article.find_all("li")
                    for item in items:
                        date = str(
                            item.find("div", class_="wp-block-post-date")
                            .find("time")
                            .get("datetime", "")
                        )
                        headline = item.find("h2", class_="wp-block-post-title")
                        link = headline.find("a")["href"]
                        headline = headline.text.strip()

                        article_response = requests.get(link, timeout=10)
                        article_response.raise_for_status()
                        soup = BeautifulSoup(article_response.text, "html.parser")
                        content_div = soup.find(
                            "div",
                            class_="entry-content wp-block-post-content has-global-padding is-layout-constrained wp-block-post-content-is-layout-constrained",
                        )
                        content = (
                            content_div.get_text()
                            .split("Briefings & Statements", 1)[-1]
                            .strip()
                        )
                        article_data.append(
                            {
                                "content": content,
                                "headline": headline,
                                "link": link,
                                "date": datetime.fromisoformat(date),
                            }
                        )

            except (requests.RequestException, IndexError, TypeError) as e:
                print(f"Error scraping primary URL: {e}")
                return []
        return article_data

    def fetch_body_from_url(self, url):
        # Implementation remains the same as before
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)
            return "Body text not found."
        except requests.RequestException as e:
            print(f"Error fetching body from {url}: {e}")
            return "Body text could not be fetched."


class WhiteHouseRssScrapper(DataSource):
    def __init__(self, url: str) -> None:
        self.url = url

    def get_releases(self):
        print(f"Scraping fallback RSS feed: {self.url}")
        try:
            feed = feedparser.parse(self.url)
            releases = []
            for entry in feed.entries:
                published_date = datetime.strptime(
                    entry.published, "%a, %d %b %Y %H:%M:%S %z"
                ).strftime("%Y-%m-%d")
                body = self.fetch_body_from_url(entry.link)

                releases.append(
                    {
                        "headline": entry.title,
                        "body": body,
                        "source": "whitehouse_rss",
                        "published_date": published_date,
                        "url": entry.link,
                    }
                )
            return releases
        except Exception as e:
            print(f"Error scraping RSS feed: {e}")
            return []

    def fetch_body_from_url(self, url):
        # Same body fetch logic
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)
            return "Body text not found."
        except requests.RequestException as e:
            print(f"Error fetching body from {url}: {e}")
            return "Body text could not be fetched."


class IngestionService:
    def __init__(self, primary_source: DataSource, fallback_source: DataSource):
        self.primary_source = primary_source
        self.fallback_source = fallback_source

    def get_all_releases(self):
        releases = self.primary_source.get_data()
        if not releases:
            releases = self.fallback_source.get_data()
        return releases


def pull_archive() -> pd.DataFrame:
    archive_links = [
        BidenArchiveHTMLScrapper(
            "https://bidenwhitehouse.archives.gov/briefing-room/press-briefings/"
        ),
        # ObamaArchiveHTMLScrapper(
        #    "https://obamawhitehouse.archives.gov/briefing-room/press-briefings"
        # ),
        # BushArchiveHTMLScrapper(
        #    "https://georgewbush-whitehouse.archives.gov/news/briefings/"
        # ),
    ]
    archive_df: pd.DataFrame = pd.DataFrame()

    for data_source in archive_links:

        ingestion_service = IngestionService(
            primary_source=data_source, fallback_source=data_source
        )

        data = ingestion_service.get_all_releases()
        if data:
            print(f"Successfully scraped {len(data)} press releases.")
            df = pd.DataFrame(data)
            df["year"] = df["date"].dt.strftime("%Y")
            df["month"] = df["date"].dt.strftime("%m")
            df["day"] = df["date"].dt.strftime("%d")

            archive_df = pd.concat([archive_df, df])

    return archive_df


def handler(event, context) -> pd.DataFrame:
    # TODO: handle archive list
    primary_url = "https://www.whitehouse.gov/briefings-statements/"
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
        # Add partitioning columns based on the published date
        df["year"] = df["date"].dt.strftime("%Y")
        df["month"] = df["date"].dt.strftime("%m")
        df["day"] = df["date"].dt.strftime("%d")

        return df


if __name__ == "__main__":
    pull_archive()
