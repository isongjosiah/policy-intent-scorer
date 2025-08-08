from abc import abstractmethod, ABC
from functools import partialmethod
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import asyncio
from typing import List, Dict, Optional
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(ABC):
    @abstractmethod
    async def get_data(self) -> List:
        return []


class WhiteHouseHTMLScraper(DataSource):
    """Async scraper for White House website articles with pagination support."""

    def __init__(self, url: str, timeout: int = 10, max_concurrent: int = 5) -> None:
        self.base_url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def get_data(self) -> List[Dict]:
        """Main method to scrape all articles from paginated pages."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            return await self._scrape_all_pages(session)

    async def _scrape_all_pages(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Scrape all pages sequentially to respect pagination order."""
        all_articles = []
        current_url = self.base_url

        while current_url:
            try:
                logger.info(f"Scraping URL: {current_url}")
                page_articles, next_url = await self._scrape_page_with_pagination(
                    session, current_url
                )
                all_articles.extend(page_articles)
                current_url = next_url
            except Exception as e:
                logger.error(f"Error scraping page {current_url}: {e}")
                break

        return all_articles

    async def _scrape_page_with_pagination(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[List[Dict], Optional[str]]:
        """Scrape articles from a single page and get next page URL."""
        soup = await self._get_soup(session, url)
        if not soup:
            return [], None

        # Get articles and next page URL concurrently
        articles_task = self._scrape_articles_from_page(session, soup)
        next_url_task = self._get_next_page_url_async(soup)

        articles, next_url = await asyncio.gather(articles_task, next_url_task)
        return articles, next_url

    async def _scrape_articles_from_page(
        self, session: aiohttp.ClientSession, soup: BeautifulSoup
    ) -> List[Dict]:
        """Extract and fetch full content for all articles on a page."""
        articles_container = self._find_articles_container(soup)
        if not articles_container:
            logger.warning("No articles container found on page")
            return []

        # Extract basic article info
        article_items = self._extract_article_items(articles_container)

        # Fetch full content for all articles concurrently
        tasks = [self._fetch_complete_article(session, item) for item in article_items]

        articles = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        return [article for article in articles if isinstance(article, dict)]

    async def _get_soup(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML content asynchronously."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    return BeautifulSoup(html, "html.parser")
            except aiohttp.ClientError as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

    async def _get_next_page_url_async(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL of the next page (async wrapper for sync operation)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_next_page_url_sync, soup
        )

    def _get_next_page_url_sync(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL of the next page (synchronous parsing)."""
        # Get current page number
        current_page_elem = soup.find("span", class_="page-numbers current")
        if not current_page_elem:
            return None

        try:
            current_page = int(current_page_elem.text.strip())
        except (ValueError, AttributeError):
            logger.error("Could not parse current page number")
            return None

        # Find next page link
        page_links = soup.find_all("a", class_="page-numbers")
        for link in page_links:
            try:
                page_num = int(link.text.strip())
                if page_num > current_page:
                    return link.get("href", "")
            except (ValueError, AttributeError):
                continue

        return None

    def _find_articles_container(self, soup: BeautifulSoup) -> Optional:
        """Find the container with articles."""
        return soup.find(
            "ul",
            class_="wp-block-post-template is-layout-flow wp-block-post-template-is-layout-flow",
        )

    def _extract_article_items(self, container) -> List[Dict[str, Optional[str]]]:
        """Extract basic article metadata from container."""
        articles = []
        items = container.find_all("li")

        for item in items:
            try:
                date_str = self._extract_article_date(item)
                headline, link = self._extract_headline_and_link(item)

                if all([date_str, headline, link]):
                    articles.append(
                        {"date_str": date_str, "headline": headline, "link": link}
                    )
                else:
                    logger.warning("Missing required article data in item")
            except Exception as e:
                logger.error(f"Error extracting article item: {e}")

        return articles

    async def _fetch_complete_article(
        self, session: aiohttp.ClientSession, article_item: Dict[str, str]
    ) -> Optional[Dict]:
        """Fetch complete article data including content."""
        try:
            content = await self._fetch_article_content(session, article_item["link"])

            return {
                "content": content,
                "headline": article_item["headline"],
                "link": article_item["link"],
                "date": datetime.fromisoformat(article_item["date_str"]),
            }

        except Exception as e:
            logger.error(
                f"Error fetching complete article {article_item.get('link', 'unknown')}: {e}"
            )
            return None

    def _extract_article_date(self, item) -> Optional[str]:
        """Extract the date from an article item."""
        try:
            date_div = item.find("div", class_="wp-block-post-date")
            time_elem = date_div.find("time")
            return time_elem.get("datetime", "")
        except AttributeError:
            return None

    def _extract_headline_and_link(self, item) -> tuple[Optional[str], Optional[str]]:
        """Extract headline and link from an article item."""
        try:
            headline_elem = item.find("h2", class_="wp-block-post-title")
            link_elem = headline_elem.find("a")
            headline = headline_elem.text.strip()
            link = link_elem["href"]
            return headline, link
        except (AttributeError, KeyError):
            return None, None

    async def _fetch_article_content(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Fetch the full content of an article asynchronously."""
        soup = await self._get_soup(session, url)
        if not soup:
            return "Content could not be fetched."

        content_div = soup.find(
            "div",
            class_="entry-content wp-block-post-content has-global-padding is-layout-constrained wp-block-post-content-is-layout-constrained",
        )

        if not content_div:
            return "Content div not found."

        # Clean up content by removing unwanted sections
        content = content_div.get_text()
        content = content.split("Briefings & Statements", 1)[-1].strip()

        return content if content else "No content found."

    async def fetch_body_from_url(self, url: str) -> str:
        """Fetch body content from a specific URL asynchronously."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            soup = await self._get_soup(session, url)
            if not soup:
                return "Body text could not be fetched."

            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)

            return "Body text not found."


class BidenArchiveHTMLScraper(DataSource):
    """Async scraper for Biden White House Archives with pagination support."""

    def __init__(self, url: str, timeout: int = 10, max_concurrent: int = 5) -> None:
        self.base_url = url
        self.base_domain = "https://bidenwhitehouse.archives.gov"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def get_data(self) -> List[Dict]:
        """Main method to scrape all articles from paginated pages."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            return await self._scrape_all_pages(session)

    async def _scrape_all_pages(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Scrape all pages sequentially to respect pagination order."""
        all_articles = []
        current_url = self.base_url

        while current_url:
            try:
                logger.info(f"Scraping URL: {current_url}")
                page_articles, next_url = await self._scrape_page_with_pagination(
                    session, current_url
                )
                all_articles.extend(page_articles)
                current_url = next_url
            except Exception as e:
                logger.error(f"Error scraping page {current_url}: {e}")
                break

        return all_articles

    async def _scrape_page_with_pagination(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[List[Dict], Optional[str]]:
        """Scrape articles from a single page and get next page URL."""
        soup = await self._get_soup(session, url)
        if not soup:
            return [], None

        # Get articles and next page URL concurrently
        articles_task = self._scrape_articles_from_page(session, soup)
        next_url_task = self._get_next_page_url_async(soup)

        articles, next_url = await asyncio.gather(articles_task, next_url_task)
        return articles, next_url

    async def _scrape_articles_from_page(
        self, session: aiohttp.ClientSession, soup: BeautifulSoup
    ) -> List[Dict]:
        """Extract and fetch full content for all articles on a page."""
        articles_containers = self._find_articles_containers(soup)
        if not articles_containers:
            logger.warning("No articles containers found on page")
            return []

        # Extract basic article info from all containers
        article_items = []
        for container in articles_containers:
            article_items.extend(self._extract_article_items_from_container(container))

        if not article_items:
            logger.warning("No article items found in containers")
            return []

        # Fetch full content for all articles concurrently
        tasks = [self._fetch_complete_article(session, item) for item in article_items]

        articles = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_articles = []
        for article in articles:
            if isinstance(article, dict):
                valid_articles.append(article)
            elif isinstance(article, Exception):
                logger.error(f"Exception while fetching article: {article}")

        return valid_articles

    async def _get_soup(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML content asynchronously."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    return BeautifulSoup(html, "html.parser")
            except aiohttp.ClientError as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

    async def _get_next_page_url_async(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL of the next page (async wrapper for sync operation)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_next_page_url_sync, soup
        )

    def _get_next_page_url_sync(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL of the next page (synchronous parsing)."""
        # Get current page number
        current_page_elem = soup.find("span", class_="page-numbers current")
        if not current_page_elem:
            logger.warning("No current page element found")
            return None

        try:
            # Parse "Page X" format
            current_page_text = current_page_elem.text.strip()
            current_page = int(current_page_text.split(" ")[1])
        except (ValueError, AttributeError, IndexError):
            logger.error(
                f"Could not parse current page number from: {current_page_elem.text if current_page_elem else 'None'}"
            )
            return None

        # Find next page link
        page_links = soup.find_all("a", class_="page-numbers")
        for link in page_links:
            try:
                page_text = link.text.strip()
                page_num = int(page_text.split(" ")[1])
                if page_num > current_page:
                    href = link.get("href", "")
                    return f"{self.base_domain}{href}" if href else None
            except (ValueError, AttributeError, IndexError):
                continue

        return None

    def _find_articles_containers(self, soup: BeautifulSoup) -> List:
        """Find all containers with articles."""
        return soup.find_all(
            "div", class_="article-wrapper col col-xs-12 col-md-8 col-lg-6 offset-lg-3"
        )

    def _extract_article_items_from_container(
        self, container
    ) -> List[Dict[str, Optional[str]]]:
        """Extract basic article metadata from a single container."""
        articles = []
        items = container.find_all("article")

        for item in items:
            try:
                date_str = self._extract_article_date(item)
                headline, link = self._extract_headline_and_link(item)

                if all([date_str, headline, link]):
                    articles.append(
                        {"date_str": date_str, "headline": headline, "link": link}
                    )
                else:
                    logger.warning(
                        f"Missing required article data: date={date_str}, headline={headline}, link={link}"
                    )
            except Exception as e:
                logger.error(f"Error extracting article item: {e}")

        return articles

    async def _fetch_complete_article(
        self, session: aiohttp.ClientSession, article_item: Dict[str, str]
    ) -> Optional[Dict]:
        """Fetch complete article data including content."""
        try:
            content = await self._fetch_article_content(session, article_item["link"])

            return {
                "content": content,
                "headline": article_item["headline"],
                "link": article_item["link"],
                "date": datetime.fromisoformat(article_item["date_str"]),
            }

        except Exception as e:
            logger.error(
                f"Error fetching complete article {article_item.get('link', 'unknown')}: {e}"
            )
            return None

    def _extract_article_date(self, item) -> Optional[str]:
        """Extract the date from an article item."""
        try:
            meta_div = item.find("div", class_="news-item__meta shared-meta")
            time_elem = meta_div.find("time")
            return time_elem.get("datetime", "")
        except AttributeError:
            logger.error("Could not find date element in article item")
            return None

    def _extract_headline_and_link(self, item) -> tuple[Optional[str], Optional[str]]:
        """Extract headline and link from an article item."""
        try:
            headline_elem = item.find("h2", class_="news-item__title-container")
            link_elem = headline_elem.find("a")
            headline = headline_elem.text.strip()
            href = link_elem["href"]
            link = f"{self.base_domain}{href}"
            return headline, link
        except (AttributeError, KeyError) as e:
            logger.error(f"Error extracting headline and link: {e}")
            return None, None

    async def _fetch_article_content(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Fetch the full content of an article asynchronously."""
        soup = await self._get_soup(session, url)
        if not soup:
            return "Content could not be fetched."

        content_section = soup.find("section", class_="body-content")

        if not content_section:
            logger.warning(f"Content section not found for URL: {url}")
            return "Content section not found."

        # Clean up content by removing unwanted sections
        content = content_section.get_text()
        content = content.split("Briefings & Statements", 1)[-1].strip()

        return content if content else "No content found."

    async def fetch_body_from_url(self, url: str) -> str:
        """Fetch body content from a specific URL asynchronously."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            soup = await self._get_soup(session, url)
            if not soup:
                return "Body text could not be fetched."

            body_content = soup.find("div", class_="body-content")
            if body_content:
                return body_content.get_text(separator=" ", strip=True)

            return "Body text not found."


class ObamaArchiveHTMLScraper(DataSource):
    """Async scraper for Obama White House Archives with pagination support."""

    def __init__(self, url: str, timeout: int = 10, max_concurrent: int = 5) -> None:
        self.base_url = url
        self.base_domain = "https://obamawhitehouse.archives.gov"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.date_format = "%B %d, %Y"

    async def get_data(self) -> List[Dict]:
        """Main method to scrape all articles from paginated pages."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            return await self._scrape_all_pages(session)

    async def _scrape_all_pages(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Scrape all pages sequentially to respect pagination order."""
        all_articles = []
        current_url = self.base_url

        while current_url:
            try:
                logger.info(f"Scraping URL: {current_url}")
                page_articles, next_url = await self._scrape_page_with_pagination(
                    session, current_url
                )
                all_articles.extend(page_articles)
                current_url = next_url
            except Exception as e:
                logger.error(f"Error scraping page {current_url}: {e}")
                break

        return all_articles

    async def _scrape_page_with_pagination(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[List[Dict], Optional[str]]:
        """Scrape articles from a single page and get next page URL."""
        soup = await self._get_soup(session, url)
        if not soup:
            return [], None

        # Get articles and next page URL concurrently
        articles_task = self._scrape_articles_from_page(session, soup)
        next_url_task = self._get_next_page_url_async(soup)

        articles, next_url = await asyncio.gather(articles_task, next_url_task)
        return articles, next_url

    async def _scrape_articles_from_page(
        self, session: aiohttp.ClientSession, soup: BeautifulSoup
    ) -> List[Dict]:
        """Extract and fetch full content for all articles on a page."""
        articles_containers = self._find_articles_containers(soup)
        if not articles_containers:
            logger.warning("No articles containers found on page")
            return []

        # Extract basic article info from all containers
        article_items = []
        for container in articles_containers:
            article_items.extend(self._extract_article_items_from_container(container))

        if not article_items:
            logger.warning("No article items found in containers")
            return []

        # Fetch full content for all articles concurrently
        tasks = [self._fetch_complete_article(session, item) for item in article_items]

        articles = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_articles = []
        for article in articles:
            if isinstance(article, dict):
                valid_articles.append(article)
            elif isinstance(article, Exception):
                logger.error(f"Exception while fetching article: {article}")

        return valid_articles

    async def _get_soup(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML content asynchronously."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    return BeautifulSoup(html, "html.parser")
            except aiohttp.ClientError as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

    async def _get_next_page_url_async(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL of the next page (async wrapper for sync operation)."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_next_page_url_sync, soup
        )

    def _get_next_page_url_sync(self, soup: BeautifulSoup) -> Optional[str]:
        """Find the URL of the next page (synchronous parsing)."""
        try:
            # Look for the "next" page link
            next_page_elem = soup.find("li", class_="pager-next last")
            if not next_page_elem:
                logger.info("No next page found - reached end of pagination")
                return None

            next_link = next_page_elem.find("a")
            if not next_link:
                return None

            href = next_link.get("href", "")
            return f"{self.base_domain}{href}" if href else None

        except Exception as e:
            logger.error(f"Error finding next page URL: {e}")
            return None

    def _find_articles_containers(self, soup: BeautifulSoup) -> List:
        """Find all containers with articles."""
        return soup.find_all("div", class_="view-content")

    def _extract_article_items_from_container(
        self, container
    ) -> List[Dict[str, Optional[str]]]:
        """Extract basic article metadata from a single container."""
        articles = []
        items = container.find_all("div", class_="views-row")

        for item in items:
            try:
                date_str = self._extract_article_date(item)
                headline, link = self._extract_headline_and_link(item)

                if all([date_str, headline, link]):
                    articles.append(
                        {"date_str": date_str, "headline": headline, "link": link}
                    )
                else:
                    logger.warning(
                        f"Missing required article data: date={date_str}, headline={headline}, link={link}"
                    )
            except Exception as e:
                logger.error(f"Error extracting article item: {e}")

        return articles

    async def _fetch_complete_article(
        self, session: aiohttp.ClientSession, article_item: Dict[str, str]
    ) -> Optional[Dict]:
        """Fetch complete article data including content."""
        try:
            content = await self._fetch_article_content(session, article_item["link"])

            # Parse the date string
            parsed_date = self._parse_date(article_item["date_str"])
            if not parsed_date:
                logger.warning(f"Could not parse date: {article_item['date_str']}")
                return None

            return {
                "content": content,
                "headline": article_item["headline"],
                "link": article_item["link"],
                "date": parsed_date,
            }

        except Exception as e:
            logger.error(
                f"Error fetching complete article {article_item.get('link', 'unknown')}: {e}"
            )
            return None

    def _extract_article_date(self, item) -> Optional[str]:
        """Extract the date from an article item."""
        try:
            date_div = item.find("div", class_="views-field-created")
            if not date_div:
                return None
            return date_div.text.strip()
        except AttributeError:
            logger.error("Could not find date element in article item")
            return None

    def _extract_headline_and_link(self, item) -> tuple[Optional[str], Optional[str]]:
        """Extract headline and link from an article item."""
        try:
            # Find headline
            headline_elem = item.find("h3", class_="field-content")
            if not headline_elem:
                return None, None
            headline = headline_elem.text.strip()

            # Find link
            link_elem = item.find("a")
            if not link_elem:
                return None, None
            href = link_elem.get("href", "")
            link = f"{self.base_domain}{href}" if href else None

            return headline, link
        except (AttributeError, KeyError) as e:
            logger.error(f"Error extracting headline and link: {e}")
            return None, None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object."""
        try:
            return datetime.strptime(date_str, self.date_format)
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
            return None

    async def _fetch_article_content(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Fetch the full content of an article asynchronously."""
        soup = await self._get_soup(session, url)
        if not soup:
            return "Content could not be fetched."

        # Look for the specific content field used by Obama archives
        content_div = soup.find(
            "div",
            class_="field field-name-field-forall-body field-type-text-long field-label-hidden forall-body",
        )

        if not content_div:
            logger.warning(f"Content div not found for URL: {url}")
            # Try alternative content selectors
            alternative_selectors = [
                {"class": "field-name-field-forall-body"},
                {"class": "forall-body"},
                {"class": "field-type-text-long"},
            ]

            for selector in alternative_selectors:
                content_div = soup.find("div", selector)
                if content_div:
                    break

            if not content_div:
                return "Content div not found."

        # Clean up content by removing unwanted sections
        content = content_div.get_text()
        content = content.split("Briefings & Statements", 1)[-1].strip()

        return content if content else "No content found."

    async def fetch_body_from_url(self, url: str) -> str:
        """Fetch body content from a specific URL asynchronously."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            soup = await self._get_soup(session, url)
            if not soup:
                return "Body text could not be fetched."

            # Try multiple possible body content selectors
            body_selectors = [
                {"class": "body-content"},
                {"class": "field-name-field-forall-body"},
                {"class": "forall-body"},
            ]

            for selector in body_selectors:
                body_content = soup.find("div", selector)
                if body_content:
                    return body_content.get_text(separator=" ", strip=True)

            return "Body text not found."


class BushArchiveHTMLScraper(DataSource):
    """Async scraper for Bush White House Archives with table-based structure."""

    def __init__(
        self,
        url: str,
        timeout: int = 10,
        max_concurrent: int = 5,
        max_articles: Optional[int] = None,
    ) -> None:
        self.base_url = url
        self.base_domain = "https://georgewbush-whitehouse.archives.gov"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.max_articles = max_articles  # None for unlimited, or specify a number
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.date_format = "%b. %d, %Y"

    async def get_data(self) -> List[Dict]:
        """Main method to scrape articles from the archive table."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            return await self._scrape_articles(session)

    async def _scrape_articles(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Scrape articles from the single page table structure."""
        logger.info(f"Scraping URL: {self.base_url}")

        soup = await self._get_soup(session, self.base_url)
        if not soup:
            return []

        article_items = self._extract_article_items_from_table(soup)
        if not article_items:
            logger.warning("No article items found in table")
            return []

        # Apply article limit if specified
        if self.max_articles:
            article_items = article_items[: self.max_articles]
            logger.info(f"Limited to {len(article_items)} articles")

        # Fetch full content for all articles concurrently
        tasks = [self._fetch_complete_article(session, item) for item in article_items]

        articles = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions
        valid_articles = []
        for article in articles:
            if isinstance(article, dict):
                valid_articles.append(article)
            elif isinstance(article, Exception):
                logger.error(f"Exception while fetching article: {article}")

        logger.info(f"Successfully scraped {len(valid_articles)} articles")
        return valid_articles

    async def _get_soup(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[BeautifulSoup]:
        """Fetch and parse HTML content asynchronously."""
        async with self.semaphore:  # Limit concurrent requests
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    return BeautifulSoup(html, "html.parser")
            except aiohttp.ClientError as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

    def _extract_article_items_from_table(
        self, soup: BeautifulSoup
    ) -> List[Dict[str, Optional[str]]]:
        """Extract article metadata from the archive table."""
        articles = []

        # Find the archive table
        archive_table = soup.find("table", class_="archive")
        if not archive_table:
            logger.error("Archive table not found")
            return []

        # Get all table rows
        rows = archive_table.find_all("tr")
        if not rows:
            logger.error("No table rows found in archive table")
            return []

        for row in rows:
            try:
                date_str, headline, link = self._extract_row_data(row)

                if all([date_str, headline, link]):
                    articles.append(
                        {"date_str": date_str, "headline": headline, "link": link}
                    )
                    logger.debug(f"Extracted article: {headline}")
                else:
                    # Skip rows without complete data (like header rows)
                    continue

            except Exception as e:
                logger.error(f"Error extracting data from table row: {e}")
                continue

        logger.info(f"Extracted {len(articles)} articles from table")
        return articles

    def _extract_row_data(
        self, row
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract date, headline, and link from a table row."""
        # Extract date
        date_cell = row.find("td", class_="archive-date-cell")
        if not date_cell:
            return None, None, None
        date_str = date_cell.text.strip()

        # Extract headline and link
        link_elem = row.find("a")
        if not link_elem:
            return None, None, None

        headline = link_elem.text.strip()
        href = link_elem.get("href", "")
        link = f"{self.base_domain}{href}" if href else None

        return date_str, headline, link

    async def _fetch_complete_article(
        self, session: aiohttp.ClientSession, article_item: Dict[str, str]
    ) -> Optional[Dict]:
        """Fetch complete article data including content."""
        try:
            logger.debug(f"Fetching content for: {article_item['headline']}")
            content = await self._fetch_article_content(session, article_item["link"])

            # Parse the date string
            parsed_date = self._parse_date(article_item["date_str"])
            if not parsed_date:
                logger.warning(f"Could not parse date: {article_item['date_str']}")
                return None

            return {
                "content": content,
                "headline": article_item["headline"],
                "link": article_item["link"],
                "date": parsed_date,
            }

        except Exception as e:
            logger.error(
                f"Error fetching complete article {article_item.get('link', 'unknown')}: {e}"
            )
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object."""
        try:
            return datetime.strptime(date_str, self.date_format)
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
            # Try alternative date formats that might be used
            alternative_formats = ["%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"]
            for alt_format in alternative_formats:
                try:
                    return datetime.strptime(date_str, alt_format)
                except (ValueError, TypeError):
                    continue
            return None

    async def _fetch_article_content(
        self, session: aiohttp.ClientSession, url: str
    ) -> str:
        """Fetch the full content of an article asynchronously."""
        soup = await self._get_soup(session, url)
        if not soup:
            return "Content could not be fetched."

        # The original code looks for the first <p> tag, but let's be more comprehensive
        content_selectors = [
            # Try specific content containers first
            {"class": "content"},
            {"class": "article-content"},
            {"class": "body-content"},
            {"id": "content"},
            # Fallback to first paragraph if specific containers not found
            None,  # This will trigger the <p> tag search
        ]

        content_div = None
        for selector in content_selectors:
            if selector is None:
                # Fallback to first <p> tag (original behavior)
                content_div = soup.find("p")
            else:
                content_div = soup.find("div", selector)

            if content_div:
                break

        if not content_div:
            logger.warning(f"No content found for URL: {url}")
            return "No content found."

        # Extract text content
        content = content_div.get_text(separator=" ", strip=True)

        # Clean up content
        content = content.strip()

        return content if content else "No content found."

    async def fetch_body_from_url(self, url: str) -> str:
        """Fetch body content from a specific URL asynchronously."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            soup = await self._get_soup(session, url)
            if not soup:
                return "Body text could not be fetched."

            # Try multiple possible body content selectors
            body_selectors = [
                {"class": "body-content"},
                {"class": "content"},
                {"class": "article-content"},
                {"id": "content"},
            ]

            for selector in body_selectors:
                body_content = soup.find("div", selector)
                if body_content:
                    return body_content.get_text(separator=" ", strip=True)

            # Fallback to first <p> tag
            p_tag = soup.find("p")
            if p_tag:
                return p_tag.get_text(separator=" ", strip=True)

            return "Body text not found."


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
    archive_links: List[DataSource] = [
        WhiteHouseHTMLScraper("https://www.whitehouse.gov/briefings-statements/"),
        BidenArchiveHTMLScraper(
            "https://bidenwhitehouse.archives.gov/briefing-room/press-briefings/"
        ),
        ObamaArchiveHTMLScraper(
            "https://obamawhitehouse.archives.gov/briefing-room/press-briefings"
        ),
        BushArchiveHTMLScraper(
            "https://georgewbush-whitehouse.archives.gov/news/briefings/"
        ),
    ]
    archive_df: pd.DataFrame = pd.DataFrame()

    for data_source in archive_links:

        ingestion_service = IngestionService(
            primary_source=data_source, fallback_source=data_source
        )

        data = ingestion_service.get_all_releases()
        with open(f"data/{data_source.url}.json", "w") as f:
            f.write(f"{data}")

        if data:
            print(f"Successfully scraped {len(data)} press releases.")
            df = pd.DataFrame(data)
            archive_df = pd.concat([archive_df, df])

        archive_df.to_parquet("data/archive.parquet", index=False)
        archive_df.to_csv("data/archive.csv", index=False)

    return archive_df


def handler(event, context):
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
    result = asyncio.run(main())
    print(f"The result is: {result}")
