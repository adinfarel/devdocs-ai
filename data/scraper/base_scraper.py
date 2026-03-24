import json
import logging
import time
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ------------------------------
#       logging setup
# ------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# ------------------------------
#       data model
# ------------------------------

@dataclass
class DocChunk:
    """
    Represents a single scraped documentation chunk.

    Attributes:
        source_url: URL of the scraped page.
        section_title: H1 title of the page.
        hierarchy: Breadcrumb path, e.g., ["Learn", "Tutorial - User Guide"].
        content: Clean scraped text, ready to enter the RAG pipeline.
        doc_source: Identifier of where this document came from, e.g., "fastapi".
    """
    source_url: str
    section_title: str
    hierarchy: list[str]
    content: str
    doc_source: str = "unknown"

# ------------------------------
#       base scraper
# ------------------------------

class BaseScraper(ABC):
    """
    Abstract base class for all documentation scrapers.

    Providing shared infrastructure:
    - HTTP session management
    - HTML fetching with proper error handling
    - Rate limiting
    - JSON persistence

    Subclasses must implement:
    - _parse_page(url) → Optional[DocChunk]
    - run() → None
    - BASE_URL as class variable
    """
    BASE_URL = ""
    
    def __init__(self, output_path: str, doc_source: str) -> None:
        """
        Args:
            output_path: Path file JSON output, e.g. "data/chunks/fastapi_chunks.json"
            doc_source:  Identifier string, e.g. "fastapi", "pytorch"
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.doc_source = doc_source
        
        self.session    = requests.Session()
        self.session.headers.update({
            "User-Agent": "devdocs-ai-scraper/1.0"
        })
        
        self.visited: set[str]    = set()
        self.chunks: list[DocChunk] = []
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    # -------- HTTP ---------------
    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch the URL and return a BeautifulSoup object.

        Every type of network error is handled explicitly —
        nothing is swallowed silently.

        Args:
        url: Full URL to fetch.

        Returns:
        BeautifulSoup if successful, None if failed.
        """
        try:
            response    = self.session.get(url=url, timeout=10)
            
            response.raise_for_status()
            
            return BeautifulSoup(response.text, "html.parser")
        
        except requests.Timeout:
            self.logger.warning(
                f"Timeout fetching {url} - Skipping"
            )
            return None
        
        except requests.HTTPError as e:
            self.logger.warning(
                f"HTTP {e.response.status_code} for {url} - Skipping"
            )
            return None
        
        except requests.ConnectionError:
            self.logger.error(
                f"Connection failed for {url} - check network"
            )
            return None
        
    # ----- PERSISTENCE -------
    def _save(self) -> None:
        """
        Serialize all chunks to a JSON file at output_path.

        asdict() recursively converts the DocChunk dataclass to a dict,
        including nested list[str] in the field hierarchy.
        """
        if not self.chunks:
            self.logger.warning(f"No chunks to save - scraping may have failed")
            return
        
        data    = [asdict(chunk) for chunk in self.chunks]
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        self.logger.info(
            f"Saved {len(self.chunks)} chunks -> {self.output_path}"
        )
    
    # ------- ABS METHOD ---------
    @abstractmethod
    def _parse_page(self, url):
        """
        Extract DocChunk from a single documentation page.

        Must be implemented for each subclass because the HTML structure of each docs site is different.

        Args:
        url: Full URL of the page to be parsed.

        Returns:
        DocChunk if the content extraction was successful, None if it failed
        or the page is irrelevant.
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """
        Entry point to start the scraping process.

        It must be implemented for each subclass because the crawling flow

        of each site's docs is different — some use a sitemap,
        others crawl from the sidebar nav.
        """
        pass