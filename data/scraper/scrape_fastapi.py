import time
from typing import Optional
from pathlib import Path
import sys
from urllib.parse import urljoin
from tqdm import tqdm

from bs4 import BeautifulSoup

from data.scraper.base_scraper import BaseScraper, DocChunk

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------
#       fastapi scraper
# ------------------------------
class FastAPIScraper(BaseScraper):
    """
    Scraper for FastAPI documentation.

    Crawls all pages from three main sections:
    - Tutorial (fastapi.tiangolo.com/tutorial/)
    - Advanced User Guide (fastapi.tiangolo.com/advanced/)
    - Reference (fastapi.tiangolo.com/reference/)

    Inherit BaseScraper for HTTP sessions, error handling,
    rate limiting, and JSON persistence.
    """
    
    BASE_URL = "https://fastapi.tiangolo.com"
    
    SECTIONS = [
        "/tutorial/",
        "/advanced/",
        "/reference/",
    ]
    
    MIN_CONTENT_LENGTH = 200
    REQ_DELAY = 0.5
    
    def __init__(self) -> None:
        """
        Initialize FastAPIScraper with the specified output path and doc_source
        — the user does not need to specify it manually.
        """
        super().__init__(
            output_path=r'data/chunks/fastapi_chunks.json',
            doc_source='fastapi'
        )
    
    # ---------- LINK CONNECTION -------------
    def _extract_links_from_section(self, section_path: str) -> list[str]:
        """
        Get all page links from a single FastAPI docs section.

        FastAPI docs use MkDocs Material — all navigation links
        are in the sidebar with the selector nav.md-nav--primary a.md-nav__link.

        Args:
        section_path: The section path, e.g., "/tutorial/"

        Returns:
        List of full URLs that have not been visited.
        """
        section_url     = self.BASE_URL + section_path
        soup            = self._get_soup(section_url)
        
        if soup is None:
            self.logger.warning(
                f"Could not fetch section {section_url} - Skipping"
            )
            return []
        
        links           = []
        
        for a_tag in tqdm(soup.select("nav.md-nav--primary a.md-nav__link"), desc=f"Fetching link at {section_url}"):
            href        = a_tag.get("href", "").strip()
            
            if not href or href.startswith("javascript"):
                continue
            
            if href.startswith("#"):
                continue
            
            if href.startswith("http"):
                continue
            elif href.startswith("/"):
                full_url = self.BASE_URL + href
            else:
                full_url = urljoin(section_url, href)
            
            full_url    = full_url.split("#")[0]
            
            if not full_url.startswith(self.BASE_URL):
                continue
            
            url_path    = full_url.replace(self.BASE_URL, "")
            if not url_path.startswith(section_path):
                continue
            
            if full_url in self.visited:
                continue
            
            if full_url not in links:
                links.append(full_url)
        
        self.logger.info(
            f"Found {len(links)} links in section {section_path}"
        )
        return links
    
    # ------- HIERARCHY -------------
    def _extract_hierarchy(self, soup: BeautifulSoup) -> list[str]:
        """
        Extract the breadcrumb hierarchy from the page.

        From inspect HTML, we know the breadcrumbs are at:
        nav.md-path ol.md-path__list li

        Example output: ["Learn", "Tutorial - User Guide"]

        Args:
        soup: BeautifulSoup object for the fetched page.

        Returns:
        List of breadcrumb strings. Empty list if not found.
        """
        breadcrumb_items    = soup.select("nav.md-path ol.md-path__list li")
        
        hierarchy           = [
            item.get_text(strip=True)
            for item in breadcrumb_items
            if item.get_text(strip=True)
        ]
        
        return hierarchy

    # ------- PAGE PARSING -------------
    def _parse_page(self, url: str) -> Optional[DocChunk]:
        """
        Extracts DocChunk from a single FastAPI docs page.

        Implementation of the abstract method BaseScraper._parse_page().
        Logic specific to the HTML structure of FastAPI docs (MkDocs Material).

        Args:
        url: Full URL of the page to be parsed.

        Returns:
        DocChunk if successful, None if the page is invalid
        or the content is too short.
        """
        soup        = self._get_soup(url)
        if soup is None:
            return None
        
        
        article     = soup.select_one("article.md-content__inner")
        if article is None:
            self.logger.debug(f"No article content found at {url} - Skipping")
            return None
        
        for noise in article.select(".headerlink"):
            noise.decompose()
        
        h1          = article.select_one("h1")
        if h1:
            title   = h1.get_text(strip=True)
        else:
            title   = url.rstrip("/").split("/")[-1]
            self.logger.debug(f"No h1 found at {url}, using URL segment: {title}")
        
        hierarchy   = self._extract_hierarchy(soup)
        
        for noise in article.select(".admonition-title"):
            noise.decompose()
        
        for noise in article.select(".md-code__nav"):
            noise.decompose()
        
        
        content     = article.get_text(separator="\n", strip=True)
        
        if len(content) < self.MIN_CONTENT_LENGTH:
            self.logger.debug(
                f"Content too short ({len(content)} chars) at {url} - skipping"
            )
            return None
        
        return DocChunk(
            source_url=url,
            section_title=title,
            hierarchy=hierarchy,
            content=content,
            doc_source=self.doc_source
        )
    
    # ------ MAIN ENTRY POINT ----------
    def run(self) -> None:
        """
        Entry point scraping — crawl all sections of the FastAPI docs.

        Flow:
        1. For each section, collect all links from the sidebar.
        2. For each link, parse the page → DocChunk.
        3. Sleep between requests (rate limiting).
        4. Once complete, save to JSON.
        """
        self.logger.info("Starting FastAPI docs scraper...")
        self.logger.info(f"Sections: {self.SECTIONS}")
        
        for section in self.SECTIONS:
            self.logger.info(f"----- Crawling section: {section} ------")
            
            links   = self._extract_links_from_section(section)
            
            for url in links:
                
                if url in self.visited:
                    continue
                
                self.visited.add(url)
                
                chunk = self._parse_page(url)
                
                if chunk:
                    self.chunks.append(chunk)
                    self.logger.info(
                        f"[{len(self.chunks)} {chunk.section_title}]"
                    )
                else:
                    self.logger.debug(f"Skipped: {url}")
                
                
                time.sleep(self.REQ_DELAY)
        
        self.logger.info(
            f"Crawling complete. Total chunks: {len(self.chunks)}"
        )
        
        self._save()


# ------ ENTRY POINT --------
if __name__ == "__main__":
    scraper = FastAPIScraper()
    scraper.run()