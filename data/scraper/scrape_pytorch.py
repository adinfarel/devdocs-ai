import time
import sys
from typing import Optional
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm
from bs4 import BeautifulSoup
from data.scraper.base_scraper import BaseScraper, DocChunk

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------
#       pytorch scraper
# -----------------------------
class PyTorchScraper(BaseScraper):
    """
    Scraper for PyTorch documentation.
    
    Crawls all pages from two main sections:
    - Tutorial (pytorch.org/tutorial/)
    - Docs (pytorch.org/docs/stable/)
    
    Inherit BaseScraper for HTTP sessions, error handling,
    rate limiting, and JSON persistence.
    """
    BASE_URL    = "https://docs.pytorch.org/docs/stable"
    
    SECTIONS    = [
        '/'
    ]
    MAX_PAGES   = 300
    
    MIN_CONTENT_LENGTH = 200
    REQ_DELAY   = 1
    
    def __init__(self, output_path: str = f"data/chunks/pytorch_chunks.json", doc_source: str = "pytorch"):
        """
        Initialize PyTorchScraper with the specified output path and doc_source
        — the user does not need to specify it manually (Optional).
        """
        super().__init__(
            output_path=output_path,
            doc_source=doc_source
        )
    
    # ------- LINK CONNECTIONS ---------
    def _extract_links_from_section(self, section_path: str) -> list[str]:
        """
        Get all page links from a single PyTorch docs section.
        
        Args:
        section_path: The section path, e.g., "/tutorial/"

        Returns:
        List of full URLs that have not been visited.
        """
        section_url         = self.BASE_URL + section_path
        to_visit = [section_url]
        links = []
        visited_local = set()
        while to_visit:
            if len(visited_local) >= self.MAX_PAGES:
                self.logger.info(f"Get links have been maximum pages can we get {self.MAX_PAGES}")
                break
            
            url = to_visit.pop()

            if url in visited_local:
                continue

            visited_local.add(url)

            soup = self._get_soup(url)
            
            if soup is None:
                continue

            for a in tqdm(soup.select("article.bd-article a.reference.internal"), desc=f"Fetch link {len(visited_local)} at {url}"):
                href = a.get("href", "").strip()

                if not href:
                    continue

                if href.startswith("#"):
                    continue

                full = urljoin(url, href)
                full = full.split("#")[0]

                if not full.startswith(self.BASE_URL):
                    continue

                if full not in visited_local:
                    to_visit.append(full)
                    links.append(full)
        
        self.logger.info(
            f"Founds {len(links)} links in section {section_path}"
        )
        return links

    # ------- HIERARCHY ----------
    def _extract_hierarchy(self, soup: BeautifulSoup) -> list[str]:
        breadcrumbs     = soup.select("nav.bd-breadcrumbs li.breadcrumb-item a")
        hierarchy       = [
            item.get_text(strip=True)
            for item in breadcrumbs
            if item.get_text(strip=True)
        ]
        
        return hierarchy if hierarchy else ["PyTorch", "Docs"]
    
    # ------- PAGE PARSING ------
    def _parse_page(self, url: str) -> list[DocChunk]:
        """
        Extracts DocChunk from a single PyTorch docs page.

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
        
        article     = soup.select_one("article.bd-article")
        if article is None:
            self.logger.debug(
                f"No article content found at {url} - Skipping"
            )
            return None
        
        for noise in article.select(".headerlink"):
            noise.decompose()

        for noise in article.select(".admonition-title"):
            noise.decompose()

        for noise in article.select(".copybutton"):
            noise.decompose()
        
        h1 = article.select_one("h1")
        if h1:
            title   = h1.get_text(strip=True)
        else:
            title   = url.rstrip("/").split("/")[-1].replace(".html", "")
            self.logger.debug(f"No h1 at {url}, using URL segment: {title}")

        hierarchy   = self._extract_hierarchy(soup)
        
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
    
    # ------ MAIN ENTRY POINT ---------
    def run(self) -> None:
        """
        Entry point scraping — crawl all sections of the FastAPI docs.

        Flow:
        1. For each section, collect all links from the sidebar.
        2. For each link, parse the page → DocChunk.
        3. Sleep between requests (rate limiting).
        4. Once complete, save to JSON.
        """
        self.logger.info("Starting PyTorch docs scraper...")
        self.logger.info(f"Sections: {self.SECTIONS}")
        
        for section in self.SECTIONS:
            self.logger.info(f"----- Crawling section: {section} ------")
            
            links   = self._extract_links_from_section(section)
            
            for url in links:
                if len(self.visited) >= self.MAX_PAGES:
                    self.logger.info(
                        f"Fetch links has been maximal, Total max {self.MAX_PAGES} link that get {len(self.chunks)}"
                    )
                    break
                
                if "generated/" not in url:
                    continue
                
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
    scraper = PyTorchScraper()
    scraper.run()