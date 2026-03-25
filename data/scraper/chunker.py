import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class TextChunk:
    """
    Represents a single chunk of text ready for embedding and indexing.

    Attributes:
        chunk_id: Unique identifier, format: "{doc_source}_{page_idx}_{chunk_idx}"
        source_url: URL of the source page — for later SourceCard.tsx
        section_title: H1 title of the source page
        hierarchy: Breadcrumb of the source page
        content: Text of this chunk — which will be embedded
        doc_source: "fastapi" or "pytorch"
        chunk_index: Position of this chunk within the page (0, 1, 2, ...)
        total_chunks: Total chunks from the same page
    """
    chunk_id: str
    source_url: str
    section_title: str
    hierarchy: list[str]
    content: str
    doc_source: str
    chunk_index: int
    total_chunks: int

class Chunker:
    """
    Splits raw DocChunk content into smaller TextChunks
    ready for embedding and vector store indexing.

    Reads from scraped JSON files produced by FastAPIScraper
    and PyTorchScraper, writes chunked JSON to output path.

    Usage:
        chunker = Chunker()
        chunker.process("data/chunks/fastapi_chunks.json",
                        "data/chunks/fastapi_chunked.json")
    """
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """
        Args:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        """
        self.logger         = logging.getLogger(self.__class__.__name__)
        self.splitter       = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.logger.info(
            f"Chunker initialized - size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def _split_doc(
        self,
        doc: dict,
        page_idx: int,
    ) -> list[TextChunk]:
        """
        Split a raw DocChunk dict into a list of TextChunks.

        Args:
        doc: Dict from JSON scraper — one docs page.
        page_idx: Index of this page in the JSON file (for chunk_id).

        Returns:
        List of TextChunks. Empty list if content is empty.
        """
        content     = doc.get("content", "").strip()
        
        if not content:
            self.logger.debug(
                f"Empty content for {doc.get("source_url", "unknown")} - Skipping"
            )
            return []
        
        splits      = self.splitter.split_text(content)
        
        total       = len(splits)
        chunks      = []
        
        for chunk_idx, split_text in enumerate(splits):
            chunk   = TextChunk(
                chunk_id=f"{doc['doc_source']}_{page_idx}_{chunk_idx}",
                source_url=doc["source_url"],
                section_title=doc['section_title'],
                hierarchy=doc.get('hierarchy', []),
                content=split_text,
                doc_source=doc['doc_source'],
                chunk_index=chunk_idx,
                total_chunks=total
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def process(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """
        Read raw JSON chunks, split all, write results to JSON output.

        Args:
            input_path: Path to the JSON file of the scraper,
                         e.g. "data/chunks/fastapi_chunks.json"
            output_path: JSON chunked output path,
                         e.g. "data/chunks/fastapi_chunked.json"
        """
        input_path      = Path(input_path)
        output_path     = Path(output_path)
        
        if not input_path.exists():
            self.logger.error(
                f"Input file not found: {input_path}"
            )
            return
        
        self.logger.info(f"Reading from {input_path}...")
        
        try:
            with open(input_path, 'r', encoding="utf-8") as f:
                raw_docs    = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON from {input_path}: {e}"
            )
            return
        
        self.logger.info(f"Loaded {len(raw_docs)} raw docs")
        
        all_chunks: list[TextChunk] = []
        
        for page_idx, doc in enumerate(raw_docs):
            chunks      = self._split_doc(doc=doc, page_idx=page_idx)
            all_chunks.extend(chunks)
            
            if (page_idx + 1) % 20 == 0:
                self.logger.info(
                    f"Processed {page_idx + 1}/{len(raw_docs)} docs"
                    f"- {len(all_chunks)} chunks so far"
                )
            
        self.logger.info(
            f"Chunking complete - {len(raw_docs)} docs -> {len(all_chunks)} chunks"
        )
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data    = [asdict(chunk) for chunk in all_chunks]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )
        
        self.logger.info(f"Saved -> {output_path}")

if __name__ == "__main__":
    chunker = Chunker(chunk_size=512, chunk_overlap=50)

    chunker.process(
        input_path="data/chunks/fastapi_chunks.json",
        output_path="data/chunks/fastapi_chunked.json",
    )

    chunker.process(
        input_path="data/chunks/pytorch_chunks.json",
        output_path="data/chunks/pytorch_chunked.json",
    )