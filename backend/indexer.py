import logging
import sys
import json
from typing import Optional
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


    
class Indexer:
    """
    Wrapper all piece, Embedder for Vector Represtation and VectorStore for Vector Databases
    
    Putting all the pieces together starting from:
    - Load chunked docs from data/chunks, e.g. data/chunks/fastapi_chunked.json
    - Embedding all content from chunked docs
    - Store metadata and vector into Qdrant with wrapper VectorStore
    """
    def __init__(self):
        """
        Initialized Embedder and VectorStore
        """
        self.logger         = logging.getLogger(self.__class__.__name__)
        self.embedder       = Embedder()
        self.vector_store   = VectorStore()
        self.logger.info(
            f"Initialized Embedder and VectorStore"
        )
    
    def _extract_all_contents(
        self,
        chunks_docs: list[dict]
    ) -> list[str]:
        """
        Extract all contents from list of dict documentation
        
        Args:
          - chunks_docs: all docs chunked
        
        Returns:
          - given list of strings for embedding in the next section
        """
        if not chunks_docs:
            self.logger.warning(
                f"Chunked docs empty, please check it again"
            )
            return []
        
        all_content     = [
            c['content'] for c in chunks_docs
        ]
        
        return all_content
    
    def index_file(self, chunked_path: str) -> None:
        """
        Load a chunked JSON file, embed its original content, and send it to the server.

        Args:
            chunked_path: The path to the chunked JSON, 
                        e.g. "data/chunks/fastapi_chunked.json
        """
        path    = Path(chunked_path)
        if not path.exists():
            self.logger.error(f"File not found: {path}")
            return
        
        self.logger.info(f"Loading {path}...")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks  = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"No chunks found in {path} - skipping")
            return
        
        self.logger.info(f"Loaded {len(chunks)} chunks from {path.name}")
        
        contents    = self._extract_all_contents(chunks)
        
        if not contents:
            self.logger.warning(f"No content to embed - skipping")
            return
        
        self.logger.info(f"Embedding {len(contents)} chunks...")
        vectors     = self.embedder.embed_chunks(contents, show_progress_bar=True)
        
        # Push to Qdrant
        self.vector_store.index_chunks(chunks=chunks, vectors=vectors)
        
        self.logger.info(
            f"Done indexing {path.name}"
            f"{len(chunks)} chunks in Qdrant"
        )
    
    def run(
        self,
        recreate_collection: bool = False,
    ) -> None:
        """
        Index all chunked documents into Qdrant.

        Sequence:
        1. Create collection (skip if it already exists, unless recreate=True)
        2. Index fastapi_chunked.json
        3. Index pytorch_chunked.json

        Args:
            recreate_collection: If True, delete the old collection
                                 and create a new one from scratch.
                                 Use this if you're rescraping from scratch.
        """
        self.logger.info(f"Starting indexer...")
        
        # Build collection Qdrant
        self.vector_store.create_collection(
            embedding_dim=Embedder.EMBEDDING_DIM,
            recreate=recreate_collection,
        )
        
        chunked_files   = [
            "data/chunks/fastapi_chunked.json",
            "data/chunks/pytorch_chunked.json",
        ]
        
        for file_path in chunked_files:
            self.index_file(file_path)
        
        
        total   = self.vector_store.count()
        self.logger.info(
            f"Indexing complete - {total} total points in Qdrant"
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Index chunked docs to Qdrant",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection from scratch",
    )
    
    args    = parser.parse_args()
    
    indexer = Indexer()
    indexer.run(recreate_collection=args.recreate)