import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

class BM25Store:
    """
    Sparse retrieval using BM25 over documentation chunks.

    Complements dense retrieval (Qdrant) by handling exact
    keyword matches — critical for DevOps technical terms,
    CLI flags, and command syntax.

    Index is built from tokenized chunk content, persisted
    to disk with pickle, and loaded on subsequent runs.

    Usage:
        store = BM25Store()
        store.build(chunks)
        store.save()

        # later / on restart
        store.load()
        results = store.search("kubectl rollout restart", top_k=20)
    """
    
    def __init__(
        self, 
        index_path: str = "data/bm25_index.pkl"
    ) -> None:
        """
        Args:
            index_path: Path for save/load pickle index.
        """
        
        self.logger     = logging.getLogger(self.__class__.__name__)
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.bm25: Optional[BM25Okapi] = None
        
        self.chunks: list[str]  = []
        
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into a list of lowercase words.

        Simple whitespace tokenization — sufficient for technical documents
        because we want to keep terms like "rollout" and "restart"
        as separate tokens, rather than breaking them down further.

        Arguments:
        text: Raw content string.

        Returns:
        List of lowercase tokens.
        """
        
        return text.lower().split()
    
    def build(self, chunks: list[str]) -> None:
        """
        Build a BM25 index from a list of chunk dicts.

        Tokenize all content and pass it to the BM25Okapi constructor.
        BM25Okapi will calculate the IDF for all terms in the corpus and store it internally.

        Args:
        chunks: List of chunk dicts — same format as
        those loaded from chunked JSON.
        """
        if not chunks:
            self.logger.error(f"Cannot build BM25 index from empty chunks list")
            return
        
        self.logger.info(f"Building BM25 index from {len(chunks)} chunks...")
        
        self.chunks = chunks
        
        tokenized_corpus = [
            self._tokenize(chunk['content'])
            for chunk in chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus) # O(N x avg_token)
        
        self.logger.info(
            f"BM25 index built - {len(chunks)} documents."
            f"Vocab size: {len(self.bm25.idf)} terms"
        )
        
    def save(self) -> None:
        """
        Persist BM25 index and chunks to disk via pickle.

        Called after build() — to avoid rebuilding every time the server restarts.
        """
        if self.bm25 is None:
            self.logger.error(
                f"No index to save - cell build() first"
            )
            return
        
        payload = {
            "bm25": self.bm25,
            "chunks": self.chunks,
        }
        
        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f)
        
        self.logger.info(f"BM25 index saved -> {self.index_path}")
        
    def load(self) -> bool:
        """
        Loads the BM25 index and chunks from disk.

        Returns:
        True if the load was successful, False if the file does not exist.
        The caller can decide: if False → call build() first.
        """
        if not self.index_path.exists():
            self.logger.warning(
                f"No saved index at {self.index_path} - call build() first"
            )
            return False
        
        self.logger.info(f"Loading BM25 index from {self.index_path}")
        
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)
        
        self.bm25   = payload["bm25"]
        self.chunks = payload["chunks"]
        
        self.logger.info(
            f"BM25 Index loaded - {len(self.chunks)} documents"
        )
        
        return True

    def search(
        self,
        query: str, 
        top_k: int = 20,
    ) -> list[dict]:
        """
        Search the BM25 index with a query string.

        Args:
        query: Raw query string — will be tokenized the same as when building the index.

        top_k: The number of top results returned.

        Returns:
        List of dicts with the score field + all metadata chunks.

        Sorted by BM25 score descending order.

        Empty list if the index has not yet been built.
        """
        if self.bm25 is None:
            self.logger.error(f"BM25 index not built - call build() or load() first")
            return []
        
        if not query or not query.strip():
            self.logger.info("Empty query received")
            
        tokenized_query = self._tokenize(query)
        
        scores          = self.bm25.get_scores(tokenized_query)
        
        top_indices     = scores.argsort()[::-1][:top_k]
        
        results         = []
        for idx in top_indices:
            score       = scores[idx]
            
            if score == 0:
                continue
            
            chunk       = self.chunks[idx]
            results.append({
                "score":         float(score),
                "chunk_id":      chunk["chunk_id"],
                "source_url":    chunk["source_url"],
                "section_title": chunk["section_title"],
                "hierarchy":     chunk["hierarchy"],
                "content":       chunk["content"],
                "doc_source":    chunk["doc_source"],
                "chunk_index":   chunk["chunk_index"],
                "total_chunks":  chunk["total_chunks"],
            })
            
        self.logger.debug(
            f"BM25 search '{query[:50]}' -> {len(results)} results"
        )
        
        return results
    
if __name__ == "__main__":
    
    # ----- LOAD CHUNKED DOCS -------
    all_chunks = []
    for path in ["data/chunks/fastapi_chunked.json",
                 "data/chunks/pytorch_chunked.json"]:
        with open(path, "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
    
    print(f"Total chunks {len(all_chunks)}")
    
    store    = BM25Store()
    
    store.build(all_chunks)
    store.save()
    
    
    store2   = BM25Store()
    store2.load()
    
    results = store2.search("dependency injection FastAPI", top_k=5)
    for r in results:
        print(f"[{r['score']:.4f}] {r['section_title']} ({r['doc_source']})")

    print("---")

    results2 = store2.search("kubectl rollout restart", top_k=5)
    for r in results2:
        print(f"[{r['score']:.4f}] {r['section_title']} ({r['doc_source']})")