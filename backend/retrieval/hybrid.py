import logging
from typing import Optional
import sys
from pathlib import Path

# ---- SETUP LOGGER ------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# ----- SETUP PACKAGE -----
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.embedder import Embedder

class HybridRetriever:
    """
    Combines dense retrieval (Qdrant) and sparse retrieval (BM25)
    using Reciprocal Rank Fusion (RRF).

    Neither retriever is strictly better — dense handles semantic
    similarity, BM25 handles exact keyword matches. RRF fuses
    both result lists using rank position only, making it
    scale-invariant across different scoring systems.

    Usage:
        retriever = HybridRetriever(embedder, vector_store, bm25_store)
        results = retriever.search("how to use dependency injection", top_k=20)
    """
    
    RRF_K       = 60
    
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ) -> None:
        """
        Args:
            embedder:     For convert query string to dense vector.
            vector_store: Qdrant wrapper for dense retrieval.
            bm25_store:   BM25 wrapper for sparse retrieval.
        """
        self.logger     = logging.getLogger(self.__class__.__name__)
        self.embedder   = embedder
        self.vector_store = vector_store
        self.bm25_store   = bm25_store
    
    def _rrf(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        Fuse dense and sparse results using RRF.

        Each chunk gets an RRF score from each retriever that returns it. The scores are summed —
        chunks that appear in both retrievers get a bonus
        from both.

        Args:
        dense_results: Output from vector_store.search()
        sparse_results: Output from bm25_store.search()

        Returns:
        Merged and re-ranked list of dicts, sorted by
        RRF score descending.
        """
        
        rrf_scores: dict[str, dict] = {}
        
        # ------- PROCESS DENSE ---------
        for rank, result in enumerate(dense_results, start=1):
            chunk_id    = result['chunk_id']
            rrf_score   = 1.0 / (self.RRF_K + rank)
            
            if chunk_id not in rrf_scores:
                
                rrf_scores[chunk_id] = {
                    "rrf_score": rrf_score,
                    **{k: v for k, v in result.items() if k != "score"},
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                }
            else:
                rrf_scores[chunk_id]['rrf_score'] += rrf_scores
                rrf_scores[chunk_id]['dense_score'] = result["score"]
        
        # ------ PROCESS SPARSE ---------
        for rank, result in enumerate(sparse_results):
            chunk_id  = result["chunk_id"]
            rrf_score = 1.0 / (self.RRF_K + rank + 1)

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    "rrf_score":    rrf_score,
                    **{k: v for k, v in result.items() if k != "score"},
                    "dense_score":  0.0,
                    "sparse_score": result["score"],
                }
            else:
                rrf_scores[chunk_id]["rrf_score"]    += rrf_score
                rrf_scores[chunk_id]["sparse_score"]  = result["score"]
        
        fused       = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        return fused

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_source: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search: dense + BM25 → RRF fusion.

        Args:
        query: Raw query string from the user.
        top_k: Number of final results returned.
        doc_source: Optional filter — "fastapi" or "pytorch".

        Returns:
        List of dicts sorted by RRF score descending.
        Each dict has the following fields: rrf_score, dense_score,
        sparse_score, and all chunk metadata.
        """
        if not query or not query.strip():
            self.logger.warning("Empty query received")
            return []
        
        self.logger.debug(f"Hybrid search: '{query[:60]}'")
        
        query_vector = self.embedder.embed_query(query)
        
        # Retrieve more than request so that fusion has lots material
        candidate_k = top_k * 2
        
        dense_results  = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
            doc_source=doc_source,
        )
        
        sparse_results = self.bm25_store.search(
            query=query,
            top_k=candidate_k,
        )
        
        self.logger.debug(
            f"Dense: {len(dense_results)} results, "
            f"Sparse: {len(sparse_results)} results"
        )
        
        fused = self._rrf_fusion(dense_results, sparse_results)
        
        final = fused[:top_k]
        
        self.logger.debug(
            f"After RRF fusion: {len(fused)} unique chunks → top {len(final)}"
        )

        return final

if __name__ == "__main__":
    embedder    = Embedder()
    vector_store= VectorStore()
    bm25_store  = BM25Store()
    
    loaded      = bm25_store.load()
    if not loaded:
        print("BM25 index not found — run bm25_store.py first")
        exit(1)

    retriever = HybridRetriever(embedder, vector_store, bm25_store)
    
    queries = [
        "how to use dependency injection in FastAPI",
        "autograd and backpropagation in PyTorch",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retriever.search(query, top_k=5)
        for r in results:
            print(
                f"[RRF={r['rrf_score']:.4f} | "
                f"dense={r['dense_score']:.3f} | "
                f"sparse={r['sparse_score']:.2f}] "
                f"{r['section_title']} ({r['doc_source']})"
            )