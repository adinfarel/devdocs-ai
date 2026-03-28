import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

from sentence_transformers import CrossEncoder

class Reranker:
    """
    Cross-encoder reranker for precision improvement after
    hybrid retrieval.

    Takes top-K candidates from hybrid search and re-scores
    each query-chunk pair using full attention — the query
    and chunk are encoded together, allowing the model to
    see direct token-level interactions.

    This is the second stage of two-stage retrieval:
    Stage 1 (hybrid search) → recall, fast, returns top-20
    Stage 2 (reranker)      → precision, slower, returns top-5

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
        - Trained on MS MARCO passage ranking dataset
        - Output: single relevance score (not normalized)
        - Lightweight, runs on CPU in acceptable time for top-20
    """
    
    MODEL_NAME      = "cross-encoder/ms-marco-MiniLM-L6-v2"
    
    def __init__(self):
        """
        Load cross-encoder model.
        Downloaded on first run (~80MB), cached after.
        """
        self.logger     = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            f"Loading cross-encoder: {self.MODEL_NAME}"
        )
        
        self.model      = CrossEncoder(self.MODEL_NAME)
        
        self.logger.info("Cross-encoder loaded")
    
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Reorder candidate chunks using cross-encoder scores.

        Arguments:
        query: The raw query string from the user.
        candidate: The output of HybridRetriever.search() —
        a list of dicts with a 'content' field.
        top_k: The number of chunks returned after reranking.
        score_threshold: If set, only returns chunks with a cross-encoder score above this threshold.
        None means always return the top_k chunks.

        Returns:
        A list of dicts sorted by cross-encoder score in descending order.
        Each dict can have a 'rerank_score' field added.
        The list is empty if candidate is empty.
        """
        
        if not candidates:
            self.logger.info("No candidate to rerank")
            return []
        
        if not query or not query.strip():
            self.logger.info(f"Empty query received")
            return []
        
        pairs = [
            [query, candidate['content']]
            for candidate in candidates
        ]
        
        self.logger.debug(
            f"Reranking {len(pairs)}, candidates for query: '{query[:60]}'"
        )
        
        scores = self.model.predict(pairs)
        
        scored = []
        for candidate, score in zip(candidates, scores):
            entry   = {**candidate, "rerank_score": score}
            scored.append(entry)
        
        scored.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if score_threshold is not None:
            before  = len(sorted)
            scored  = [s for s in scored if s['rerank_score'] > score_threshold]
            self.logger.debug(
                f"Threshold {score_threshold} filtered"
                f"{before - len(scored)} chunks"
            )
            
        result = scored[:top_k]
        
        self.logger.debug(
            f"Rerank complete — top score: {result[0]['rerank_score']:.4f}, "
            f"bottom score: {result[-1]['rerank_score']:.4f}"
            if result else "No results after reranking"
        )
        
        return result
    
    def get_confidence(self, reranked: list[dict]) -> float:
        """
        Compute the confidence score of the reranking results.

        Used by LangGraph to decide whether to trigger a fallback node or continue generating it.

        Confidence = mean rerank_score of the top results.
        Negative score → low confidence → fallback triggered.

        Args:
        reranked: Output from rerank().

        Returns:
        Mean rerank_score as a float.
        0.0 if reranked is empty.
        """
        
        if not reranked:
            return 0.0
        
        scores = [r['rerank_score'] for r in reranked]
        return sum(scores) / len(scores)
    
if __name__ == "__main__":
    from backend.retrieval.bm25_store import BM25Store
    from backend.retrieval.embedder import Embedder
    from backend.retrieval.hybrid import HybridRetriever
    from backend.retrieval.vector_store import VectorStore
    
    embedder     = Embedder()
    vector_store = VectorStore()
    bm25_store   = BM25Store()
    bm25_store.load()
    
    retriever = HybridRetriever(embedder, vector_store, bm25_store)
    reranker  = Reranker()
    
    query = "how to declare request body in FastAPI"
    
    candidates = retriever.search(query, top_k=20)
    print(f"Stage 1 — {len(candidates)} candidates from hybrid search")
    
    reranked = reranker.rerank(query, candidates, top_k=5)
    
    print(f"\nStage 2 — top 5 after reranking:")
    for r in reranked:
        print(
            f"  [{r['rerank_score']:+.4f}] "
            f"{r['section_title']} ({r['doc_source']})"
        )
        
    confidence = reranker.get_confidence(reranked)
    print(f"\nConfidence score: {confidence:.4f}")
    print(f"Fallback needed : {confidence < 0.0}")