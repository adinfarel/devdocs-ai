import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

from typing import Any
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.hybrid import HybridRetriever
from backend.reranker.cross_encoder import Reranker
from backend.generator.llm import LLMGenerator
from pipeline.state import RAGState

logger = logging.getLogger(__name__)

class RAGNodes:
    """
    Container for all LangGraph node functions.

    Each node is a method that takes RAGState and returns
    a dict of updated fields — LangGraph merges this into
    the existing state automatically.

    All dependencies (embedder, retriever, reranker, generator)
    are injected at init time — nodes are pure functions of state,
    not responsible for setup.

    Usage:
        nodes = RAGNodes()
        # pass nodes.embed_query, nodes.hybrid_search, etc.
        # to graph.add_node() in graph.py
    """
    
    CONFIDENCE_THRESHOLD = 0.0
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Initializing RAGNodes — loading all components...")

        self.embedder     = Embedder()
        self.vector_store = VectorStore()
        self.bm25_store   = BM25Store()
        
        loaded = self.bm25_store.load()
        if not loaded:
            raise RuntimeError(
                "BM25 index not found. "
                "Run: python -m backend.retrieval.bm25_store"
            )
        
        self.retriever    = HybridRetriever(
            self.embedder,
            self.vector_store,
            self.bm25_store
        )
        
        self.reranker     = Reranker()
        self.generator    = LLMGenerator()
        
        self.logger.info("RAGNodes initialized successfully.")
    
    def embed_query(self, state: RAGState) -> dict:
        """
        Node 1: Validate and log incoming queries.

        The actual embedding occurs inside HybridRetriever.search()
        —this node exists for explicit state transitions and
        early validation before retrieval.

        Updates state:
        - error: set if query is empty
        """
        query = state.get("query", "").strip()
        
        self.logger.info(f"[embed_query] Query: {query[:80]}")
        
        if not query:
            self.logger.warning(f"[embed_query] Empty query received.")
            return {
                "error": "Query cannot be empty.",
                "candidates": [],
                "reranked": [],
                "confidence": 0.0,
                "answer": "Please provide a question.",
                "sources": [],
                "fallback_triggered": False,
            }
        
        return {"error": None}
    
    def hybrid_search(self, state: RAGState) -> dict:
        """
        Node 2: Retrieve top-20 candidate chunks via hybrid search.

        Runs dense (Qdrant) and sparse (BM25) retrieval in sequence,
        fuses results with RRF.

        Updates state:
            - candidates: top-20 chunks from hybrid search
            - error: set if retrieval fails or returns empty
        """
        if state.get("error"):
            return {}
        
        query = state["query"]
        
        self.logger.info(f"[hybrid_search] Starting hybrid retrieval...")
        
        try:
            candidates = self.retriever.search(query, top_k=20)
        
        except Exception as e:
            self.logger.error(f"[hybrid_search] Retrieval error: {e}")
            return {
                "error": f"Retrieval failed: {str(e)}",
                "candidates": [],
            }
        
        if not candidates:
            self.logger.warning("[hybrid_search] No candidates found")
            return {
                "candidates": [],
                "error": "No relevant documentation found for this query.",
            }
        
        return {"candidates": candidates}

    def rerank(self, state: RAGState) -> dict:
        """
        Node 3: Rerank top-20 candidates to top-5 using cross-encoder.

        Computes confidence score from mean rerank scores —
        used by conditional edge to decide generate vs fallback.

        Updates state:
            - reranked: top-5 chunks after cross-encoder scoring
            - confidence: mean rerank score from top-5
            - error: set if reranking fails
        """
        if state.get("error"):
            return {}
        
        query = state["query"]
        candidates = state.get("candidates", [])
        
        if not candidates:
            return {
                "reranked": [],
                "confidence": 0.0
            }
        
        self.logger.info(
            f"[rerank] Reranking {len(candidates)} candidates..."
        )
        
        try:
            reranked = self.reranker.rerank(query, candidates=candidates, top_k=5)
            confidence = self.reranker.get_confidence(reranked)
        
        except Exception as e:
            self.logger.error(f"[rerank] Reranking failed: {e}")
            return {
                "error": f"Reranking failed: {str(e)}",
                "reranked": [],
                "confidence": 0.0,
            }
        
        self.logger.info(
            f"[rerank] Confidence: {confidence:.4f} "
            f"(threshold: {self.CONFIDENCE_THRESHOLD})"
        )

        return {
            "reranked":   reranked,
            "confidence": confidence,
        }
    
    def generate(self, state: RAGState) -> dict:
        """
        Node 4a: Generate answer from reranked chunks via Groq.

        Executed if confidence >= CONFIDENCE_THRESHOLD.
        Uses non-streaming generate() because LangGraph nodes must return complete state updates — streaming
        happens in the FastAPI endpoint layer, not here.

        Updates state:
        - answer: generated answer string
        - fallback_triggered: False
        """
        if state.get("error"):
            return {}

        query      = state["query"]
        reranked   = state.get("reranked", [])
        confidence = state.get("confidence", 0.0)
        
        self.logger.info(
            f"[generate] Generating answer "
            f"(confidence={confidence:.4f})..."
        )
        
        try:
            answer = self.generator.generate(query, reranked, confidence)
        except Exception as e:
            self.logger.error(f"[generate] Generation failed: {e}")
            return {
                "error":  f"Generation failed: {str(e)}",
                "answer": "An error occurred while generating the answer.",
                "fallback_triggered": False,
            }
        
        self.logger.info(
            f"[generate] Answer generated ({len(answer)} chars)"
        )

        return {
            "answer":             answer,
            "fallback_triggered": False,
        }
    
    def fallback(self, state: RAGState) -> dict:
        """
        Node 4b: Fallback handler for low-confidence retrieval.

        Executed if confidence < CONFIDENCE_THRESHOLD.
        Two sequential strategies:
        1. Query rewriting — rephrase and retry retrieval
        2. Graceful degradation — honest "not found" response

        Updates state:
            - answer: fallback message or rewritten answer
            - fallback_triggered: True
        """
        query = state["query"]
        confidence = state.get("confidence", 0.0)
        
        self.logger.warning(
            f"[fallback] Triggered — confidence {confidence:.4f} "
            f"below threshold {self.CONFIDENCE_THRESHOLD}"
        )
        
        rewrite_prompt = (
            f"Rephrase this technical question more specifically "
            f"for documentation search: {query}"
        )
        
        try:
            rewritten_query = self.generator.generate(
                query=rewrite_prompt,
                chunks=[],
                confidence=None,
            )
            rewritten_query = rewritten_query.strip()
            self.logger.info(
                f"[fallback] Rewritten query: '{rewritten_query[:80]}'"
            )
            
            new_candidates = self.retriever.search(rewritten_query, top_k=20)
            
            if new_candidates:
                new_reranked = self.reranker.rerank(
                    rewritten_query,
                    candidates=new_candidates,
                    top_k=5,
                )
                
                new_confidence = self.reranker.get_confidence(new_reranked)
                
                if new_confidence >= self.CONFIDENCE_THRESHOLD:
                    new_answer = self.generator.generate(
                        rewritten_query,
                        reranked=new_reranked,
                        confidence=new_confidence,
                    )
                    self.logger.info("[fallback] Rewrite strategy succeeded")
                    return {
                        "answer": new_answer,
                        "reranked": new_reranked,
                        "confidence": new_confidence,
                        "fallback_triggered": True,
                    }
                    
        except Exception as e:
            self.logger.error(f"[fallback] Query rewriting failed: {e}")

        self.logger.info("[fallback] Using graceful degradation")
        
        answer = (
            f"I could not find relevant documentation for: '{query}'\n\n"
            f"This may be because:\n"
            f"- The topic is not covered in the indexed documentation "
            f"(FastAPI + PyTorch docs)\n"
            f"- The question uses terminology different from the docs\n\n"
            f"Suggestions:\n"
            f"- Try rephrasing with more specific technical terms\n"
            f"- Check the official docs directly:\n"
            f"  - FastAPI: https://fastapi.tiangolo.com\n"
            f"  - PyTorch: https://pytorch.org/docs/stable"
        )
        
        return {
            "answer":             answer,
            "fallback_triggered": True,
        }
    
    def format_output(self, state: RAGState) -> dict:
        """
        Node 5: Extract source metadata from reranked chunks.

        Strips heavy 'content' fields from reranked chunks —
        The frontend only needs the URL, title, and hierarchy for the SourceCard.
        The content is already in the answer string via LLM citation.

        Updates state:
        - sources: lightweight source metadata list
        """
        reranked = state.get("reranked", [])
        
        sources = [
            {
                "url":       chunk["source_url"],
                "title":     chunk["section_title"],
                "hierarchy": chunk["hierarchy"],
                "doc_source": chunk["doc_source"],
            }
            for chunk in reranked
        ]
        
        self.logger.info(
            f"[format_output] Pipeline complete — "
            f"{len(sources)} sources, "
            f"fallback={state.get('fallback_triggered', False)}"
        )
        
        return {"sources": sources}

    def should_fallback(self, state: RAGState) -> dict:
        """
        Conditional edge function for LangGraph.

        Called after reranking a node — decides whether to proceed to generate or fallback.

        Returns:
        "generate" or "fallback" — this string must
        match the node name registered in graph.py.
        """
        
        if state.get("error"):
            return "generate"
        
        confidence = state.get("confidence", 0.0)
        candidates = state.get("candidates", [])
        
        if not candidates or confidence < self.CONFIDENCE_THRESHOLD:
            return "fallback"

        return "generate"