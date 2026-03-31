import logging
import sys
import json
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
        
    def decompose_query(self, state: RAGState) -> dict:
        """
        Node 1 (new): Decompose the query into sub-queries using LLM.

        Always executed — if the query is simple, LLM returns a list
        with one item (the original query). There is no separate classify step
        because false negatives are more dangerous than
        one extra LLM call.

        Updates state:
            - sub_queries: list of sub-query strings
            - is_multi_hop: True if LLM returns more than one sub-query
            - error: set if decomposition fails completely
        """
        if state.get("error"):
            return {}
        
        query = state['query']
        
        self.logger.info(f"[decompose_query] Decomposing query: {query[:80]}")
        
        decompose_prompt = f"""You are a query decomposition system for a technical documentation search engine.

            Your task: decompose the user query into specific sub-queries for document retrieval.

            Rules:
            1. Return ONLY a JSON array of strings — no explanation, no markdown, no preamble
            2. If the query is simple and focused on one concept, return a list with ONE item (the original query, slightly refined for search)
            3. If the query spans multiple distinct concepts, decompose into 2-3 sub-queries
            4. Each sub-query must be self-contained and searchable on its own
            5. Each sub-query must preserve the technical context (e.g. "in FastAPI", "in PyTorch")

            Examples:
            Query: "how to use dependency injection in FastAPI"
            Output: ["dependency injection Depends() FastAPI"]

            Query: "how to validate request body fields and exclude null fields from response in FastAPI"
            Output: ["validate request body fields Field() FastAPI", "exclude null fields response_model_exclude_none FastAPI"]

            Query: "how to freeze layers and use custom optimizer in PyTorch"
            Output: ["freeze layers requires_grad False PyTorch", "custom optimizer parameter groups PyTorch"]

            Now decompose this query:
            "{query}"

            Output (JSON array only):
        """
        try:
            raw = self.generator.generate_raw(
                prompt=decompose_prompt,
            )
            
            self.logger.debug(f"[decompose_query] Raw LLM output: {raw[:200]}")
            
            cleaned = raw.strip()
            
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                
                cleaned = "\n".join(lines[1:-1]).strip()
            
            sub_queries = json.loads(cleaned)
            
            if not isinstance(sub_queries, list):
                raise ValueError(f"Expected list, got {type(sub_queries)}")
            
            if not all(isinstance(q, str) for q in sub_queries):
                raise ValueError(f"All sub-queries must be strings")
            
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            
            if not sub_queries:
                self.logger.warning(
                    "[decompose_query] LLM returned empty list - "
                    "falling back to original query"
                )
                sub_queries = [query]
            
            is_multi_hop = len(sub_queries) > 1
            
            self.logger.info(
                f"[decompose_query] {"Multi-hop" if is_multi_hop else "Single-hop"}"
                f" - {len(sub_queries)} sub-queries: {sub_queries}"
            )
            
            return {
                "sub_queries": sub_queries,
                "is_multi_hop": is_multi_hop,
            }
        
        except json.JSONDecodeError as e:
            self.logger.warning(
                f"[decompose_query] JSON parse failed: {e} — "
                f"falling back to original query"
            )

            return {
                "sub_queries": [query],
                "is_multi_hop": False,
            }
        
        except Exception as e:
            self.logger.error(f"[decompose_query] Unexpected error: {e}")
            return {
                "sub_queries": [query],
                "is_multi_hop": False,
            }
                
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
    
    def multi_hop_search(
        self,
        state: RAGState
    ) -> dict:
        """
        Node 2 (new): Retrieve per sub-query and merge the results.

        Replaces the hybrid_search node for multi-hop queries.
        For single-hop queries (sub_queries with one item), the behavior is identical to the old hybrid_search node.

        Retrieval of each sub-query is executed sequentially
        (not truly parallel) due to Python's GIL limitations
        and Qdrant's free-tier rate limit. For production scale,
        can be upgraded to asyncio.gather().

        Updates state:
        - sub_results: list of retrieval results per sub-query
        - candidates: merged + deduplicated from all sub_results
        - error: set if all sub-query retrievals fail
        """
        if state.get("error"):
            return {}
        
        sub_queries = state.get("sub_queries", [])
        
        if not sub_queries:
            self.logger.warning(
                f"[multi_hop_search] No sub-queries found - "
                "falling back to original query"
            )
            sub_queries = [state["query"]]
        
        self.logger.info(
            f"[multi_hop_search] Retrieving for "
            f"{len(sub_queries)} sub-queries..."
        )
        
        sub_results = []
        successful = 0
        
        for i, sub_query in enumerate(sub_queries):
            self.logger.info(
                f"[multi_hop_search] Sub-query {i+1}/{len(sub_queries)}"
                f"'{sub_query[:80]}'"
            )
            
            try:
                results = self.retriever.search(sub_query, top_k=20)
                sub_results.append(results)
                successful += 1
                
                self.logger.info(
                    f"[multi_hop_query] Sub-query {i+1}"
                    f"{len(results)} results"
                )
            except Exception as e:
                self.logger.error(
                    f"[multi_hop_search] Sub-query {i+1} failed: {e} - "
                    f"appending empty results"
                )
                sub_results.append([])
            
        if successful == 0:
            self.logger.error(
                "[multi_hop_search] All sub-queries failed"
            )
            return {
                "sub_results": sub_results,
                "candidates": [],
                "error": "All retrieval attempts failed"
            }
        
        seen_chunk_ids: set[str] = set()
        merged: list[dict] = []
        
        for i, results in enumerate(sub_results):
            for chunk in results:
                chunk_id = chunk.get("chunk_id", "")
                
                if chunk_id in seen_chunk_ids:
                    self.logger.debug(
                        f"[multi_hop_search] Duplicate chunk skipped: "
                        f"{chunk_id}"
                    )
                    continue
            
                seen_chunk_ids.add(chunk_id)
                
                chunk_with_source = {
                    **chunk,
                    "from_sub_query": i,
                    "sub_query_text": sub_queries[i],
                }
                
                merged.append(chunk_with_source)
        
        self.logger.info(
            f"[multi_hop_search] Merged: {sum(len(r) for r in sub_results)}"
            f"total -> {len(merged)} unique candidates after dedup"
        )
        
        return {
            "sub_results": sub_results,
            "candidates":  merged,
        }
    
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