import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

from typing import TypedDict, Optional

class RAGState(TypedDict):
    """
    A shared state object that flows through the entire LangGraph pipeline.

    Each node reads from this state and returns a dict containing the
    updated fields — not replacing the entire state.
    LangGraph merges partial updates automatically.

    Flow:
        embed_query → hybrid_search → rerank → generate/fallback → format_output
    """
    # --- INPUT ---
    query: str
    # --- RETRIEVAL ---
    candidates: list[dict] # Output hybrid search
    reranked: list[dict] # Output cross-encoder
    confidence: float # Mean reranked score
    # --- GENERATION ---
    answer: str
    sources: list[dict]
    # --- CONTROL FLOW ---
    fallback_triggered: bool
    error: Optional[str]
    
    