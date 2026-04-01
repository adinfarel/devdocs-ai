import logging
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

from langgraph.graph import StateGraph, END
from pipeline.state import RAGState
from pipeline.nodes import RAGNodes

logger = logging.getLogger(__name__)

class RAGGraph:
    """
    LangGraph pipeline definition for devdocs-ai RAG system.

    Wires all RAGNodes into an explicit state machine:
        embed_query → hybrid_search → rerank → [conditional]
                                                    ↓
                                    generate ← confidence >= 0.0
                                    fallback ← confidence < 0.0
                                                    ↓
                                            output_format → END

    The graph is compiled once at init and reused for every
    incoming query — compilation is expensive, execution is cheap.

    Usage:
        graph = RAGGraph()
        result = graph.run("how to use dependency injection in FastAPI")
    """ 
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Building RAG pipeline graph...")
        
        self.nodes = RAGNodes()
        
        self._graph = self._build()
        
        self.logger.info("RAG pipeline graph built")
    
    def _build(self):
        """
        Define nodes, edges, and conditional routing.

        Returns compiled LangGraph executable.
        """
        builder = StateGraph(RAGState)
        
        # --- ADD NODES ---
        builder.add_node("decompose_query",   self.nodes.decompose_query)
        builder.add_node("multi_hop_search",     self.nodes.multi_hop_search)
        builder.add_node("rerank",            self.nodes.rerank)
        builder.add_node("generate",          self.nodes.generate)
        builder.add_node("fallback",          self.nodes.fallback)
        builder.add_node("format_output",     self.nodes.format_output)
        
        # --- ADD EDGES ---
        builder.set_entry_point("decompose_query")
        
        builder.add_edge("decompose_query",   "multi_hop_search")
        builder.add_edge("multi_hop_search", "rerank")
        
        builder.add_conditional_edges(
            "rerank",
            self.nodes.should_fallback,  
            {
                "generate": "generate",  
                "fallback": "fallback",  
            }
        )
        
        builder.add_edge("generate", "format_output")
        builder.add_edge("fallback", "format_output")
        
        builder.add_edge("format_output", END)
        
        return builder.compile()
    
    def run(self, query: str) -> RAGState:
        """
        Execute the full RAG pipeline for a single query.

        Args:
            query: Raw user query string.

        Returns:
            Final RAGState after all nodes have completed.
            The caller can access state["answer"], state["sources"],
            state["fallback_triggered"], state["error"].
        """
        self.logger.info(f"Running pipeline for: '{query[:80]}")
        
        initial_state: RAGState = {
            "query":              query,
            "is_multi_hop":       False,
            "sub_queries":        [],
            "sub_results":        [],
            "candidates":         [],
            "reranked":           [],
            "confidence":         0.0,
            "answer":             "",
            "sources":            [],
            "fallback_triggered": False,
            "error":              None,
        }
        
        final_state = self._graph.invoke(initial_state)
        
        self.logger.info(
            f"Pipeline complete - "
            f"fallback={final_state['fallback_triggered']}, "\
            f"error={final_state['error']}"
        )
        
        return final_state

# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     load_dotenv()

#     graph = RAGGraph()

#     # test normal query
#     print("\n" + "="*60)
#     print("TEST 1 — Normal query")
#     print("="*60)
#     result = graph.run("How do I declare a request body in FastAPI?")
#     print(f"Answer:\n{result['answer']}")
#     print(f"\nSources:")
#     for s in result["sources"]:
#         print(f"  - {s['title']} ({s['url']})")
#     print(f"\nFallback triggered: {result['fallback_triggered']}")
#     print(f"Error: {result['error']}")

#     # test low relevance query
#     print("\n" + "="*60)
#     print("TEST 2 — Off-topic query (expect fallback)")
#     print("="*60)
#     result2 = graph.run("How do I configure nginx reverse proxy?")
#     print(f"Answer:\n{result2['answer'][:300]}")
#     print(f"\nFallback triggered: {result2['fallback_triggered']}")

#     # test empty query
#     print("\n" + "="*60)
#     print("TEST 3 — Empty query (expect early error)")
#     print("="*60)
#     result3 = graph.run("")
#     print(f"Answer: {result3['answer']}")
#     print(f"Error: {result3['error']}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    graph = RAGGraph()

    # test 1 — single hop
    print("\n" + "="*60)
    print("TEST 1 — Single-hop query")
    print("="*60)
    result = graph.run("How do I declare a request body in FastAPI?")
    print(f"is_multi_hop : {result['is_multi_hop']}")
    print(f"sub_queries  : {result['sub_queries']}")
    print(f"candidates   : {len(result['candidates'])} chunks")
    print(f"confidence   : {result['confidence']:.4f}")
    print(f"answer       : {result['answer'][:200]}")
    print(f"fallback     : {result['fallback_triggered']}")
    print(f"error        : {result['error']}")

    # test 2 — multi hop
    print("\n" + "="*60)
    print("TEST 2 — Multi-hop query")
    print("="*60)
    result2 = graph.run(
        "How do I validate request body fields and exclude "
        "null fields from the response in FastAPI?"
    )
    print(f"is_multi_hop : {result2['is_multi_hop']}")
    print(f"sub_queries  : {result2['sub_queries']}")
    print(f"sub_results  : {[len(r) for r in result2['sub_results']]} chunks per sub-query")
    print(f"candidates   : {len(result2['candidates'])} unique chunks after dedup")
    print(f"confidence   : {result2['confidence']:.4f}")
    print(f"answer       : {result2['answer'][:200]}")
    print(f"fallback     : {result2['fallback_triggered']}")
    print(f"error        : {result2['error']}")

    # test 3 — empty query, fail fast
    print("\n" + "="*60)
    print("TEST 3 — Empty query")
    print("="*60)
    result3 = graph.run("")
    print(f"sub_queries  : {result3['sub_queries']}")
    print(f"answer       : {result3['answer']}")
    print(f"error        : {result3['error']}")