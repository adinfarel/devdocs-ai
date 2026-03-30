import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- SETUP LOGGER ------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

# ----- SETUP PACKAGE -----
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline.graph import RAGGraph
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.bm25_store import BM25Store
from backend.retrieval.hybrid import HybridRetriever
from backend.reranker.cross_encoder import Reranker
from backend.generator.llm import LLMGenerator

# --- SETUP SCHEMA ---
class QueryRequest(BaseModel):
    """
    Request body untuk POST /query.
    """
    query: str
    doc_source: str | None = None
    
class QueryResponse(BaseModel):
    """
    Response body untuk POST /query/sync (non-streaming).
    Used in the evaluation pipeline.
    """
    answer: str
    sources: list[dict]
    confidence: float
    fallback_triggered: bool
    error: str | None

# --- SETUP COMPONENTS ---
embedder:         Embedder        = None
vector_store:     VectorStore     = None
bm25_store:       BM25Store       = None
retriever:        HybridRetriever = None
reranker:         Reranker        = None
generator:        LLMGenerator    = None
graph:            RAGGraph        = None

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — loads all ML models once
    at server startup, not per request.

    Without this, each request would reload Embedder and
    CrossEncoder from disk — a latency of thousands of ms per request.
    """
    global embedder, vector_store, bm25_store
    global retriever, reranker, generator, graph
    
    logger.info("Starting up - loading all components...")

    embedder     = Embedder()
    vector_store = VectorStore()
    bm25_store   = BM25Store()
    bm25_store.load()

    retriever    = HybridRetriever(embedder, vector_store, bm25_store)
    reranker     = Reranker()
    generator    = LLMGenerator()
    graph        = RAGGraph()
    
    logger.info("All components loaded - server ready")
    
    yield
    
    logger.info("Shutting down")

# --- SETUP FASTAPI ---
app = FastAPI(
    title="devdocs-ai",
    description="Production-grade RAG for DevOps documentation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the DevDocs AI backend! Visit /docs for API documentation."}

@app.get("/health")
async def health():
    """
    Health check endpoint for Railway deployment.
    Railway pings this endpoint every 30 seconds — if it returns a non-200,
    Railway restarts the container.
    """
    return {
        "status": "ok",
        "model": LLMGenerator.MODEL,
        "version": "0.1.0",
    }

@app.post("/query")
async def query(request: QueryRequest):
    """
    Main query endpoint — streams answer tokens one by one through SSE.

    Flow:
    1. Hybrid search → top 20 candidates
    2. Reordering → top 5 + confidence
    3. Decide: yield or return to backup choice
    4. Stream tokens to client

    SSE format:
        data: <token>\n\n
        data: [SOURCE] <json>\n\n
        data: [DONE]\n\n
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # --- EMBEDDING ---
            query_vector = embedder.embed_chunks(request.query)
            # --- HYBRID RETRIEVAL ---
            candidates = retriever.search(
                request.query,
                top_k=20,
                doc_source=request.doc_source,
            )
            
            if not candidates:
                yield "data: I could not find relevant documentation.\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # --- RERANKING ---
            reranked = reranker.rerank(request.query, candidates)
            confidence = reranker.get_confidence(reranked)
            
            if confidence < RAGGraph.CONFIDENCE_THRESHOLD \
                if hasattr(RAGGraph, 'CONFIDENCE_THRESHOLD') \
                else confidence < 0.0:
                
                fallback_msg = (
                    f"I could not find relevant documentation "
                    f"for: '{request.query}'"
                )
                
                yield f"data: {fallback_msg}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            for token in generator.stream(request.query, reranked, confidence):
                token_escaped = token.replace("\n", "<br>")
                yield f"data: {token_escaped}\n\n"
            
            
            import json
            sources = [
                {
                    "url":        chunk["source_url"],
                    "title":      chunk["section_title"],
                    "hierarchy":  chunk["hierarchy"],
                    "doc_source": chunk["doc_source"],
                }
                for chunk in reranked
            ]
            yield f"data: [SOURCES] {json.dumps(sources)}\n\n"
            yield "data: [DONE]\n\n"
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/query/sync", response_model=QueryResponse)
async def query_sync(request: QueryRequest):
    """
    Non-streaming query endpoint — return complete response.
    Evaluation pipeline is used (RAGAS requires complete string).
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = graph.run(request.query)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        fallback_triggered=result["fallback_triggered"],
        error=result["error"],
    )
