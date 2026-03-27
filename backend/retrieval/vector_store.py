import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

class VectorStore:
    """
    Wrapper around QdrantClient for vector storage and retrieval.

    Handles collection creation, chunk indexing, and
    semantic similarity search with optional metadata filtering.

    Usage:
        store = VectorStore()
        store.create_collection()
        store.index_chunks(chunks, vectors)
        results = store.search(query_vector, top_k=20)
    """
    
    COLLECTION_NAME     = "devdocs"
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant client.

        Args:
            url:     Qdrant server URL. Local dev: http://localhost:6333
                     Production: Qdrant Cloud cluster URL
            api_key: Required for Qdrant Cloud, None for local
        """
        self.logger     = logging.getLogger(self.__class__.__name__)
        self.client     = QdrantClient(url=url, api_key=api_key)
        
        self.logger.info(f"VectorStore connected -> {url}")
    
    def create_collection(
        self,
        embedding_dim: int = 384,
        recreate: bool = False,
    ):
        """
        Create a Qdrant collection to store chunk vectors.

        Args:
        embedding_dim: Must match the Embedder output dim.
        all-MiniLM-L6-v2 → 384
        recreate: If True, deletes the old collection and creates a new one.
        Useful when re-indexing from scratch.
        """
        exists       = self.client.collection_exists(self.COLLECTION_NAME)
        
        if exists and not recreate:
            self.logger.info(
                f"Collection '{self.COLLECTION_NAME}' aldready exists - Skipping"
            )
            return
        
        if exists and recreate:
            self.logger.warning(
                f"Recreating collection '{self.COLLECTION_NAME}' - all data will be lost"
            )
            self.client.delete_collection(self.COLLECTION_NAME)
        
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        
        self.logger.info(
            f"Collection '{self.COLLECTION_NAME}' created"
            f"(dim={embedding_dim}, metric=COSINE)"
        )
    
    def index_chunks(
        self,
        chunks: list[dict],
        vectors: np.ndarray,
        batch_size: int = 100,
    ) -> None:
        """
        Upload chunks and vectors to Qdrant.

        Args:
        chunks: List of chunk dicts from the JSON chunker.
        Each dict must have a chunk_id field.
        vectors: Numpy array of shape (N, 384) — result of Embedder.embed_chunks()
        Must be the same length as the chunks.
        batch_size: Upload N points per request to Qdrant.
        Smaller = safer for the free tier.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks {len(chunks)} and vectors {len(vectors)} "
                f"length mismatch"
            )
        
        total       = len(chunks)
        self.logger.info(
            f"Indexing {total} chunks to Qdrant..."
        )
        
        for batch_start in range(0, total, batch_size):
            batch_end       = min(batch_start + batch_size, total)
            
            batch_chunks    = chunks[batch_start:batch_end]
            batch_vectors   = vectors[batch_start:batch_end]
            
            points = [
                PointStruct(
                    id=abs(hash(chunk['chunk_id'])) % (6**23),
                    vector=vector.tolist(),
                    payload={
                        'chunk_id':      chunk['chunk_id'],
                        'source_url':    chunk['source_url'],
                        'section_title': chunk['section_title'],
                        'hierarchy':     chunk['hierarchy'],
                        "content":       chunk["content"],
                        "doc_source":    chunk["doc_source"],
                        "chunk_index":   chunk["chunk_index"],
                        "total_chunks":  chunk["total_chunks"],
                    },
                ) 
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points,
            )
            
            self.logger.info(
                f"Uploaded batch {batch_start}-{batch_end} / {total}"
            )
        
        self.logger.info(f"Indexing complete - {total} points in Qdrant")
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        doc_source: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic similarity search against all chunks in the collection.

        Args:
        query_vector: Embedded query, shape (384,).
        top_k: Number of top results returned.
        doc_source: Optional filter — "fastapi" or "pytorch".
        None means search across all docs.

        Returns:
        List of dicts, each containing the score + all payload fields.
        Sorted by score descending order (most relevant at index 0).
        """
        query_filter = None
        if doc_source:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_source",
                        match=MatchValue(value=doc_source),
                    )
                ]
            )
        
        # Dense search (Based on semantic (context on doc similiarity with query retrieval))
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        ).points
        
        return [
            {
                "score":         hit.score,
                "chunk_id":      hit.payload["chunk_id"],
                "source_url":    hit.payload["source_url"],
                "section_title": hit.payload["section_title"],
                "hierarchy":     hit.payload["hierarchy"],
                "content":       hit.payload["content"],
                "doc_source":    hit.payload["doc_source"],
                "chunk_index":   hit.payload["chunk_index"],
                "total_chunks":  hit.payload["total_chunks"],
            }
            for hit in results
        ]
    
    def count(self) -> int:
        """
        Returns the number of points stored in the collection.
        Useful for validation after indexing.
        """
        result  = self.client.count(
            collection_name=self.COLLECTION_NAME,
            exact=True
        )
        
        return result.count

# ------- SMOKE TEST ---------
if __name__ == "__main__":
    from backend.retrieval.embedder import Embedder
    
    embedder        = Embedder()
    store           = VectorStore()

    store.create_collection(
        embedding_dim=embedder.EMBEDDING_DIM
    )
    
    # dummy chunks
    dummy_chunks    = [
        {
            "chunk_id":      "fastapi_0_0",
            "source_url":    "https://fastapi.tiangolo.com/tutorial/dependencies/",
            "section_title": "Dependencies",
            "hierarchy":     ["Learn", "Tutorial"],
            "content":       "FastAPI dependency injection allows reuse of logic across endpoints",
            "doc_source":    "fastapi",
            "chunk_index":   0,
            "total_chunks":  1,
        },
        {
            "chunk_id":      "fastapi_1_0",
            "source_url":    "https://fastapi.tiangolo.com/tutorial/body/",
            "section_title": "Request Body",
            "hierarchy":     ["Learn", "Tutorial"],
            "content":       "Request body is declared using Pydantic models in FastAPI",
            "doc_source":    "fastapi",
            "chunk_index":   0,
            "total_chunks":  1,
        },
    ]
    
    # index + embed
    contents        = [c['content'] for c in dummy_chunks]
    vectors         = embedder.embed_chunks(contents)
    store.index_chunks(chunks=dummy_chunks, vectors=vectors)
    
    print(f"Total points in Qdrant: {store.count()}")
    
    # search 
    query_vec       = embedder.embed_query("how to use dependency injection")
    results         = store.search(query_vec, top_k=2)
    
    for r in results:
        print(f"[{r['score']:.4f}] {r['section_title']} - {r['source_url']}")