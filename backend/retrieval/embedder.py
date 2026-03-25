import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Wrapper around SentenceTransformer for text embedding.

    Converts text or list of texts into dense vectors
    for semantic similarity search in Qdrant.

    Model: all-MiniLM-L6-v2
        - Output dimension: 384
        - Max input tokens: 256
        - Fast, lightweight, good for retrieval tasks
    """
    MODEL_NAME      = "all-MiniLM-L6-v2"
    EMBEDDING_DIM   = 384
    
    def __init__(
        self,
    ) -> None:
        """
        Load the sentence transformer model.
        Model is downloaded on first run (~80MB), cached after.
        """
        self.logger         = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loading embedding model: {self.MODEL_NAME}")
        
        self.model          = SentenceTransformer(self.MODEL_NAME)
        
        self.logger.info(
            f"Model loaded - embeddind dim {self.EMBEDDING_DIM}"
        )
    
    def embed(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Embed a single text or list of texts into dense vectors.

        Args:
        texts: Single string or list of strings.
        batch_size: Number of texts to process at once.
        Larger = faster but requires more RAM.
        show_progress: Show a progress bar for large batches.

        Returns:
        numpy array shape(384,) for single string input,
        shape(N,384) for list of N strings input.
        """
        is_single   = isinstance(texts, str)
        if is_single:
            texts   = [texts]
        
        self.logger.debug(f"Embedding {len(texts)} texts...")
        
        vectors     = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        if is_single:
            return vectors[0]
        
        return vectors
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.
        Shortcut to embed() for clarity in the pipeline code.

        Args:
        query: User query string.

        Returns:
        numpy array shape (384,).
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        return self.embed(query)
    
    def embed_chunks(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Embed a single query string for retrieval.
        Shortcut to embed() for clarity in the pipeline code.

        Args:
        query: User query string.

        Returns:
        numpy array shape(384,).
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        return self.embed(texts, show_progress_bar=show_progress_bar)

if __name__ == "__main__":
    embedder = Embedder()
    
    # test single query
    vec      = embedder.embed_query("how to use depedency injection in FastAPI")
    print(f"Query vector shape: {vec.shape}")
    print(f"Query vector norm : {np.linalg.norm(vec):.4f}")
    
    # test batch
    chunks   = [
        "FastAPI dependency injection allows reuse of logic",
        "Request body is declared using Pydantic models",
        "Path parameters are part of the URL path",
    ]
    vecs     = embedder.embed_chunks(chunks)
    print(f"Batch vector shape: {vecs.shape}")
    
    # test cosine
    scores   = vecs @ vec # (3, 384) @ (1, 384).T
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"[{score:.4f}] {chunk[:50]}")