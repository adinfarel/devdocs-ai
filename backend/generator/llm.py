import logging
import os
from typing import Optional, Generator

from groq import Groq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

class LLMGenerator:
    """
    Groq-powered LLM generator for RAG response generation.

    Takes reranked chunks as context and generates a grounded
    answer using Groq inference API (llama-3.1-8b-instant).

    Supports both streaming (token by token) and non-streaming
    for flexibility — streaming used in production for UX,
    non-streaming used in evaluation pipeline.

    Usage:
        generator = LLMGenerator()

        # streaming — for FastAPI SSE endpoint
        for token in generator.stream(query, reranked_chunks):
            yield token

        # non-streaming — for evaluation
        answer = generator.generate(query, reranked_chunks)
    """

    MODEL       = "llama-3.1-8b-instant"
    MAX_TOKENS  = 1024
    
    LOW_CONFIDENCE_THRESHOLD = 0.0
    
    def __init__(self):
        """
        Initialize Groq client dari environment variable.
        GROQ_API_KEY must there is in .env
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        api_key     = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Copy .env.example to .env and add your key."
            )
        
        self.client = Groq(api_key=api_key)
        self.logger.info(f"LLMGenerator initialized - model: {self.MODEL}")
        
    def _build_context(self, chunks: list[dict]) -> str:
        """
        Formats the reordered chunks into a context string for the prompt.

        Each chunk is formatted with the source URL and section title
        as headers — this is what LLM uses to cite the source.

        Arguments:
        chunk: Output from Reranker.rerank()

        Returns:
        The formatted context string, ready to be inserted into the prompt.
        """
        if not chunks:
            return "No relevant documentation found"
        
        parts = []
        for i, chunk in enumerate(chunks, 1):
            part = (
                f"[Source {i}]\n"
                f"URL: {chunk['source_url']}\n"
                f"Section: {chunk['section_title']}\n"
                f"Content: {chunk['content']}"
            )
            parts.append(part)
        
        return "\n\n---\n\n".join(parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        confidence: Optional[float] = None,
    ) -> list[dict]:
        """
        Build messages array for Groq API.

        Args:
            query: User query string.
            context: Formatted context of _build_context().
            confidence: Optional reranker confidence score.
                        If it's low, add a disclaimer to the prompt.

        Returns:
            List of message dicts: [system_message, user_message]
        """
        confidence_note = ""
        if confidence is not None and confidence < self.LOW_CONFIDENCE_THRESHOLD:
            confidence_note = (
                f"\nNote: Retrieved context may not be highly relevant"
                f"to this query. If the answer is not clear from the "
                f"context, say so explicity.\n"
            )
        
        system_prompt   = f"""
        You are a precise technical documentation assistant for DevOps and ML developers.
        
        Your task is to answer questions based STRICTLY on the provided documentation context.

        Rules:
        1. Answer ONLY from the provided context — do not use outside knowledge
        2. If the answer is not in the context, say: "I could not find this in the available documentation."
        3. When citing information, reference the source like [Source 1] or [Source 2]
        4. Be concise and technical — the user is a developer
        5. If showing code examples, preserve them exactly as they appear in the context
        6. Do not fabricate API endpoints, parameters, or behaviors{confidence_note}
        """
        
        user_message = f"""Documentation context:

        {context}

        ---

        Question: {query}"""
        
        return [
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_message}
        ]
        
    def generate(
        self,
        query: str,
        chunks: list[dict],
        confidence: Optional[float] = None,
    ) -> str:
        """
        Generates a complete (non-streaming) answer string.
        Used by the evaluation pipeline — RAGAS requires the complete string, not the generator.
        Args:
            query: The user's query.
            chunks: The reordered chunks from the Reranker.
            confidence: An optional confidence score from the Reranker.

        Returns:
            The complete answer string.
        """
        context     = self._build_context(chunks=chunks)
        messages    = self._build_messages(query=query, context=context, confidence=confidence)
        
        self.logger.debug(f"Generating answer for: {query[:60]}")
        
        response    = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=0.1
        )
        
        answer      = response.choices[0].message.content
        self.logger.debug(f"Generated {len(answer)} chars")

        return answer
    
    def generate_raw(
        self,
        prompt: str,
    ) -> str:
        """
        Generates a solution from the raw prompt without the RAG system prompt.

        Used for internal tasks such as query decomposition that do not require RAG foundation constraints.

        Arguments:
        prompt: The full prompt string, directly to the user role.

        Returns:
        The raw solution string from the LLM.
        """
        response   = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role":"user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.0,
        )
        
        return response.choices[0].message.content
    
    def stream(
        self,
        query: str,
        chunks: list[dict],
        confidence: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """
        Streams the answer token by token (generator function).

        Used by the FastAPI SSE endpoint — each token is
        yielded directly to the client without waiting for completion.

        Arguments:
            request: User request.
            chunk: Reranked chunk from Reranker.
            confidence: Optional confidence score from Reranker.

        Result:
            String of tokens one by one.
        """
        context     = self._build_context(chunks=chunks)
        messages    = self._build_messages(query=query, context=context, confidence=confidence)
        
        self.logger.debug(f"Streaming answer for: '{query[:60]}'")
        
        response    = self.client.chat.completions.create(
            model=self.MODEL,
            messages=messages,
            max_tokens=self.MAX_TOKENS,
            temperature=0.1,
            stream=True,
        )
        
        for chunk in response:
            token   = chunk.choices[0].delta.content
            
            if token is not None:
                yield token
                
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from backend.retrieval.bm25_store import BM25Store
    from backend.retrieval.embedder import Embedder
    from backend.retrieval.hybrid import HybridRetriever
    from backend.retrieval.vector_store import VectorStore
    from backend.reranker.cross_encoder import Reranker
    
    embedder     = Embedder()
    vector_store = VectorStore()
    bm25_store   = BM25Store()
    bm25_store.load()
    
    retriever  = HybridRetriever(embedder, vector_store, bm25_store)
    reranker   = Reranker()
    generator  = LLMGenerator()
    
    query = "How do I declare a request body in FastAPI?"
    
    # stage 1 — hybrid search
    candidates = retriever.search(query, top_k=20)
    
    # stage 2 — rerank
    reranked   = reranker.rerank(query, candidates, top_k=5)
    confidence = reranker.get_confidence(reranked)
    
    print(f"Query: {query}")
    print(f"Confidence: {confidence:.4f}")
    print(f"\nStreaming answer:\n{'-'*60}")
    
    # test streaming
    for token in generator.stream(query, reranked, confidence):
        print(token, end="", flush=True)

    print(f"\n{'-'*60}")

    # test non-streaming
    answer = generator.generate(query, reranked, confidence)
    print(f"\nFull answer ({len(answer)} chars):\n{answer[:200]}...")