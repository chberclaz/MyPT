"""
RAG Pipeline - glues retriever and model for augmented generation.

The pipeline:
1. Takes a user question
2. Retrieves relevant context from the index
3. Builds a prompt with context
4. Generates an answer using the model

Usage:
    from core.rag import RAGPipeline, Retriever
    from core import load_model
    
    model = load_model("checkpoints/my_model")
    retriever = Retriever("workspace/index/latest")
    pipeline = RAGPipeline(model, retriever)
    
    answer = pipeline.answer("What is the capital of France?")
"""

from typing import Optional, List, Dict, Any, Callable

from .retriever import Retriever
from .tags import (
    build_rag_prompt, 
    format_context_block, 
    wrap_context,
    ASSISTANT_CLOSE,
)

# Import audit logging (optional - doesn't fail if not available)
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    Args:
        model: GPT model instance with generate() method
        retriever: Retriever instance with loaded index
        default_system: Default system prompt (optional)
        default_top_k: Default number of chunks to retrieve
    """
    
    def __init__(
        self,
        model,
        retriever: Retriever,
        default_system: str = "",
        default_top_k: int = 5,
    ):
        self.model = model
        self.retriever = retriever
        self.default_system = default_system
        self.default_top_k = default_top_k
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Search query
            top_k: Number of chunks (default: self.default_top_k)
            min_score: Minimum similarity threshold
            
        Returns:
            List of chunk dicts with text, score, source
        """
        k = top_k or self.default_top_k
        chunks = self.retriever.retrieve_with_threshold(query, top_k=k, min_score=min_score)
        
        # Audit: RAG retrieval
        if AUDIT_AVAILABLE:
            avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks) if chunks else 0
            audit.rag("retrieve", 
                     query_length=len(query),
                     chunks_retrieved=len(chunks),
                     top_k=k,
                     min_score=min_score,
                     avg_score=f"{avg_score:.3f}",
                     details=query[:100] + ("..." if len(query) > 100 else ""))
        
        return chunks
    
    def build_prompt(
        self,
        question: str,
        context_chunks: Optional[List[Dict]] = None,
        system: Optional[str] = None,
        include_source: bool = True
    ) -> str:
        """
        Build a complete RAG prompt.
        
        Args:
            question: User's question
            context_chunks: Retrieved chunks (or None to retrieve automatically)
            system: System prompt (or None to use default)
            include_source: Include source attribution in context
            
        Returns:
            Complete prompt string ready for generation
        """
        sys_prompt = system if system is not None else self.default_system
        
        return build_rag_prompt(
            question=question,
            context_chunks=context_chunks,
            system=sys_prompt if sys_prompt else None,
            include_source=include_source,
        )
    
    def answer(
        self,
        question: str,
        system: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        max_new_tokens: int = 256,
        include_source: bool = True,
        return_context: bool = False,
        temperature: float = 0.7,
        top_k_sampling: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str | tuple:
        """
        Generate a RAG-augmented answer.
        
        Args:
            question: User's question
            system: Optional system prompt override
            top_k: Number of context chunks to retrieve
            min_score: Minimum similarity for retrieved chunks
            max_new_tokens: Maximum tokens to generate
            include_source: Include source attribution in context
            return_context: If True, return (answer, context_chunks) tuple
            temperature: Sampling temperature (0.0=deterministic, 1.0=neutral)
            top_k_sampling: Only sample from top K tokens (0=disabled)
            top_p: Nucleus sampling threshold (1.0=disabled)
            repetition_penalty: Penalize repeated tokens (1.0=disabled)
            
        Returns:
            Generated answer string, or (answer, chunks) if return_context=True
        """
        # Retrieve context
        chunks = self.retrieve_context(question, top_k=top_k, min_score=min_score)
        
        # Build prompt
        prompt = self.build_prompt(
            question=question,
            context_chunks=chunks,
            system=system,
            include_source=include_source,
        )
        
        # Generate answer
        full_response = self.model.generate(
            prompt, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sampling,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        # Extract just the assistant's response (after the prompt)
        # The prompt ends with <myPT_assistant>, response follows
        if prompt in full_response:
            answer = full_response[len(prompt):]
        else:
            answer = full_response
        
        # Clean up: remove closing tag if present
        if ASSISTANT_CLOSE in answer:
            answer = answer.split(ASSISTANT_CLOSE)[0]
        
        answer = answer.strip()
        
        # Audit: RAG answer generation
        if AUDIT_AVAILABLE:
            audit.rag("answer",
                     context_chunks=len(chunks),
                     answer_length=len(answer),
                     max_new_tokens=max_new_tokens,
                     temperature=temperature,
                     details=answer[:100] + ("..." if len(answer) > 100 else ""))
        
        if return_context:
            return answer, chunks
        return answer
    
    def answer_with_sources(
        self,
        question: str,
        system: Optional[str] = None,
        top_k: Optional[int] = None,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Generate answer with full source information.
        
        Returns:
            Dict with: answer, sources (list), prompt, scores
        """
        answer, chunks = self.answer(
            question=question,
            system=system,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            return_context=True,
        )
        
        return {
            "answer": answer,
            "sources": [
                {
                    "filename": c["source"].get("filename", "unknown"),
                    "file": c["source"].get("file", ""),
                    "score": c["score"],
                    "text_preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                }
                for c in chunks
            ],
            "num_sources": len(chunks),
        }
    
    def reload_index(self, index_dir: Optional[str] = None) -> None:
        """Hot-reload the retriever's index."""
        self.retriever.reload_index(index_dir)
    
    @property
    def index_loaded(self) -> bool:
        """Check if retriever has a loaded index."""
        return self.retriever.is_loaded


class StreamingRAGPipeline(RAGPipeline):
    """
    RAG Pipeline with streaming generation support.
    
    For models that support token-by-token generation callbacks.
    (Future extension point)
    """
    
    def answer_stream(
        self,
        question: str,
        on_token: Callable[[str], None],
        **kwargs
    ):
        """
        Stream answer token by token.
        
        Args:
            question: User's question
            on_token: Callback for each generated token
            **kwargs: Same as answer()
        """
        # TODO: Implement when model supports streaming
        # For now, fall back to non-streaming
        answer = self.answer(question, **kwargs)
        on_token(answer)
        return answer

