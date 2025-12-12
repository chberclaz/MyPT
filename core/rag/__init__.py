"""
RAG (Retrieval-Augmented Generation) module.

Components:
- tags: Helper functions for special token formatting
- retriever: Load index and retrieve relevant chunks
- pipeline: Glue retriever + model for RAG generation

Usage:
    from core.rag import Retriever, RAGPipeline
    
    retriever = Retriever("workspace/index/latest")
    pipeline = RAGPipeline(model, retriever)
    answer = pipeline.answer("What is X?")
"""

from .tags import (
    SYSTEM_OPEN, SYSTEM_CLOSE,
    CONTEXT_OPEN, CONTEXT_CLOSE,
    USER_OPEN, USER_CLOSE,
    ASSISTANT_OPEN, ASSISTANT_CLOSE,
    EOT,
    wrap_system, wrap_context, wrap_user, wrap_assistant,
)
from .retriever import Retriever
from .pipeline import RAGPipeline

__all__ = [
    # Tags
    "SYSTEM_OPEN", "SYSTEM_CLOSE",
    "CONTEXT_OPEN", "CONTEXT_CLOSE", 
    "USER_OPEN", "USER_CLOSE",
    "ASSISTANT_OPEN", "ASSISTANT_CLOSE",
    "EOT",
    "wrap_system", "wrap_context", "wrap_user", "wrap_assistant",
    # Components
    "Retriever",
    "RAGPipeline",
]

