"""
Document processing module for RAG.

Handles:
- Document loading (TXT, MD, future: PDF)
- Text chunking with overlap
- Metadata extraction

Designed to be extensible for future Glean-like workspace features:
- Document watchers
- Incremental indexing
- Document relationships
"""

from .chunker import TextChunker, Chunk
from .loader import DocumentLoader, Document

__all__ = ["TextChunker", "Chunk", "DocumentLoader", "Document"]

