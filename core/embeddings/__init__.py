"""
Embeddings module - pluggable text embedding backends.

Supports:
- LocalEmbedder: Fast, CPU-based hashed n-gram embeddings (no deps)
- Future: SentenceTransformerEmbedder, OpenAIEmbedder, etc.
"""

from .base import BaseEmbedder
from .local_embedder import LocalEmbedder

__all__ = ["BaseEmbedder", "LocalEmbedder"]

