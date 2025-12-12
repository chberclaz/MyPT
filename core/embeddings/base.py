"""
Abstract base class for text embedders.

All embedder implementations must inherit from BaseEmbedder.
This allows swapping embedding backends without changing RAG code.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class BaseEmbedder(ABC):
    """
    Abstract embedder interface.
    
    Subclasses must implement:
    - encode(text) -> vector
    - encode_batch(texts) -> matrix
    - dim property
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Return embedding dimension."""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text string to a normalized vector.
        
        Args:
            text: Input text string
            
        Returns:
            np.ndarray of shape (dim,), L2-normalized
        """
        pass
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts efficiently.
        
        Default implementation calls encode() in a loop.
        Subclasses may override for batch optimization.
        
        Args:
            texts: List of input strings
            
        Returns:
            np.ndarray of shape (len(texts), dim)
        """
        return np.stack([self.encode(t) for t in texts])
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Assumes vectors are already L2-normalized.
        """
        return float(np.dot(vec1, vec2))
    
    def similarity_matrix(self, query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
        """
        Compute similarities between query and all corpus vectors.
        
        Args:
            query_vec: Shape (dim,)
            corpus_vecs: Shape (N, dim)
            
        Returns:
            np.ndarray of shape (N,) with similarity scores
        """
        return corpus_vecs @ query_vec

