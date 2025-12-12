"""
Fast, dependency-free local text embedder using hashed character n-grams.

This embedder is designed for:
- Speed: Can embed thousands of chunks per second on CPU
- Simplicity: No external models or dependencies
- Determinism: Same input always produces same output
- Portability: Works offline, pure Python + numpy

Trade-off: Lower semantic quality than neural embedders, but sufficient
for basic retrieval and extremely fast for prototyping.
"""

import numpy as np
import hashlib
from typing import List, Tuple

from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Hashed character n-gram embedder.
    
    Converts text to a fixed-dimension vector by:
    1. Extracting character n-grams (e.g., bigrams, trigrams, 4-grams)
    2. Hashing each n-gram to a bucket index
    3. Accumulating counts in a fixed-size vector
    4. L2-normalizing for cosine similarity
    
    Args:
        dim: Embedding dimension (default: 256)
        ngram_range: Tuple of (min_n, max_n) for n-gram sizes (default: (2, 5))
        lowercase: Whether to lowercase text before embedding (default: True)
    """
    
    def __init__(
        self, 
        dim: int = 256, 
        ngram_range: Tuple[int, int] = (2, 5),
        lowercase: bool = True
    ):
        self._dim = dim
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        
        # Precompute hash seed for consistency
        self._hash_seed = b"myPT_embedder_v1"
    
    @property
    def dim(self) -> int:
        return self._dim
    
    def _hash_ngram(self, ngram: str) -> int:
        """Hash n-gram to bucket index using MD5."""
        data = self._hash_seed + ngram.encode('utf-8', errors='replace')
        hash_bytes = hashlib.md5(data).digest()
        # Use first 8 bytes as integer
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='little')
        return hash_int % self._dim
    
    def _extract_ngrams(self, text: str) -> List[str]:
        """Extract all n-grams from text."""
        if self.lowercase:
            text = text.lower()
        
        ngrams = []
        min_n, max_n = self.ngram_range
        
        for n in range(min_n, max_n + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])
        
        return ngrams
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to fixed-dimension normalized vector.
        
        Args:
            text: Input text string
            
        Returns:
            np.ndarray of shape (dim,), L2-normalized
        """
        vec = np.zeros(self._dim, dtype=np.float32)
        
        ngrams = self._extract_ngrams(text)
        for ngram in ngrams:
            idx = self._hash_ngram(ngram)
            vec[idx] += 1.0
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        
        return vec
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts efficiently.
        
        Args:
            texts: List of input strings
            
        Returns:
            np.ndarray of shape (len(texts), dim)
        """
        # For this simple embedder, just stack individual encodings
        # More sophisticated embedders might batch for GPU efficiency
        return np.stack([self.encode(t) for t in texts])
    
    def get_config(self) -> dict:
        """Return configuration for serialization."""
        return {
            "type": "LocalEmbedder",
            "dim": self._dim,
            "ngram_range": list(self.ngram_range),
            "lowercase": self.lowercase,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> "LocalEmbedder":
        """Create embedder from configuration dict."""
        return cls(
            dim=config.get("dim", 256),
            ngram_range=tuple(config.get("ngram_range", [2, 5])),
            lowercase=config.get("lowercase", True),
        )

