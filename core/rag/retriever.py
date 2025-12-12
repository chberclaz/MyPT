"""
Document retriever for RAG.

Loads pre-built embedding indexes and retrieves relevant chunks
for a given query using cosine similarity.

Features:
- Memory-mapped index loading for instant startup
- Hot-reload capability for index updates
- Full provenance tracking (source file, lines, etc.)
- Pluggable embedder backend

Usage:
    retriever = Retriever("workspace/index/latest")
    results = retriever.retrieve("What is machine learning?", top_k=5)
    
    for r in results:
        print(f"[{r['score']:.2f}] {r['source']['filename']}: {r['text'][:100]}...")
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any

from core.embeddings import LocalEmbedder, BaseEmbedder


class Retriever:
    """
    Document chunk retriever using cosine similarity.
    
    Args:
        index_dir: Path to index directory (or None for lazy loading)
        embedder: Embedder instance (default: LocalEmbedder)
        mmap_mode: Memory-map mode for embeddings ('r' for read-only, None to load into RAM)
    """
    
    def __init__(
        self, 
        index_dir: Optional[str] = None,
        embedder: Optional[BaseEmbedder] = None,
        mmap_mode: str = 'r'
    ):
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.index_config: Dict[str, Any] = {}
        self.index_dir: Optional[str] = None
        self.mmap_mode = mmap_mode
        
        # Use provided embedder or create default
        self.embedder = embedder or LocalEmbedder()
        
        if index_dir:
            self.load_index(index_dir)
    
    def load_index(self, index_dir: str) -> None:
        """
        Load embedding index from directory.
        
        Expected files:
        - embeddings.npy: Shape (N, D) normalized vectors
        - meta.jsonl: One JSON object per line with chunk info
        - config.json: Index configuration (optional)
        
        Args:
            index_dir: Path to index directory
        """
        if not os.path.isdir(index_dir):
            raise ValueError(f"Index directory not found: {index_dir}")
        
        embeddings_path = os.path.join(index_dir, "embeddings.npy")
        meta_path = os.path.join(index_dir, "meta.jsonl")
        config_path = os.path.join(index_dir, "config.json")
        
        if not os.path.exists(embeddings_path):
            raise ValueError(f"Embeddings file not found: {embeddings_path}")
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata file not found: {meta_path}")
        
        # Load embeddings (memory-mapped for large indexes)
        self.embeddings = np.load(embeddings_path, mmap_mode=self.mmap_mode)
        
        # Load metadata
        self.metadata = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        
        # Load config if exists
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.index_config = json.load(f)
        
        self.index_dir = index_dir
        
        # Validate
        if len(self.metadata) != len(self.embeddings):
            raise ValueError(
                f"Mismatch: {len(self.embeddings)} embeddings vs {len(self.metadata)} metadata entries"
            )
        
        print(f"Loaded index: {len(self.metadata)} chunks from {index_dir}")
    
    def reload_index(self, index_dir: Optional[str] = None) -> None:
        """
        Hot-reload index (e.g., after new documents added).
        
        Args:
            index_dir: New index path, or None to reload current
        """
        path = index_dir or self.index_dir
        if not path:
            raise ValueError("No index directory specified")
        
        # Clear memory-mapped arrays
        self.embeddings = None
        self.metadata = []
        
        self.load_index(path)
        print(f"Index reloaded: {len(self.metadata)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: text, score, source, chunk_id
        """
        if self.embeddings is None or len(self.metadata) == 0:
            return []
        
        # Embed query
        query_vec = self.embedder.encode(query)
        
        # Compute similarities (assumes normalized vectors)
        scores = self.embeddings @ query_vec
        
        # Get top-k indices efficiently
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            # Use argpartition for O(n) instead of O(n log n)
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            # Sort the top_k by score
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # Build results with full provenance
        results = []
        for idx in top_indices:
            meta = self.metadata[idx]
            results.append({
                "text": meta.get("text", ""),
                "score": float(scores[idx]),
                "chunk_id": meta.get("chunk_id", idx),
                "source": meta.get("source", {}),
            })
        
        return results
    
    def retrieve_with_threshold(
        self, 
        query: str, 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks above a minimum similarity threshold.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of results with score >= min_score
        """
        results = self.retrieve(query, top_k=top_k)
        return [r for r in results if r["score"] >= min_score]
    
    @property
    def num_chunks(self) -> int:
        """Return number of indexed chunks."""
        return len(self.metadata)
    
    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self.embeddings is not None and len(self.metadata) > 0
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID."""
        for meta in self.metadata:
            if meta.get("chunk_id") == chunk_id:
                return meta
        return None
    
    def search_by_source(self, filename: str) -> List[Dict[str, Any]]:
        """Find all chunks from a specific source file."""
        results = []
        for meta in self.metadata:
            source = meta.get("source", {})
            if filename in source.get("file", "") or filename == source.get("filename", ""):
                results.append(meta)
        return results

