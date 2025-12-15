"""
WorkspaceEngine - Central workspace abstraction for agentic RAG.

Manages documents, chunks, and provides search capabilities.
Wraps the Phase 3A RAG retriever for search functionality.

Usage:
    engine = WorkspaceEngine("workspace/", "workspace/index/latest")
    
    # List all documents
    docs = engine.list_docs()
    
    # Search by query
    chunks = engine.search("machine learning", top_k=5)
    
    # Get specific document
    doc = engine.get_doc("doc_123")
    text = engine.get_doc_text("doc_123")
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Document:
    """
    Represents a document in the workspace.
    
    Attributes:
        doc_id: Unique identifier (hash of path)
        title: Document title (filename without extension)
        path: Full path to document
        created_at: Creation timestamp
        updated_at: Last modification timestamp
    """
    doc_id: str
    title: str
    path: str
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "path": self.path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.
    
    Attributes:
        chunk_id: Unique identifier
        doc_id: Parent document ID
        text: Chunk text content
        position: Chunk index within document
        score: Similarity score (from search)
        metadata: Additional metadata (source info, etc.)
    """
    chunk_id: str
    doc_id: str
    text: str
    position: int = 0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "position": self.position,
            "score": self.score,
            "metadata": self.metadata,
        }


class WorkspaceEngine:
    """
    Central workspace abstraction managing documents and search.
    
    Args:
        base_dir: Root directory of workspace (contains docs/)
        index_dir: Directory where RAG index lives (embeddings + meta)
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.text', '.markdown'}
    
    def __init__(self, base_dir: str, index_dir: str):
        self.base_dir = Path(base_dir)
        self.index_dir = Path(index_dir)
        self.docs_dir = self.base_dir / "docs"
        
        # Document cache
        self._documents: Dict[str, Document] = {}
        self._doc_id_to_path: Dict[str, str] = {}
        
        # Retriever (lazy loaded)
        self._retriever = None
        
        # Scan documents on init
        self._scan_documents()
        
        # Load retriever if index exists
        if self.index_dir.exists():
            self._load_retriever()
    
    def _generate_doc_id(self, path: str) -> str:
        """Generate a stable document ID from path."""
        # Use relative path for stable IDs
        try:
            rel_path = Path(path).relative_to(self.base_dir)
        except ValueError:
            rel_path = Path(path)
        return hashlib.md5(str(rel_path).encode()).hexdigest()[:12]
    
    def _scan_documents(self) -> None:
        """Scan workspace/docs for documents."""
        self._documents.clear()
        self._doc_id_to_path.clear()
        
        if not self.docs_dir.exists():
            return
        
        for file_path in self.docs_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self._load_document_metadata(str(file_path))
                if doc:
                    self._documents[doc.doc_id] = doc
                    self._doc_id_to_path[doc.doc_id] = str(file_path)
    
    def _load_document_metadata(self, path: str) -> Optional[Document]:
        """Load document metadata from file."""
        try:
            file_path = Path(path)
            stat = file_path.stat()
            
            doc_id = self._generate_doc_id(path)
            title = file_path.stem  # Filename without extension
            
            return Document(
                doc_id=doc_id,
                title=title,
                path=str(file_path),
                created_at=stat.st_ctime,
                updated_at=stat.st_mtime,
            )
        except Exception as e:
            print(f"Error loading document metadata for {path}: {e}")
            return None
    
    def _load_retriever(self) -> None:
        """Load the RAG retriever."""
        try:
            from core.rag import Retriever
            self._retriever = Retriever(str(self.index_dir))
        except Exception as e:
            print(f"Warning: Could not load retriever from {self.index_dir}: {e}")
            self._retriever = None
    
    # ==================== Document Management ====================
    
    def list_docs(self) -> List[Document]:
        """
        List all documents in the workspace.
        
        Returns:
            List of Document objects
        """
        return list(self._documents.values())
    
    def get_doc(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        return self._documents.get(doc_id)
    
    def get_doc_text(self, doc_id: str) -> Optional[str]:
        """
        Get full text content of a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text or None if not found
        """
        path = self._doc_id_to_path.get(doc_id)
        if not path:
            return None
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading document {doc_id}: {e}")
            return None
    
    def get_doc_by_title(self, title: str) -> Optional[Document]:
        """Find document by title (case-insensitive)."""
        title_lower = title.lower()
        for doc in self._documents.values():
            if doc.title.lower() == title_lower:
                return doc
        return None
    
    def refresh(self) -> None:
        """Rescan documents directory."""
        self._scan_documents()
    
    # ==================== Search / RAG ====================
    
    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of Chunk objects with scores
        """
        if self._retriever is None:
            print("Warning: No retriever loaded. Run build_rag_index.py first.")
            return []
        
        results = self._retriever.retrieve(query, top_k=top_k)
        
        chunks = []
        for i, r in enumerate(results):
            source = r.get("source", {})
            
            # Try to find doc_id from source file
            source_file = source.get("file", "")
            doc_id = self._generate_doc_id(source_file) if source_file else f"unknown_{i}"
            
            chunk = Chunk(
                chunk_id=str(r.get("chunk_id", i)),
                doc_id=doc_id,
                text=r.get("text", ""),
                position=source.get("start_line", 0),
                score=r.get("score", 0.0),
                metadata={
                    "filename": source.get("filename", ""),
                    "file": source_file,
                    "start_line": source.get("start_line", 0),
                    "end_line": source.get("end_line", 0),
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get a specific chunk by ID.
        
        Note: This requires the retriever metadata.
        """
        if self._retriever is None:
            return None
        
        meta = self._retriever.get_chunk_by_id(int(chunk_id))
        if meta is None:
            return None
        
        source = meta.get("source", {})
        return Chunk(
            chunk_id=chunk_id,
            doc_id=self._generate_doc_id(source.get("file", "")),
            text=meta.get("text", ""),
            position=source.get("start_line", 0),
            score=0.0,
            metadata=source,
        )
    
    def reload_index(self, index_dir: Optional[str] = None) -> None:
        """
        Reload the RAG index.
        
        Args:
            index_dir: New index directory, or None to reload current
        """
        if index_dir:
            self.index_dir = Path(index_dir)
        self._load_retriever()
    
    # ==================== Properties ====================
    
    @property
    def num_docs(self) -> int:
        """Number of documents in workspace."""
        return len(self._documents)
    
    @property
    def has_index(self) -> bool:
        """Whether RAG index is loaded."""
        return self._retriever is not None and self._retriever.is_loaded
    
    @property
    def num_chunks(self) -> int:
        """Number of indexed chunks."""
        return self._retriever.num_chunks if self._retriever else 0



