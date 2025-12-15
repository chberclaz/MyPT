"""
Workspace API Router - Handles workspace and index operations

Endpoints:
- GET /info - Get workspace info (docs, chunks, status)
- POST /rebuild-index - Rebuild the RAG index
- GET /documents - List documents
"""

import os
import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


class RebuildIndexRequest(BaseModel):
    docs_dir: Optional[str] = None


def get_workspace_dir() -> Path:
    """Get the workspace directory."""
    return PROJECT_ROOT / "workspace"


def get_docs_dir() -> Path:
    """Get the documents directory."""
    return get_workspace_dir() / "docs"


def get_index_dir() -> Path:
    """Get the index directory."""
    return get_workspace_dir() / "index" / "latest"


@router.get("/info")
async def get_workspace_info():
    """Get workspace information including document count and index status."""
    workspace_dir = get_workspace_dir()
    docs_dir = get_docs_dir()
    index_dir = get_index_dir()
    
    # Count documents
    documents = []
    if docs_dir.exists():
        for f in docs_dir.glob("**/*"):
            if f.is_file() and f.suffix in (".txt", ".md", ".rst"):
                documents.append({
                    "doc_id": str(hash(str(f)))[-8:],
                    "title": f.stem,
                    "path": str(f.relative_to(workspace_dir))
                })
    
    # Check index status
    num_chunks = 0
    last_updated = None
    has_index = False
    
    if index_dir.exists():
        # Check for embeddings file
        embeddings_file = index_dir / "embeddings.npy"
        meta_file = index_dir / "meta.jsonl"
        
        if embeddings_file.exists():
            has_index = True
            last_updated = embeddings_file.stat().st_mtime
            
            # Count chunks from meta file
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    num_chunks = sum(1 for _ in f)
    
    return {
        "workspace_dir": str(workspace_dir),
        "documents": documents,
        "num_docs": len(documents),
        "num_chunks": num_chunks,
        "has_index": has_index,
        "last_updated": last_updated
    }


@router.post("/rebuild-index")
async def rebuild_index(request: RebuildIndexRequest):
    """Rebuild the RAG index from documents."""
    docs_dir = Path(request.docs_dir) if request.docs_dir else get_docs_dir()
    index_dir = get_index_dir()
    
    if not docs_dir.exists():
        raise HTTPException(status_code=400, detail=f"Documents directory not found: {docs_dir}")
    
    # Create index directory
    index_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import RAG components
        from core.document import DocumentLoader, TextChunker
        from core.embeddings import LocalEmbedder
        from core.rag import Retriever
        
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_directory(str(docs_dir))
        
        if not documents:
            return {
                "success": True,
                "message": "No documents found to index",
                "num_chunks": 0
            }
        
        # Chunk documents
        chunker = TextChunker(chunk_size=500, overlap=50)
        all_chunks = []
        
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return {
                "success": True,
                "message": "No chunks generated from documents",
                "num_chunks": 0
            }
        
        # Create embeddings
        embedder = LocalEmbedder()
        texts = [chunk.text for chunk in all_chunks]
        embeddings = embedder.embed(texts)
        
        # Save index
        import numpy as np
        import json
        
        # Save embeddings
        np.save(str(index_dir / "embeddings.npy"), embeddings)
        
        # Save metadata
        with open(index_dir / "meta.jsonl", 'w') as f:
            for chunk in all_chunks:
                meta = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "position": chunk.position,
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(meta) + "\n")
        
        # Save config
        config = {
            "docs_dir": str(docs_dir),
            "num_docs": len(documents),
            "num_chunks": len(all_chunks),
            "chunk_size": 500,
            "overlap": 50
        }
        with open(index_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return {
            "success": True,
            "message": f"Index rebuilt successfully",
            "num_docs": len(documents),
            "num_chunks": len(all_chunks)
        }
        
    except ImportError as e:
        # RAG components not available
        return {
            "success": False,
            "error": f"RAG components not available: {e}",
            "num_chunks": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {e}")


@router.get("/documents")
async def list_documents():
    """List all documents in the workspace."""
    docs_dir = get_docs_dir()
    
    documents = []
    if docs_dir.exists():
        for f in docs_dir.glob("**/*"):
            if f.is_file() and f.suffix in (".txt", ".md", ".rst"):
                documents.append({
                    "doc_id": str(hash(str(f)))[-8:],
                    "title": f.stem,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime
                })
    
    return {"documents": documents, "total": len(documents)}

