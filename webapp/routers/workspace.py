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
from typing import Optional, Set
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.logging_config import DebugLogger, is_debug_mode
from webapp.auth import require_user, require_admin, User
from core.document.loader import get_supported_formats

router = APIRouter()
log = DebugLogger("workspace")


def get_supported_extensions() -> Set[str]:
    """Get set of supported file extensions based on available libraries."""
    return {ext for ext, available in get_supported_formats().items() if available}


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
async def get_workspace_info(user: User = Depends(require_user)):
    """Get workspace information including document count and index status."""
    log.request("GET", "/info", user=user.username)
    
    workspace_dir = get_workspace_dir()
    docs_dir = get_docs_dir()
    index_dir = get_index_dir()
    
    log.workspace("scan", workspace=str(workspace_dir))
    
    # Count documents - use dynamic extension list based on available libraries
    supported_exts = get_supported_extensions()
    documents = []
    if docs_dir.exists():
        for f in docs_dir.glob("**/*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
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
    
    if is_debug_mode():
        log.section("WORKSPACE INFO")
        print(f"  Workspace:  {workspace_dir}")
        print(f"  Docs Dir:   {docs_dir}")
        print(f"  Index Dir:  {index_dir}")
        print(f"  Documents:  {len(documents)}")
        print(f"  Chunks:     {num_chunks}")
        print(f"  Has Index:  {has_index}")
        if documents:
            print(f"  Files:")
            for doc in documents:
                print(f"    - {doc['title']} ({doc['path']})")
        log.section("END WORKSPACE INFO")
    
    log.response(200, docs=len(documents), chunks=num_chunks, indexed=has_index)
    
    return {
        "workspace_dir": str(workspace_dir),
        "documents": documents,
        "num_docs": len(documents),
        "num_chunks": num_chunks,
        "has_index": has_index,
        "last_updated": last_updated
    }


@router.post("/rebuild-index")
async def rebuild_index(request: RebuildIndexRequest, user: User = Depends(require_admin)):
    """Rebuild the RAG index from documents - admin only."""
    log.section("REBUILD INDEX")
    log.request("POST", "/rebuild-index", user=user.username)
    
    docs_dir = Path(request.docs_dir) if request.docs_dir else get_docs_dir()
    index_dir = get_index_dir()
    
    if is_debug_mode():
        print(f"  Source:      {docs_dir}")
        print(f"  Destination: {index_dir}")
    
    log.rag("init", docs_dir=str(docs_dir), index_dir=str(index_dir))
    
    if not docs_dir.exists():
        log.error(f"Documents directory not found: {docs_dir}")
        raise HTTPException(status_code=400, detail=f"Documents directory not found: {docs_dir}")
    
    # Create index directory
    index_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import RAG components
        from core.document import DocumentLoader, TextChunker
        from core.embeddings import LocalEmbedder
        from core.rag import Retriever
        
        if is_debug_mode():
            log.section("LOADING DOCUMENTS")
        
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_directory(str(docs_dir))
        
        if is_debug_mode():
            print(f"  Loaded {len(documents)} documents:")
            for doc in documents:
                print(f"    - {doc.filename} ({len(doc.text)} chars)")
        
        if not documents:
            log.warning("No documents found to index")
            return {
                "success": True,
                "message": "No documents found to index",
                "num_chunks": 0
            }
        
        if is_debug_mode():
            log.section("CHUNKING DOCUMENTS")
        
        # Chunk documents
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        all_chunks = []
        
        for doc in documents:
            chunks = chunker.chunk_document(doc, start_chunk_id=len(all_chunks))
            all_chunks.extend(chunks)
            if is_debug_mode():
                print(f"    {doc.filename}: {len(chunks)} chunks")
        
        if is_debug_mode():
            print(f"  Total chunks: {len(all_chunks)}")
        
        if not all_chunks:
            log.warning("No chunks generated from documents")
            return {
                "success": True,
                "message": "No chunks generated from documents",
                "num_chunks": 0
            }
        
        if is_debug_mode():
            log.section("GENERATING EMBEDDINGS")
            print(f"  Creating embeddings for {len(all_chunks)} chunks...")
        
        # Create embeddings
        embedder = LocalEmbedder()
        texts = [chunk.text for chunk in all_chunks]
        embeddings = embedder.encode_batch(texts)
        
        if is_debug_mode():
            print(f"  Embeddings shape: {embeddings.shape}")
        
        if is_debug_mode():
            log.section("SAVING INDEX")
        
        # Save index
        import numpy as np
        import json
        
        # Save embeddings
        np.save(str(index_dir / "embeddings.npy"), embeddings)
        if is_debug_mode():
            print(f"  Saved: embeddings.npy")
        
        # Save metadata
        with open(index_dir / "meta.jsonl", 'w') as f:
            for chunk in all_chunks:
                # Chunk has: chunk_id, text, source (dict with file, filename, start_char, etc.)
                meta = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,  # Contains file, filename, start_line, end_line, etc.
                }
                f.write(json.dumps(meta) + "\n")
        
        if is_debug_mode():
            print(f"  Saved: meta.jsonl ({len(all_chunks)} entries)")
        
        # Save config
        config = {
            "docs_dir": str(docs_dir),
            "num_docs": len(documents),
            "num_chunks": len(all_chunks),
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        with open(index_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        if is_debug_mode():
            print(f"  Saved: config.json")
            log.section("INDEX COMPLETE")
            print(f"  ✓ {len(documents)} documents")
            print(f"  ✓ {len(all_chunks)} chunks")
            print(f"  ✓ {embeddings.shape[1]}-dim embeddings")
        
        log.info(f"Index rebuilt: {len(documents)} docs, {len(all_chunks)} chunks")
        
        return {
            "success": True,
            "message": f"Index rebuilt successfully",
            "num_docs": len(documents),
            "num_chunks": len(all_chunks)
        }
        
    except ImportError as e:
        log.error(f"RAG components not available: {e}")
        if is_debug_mode():
            import traceback
            traceback.print_exc()
        return {
            "success": False,
            "error": f"RAG components not available: {e}",
            "num_chunks": 0
        }
    except Exception as e:
        log.error(f"Failed to rebuild index: {e}")
        if is_debug_mode():
            import traceback
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {e}")


@router.get("/documents")
async def list_documents(user: User = Depends(require_user)):
    """List all documents in the workspace."""
    log.request("GET", "/documents", user=user.username)
    docs_dir = get_docs_dir()
    
    supported_exts = get_supported_extensions()
    documents = []
    if docs_dir.exists():
        for f in docs_dir.glob("**/*"):
            if f.is_file() and f.suffix.lower() in supported_exts:
                documents.append({
                    "doc_id": str(hash(str(f)))[-8:],
                    "title": f.stem,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime
                })
    
    if is_debug_mode():
        log.section("DOCUMENT LIST")
        for doc in documents:
            print(f"  - {doc['title']} ({doc['size']} bytes)")
        log.section("END DOCUMENT LIST")
    
    log.response(200, count=len(documents))
    return {"documents": documents, "total": len(documents)}
