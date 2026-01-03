"""
Workspace module for agentic RAG.

Provides:
- WorkspaceEngine: Document and chunk management
- WorkspaceTools: Tool implementations for agent
- Document, Chunk: Data models

Usage:
    from core.workspace import WorkspaceEngine, WorkspaceTools
    
    engine = WorkspaceEngine("workspace/", "workspace/index/latest")
    tools = WorkspaceTools(engine)
    
    # Search documents
    results = tools.execute("workspace.search", {"query": "...", "top_k": 5})
"""

from .engine import WorkspaceEngine, Document, Chunk
from .tools import WorkspaceTools, TOOL_REGISTRY

__all__ = [
    "WorkspaceEngine",
    "Document", 
    "Chunk",
    "WorkspaceTools",
    "TOOL_REGISTRY",
]






