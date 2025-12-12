"""
Workspace tools for agentic RAG.

Implements the workspace.* tool set that the model can call:
- workspace.search: Search documents by semantic similarity
- workspace.list_docs: List all documents
- workspace.get_doc: Get full document text
- workspace.summarize: Summarize a document or text

Each tool:
- Takes a dict as input (from parsed JSON)
- Returns a dict that's JSON-serializable
- Is deterministic/pure

Usage:
    from core.workspace import WorkspaceEngine, WorkspaceTools
    
    engine = WorkspaceEngine("workspace/", "workspace/index/latest")
    tools = WorkspaceTools(engine)
    
    result = tools.execute("workspace.search", {"query": "python", "top_k": 5})
"""

from typing import Dict, Any, Optional, Callable
from .engine import WorkspaceEngine


class WorkspaceTools:
    """
    Tool implementations for workspace operations.
    
    Args:
        engine: WorkspaceEngine instance
        model: Optional model for summarization (if None, uses extractive summary)
    """
    
    def __init__(self, engine: WorkspaceEngine, model=None):
        self.engine = engine
        self.model = model  # For generative summarization
        
        # Build tool registry
        self._tools: Dict[str, Callable] = {
            "workspace.search": self.workspace_search,
            "workspace.list_docs": self.workspace_list_docs,
            "workspace.get_doc": self.workspace_get_doc,
            "workspace.summarize": self.workspace_summarize,
        }
    
    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Tool name (e.g., "workspace.search")
            args: Tool arguments as dict
            
        Returns:
            Tool result as dict
            
        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self._tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self._tools.keys()),
            }
        
        try:
            return self._tools[tool_name](args)
        except Exception as e:
            return {
                "error": str(e),
                "tool": tool_name,
            }
    
    def list_tools(self) -> list:
        """List available tool names."""
        return list(self._tools.keys())
    
    # ==================== Tool Implementations ====================
    
    def workspace_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search documents by semantic similarity.
        
        Args:
            query (str): Search query
            top_k (int, optional): Number of results (default: 5)
            
        Returns:
            {
                "documents": [
                    {"chunk_id": str, "doc_id": str, "text": str, "score": float, ...},
                    ...
                ],
                "total": int
            }
        """
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))
        
        if not query:
            return {"error": "Missing required argument: query", "documents": []}
        
        chunks = self.engine.search(query, top_k=top_k)
        
        return {
            "documents": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "score": round(c.score, 4),
                    "filename": c.metadata.get("filename", ""),
                    "position": c.position,
                }
                for c in chunks
            ],
            "total": len(chunks),
        }
    
    def workspace_list_docs(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        List all documents in the workspace.
        
        Args:
            (none)
            
        Returns:
            {
                "documents": [
                    {"doc_id": str, "title": str, "path": str},
                    ...
                ],
                "total": int
            }
        """
        docs = self.engine.list_docs()
        
        return {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "path": d.path,
                }
                for d in docs
            ],
            "total": len(docs),
        }
    
    def workspace_get_doc(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get full document text.
        
        Args:
            doc_id (str): Document ID
            OR
            title (str): Document title
            
        Returns:
            {
                "doc_id": str,
                "title": str,
                "text": str,
                "length": int
            }
        """
        doc_id = args.get("doc_id")
        title = args.get("title")
        
        doc = None
        if doc_id:
            doc = self.engine.get_doc(doc_id)
        elif title:
            doc = self.engine.get_doc_by_title(title)
        
        if doc is None:
            return {
                "error": f"Document not found: {doc_id or title}",
                "doc_id": doc_id,
                "title": title,
            }
        
        text = self.engine.get_doc_text(doc.doc_id)
        
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": text or "",
            "length": len(text) if text else 0,
        }
    
    def workspace_summarize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a document or provided text.
        
        Args:
            doc_id (str, optional): Document ID to summarize
            text (str, optional): Text to summarize (if no doc_id)
            max_length (int, optional): Max summary length (default: 500)
            
        Returns:
            {
                "summary": str,
                "source": "doc_id" | "text",
                "original_length": int
            }
        """
        doc_id = args.get("doc_id")
        text = args.get("text")
        max_length = int(args.get("max_length", 500))
        
        source_text = None
        source_type = None
        
        if doc_id:
            source_text = self.engine.get_doc_text(doc_id)
            source_type = "doc_id"
            if source_text is None:
                return {"error": f"Document not found: {doc_id}"}
        elif text:
            source_text = text
            source_type = "text"
        else:
            return {"error": "Missing required argument: doc_id or text"}
        
        # Generate summary
        if self.model is not None:
            # Use model for generative summary
            summary = self._generate_summary(source_text, max_length)
        else:
            # Extractive summary (first N chars + ellipsis)
            summary = self._extractive_summary(source_text, max_length)
        
        return {
            "summary": summary,
            "source": source_type,
            "original_length": len(source_text),
        }
    
    def _extractive_summary(self, text: str, max_length: int) -> str:
        """Simple extractive summary: first paragraph(s) up to max_length."""
        # Split into paragraphs
        paragraphs = text.strip().split('\n\n')
        
        summary = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_length + len(para) > max_length:
                if not summary:
                    # At least include truncated first paragraph
                    summary.append(para[:max_length - 3] + "...")
                break
            
            summary.append(para)
            current_length += len(para) + 2  # +2 for newlines
        
        return '\n\n'.join(summary) if summary else text[:max_length]
    
    def _generate_summary(self, text: str, max_length: int) -> str:
        """Use model to generate summary (if model available)."""
        if self.model is None:
            return self._extractive_summary(text, max_length)
        
        # Build a summarization prompt
        from core.rag.tags import wrap_context, wrap_user, ASSISTANT_OPEN
        
        prompt = (
            wrap_context(text[:2000]) + "\n" +  # Limit context
            wrap_user("Summarize the above text concisely.") + "\n" +
            ASSISTANT_OPEN
        )
        
        try:
            response = self.model.generate(prompt, max_new_tokens=max_length // 4)
            # Extract just the generated part
            if ASSISTANT_OPEN in response:
                response = response.split(ASSISTANT_OPEN)[-1]
            return response.strip()[:max_length]
        except Exception as e:
            print(f"Summary generation failed: {e}")
            return self._extractive_summary(text, max_length)


# ==================== Tool Registry ====================

def get_tool_registry(tools: WorkspaceTools) -> Dict[str, Callable]:
    """
    Get tool registry mapping names to functions.
    
    This is used by AgentController to dispatch tool calls.
    """
    return {
        "workspace.search": tools.workspace_search,
        "workspace.list_docs": tools.workspace_list_docs,
        "workspace.get_doc": tools.workspace_get_doc,
        "workspace.summarize": tools.workspace_summarize,
    }


# Tool metadata for documentation / prompts
TOOL_REGISTRY = {
    "workspace.search": {
        "description": "Search documents by semantic similarity",
        "args": {
            "query": {"type": "str", "required": True, "description": "Search query"},
            "top_k": {"type": "int", "required": False, "default": 5, "description": "Number of results"},
        },
    },
    "workspace.list_docs": {
        "description": "List all documents in the workspace",
        "args": {},
    },
    "workspace.get_doc": {
        "description": "Get full text of a document",
        "args": {
            "doc_id": {"type": "str", "required": False, "description": "Document ID"},
            "title": {"type": "str", "required": False, "description": "Document title"},
        },
    },
    "workspace.summarize": {
        "description": "Summarize a document or text",
        "args": {
            "doc_id": {"type": "str", "required": False, "description": "Document ID to summarize"},
            "text": {"type": "str", "required": False, "description": "Text to summarize"},
            "max_length": {"type": "int", "required": False, "default": 500, "description": "Max summary length"},
        },
    },
}


def get_tools_prompt() -> str:
    """
    Generate a system prompt describing available tools.
    
    This can be included in the system message for the agent.
    """
    lines = ["You have access to the following workspace tools:", ""]
    
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"**{name}**: {info['description']}")
        if info.get("args"):
            lines.append("  Arguments:")
            for arg_name, arg_info in info["args"].items():
                req = "required" if arg_info.get("required") else "optional"
                lines.append(f"    - {arg_name} ({arg_info['type']}, {req}): {arg_info.get('description', '')}")
        lines.append("")
    
    lines.append("To use a tool, output a toolcall block with JSON arguments.")
    
    return "\n".join(lines)

