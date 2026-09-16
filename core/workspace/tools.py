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

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import re
from .engine import WorkspaceEngine
from .summarize import clip_indexed_for_context, SUMMARIZE_CITE_BUDGET


SEARCH_TOP_K_DEFAULT = 2
SEARCH_TOP_K_MAX = 2
SEARCH_SNIPPET_CHARS = 280
LIST_DOCS_MAX = 12
_SEARCH_TERM = re.compile(r"[A-Za-z0-9ÄÖÜäöüß]{3,}")
_SEARCH_STOP = frozenset({
    "who", "what", "when", "where", "why", "how", "the", "and", "for",
    "are", "was", "can", "does", "did", "you", "our", "its", "this",
    "that", "have", "has", "not", "any", "all", "get", "read", "open",
    "file", "with", "from", "about", "please", "tell", "look", "find",
    "search", "docs", "document", "documents", "workspace", "which",
    "wer", "was", "wie", "wo", "warum", "der", "die", "das", "und",
    "oder", "ein", "eine", "ist", "sind", "suche", "nach", "ueber",
    "über", "bitte", "mich", "uns", "unsere",
})


def search_match_terms(query: str) -> List[str]:
    """Distinctive tokens from a search query (no stopwords, no quotes)."""
    raw = (query or "").replace('"', " ").replace("'", " ")
    out = []
    seen = set()
    for w in _SEARCH_TERM.findall(raw):
        lw = w.lower()
        if lw in _SEARCH_STOP or lw in seen:
            continue
        seen.add(lw)
        out.append(lw)
    return out


def lexical_overlap(query: str, *blobs: str) -> int:
    """How many distinctive query terms appear in title/filename/text."""
    terms = search_match_terms(query)
    if not terms:
        return 0
    hay = " ".join(b or "" for b in blobs).lower()
    return sum(1 for t in terms if t in hay)


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
        try:
            top_k = int(args.get("top_k", SEARCH_TOP_K_DEFAULT))
        except (TypeError, ValueError):
            top_k = SEARCH_TOP_K_DEFAULT
        top_k = max(1, min(top_k, SEARCH_TOP_K_MAX))

        if not query:
            return {"error": "Missing required argument: query", "documents": [], "total": 0}

        terms = search_match_terms(query)
        if not terms:
            return {"documents": [], "total": 0}

        chunks = self.engine.search(query, top_k=max(8, top_k * 4))

        # One hit per document. Drop neighbors that share no query terms —
        # dense top-k is never empty, even for names that are not in the index.
        best_by_doc = {}
        for c in chunks:
            filename = c.metadata.get("filename") or ""
            title = ""
            doc = self.engine.get_doc(c.doc_id)
            if doc:
                title = doc.title or ""
            if not title:
                title = filename.rsplit(".", 1)[0] if filename else ""
            overlap = lexical_overlap(query, c.text or "", title, filename)
            if overlap < 1:
                continue
            prev = best_by_doc.get(c.doc_id)
            key = (overlap, c.score)
            if prev is None or key > prev[0]:
                best_by_doc[c.doc_id] = (key, c, title, filename)
        ordered = sorted(best_by_doc.values(), key=lambda x: x[0], reverse=True)
        picked = ordered[:top_k]

        hits = []
        for i, (_, c, title, filename) in enumerate(picked, 1):
            snippet = " ".join((c.text or "").split())
            if len(snippet) > SEARCH_SNIPPET_CHARS:
                snippet = snippet[: SEARCH_SNIPPET_CHARS - 3].rsplit(" ", 1)[0] + "..."
            hits.append({
                "rank": i,
                "title": title,
                "filename": filename or c.metadata.get("filename", ""),
                "snippet": snippet,
                "doc_id": c.doc_id,
            })

        return {
            "documents": hits,
            "total": len(hits),
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
        shown = docs[:LIST_DOCS_MAX]
        return {
            "documents": [
                {"doc_id": d.doc_id, "title": d.title, "filename": Path(d.path).name}
                for d in shown
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
        
        text = self.engine.get_doc_text(doc.doc_id) or ""
        clipped = clip_indexed_for_context(text, SUMMARIZE_CITE_BUDGET)

        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "text": clipped,
            "length": len(text),
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
        
        # Never feed the raw full doc into the 4096 window. Cut the indexed
        # extract first (headings + head + tail), then summarize that.
        from core.workspace.summarize import (
            SUMMARIZE_MODEL_INPUT_BUDGET,
            clip_indexed_for_context,
        )
        extract = clip_indexed_for_context(source_text, SUMMARIZE_MODEL_INPUT_BUDGET)
        lang = "de" if args.get("lang") == "de" else "en"
        summary = self._extractive_summary(extract, max_length, lang=lang)
        
        return {
            "summary": summary,
            "source": source_type,
            "original_length": len(source_text),
            "indexed_chars": len(extract),
        }
    
    def _extractive_summary(self, text: str, max_length: int, lang: str = "en") -> str:
        """Structured abstract: title + headings + lead sentences. Shared with SFT."""
        from core.workspace.summarize import structured_summary
        return structured_summary(text, max_length=max_length, lang=lang)
    
    def _generate_summary(self, text: str, max_length: int) -> str:
        """Use model to generate summary (if model available)."""
        if self.model is None:
            return self._extractive_summary(text, max_length)
        
        # Build a summarization prompt
        from core.rag.tags import wrap_context, wrap_user, ASSISTANT_OPEN
        
        from core.workspace.summarize import (
            CONTEXT_LIMIT,
            clip_indexed_for_context,
            n_tokens,
        )
        # Leave room for the user ask + ~128 new tokens inside 4096.
        ask = "Summarize the above text concisely."
        overhead = n_tokens(ask) + 128 + 64
        extract = clip_indexed_for_context(text, max(64, CONTEXT_LIMIT - overhead))
        prompt = (
            wrap_context(extract) + "\n" +
            wrap_user(ask) + "\n" +
            ASSISTANT_OPEN
        )
        
        try:
            # Use factual/focused settings for summarization
            response = self.model.generate(
                prompt, 
                max_new_tokens=max_length // 4,
                temperature=0.3,      # Focused
                top_k=20,             # Restricted vocabulary
                top_p=0.9,            # Tight nucleus
                repetition_penalty=1.2  # Avoid repetitive summaries
            )
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
            "top_k": {"type": "int", "required": False, "default": 2, "description": "Number of results (max 2)"},
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



