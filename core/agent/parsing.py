"""
Toolcall parsing utilities.

Detects and parses <myPT_toolcall> blocks in model output,
and renders <myPT_toolresult> blocks.

Format:
    <myPT_toolcall>{"name": "workspace.search", "query": "...", "top_k": 5}</myPT_toolcall>
    
    <myPT_toolresult>{"documents": [...], "total": 5}</myPT_toolresult>

Usage:
    from core.agent.parsing import find_toolcall, render_toolresult
    
    text = model.generate(prompt)
    toolcall = find_toolcall(text)
    
    if toolcall:
        result = tools.execute(toolcall.name, toolcall.arguments)
        result_block = render_toolresult(result)
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from core.special_tokens import SPECIAL_TOKEN_STRINGS


# Get tokens from special_tokens.py
TOOLCALL_OPEN = SPECIAL_TOKEN_STRINGS["myPT_toolcall_open"]
TOOLCALL_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_toolcall_close"]
TOOLRESULT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"]
TOOLRESULT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"]


@dataclass
class ToolCall:
    """
    Parsed toolcall data.
    
    Attributes:
        name: Tool name (e.g., "workspace.search")
        arguments: Tool arguments as dict
        raw_block: Full matched string including tags
    """
    name: str
    arguments: Dict[str, Any]
    raw_block: str
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
        }


def find_toolcall(text: str) -> Optional[ToolCall]:
    """
    Find and parse the last toolcall block in text.
    
    Looks for:
        <myPT_toolcall>{"name": "...", ...}</myPT_toolcall>
    
    Args:
        text: Model output text
        
    Returns:
        ToolCall if found, None otherwise
    """
    # Escape special regex chars in tokens
    open_escaped = re.escape(TOOLCALL_OPEN)
    close_escaped = re.escape(TOOLCALL_CLOSE)
    
    # Pattern: <myPT_toolcall> ... </myPT_toolcall>
    # Use non-greedy match and capture content
    pattern = f"{open_escaped}(.*?){close_escaped}"
    
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    if not matches:
        return None
    
    # Get the last match (in case of multiple toolcalls)
    match = matches[-1]
    raw_block = match.group(0)
    json_content = match.group(1).strip()
    
    # Parse JSON
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        # Try to extract name from malformed JSON
        print(f"Warning: Could not parse toolcall JSON: {e}")
        return None
    
    # Extract name (required)
    name = data.get("name")
    if not name:
        print("Warning: Toolcall missing 'name' field")
        return None
    
    # Everything else is arguments
    arguments = {k: v for k, v in data.items() if k != "name"}
    
    return ToolCall(
        name=name,
        arguments=arguments,
        raw_block=raw_block,
    )


def find_all_toolcalls(text: str) -> list:
    """
    Find all toolcall blocks in text.
    
    Returns:
        List of ToolCall objects
    """
    open_escaped = re.escape(TOOLCALL_OPEN)
    close_escaped = re.escape(TOOLCALL_CLOSE)
    pattern = f"{open_escaped}(.*?){close_escaped}"
    
    toolcalls = []
    for match in re.finditer(pattern, text, re.DOTALL):
        raw_block = match.group(0)
        json_content = match.group(1).strip()
        
        try:
            data = json.loads(json_content)
            name = data.get("name")
            if name:
                arguments = {k: v for k, v in data.items() if k != "name"}
                toolcalls.append(ToolCall(name=name, arguments=arguments, raw_block=raw_block))
        except json.JSONDecodeError:
            continue
    
    return toolcalls


def has_toolcall(text: str) -> bool:
    """Check if text contains a toolcall block."""
    return TOOLCALL_OPEN in text and TOOLCALL_CLOSE in text


def render_toolcall(name: str, arguments: Dict[str, Any]) -> str:
    """
    Render a toolcall block.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        
    Returns:
        Formatted toolcall string
    """
    data = {"name": name, **arguments}
    json_str = json.dumps(data, ensure_ascii=False)
    return f"{TOOLCALL_OPEN}{json_str}{TOOLCALL_CLOSE}"


def render_toolresult(result: Dict[str, Any], pretty: bool = False) -> str:
    """
    Render a toolresult block.
    
    Args:
        result: Tool result dict
        pretty: Whether to pretty-print JSON
        
    Returns:
        Formatted toolresult string
    """
    if pretty:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        json_str = json.dumps(result, ensure_ascii=False)
    
    return f"{TOOLRESULT_OPEN}{json_str}{TOOLRESULT_CLOSE}"


def truncate_toolresult(result: Dict[str, Any], max_chars: int = 2000) -> Dict[str, Any]:
    """
    Truncate large tool results to fit context.
    
    Args:
        result: Tool result dict
        max_chars: Maximum characters in result
        
    Returns:
        Truncated result dict
    """
    json_str = json.dumps(result, ensure_ascii=False)
    
    if len(json_str) <= max_chars:
        return result
    
    # For documents/chunks, truncate the text fields
    if "documents" in result and isinstance(result["documents"], list):
        truncated_docs = []
        remaining = max_chars - 100  # Reserve space for metadata
        
        for doc in result["documents"]:
            doc_copy = doc.copy()
            if "text" in doc_copy and len(doc_copy["text"]) > 300:
                doc_copy["text"] = doc_copy["text"][:297] + "..."
            
            doc_json = json.dumps(doc_copy)
            if remaining - len(doc_json) < 0:
                truncated_docs.append({"truncated": True, "remaining": len(result["documents"]) - len(truncated_docs)})
                break
            
            truncated_docs.append(doc_copy)
            remaining -= len(doc_json)
        
        return {**result, "documents": truncated_docs, "truncated": True}
    
    # For text fields
    if "text" in result and len(result["text"]) > max_chars - 100:
        return {**result, "text": result["text"][:max_chars - 103] + "...", "truncated": True}
    
    if "summary" in result and len(result["summary"]) > max_chars - 100:
        return {**result, "summary": result["summary"][:max_chars - 103] + "...", "truncated": True}
    
    return result


def extract_text_after_toolresult(text: str) -> str:
    """
    Extract text that comes after the last toolresult block.
    
    This is typically the final answer after tool execution.
    """
    close_escaped = re.escape(TOOLRESULT_CLOSE)
    
    # Find last toolresult close tag
    matches = list(re.finditer(close_escaped, text))
    
    if not matches:
        return text
    
    last_match = matches[-1]
    return text[last_match.end():].strip()


_WORD = re.compile(r"[a-z0-9]{4,}", re.I)


def _hit_blob(hit: Dict[str, Any]) -> str:
    return " ".join(
        str(hit.get(k) or "")
        for k in ("title", "filename", "snippet", "text")
    )


def bind_doc_id_from_search(
    generated: str,
    requested_id: Optional[str],
    hits: list,
) -> tuple:
    """Map a get_doc id onto the search hit the model actually named.

    750M copies a nearby hex id from a JSON list. Prefer the hit whose
    title/snippet uniquely overlaps the think text; otherwise always use
    rank 1. Do not trust a requested id just because it appears later in
    the same catalog (that is the next-id-after-snippet failure).
    """
    if not hits:
        return requested_id or "", "no_hits"

    ids = [h.get("doc_id") for h in hits if h.get("doc_id")]
    think_words = set(_WORD.findall((generated or "").lower()))

    scored = []
    for hit in hits:
        blob_words = set(_WORD.findall(_hit_blob(hit).lower()))
        score = len(think_words & blob_words)
        scored.append((score, hit))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0

    if best_score >= 2 and best_score > second:
        chosen = best.get("doc_id") or ""
        if chosen and chosen != requested_id:
            return chosen, "think_overlap"
        if chosen:
            return chosen, "think_overlap_same"

    # Generic think ("I'll retrieve the document") copies the next hex id
    # after the snippet — that is hit 2. Always use rank 1 unless think
    # uniquely named another file.
    return (ids[0] if ids else (requested_id or "")), "rank1_default"


def prompt_has_doc_body(prompt: str) -> bool:
    """True if the last toolresult in a tagged prompt already has document text.

    Decode-stop uses this to ban ``<myPT_toolcall>`` after get_doc/summarize.
    Search catalogs (snippets only) stay False so bind/chain can still call get_doc.
    """
    if not prompt:
        return False
    end = prompt.rfind(TOOLRESULT_CLOSE)
    if end < 0:
        return False
    start = prompt.rfind(TOOLRESULT_OPEN, 0, end)
    if start < 0:
        return False
    raw = prompt[start + len(TOOLRESULT_OPEN) : end].strip()
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return '"text":' in raw or '"summary":' in raw
    if not isinstance(obj, dict):
        return False
    return bool(obj.get("text")) or bool(obj.get("summary"))


def history_has_get_doc_body(history: list) -> bool:
    """True if a successful get_doc toolresult is already in the transcript.

    Used to refuse a second get_doc/summarize. Does not stitch the answer.
    """
    for h in history or []:
        if isinstance(h, dict):
            role, name, content = h.get("role"), h.get("name"), h.get("content") or ""
        else:
            role = getattr(h, "role", None)
            name = getattr(h, "name", None)
            content = getattr(h, "content", None) or ""
        if role != "toolresult" or name != "workspace.get_doc":
            continue
        if '"text"' in content:
            return True
        try:
            obj = json.loads(content)
        except (TypeError, ValueError):
            continue
        if isinstance(obj, dict) and obj.get("text"):
            return True
    return False


_EMPTY_INDEX_LINE = re.compile(r"^\s*\[\d+\]\s*$")
_INDEX_TOKEN = re.compile(r"\[\d+\]")


def is_collapsed_generation(text: str) -> bool:
    """True when the model dumped a think/citation template instead of an answer.

    Typical failure: unclosed <myPT_think> plus `[1] … [2] [3] … [58]`.
    """
    if not text or not str(text).strip():
        return True
    raw = str(text)
    if "<myPT_think>" in raw and "</myPT_think>" not in raw:
        return True
    lines = raw.splitlines()
    empty_index = sum(1 for ln in lines if _EMPTY_INDEX_LINE.match(ln))
    if empty_index >= 4:
        return True
    index_tokens = _INDEX_TOKEN.findall(raw)
    letters = sum(ch.isalpha() for ch in raw)
    if len(index_tokens) >= 6 and letters < 80:
        return True
    return False


def fallback_answer_from_hits(hits: list) -> str:
    """Extractive answer from compact search hits when generation collapses."""
    if not hits:
        return ""
    best = max(hits, key=lambda h: len((h.get("snippet") or h.get("text") or "")))
    title = (best.get("title") or best.get("filename") or "the workspace").strip()
    snip = (best.get("snippet") or best.get("text") or "").strip()
    if not snip:
        return ""
    return f"From {title}: {snip}"







