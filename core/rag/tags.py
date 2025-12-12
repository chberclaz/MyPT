"""
Special token tag helpers for RAG and chat formatting.

Uses definitions from core/special_tokens.py to ensure consistency
across the entire codebase. Never hardcode tag strings elsewhere!

Usage:
    from core.rag.tags import wrap_system, wrap_user, wrap_assistant, wrap_context
    
    prompt = wrap_system("You are helpful.") + wrap_context(docs) + wrap_user(q)
"""

from core.special_tokens import SPECIAL_TOKEN_STRINGS

# ============================================================
# Tag constants - import these instead of hardcoding strings
# ============================================================

SYSTEM_OPEN = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
SYSTEM_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_system_close"]

CONTEXT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_context_open"]
CONTEXT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_context_close"]

USER_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
USER_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_close"]

ASSISTANT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
ASSISTANT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]

ASSISTANT_CONTEXT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_context_open"]
ASSISTANT_CONTEXT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_context_close"]

TOOL_CALL_OPEN = SPECIAL_TOKEN_STRINGS["myPT_tool_call_open"]
TOOL_CALL_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_tool_call_close"]

TOOL_RESULT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_tool_result_open"]
TOOL_RESULT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_tool_result_close"]

EOT = SPECIAL_TOKEN_STRINGS["myPT_eot"]


# ============================================================
# Wrapper functions - use these to format content
# ============================================================

def wrap_system(text: str) -> str:
    """Wrap text in system tags."""
    return f"{SYSTEM_OPEN}{text}{SYSTEM_CLOSE}"


def wrap_context(text: str) -> str:
    """Wrap text in user context tags (for RAG documents)."""
    return f"{CONTEXT_OPEN}{text}{CONTEXT_CLOSE}"


def wrap_user(text: str) -> str:
    """Wrap text in user tags."""
    return f"{USER_OPEN}{text}{USER_CLOSE}"


def wrap_assistant(text: str) -> str:
    """Wrap text in assistant tags."""
    return f"{ASSISTANT_OPEN}{text}{ASSISTANT_CLOSE}"


def wrap_tool_call(text: str) -> str:
    """Wrap text in tool call tags."""
    return f"{TOOL_CALL_OPEN}{text}{TOOL_CALL_CLOSE}"


def wrap_tool_result(text: str) -> str:
    """Wrap text in tool result tags."""
    return f"{TOOL_RESULT_OPEN}{text}{TOOL_RESULT_CLOSE}"


# ============================================================
# Prompt builders - higher-level formatting
# ============================================================

def format_context_block(chunks: list, include_source: bool = True) -> str:
    """
    Format retrieved chunks into a context block.
    
    Args:
        chunks: List of dicts with 'text' and optionally 'source' info
        include_source: Whether to include source attribution
        
    Returns:
        Formatted string ready to wrap with wrap_context()
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        
        if include_source and "source" in chunk:
            source = chunk["source"]
            filename = source.get("filename", source.get("file", "unknown"))
            lines.append(f"[{i}] ({filename}) {text}")
        else:
            lines.append(f"[{i}] {text}")
    
    return "\n".join(lines)


def build_rag_prompt(
    question: str,
    context_chunks: list = None,
    system: str = None,
    include_source: bool = True
) -> str:
    """
    Build a complete RAG prompt ready for generation.
    
    Args:
        question: User's question
        context_chunks: List of retrieved document chunks
        system: Optional system message
        include_source: Include source attribution in context
        
    Returns:
        Complete prompt string ending with open assistant tag
    """
    parts = []
    
    if system:
        parts.append(wrap_system(system))
    
    if context_chunks:
        ctx_text = format_context_block(context_chunks, include_source)
        parts.append(wrap_context(ctx_text))
    
    parts.append(wrap_user(question))
    parts.append(ASSISTANT_OPEN)  # Open tag for generation to continue
    
    return "\n".join(parts)

