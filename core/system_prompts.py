"""
System Prompts for MyPT

Central location for all system prompts used in training and inference.
Do not copy these strings into scripts — import the constants (or a preset).

Phase split (mask-ratio and train/eval match):

    Phase 1-2  CONVERSATION_SYSTEM_PROMPT   minimal, no tools
    Phase 3-4  CHAT_SYSTEM_PROMPT           chat/RAG, no tools
    Phase 5-6  AGENTIC_STANDARD_PROMPT      workspace tools + toolcall syntax

Usage:
    from core.system_prompts import (
        CONVERSATION_SYSTEM_PROMPT,
        CHAT_SYSTEM_PROMPT,
        AGENTIC_SYSTEM_PROMPT,
        resolve_system_prompt,
    )

    episode["system"] = CHAT_SYSTEM_PROMPT          # Phase 3-4 generators
    prompt = resolve_system_prompt(preset="chat")   # packers / eval
    controller = AgentController(model, tools, system_prompt=AGENTIC_SYSTEM_PROMPT)
"""

from typing import Optional


# =============================================================================
# Phase 1-2: Conversational System Prompt (no tools)
# =============================================================================
#
# Used for:
#   - Format lock and operator SFT (Phase 1-2)
#   - Operator eval with a system prompt (not --no_system_prompt)
#
# Keep this short: every token is masked (loss=0) and eats supervised signal.
# =============================================================================

CONVERSATION_SYSTEM_PROMPT = """You are MyPT."""
#CONVERSATION_SYSTEM_PROMPT = """You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly."""


# =============================================================================
# Phase 3-4: Chat / RAG System Prompt (no tools)
# =============================================================================
#
# Slightly longer than the Phase 1-2 prompt, but still compact.
# Phase 3-4 episodes (RAG chat, multi-turn) are long enough that
# ~15 masked system tokens don't significantly hurt loss mask %.
#
# DO NOT put tool names or <myPT_toolcall> here — that bloats the mask-0
# prefix and was the reason chat SFT previously used a tool prompt too early.
# =============================================================================

CHAT_SYSTEM_PROMPT = """You are MyPT, a helpful assistant. Answer based on the provided context when available."""

# A few short variants for surface diversity (all ~15-20 tokens)
CHAT_SYSTEM_PROMPTS = [
    CHAT_SYSTEM_PROMPT,
    "You are MyPT. Use the provided context to answer accurately.",
    "You are MyPT, a helpful assistant. Be concise and cite sources when possible.",
    "You are MyPT. Answer questions based on the given context.",
]


# =============================================================================
# Phase 5-6: Agentic System Prompt (with tools)
# =============================================================================
#
# Used for:
#   - Toolcall / agentic SFT (Phase 5-6)
#   - AgentController runtime (RAG workspace)
#
# Variants:
#   COMPACT (~50 tokens): Best for small models / limited context
#   STANDARD (~80 tokens): Balance of brevity and clarity
#   VERBOSE (~120 tokens): More explicit instructions
#
# After SFT training, the model learns tool-calling patterns from examples.
# The system prompt only needs to:
#   1. List available tools (prevent hallucinated tool names)
#   2. Show the exact toolcall syntax
#
# =============================================================================

# Compact version - minimal tokens, relies on SFT training
AGENTIC_COMPACT_PROMPT = """Tools: workspace.search(query), workspace.list_docs(), workspace.get_doc(doc_id|title), workspace.summarize(doc_id|text)
Use: <myPT_toolcall>{"name": "...", ...}</myPT_toolcall>"""


# Standard version - good balance (RECOMMENDED)
AGENTIC_STANDARD_PROMPT = """You are MyPT. Answer questions using workspace tools when needed.

Tools:
- workspace.search(query, top_k=5) - find relevant documents
- workspace.list_docs() - list all documents
- workspace.get_doc(doc_id or title) - get document text
- workspace.summarize(doc_id or text) - summarize content

Format: <myPT_toolcall>{"name": "workspace.search", "query": "..."}</myPT_toolcall>"""


# Verbose version - explicit instructions
AGENTIC_VERBOSE_PROMPT = """You are MyPT, a helpful workspace assistant.

You have access to workspace tools to search, read, and summarize documents.

To use a tool, output a toolcall block like:
<myPT_toolcall>{"name": "workspace.search", "query": "your search", "top_k": 5}</myPT_toolcall>

Available tools:
- workspace.search: Search documents by query
- workspace.list_docs: List all documents
- workspace.get_doc: Get full document text (args: doc_id or title)
- workspace.summarize: Summarize a document (args: doc_id or text)

After receiving tool results, provide a helpful answer to the user."""


# =============================================================================
# Defaults - CHANGE THESE TO SWITCH BETWEEN PROMPT VARIANTS
# =============================================================================

# For Phase 1-2 SFT (minimal, maximizes loss mask %)
DEFAULT_CONVERSATION_PROMPT = CONVERSATION_SYSTEM_PROMPT

# For Phase 3-4 SFT (slightly longer, episodes are big enough)
DEFAULT_CHAT_PROMPT = CHAT_SYSTEM_PROMPT

# =============================================================================
# DEFAULT AGENTIC PROMPT SELECTION
# =============================================================================
# Uncomment ONE of the following lines to set the default agentic prompt:
#
# DEFAULT_AGENTIC_PROMPT = AGENTIC_COMPACT_PROMPT   # ~50 tokens, minimal
DEFAULT_AGENTIC_PROMPT = AGENTIC_STANDARD_PROMPT   # ~80 tokens, balanced (RECOMMENDED)
# DEFAULT_AGENTIC_PROMPT = AGENTIC_VERBOSE_PROMPT  # ~120 tokens, explicit
#
# =============================================================================

# Legacy alias for backwards compatibility
AGENTIC_SYSTEM_PROMPT = DEFAULT_AGENTIC_PROMPT

SYSTEM_PROMPT_PRESETS = {
    "conversation": CONVERSATION_SYSTEM_PROMPT,
    "chat": CHAT_SYSTEM_PROMPT,
    "agentic": DEFAULT_AGENTIC_PROMPT,
}


def resolve_system_prompt(
    preset: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    """Resolve the system prompt text for packing or eval.

    ``override`` wins when set. Otherwise ``preset`` must be a key of
    SYSTEM_PROMPT_PRESETS. Default (neither set) is conversation (Phase 1-2).
    """
    if override:
        return override
    if preset:
        if preset not in SYSTEM_PROMPT_PRESETS:
            raise ValueError(
                f"Unknown system_prompt_preset {preset!r}; "
                f"expected one of {sorted(SYSTEM_PROMPT_PRESETS)}"
            )
        return SYSTEM_PROMPT_PRESETS[preset]
    return CONVERSATION_SYSTEM_PROMPT
