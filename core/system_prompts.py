"""
System Prompts for MyPT

Central location for all system prompts used in training and inference.

Usage:
    from core.system_prompts import CONVERSATION_SYSTEM_PROMPT, AGENTIC_SYSTEM_PROMPT
    
    # For Phase 3a SFT (conversational, no tools)
    episode["system"] = CONVERSATION_SYSTEM_PROMPT
    
    # For Phase 3b SFT (agentic, with tools)
    controller = AgentController(model, tools, system_prompt=AGENTIC_SYSTEM_PROMPT)
"""

# =============================================================================
# Phase 3a: Conversational System Prompt (no tools)
# =============================================================================
# 
# Used for:
#   - Chat SFT training (Phase 3a)x
#   - Simple Q&A inference (no RAG/tools)
#
# Characteristics:
#   - No tool mentions
#   - Focus on response quality
#   - ~50 tokens (fits easily in context)
#
# =============================================================================

CONVERSATION_SYSTEM_PROMPT = """You are MyPT, an offline, privacy-focused assistant. Be concise: 1-2 sentences. Follow instructions exactly. If unsure, state assumptions briefly."""


# =============================================================================
# Phase 3b: Agentic System Prompt (with tools)
# =============================================================================
#
# Used for:
#   - Agentic SFT training (Phase 3b)
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

# For Phase 3a SFT and simple inference
DEFAULT_CONVERSATION_PROMPT = CONVERSATION_SYSTEM_PROMPT

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

