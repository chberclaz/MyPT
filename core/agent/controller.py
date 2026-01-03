"""
AgentController - Orchestrates model + tools for agentic RAG.

Runs a loop:
1. Build prompt from history
2. Generate with model
3. If toolcall found → execute → inject result → continue
4. Else → return final answer

Usage:
    from core.agent import AgentController
    from core.workspace import WorkspaceEngine, WorkspaceTools
    
    engine = WorkspaceEngine("workspace/", "workspace/index/latest")
    tools = WorkspaceTools(engine)
    controller = AgentController(model, tools)
    
    result = controller.run([
        {"role": "user", "content": "Find and summarize docs about X"}
    ])
    
    print(result["content"])  # Final answer
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.special_tokens import SPECIAL_TOKEN_STRINGS
from .parsing import (
    find_toolcall, 
    has_toolcall, 
    render_toolresult,
    truncate_toolresult,
    TOOLCALL_OPEN,
    TOOLCALL_CLOSE,
)

# Import audit logging (optional)
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# Get tokens from special_tokens.py
SYSTEM_OPEN = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
SYSTEM_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_system_close"]
USER_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
USER_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_close"]
ASSISTANT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
ASSISTANT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]
TOOLRESULT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"]
TOOLRESULT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"]
EOT = SPECIAL_TOKEN_STRINGS["myPT_eot"]


# =============================================================================
# System Prompt Options
# =============================================================================
# 
# COMPACT (~50 tokens): Best for small models / limited context
# STANDARD (~80 tokens): Balance of brevity and clarity  
# VERBOSE (~120 tokens): More explicit instructions for less-trained models
#
# After SFT training, the model learns tool-calling patterns from examples.
# The system prompt only needs to:
#   1. List available tools (prevent hallucinated tool names)
#   2. Show the exact toolcall syntax
#
# =============================================================================

# Compact version - minimal tokens, relies on SFT training
COMPACT_SYSTEM_PROMPT = """Tools: workspace.search(query), workspace.list_docs(), workspace.get_doc(doc_id|title), workspace.summarize(doc_id|text)
Use: <myPT_toolcall>{"name": "...", ...}</myPT_toolcall>"""

# Standard version - good balance
STANDARD_SYSTEM_PROMPT = """You are MyPT. Answer questions using workspace tools when needed.

Tools:
- workspace.search(query, top_k=5) - find relevant documents
- workspace.list_docs() - list all documents  
- workspace.get_doc(doc_id or title) - get document text
- workspace.summarize(doc_id or text) - summarize content

Format: <myPT_toolcall>{"name": "workspace.search", "query": "..."}</myPT_toolcall>"""

# Verbose version - explicit instructions (original)
VERBOSE_SYSTEM_PROMPT = """You are MyPT, a helpful workspace assistant.

You have access to workspace tools to search, read, and summarize documents.

To use a tool, output a toolcall block like:
<myPT_toolcall>{"name": "workspace.search", "query": "your search", "top_k": 5}</myPT_toolcall>

Available tools:
- workspace.search: Search documents by query
- workspace.list_docs: List all documents
- workspace.get_doc: Get full document text (args: doc_id or title)
- workspace.summarize: Summarize a document (args: doc_id or text)

After receiving tool results, provide a helpful answer to the user."""

# Default - use STANDARD for balance of clarity and efficiency
DEFAULT_SYSTEM_PROMPT = STANDARD_SYSTEM_PROMPT


@dataclass
class Message:
    """A message in the conversation history."""
    role: str  # "system", "user", "assistant", "toolcall", "toolresult"
    content: str
    name: Optional[str] = None  # Tool name for toolcall/toolresult
    
    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


def _debug_section(title: str, verbose: bool):
    """Print a debug section header."""
    if verbose:
        width = 60
        padding = (width - len(title) - 2) // 2
        print(f"\n{'═' * padding} {title} {'═' * padding}")


def _debug_end_section(title: str, verbose: bool):
    """Print a debug section footer."""
    if verbose:
        width = 60
        padding = (width - len(f"END {title}") - 2) // 2
        print(f"{'═' * padding} END {title} {'═' * padding}")


def _debug_prompt(prompt: str, verbose: bool, label: str = "FULL PROMPT"):
    """Print the full prompt with line numbers."""
    if verbose:
        _debug_section(label, True)
        lines = prompt.split('\n')
        for i, line in enumerate(lines, 1):
            # Show full line but truncate extremely long ones
            if len(line) > 300:
                print(f"  {i:3d} | {line[:300]}...")
            else:
                print(f"  {i:3d} | {line}")
        print(f"\n  [Total: {len(prompt)} chars, {len(lines)} lines]")
        _debug_end_section(label, True)


def _debug_generation(output: str, verbose: bool, label: str = "MODEL OUTPUT"):
    """Print the generated output."""
    if verbose:
        _debug_section(label, True)
        lines = output.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 300:
                print(f"  {i:3d} | {line[:300]}...")
            else:
                print(f"  {i:3d} | {line}")
        print(f"\n  [Total: {len(output)} chars]")
        _debug_end_section(label, True)


def _debug_tool_call(name: str, arguments: dict, result: Any, verbose: bool):
    """Print a tool call with full details."""
    if verbose:
        _debug_section(f"TOOL CALL: {name}", True)
        print(f"  Arguments:")
        for k, v in arguments.items():
            print(f"    {k}: {v}")
        print(f"\n  Result:")
        result_str = json.dumps(result, indent=4, ensure_ascii=False)
        lines = result_str.split('\n')
        for line in lines[:30]:  # Show first 30 lines
            print(f"    {line}")
        if len(lines) > 30:
            print(f"    ... ({len(lines) - 30} more lines)")
        _debug_end_section(f"TOOL CALL: {name}", True)


class AgentController:
    """
    Orchestrates model generation with tool execution.
    
    Args:
        model: Model with generate() method
        tools: WorkspaceTools instance
        system_prompt: System prompt (default: DEFAULT_SYSTEM_PROMPT)
        max_result_chars: Max chars for tool results (default: 2000)
        safety_margin: Token buffer below block_size (default: 64)
    """
    
    def __init__(
        self, 
        model,
        tools,
        system_prompt: str = None,
        max_result_chars: int = 2000,
        safety_margin: int = 64,
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_result_chars = max_result_chars
        self.safety_margin = safety_margin
        
        # Get block_size and tokenizer from model
        self.block_size = getattr(model, 'config', None)
        if self.block_size:
            self.block_size = getattr(self.block_size, 'block_size', 1024)
        else:
            self.block_size = 1024  # Default fallback
        
        # Get tokenizer from model if available
        self.tokenizer = getattr(model, 'tokenizer', None)
    
    def build_prompt(self, history: List[Dict[str, Any]]) -> str:
        """
        Build prompt string from conversation history.
        
        Args:
            history: List of message dicts with "role" and "content"
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        # Add system prompt if not already in history
        has_system = any(m.get("role") == "system" for m in history)
        if not has_system and self.system_prompt:
            parts.append(f"{SYSTEM_OPEN}{self.system_prompt}{SYSTEM_CLOSE}")
        
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            name = msg.get("name", "")
            
            if role == "system":
                parts.append(f"{SYSTEM_OPEN}{content}{SYSTEM_CLOSE}")
            
            elif role == "user":
                parts.append(f"{USER_OPEN}{content}{USER_CLOSE}")
            
            elif role == "assistant":
                parts.append(f"{ASSISTANT_OPEN}{content}{ASSISTANT_CLOSE}")
            
            elif role == "assistant_toolcall":
                # Assistant message containing a toolcall
                parts.append(f"{ASSISTANT_OPEN}{content}{ASSISTANT_CLOSE}")
            
            elif role == "toolresult":
                parts.append(f"{TOOLRESULT_OPEN}{content}{TOOLRESULT_CLOSE}")
        
        # Add open assistant tag for generation
        parts.append(ASSISTANT_OPEN)
        
        return "\n".join(parts)
    
    def _trim_history_to_block_size(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Trim history until the rendered prompt fits within block_size.
        
        Prevents prompt overflow by dropping oldest non-system messages first.
        Always preserves:
        - System message (first element if role == "system")
        - At least the most recent user message
        
        Args:
            history: Conversation history
            
        Returns:
            Trimmed history that fits within block_size - safety_margin tokens
        """
        # Safety fallback: if no tokenizer, can't measure tokens
        if self.tokenizer is None:
            return history
        
        working = list(history)
        max_tokens = self.block_size - self.safety_margin
        
        while True:
            # Build prompt and measure token count
            prompt = self.build_prompt(working)
            try:
                token_ids = self.tokenizer.encode(prompt)
                token_count = len(token_ids)
            except Exception:
                # If encoding fails, return as-is
                return working
            
            # Check if we fit within budget
            if token_count <= max_tokens:
                return working
            
            # If nothing left to trim (keep at least system + 1 message), stop
            if len(working) <= 2:
                return working
            
            # Determine which message to drop:
            # Preserve system message at index 0 if present
            if working[0].get("role") == "system":
                drop_index = 1  # Drop second message (oldest non-system)
            else:
                drop_index = 0  # No system message, drop oldest
            
            working.pop(drop_index)
    
    def run(
        self, 
        history: List[Dict[str, Any]], 
        max_steps: int = 5,
        max_new_tokens: int = 256,
        verbose: bool = False,
        temperature: float = 0.3,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> Dict[str, Any]:
        """
        Run the agent loop until final answer or max_steps.
        
        Args:
            history: Initial conversation history
            max_steps: Maximum tool execution steps
            max_new_tokens: Max tokens per generation (dynamically adjusted per prompt to prevent overflow)
            verbose: Print debug info (full prompts, tool calls, etc.)
            temperature: Sampling temperature (0.0=deterministic, 1.0=neutral)
            top_k: Only sample from top K tokens (0=disabled)
            top_p: Nucleus sampling threshold (1.0=disabled)
            repetition_penalty: Penalize repeated tokens (1.0=disabled)
            
        Returns:
            Final assistant message dict with:
            - role: "assistant"
            - content: Final answer text
            - tool_calls: List of tool calls made
            - steps: Number of steps taken
        """
        # Make a copy to avoid mutating input
        history = [msg.copy() for msg in history]
        tool_calls = []
        
        if verbose:
            _debug_section("AGENT RUN START", True)
            print(f"  Max steps:      {max_steps}")
            print(f"  Max new tokens: {max_new_tokens}")
            print(f"  History length: {len(history)} messages")
            print(f"  Block size:     {self.block_size}")
            print(f"  Repetition pen: {repetition_penalty}")
            _debug_section("SYSTEM PROMPT", True)
            for i, line in enumerate(self.system_prompt.split('\n'), 1):
                print(f"  {i:3d} | {line}")
            _debug_end_section("SYSTEM PROMPT", True)
            
            _debug_section("INPUT HISTORY", True)
            for i, msg in enumerate(history):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if len(content) > 100:
                    print(f"  [{i+1}] {role.upper()}: {content[:100]}... ({len(content)} chars)")
                else:
                    print(f"  [{i+1}] {role.upper()}: {content}")
            _debug_end_section("INPUT HISTORY", True)
        
        for step in range(max_steps):
            # Trim history to fit within block_size (prevents overflow)
            history = self._trim_history_to_block_size(history)
            
            # Build prompt
            prompt = self.build_prompt(history)
            
            if verbose:
                _debug_section(f"STEP {step + 1}/{max_steps}", True)
                print(f"  Prompt length: {len(prompt)} chars")
                if self.tokenizer:
                    try:
                        token_count = len(self.tokenizer.encode(prompt))
                        print(f"  Prompt tokens: {token_count}")
                    except:
                        pass
                
                # Log the FULL prompt being sent to model
                _debug_prompt(prompt, True, f"PROMPT TO MODEL (Step {step + 1})")
            
            # Dynamically adjust max_new_tokens based on actual prompt length
            # to prevent KV cache overflow
            if self.tokenizer:
                try:
                    prompt_token_count = len(self.tokenizer.encode(prompt))
                    tokens_remaining = self.block_size - prompt_token_count
                    # Reserve at least 1 token for generation
                    max_safe_tokens = max(1, tokens_remaining - 1)
                    
                    if max_new_tokens > max_safe_tokens:
                        if verbose:
                            print(f"\n  [INFO] Reducing max_new_tokens from {max_new_tokens} to {max_safe_tokens} "
                                  f"(prompt={prompt_token_count} tokens, block_size={self.block_size})")
                        max_new_tokens = max_safe_tokens
                except Exception:
                    pass  # If tokenization fails, use original max_new_tokens
            
            # Generate
            try:
                if verbose:
                    print(f"\n  [Generating with max_new_tokens={max_new_tokens}, "
                          f"temp={temperature}, top_k={top_k}, top_p={top_p}, repetition penalty:{repetition_penalty}...]")
                output = self.model.generate(
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=64,
                    recent_penalty_window=max_new_tokens
                )
            except Exception as e:
                if verbose:
                    print(f"\n  [ERROR] Generation failed: {e}")
                return {
                    "role": "assistant",
                    "content": f"Error generating response: {e}",
                    "tool_calls": tool_calls,
                    "steps": step + 1,
                    "error": str(e),
                }
            
            # Log raw model output
            if verbose:
                _debug_generation(output, True, f"RAW MODEL OUTPUT (Step {step + 1})")
            
            # Extract generated part (after prompt)
            if prompt in output:
                generated = output[len(prompt):]
            else:
                generated = output
            
            # Remove closing assistant tag if present
            if ASSISTANT_CLOSE in generated:
                generated = generated.split(ASSISTANT_CLOSE)[0]
            
            generated = generated.strip()
            
            if verbose:
                _debug_generation(generated, True, f"EXTRACTED GENERATION (Step {step + 1})")
            
            # Check for toolcall
            if has_toolcall(generated):
                toolcall = find_toolcall(generated)
                
                if toolcall:
                    if verbose:
                        print(f"\n  [TOOLCALL DETECTED]")
                        print(f"    Name: {toolcall.name}")
                        print(f"    Args: {json.dumps(toolcall.arguments, indent=2)}")
                    
                    # Execute tool
                    try:
                        result = self.tools.execute(toolcall.name, toolcall.arguments)
                        
                        # Audit: Tool call success
                        if AUDIT_AVAILABLE:
                            audit.agent("tool_execute",
                                       tool=toolcall.name,
                                       status="success",
                                       step=step + 1,
                                       details=json.dumps(toolcall.arguments)[:150])
                    except Exception as e:
                        result = {"error": str(e)}
                        if verbose:
                            print(f"    [TOOL ERROR] {e}")
                        
                        # Audit: Tool call failed
                        if AUDIT_AVAILABLE:
                            audit.agent("tool_error",
                                       tool=toolcall.name,
                                       status="failed",
                                       step=step + 1,
                                       error=str(e)[:100],
                                       level=audit.AuditLevel.ERROR,
                                       details=f"Tool '{toolcall.name}' failed: {str(e)[:100]}")
                    
                    # Log full tool call and result
                    _debug_tool_call(toolcall.name, toolcall.arguments, result, verbose)
                    
                    # Truncate large results
                    result = truncate_toolresult(result, self.max_result_chars)
                    
                    # Record tool call
                    tool_calls.append({
                        "name": toolcall.name,
                        "arguments": toolcall.arguments,
                        "result": result,
                    })
                    
                    # Add to history
                    history.append({
                        "role": "assistant_toolcall",
                        "content": generated,
                        "name": toolcall.name,
                    })
                    
                    history.append({
                        "role": "toolresult",
                        "content": json.dumps(result, ensure_ascii=False),
                        "name": toolcall.name,
                    })
                    
                    if verbose:
                        print(f"\n  [Continuing to next step...]")
                    
                    # Continue loop
                    continue
            
            # No toolcall - this is the final answer
            if verbose:
                _debug_section("FINAL ANSWER", True)
                for i, line in enumerate(generated.split('\n'), 1):
                    print(f"  {i:3d} | {line}")
                _debug_end_section("FINAL ANSWER", True)
                
                print(f"\n  [Agent completed in {step + 1} step(s), {len(tool_calls)} tool call(s)]")
            
            # Audit: Agent loop completed
            if AUDIT_AVAILABLE:
                audit.agent("loop_complete",
                           steps=step + 1,
                           tool_calls=len(tool_calls),
                           answer_length=len(generated),
                           details=f"Agent completed in {step + 1} steps with {len(tool_calls)} tool calls")
            
            return {
                "role": "assistant",
                "content": generated,
                "tool_calls": tool_calls,
                "steps": step + 1,
            }
        
        # Max steps reached without final answer
        if verbose:
            print(f"\n  [WARNING] Max steps ({max_steps}) reached without final answer")
        
        return {
            "role": "assistant",
            "content": "I apologize, but I wasn't able to complete the task within the allowed steps.",
            "tool_calls": tool_calls,
            "steps": max_steps,
            "max_steps_reached": True,
        }
    
    def run_single(
        self, 
        user_input: str,
        system: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method for single-turn interaction.
        
        Args:
            user_input: User's message
            system: Optional system prompt override
            **kwargs: Passed to run()
            
        Returns:
            Agent response dict
        """
        history = []
        
        if system:
            history.append({"role": "system", "content": system})
        
        history.append({"role": "user", "content": user_input})
        
        return self.run(history, **kwargs)
