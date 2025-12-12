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


# Default system prompt for agent
DEFAULT_SYSTEM_PROMPT = """You are MyPT, a helpful workspace assistant.

You have access to workspace tools to search, read, and summarize documents.

To use a tool, output a toolcall block like:
<myPT_toolcall>{"name": "workspace.search", "query": "your search", "top_k": 5}</myPT_toolcall>

Available tools:
- workspace.search: Search documents by query
- workspace.list_docs: List all documents
- workspace.get_doc: Get full document text (args: doc_id or title)
- workspace.summarize: Summarize a document (args: doc_id or text)

After receiving tool results, provide a helpful answer to the user."""


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


class AgentController:
    """
    Orchestrates model generation with tool execution.
    
    Args:
        model: Model with generate() method
        tools: WorkspaceTools instance
        system_prompt: System prompt (default: DEFAULT_SYSTEM_PROMPT)
        max_result_chars: Max chars for tool results (default: 2000)
    """
    
    def __init__(
        self, 
        model,
        tools,
        system_prompt: str = None,
        max_result_chars: int = 2000,
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_result_chars = max_result_chars
    
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
    
    def run(
        self, 
        history: List[Dict[str, Any]], 
        max_steps: int = 5,
        max_new_tokens: int = 512,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the agent loop until final answer or max_steps.
        
        Args:
            history: Initial conversation history
            max_steps: Maximum tool execution steps
            max_new_tokens: Max tokens per generation
            verbose: Print debug info
            
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
        
        for step in range(max_steps):
            # Build prompt
            prompt = self.build_prompt(history)
            
            if verbose:
                print(f"\n--- Step {step + 1} ---")
                print(f"Prompt length: {len(prompt)} chars")
            
            # Generate
            try:
                output = self.model.generate(prompt, max_new_tokens=max_new_tokens)
            except Exception as e:
                return {
                    "role": "assistant",
                    "content": f"Error generating response: {e}",
                    "tool_calls": tool_calls,
                    "steps": step + 1,
                    "error": str(e),
                }
            
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
                print(f"Generated: {generated[:200]}...")
            
            # Check for toolcall
            if has_toolcall(generated):
                toolcall = find_toolcall(generated)
                
                if toolcall:
                    if verbose:
                        print(f"Tool call: {toolcall.name}({toolcall.arguments})")
                    
                    # Execute tool
                    try:
                        result = self.tools.execute(toolcall.name, toolcall.arguments)
                    except Exception as e:
                        result = {"error": str(e)}
                    
                    # Truncate large results
                    result = truncate_toolresult(result, self.max_result_chars)
                    
                    if verbose:
                        print(f"Tool result: {str(result)[:200]}...")
                    
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
                    
                    import json
                    history.append({
                        "role": "toolresult",
                        "content": json.dumps(result, ensure_ascii=False),
                        "name": toolcall.name,
                    })
                    
                    # Continue loop
                    continue
            
            # No toolcall - this is the final answer
            return {
                "role": "assistant",
                "content": generated,
                "tool_calls": tool_calls,
                "steps": step + 1,
            }
        
        # Max steps reached without final answer
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

