"""
Agent module for agentic RAG.

Provides:
- AgentController: Orchestrates model + tools in a loop
- ToolCall: Parsed toolcall data structure
- Parsing utilities for toolcall/toolresult blocks

Usage:
    from core.agent import AgentController, ToolCall
    from core.workspace import WorkspaceEngine, WorkspaceTools
    
    engine = WorkspaceEngine("workspace/", "workspace/index/latest")
    tools = WorkspaceTools(engine)
    controller = AgentController(model, tools)
    
    result = controller.run([
        {"role": "user", "content": "Find docs about X"}
    ])
"""

from .parsing import ToolCall, find_toolcall, render_toolcall, render_toolresult
from .controller import AgentController

__all__ = [
    "AgentController",
    "ToolCall",
    "find_toolcall",
    "render_toolcall",
    "render_toolresult",
]



