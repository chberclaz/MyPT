"""
Chat API Router - Handles chat/RAG interactions

Endpoints:
- GET /models - List available models
- GET /history - Get chat history
- POST /send - Send message and get response
- POST /clear - Clear chat history
"""

import os
import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()

# Session storage (in production, use Redis or database)
_chat_sessions = {}
_loaded_models = {}


class SendMessageRequest(BaseModel):
    message: str
    model: Optional[str] = None
    verbose: bool = False


class SendMessageResponse(BaseModel):
    content: str
    tool_calls: list = []
    steps: int = 0


def get_checkpoints_dir() -> Path:
    """Get the checkpoints directory."""
    return PROJECT_ROOT / "checkpoints"


def list_available_models() -> list[str]:
    """List all available model checkpoints."""
    checkpoints_dir = get_checkpoints_dir()
    if not checkpoints_dir.exists():
        return []
    
    models = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir():
            # Check if it has model.pt or config.json
            if (item / "model.pt").exists() or (item / "config.json").exists():
                models.append(item.name)
    
    return sorted(models)


def get_or_load_model(model_name: str):
    """Get or load a model by name."""
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    
    try:
        from core import load_model
        model = load_model(model_name)
        _loaded_models[model_name] = model
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@router.get("/models")
async def get_models():
    """List available models."""
    models = list_available_models()
    return {"models": models}


@router.get("/history")
async def get_history(session_id: str = "default"):
    """Get chat history for a session."""
    history = _chat_sessions.get(session_id, {}).get("messages", [])
    return {"messages": history}


@router.post("/send")
async def send_message(request: SendMessageRequest, session_id: str = "default"):
    """Send a message and get a response."""
    
    # Initialize session if needed
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = {"messages": [], "history": []}
    
    session = _chat_sessions[session_id]
    
    # Check if we have a model
    if not request.model:
        available = list_available_models()
        if not available:
            return SendMessageResponse(
                content="No models available. Please train a model first using the Training page.",
                tool_calls=[],
                steps=0
            )
        request.model = available[0]
    
    try:
        # Try to use the agent system if available
        model = get_or_load_model(request.model)
        
        # Check if this is an agent model (has workspace tools)
        workspace_dir = PROJECT_ROOT / "workspace"
        index_dir = workspace_dir / "index" / "latest"
        
        if index_dir.exists():
            # Use agent controller for RAG
            try:
                from core.workspace import WorkspaceEngine, WorkspaceTools
                from core.agent import AgentController
                
                engine = WorkspaceEngine(
                    base_dir=str(workspace_dir),
                    index_dir=str(index_dir)
                )
                tools = WorkspaceTools(engine)
                controller = AgentController(model, tools)
                
                # Build history for agent
                agent_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in session["messages"]
                    if msg["role"] in ("user", "assistant")
                ]
                agent_history.append({"role": "user", "content": request.message})
                
                # Run agent
                result = controller.run(
                    agent_history,
                    max_steps=5,
                    verbose=request.verbose
                )
                
                return SendMessageResponse(
                    content=result.get("content", ""),
                    tool_calls=result.get("tool_calls", []),
                    steps=result.get("steps", 0)
                )
                
            except ImportError:
                # Fall back to simple generation
                pass
            except Exception as e:
                if request.verbose:
                    print(f"Agent error: {e}")
                # Fall back to simple generation
        
        # Simple generation (no RAG)
        prompt = request.message
        response = model.generate(prompt, max_new_tokens=200)
        
        # Extract just the generated part (after the prompt)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return SendMessageResponse(
            content=response,
            tool_calls=[],
            steps=0
        )
        
    except Exception as e:
        return SendMessageResponse(
            content=f"Error generating response: {str(e)}",
            tool_calls=[],
            steps=0
        )


@router.post("/clear")
async def clear_history(session_id: str = "default"):
    """Clear chat history for a session."""
    if session_id in _chat_sessions:
        _chat_sessions[session_id] = {"messages": [], "history": []}
    return {"success": True}

