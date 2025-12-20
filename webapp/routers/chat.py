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
import time
import json
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.logging_config import DebugLogger, is_debug_mode
from webapp.auth import require_user, User

# Import audit logging
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

router = APIRouter()
log = DebugLogger("chat")

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
        log.debug(f"Checkpoints dir not found: {checkpoints_dir}")
        return []
    
    models = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir():
            if (item / "model.pt").exists() or (item / "config.json").exists():
                models.append(item.name)
    
    log.debug(f"Found {len(models)} models: {models}")
    return sorted(models)


def get_or_load_model(model_name: str):
    """Get or load a model by name, with optional torch.compile() for faster inference."""
    if model_name in _loaded_models:
        log.model("cache_hit", model_name)
        return _loaded_models[model_name]
    
    log.model("loading", model_name)
    start = time.time()
    
    try:
        import torch
        from core import load_model
        model = load_model(model_name)
        
        load_elapsed = time.time() - start
        log.model("loaded", model_name, time=f"{load_elapsed:.2f}s", device=str(model.config.device))
        
        # Compile model for faster inference (PyTorch 2.0+)
        # Only compile if torch.compile is available and we're on CUDA
        if hasattr(torch, 'compile') and model.config.device == 'cuda':
            # Suppress dynamo errors so compilation failure doesn't crash generation
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            try:
                log.info(f"Compiling model for faster inference...")
                compile_start = time.time()
                # Use 'default' mode which is more compatible than 'reduce-overhead'
                # 'reduce-overhead' requires Triton which may not be installed
                model.compile_for_inference(mode="default")
                compile_elapsed = time.time() - compile_start
                log.info(f"Model compiled in {compile_elapsed:.2f}s (first generation will trigger JIT)")
            except Exception as e:
                log.warning(f"torch.compile() skipped: {str(e)[:100]}")
                log.info("Using uncompiled model (still fast with other optimizations)")
        
        _loaded_models[model_name] = model
        return model
    except Exception as e:
        log.error(f"Failed to load model {model_name}", e)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def log_full_prompt(prompt: str, label: str = "PROMPT"):
    """Log the full prompt being sent to the model."""
    if is_debug_mode():
        log.section(f"{label} TO MODEL")
        # Print the full prompt with line numbers for debugging
        lines = prompt.split('\n')
        for i, line in enumerate(lines, 1):
            # Truncate very long lines but show them
            if len(line) > 200:
                print(f"  {i:3d} | {line[:200]}...")
            else:
                print(f"  {i:3d} | {line}")
        print(f"  --- | [Total: {len(prompt)} chars, {len(lines)} lines]")
        log.section("END PROMPT")


def log_model_response(response: str, label: str = "RESPONSE"):
    """Log the full model response."""
    if is_debug_mode():
        log.section(f"MODEL {label}")
        lines = response.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > 200:
                print(f"  {i:3d} | {line[:200]}...")
            else:
                print(f"  {i:3d} | {line}")
        print(f"  --- | [Total: {len(response)} chars]")
        log.section("END RESPONSE")


def log_tool_call(name: str, arguments: dict, result: dict):
    """Log a tool call with full details."""
    if is_debug_mode():
        log.section(f"TOOL CALL: {name}")
        print(f"  Arguments: {json.dumps(arguments, indent=2)}")
        print(f"  Result:    {json.dumps(result, indent=2)[:500]}{'...' if len(json.dumps(result)) > 500 else ''}")
        log.section("END TOOL CALL")


def log_rag_context(chunks: list, query: str):
    """Log RAG retrieval results."""
    if is_debug_mode():
        log.section("RAG CONTEXT RETRIEVAL")
        print(f"  Query: \"{query}\"")
        print(f"  Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get('text', str(chunk))[:150]
            score = chunk.get('score', 'N/A')
            print(f"  [{i}] Score: {score}")
            print(f"      {text}...")
        log.section("END RAG CONTEXT")


def log_agent_history(history: list):
    """Log the conversation history being sent to agent."""
    if is_debug_mode():
        log.section("AGENT CONVERSATION HISTORY")
        for i, msg in enumerate(history):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]
            print(f"  [{i+1}] {role.upper()}: {content}{'...' if len(msg.get('content', '')) > 200 else ''}")
        log.section("END HISTORY")


@router.get("/models")
async def get_models(user: User = Depends(require_user)):
    """List available models - requires authentication."""
    log.request("GET", "/models", user=user.username)
    models = list_available_models()
    log.response(200, count=len(models))
    return {"models": models}


@router.get("/history")
async def get_history(session_id: str = "default", user: User = Depends(require_user)):
    """Get chat history for a session - requires authentication."""
    log.request("GET", "/history", session=session_id, user=user.username)
    history = _chat_sessions.get(session_id, {}).get("messages", [])
    log.response(200, messages=len(history))
    return {"messages": history}


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    # Check X-Forwarded-For header (for proxied requests)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # Fall back to direct client
    if request.client:
        return request.client.host
    return "unknown"


@router.post("/send")
async def send_message(
    msg_request: SendMessageRequest, 
    request: Request,
    session_id: str = "default", 
    user: User = Depends(require_user)
):
    """Send a message and get a response - requires authentication."""
    log.section("NEW CHAT REQUEST")
    log.request("POST", "/send", session=session_id, model=msg_request.model, verbose=msg_request.verbose)
    
    # Get client IP for audit
    client_ip = get_client_ip(request)
    
    # Audit: User prompt received
    if AUDIT_AVAILABLE:
        audit.chat("prompt", user=user.username, ip=client_ip,
                  session=session_id, model=msg_request.model or "auto",
                  tokens_in=len(msg_request.message.split()),  # Rough word count
                  details=msg_request.message[:200] + ("..." if len(msg_request.message) > 200 else ""))
    
    # Log the full user message
    if is_debug_mode():
        log.section("USER INPUT")
        print(f"  Message: \"{msg_request.message}\"")
        print(f"  Length:  {len(msg_request.message)} chars")
        log.section("END USER INPUT")
    
    start_time = time.time()
    
    # Initialize session if needed
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = {"messages": [], "history": []}
        log.debug(f"Created new session: {session_id}")
    
    session = _chat_sessions[session_id]
    
    # Check if we have a model
    if not msg_request.model:
        available = list_available_models()
        if not available:
            log.warning("No models available")
            return SendMessageResponse(
                content="No models available. Please train a model first using the Training page.",
                tool_calls=[],
                steps=0
            )
        msg_request.model = available[0]
        log.info(f"Auto-selected model: {msg_request.model}")
    
    try:
        # Load model
        model = get_or_load_model(msg_request.model)
        
        # Log model info
        if is_debug_mode():
            log.section("MODEL INFO")
            print(f"  Name:       {msg_request.model}")
            print(f"  Layers:     {model.config.n_layer}")
            print(f"  Heads:      {model.config.n_head}")
            print(f"  Embedding:  {model.config.n_embd}")
            print(f"  Block size: {model.config.block_size}")
            print(f"  Device:     {model.config.device}")
            log.section("END MODEL INFO")
        
        # =================================================================
        # TRUE AGENTIC MODE: Always use AgentController
        # The MODEL decides whether to use tools, not the code!
        # =================================================================
        
        workspace_dir = PROJECT_ROOT / "workspace"
        index_dir = workspace_dir / "index" / "latest"
        
        log.workspace("check_index", path=str(index_dir), exists=index_dir.exists())
        log.info("Using AGENTIC mode - model decides tool usage")
        
        try:
            from core.workspace import WorkspaceEngine, WorkspaceTools
            from core.agent import AgentController
            
            # Initialize workspace engine (works even without index)
            log.rag("init_engine", docs_dir=str(workspace_dir / "docs"))
            engine = WorkspaceEngine(
                base_dir=str(workspace_dir),
                index_dir=str(index_dir)
            )
            
            if is_debug_mode():
                log.section("WORKSPACE ENGINE")
                print(f"  Workspace:  {workspace_dir}")
                print(f"  Index Dir:  {index_dir}")
                print(f"  Documents:  {engine.num_docs}")
                print(f"  Chunks:     {engine.num_chunks}")
                print(f"  Has Index:  {engine.has_index}")
                if not engine.has_index:
                    print(f"  [NOTE] No index built - workspace.search will return empty")
                    print(f"         Run 'Rebuild Index' or: python scripts/build_rag_index.py")
                log.section("END WORKSPACE")
            
            log.agent("init_tools")
            tools = WorkspaceTools(engine)
            
            log.agent("init_controller")
            controller = AgentController(model, tools)
            
            # Build history for agent
            agent_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in session["messages"]
                if msg["role"] in ("user", "assistant")
            ]
            agent_history.append({"role": "user", "content": msg_request.message})
            
            # Log the full history being sent
            log_agent_history(agent_history)
            
            log.agent("history_built", turns=len(agent_history))
            
            # Run agent - this ALWAYS wraps prompt with system instructions
            # The model sees tool descriptions and decides whether to use them
            log.section("AGENT EXECUTION START")
            log.agent("run_start", max_steps=5)
            
            # Log system prompt so user sees the full agentic context
            if is_debug_mode():
                log.section("AGENT SYSTEM PROMPT")
                print(controller.system_prompt)
                log.section("END SYSTEM PROMPT")
            
            # Run with verbose mode when debug is enabled
            result = controller.run(
                agent_history,
                max_steps=5,
                verbose=is_debug_mode()
            )
            
            # Log tool calls with full details
            tool_calls = result.get("tool_calls", [])
            steps = result.get("steps", 0)
            
            if is_debug_mode():
                if tool_calls:
                    log.section("ALL TOOL CALLS")
                    for i, tc in enumerate(tool_calls):
                        print(f"\n  === Tool Call {i+1} ===")
                        print(f"  Name: {tc.get('name', 'unknown')}")
                        print(f"  Args: {json.dumps(tc.get('arguments', {}), indent=4)}")
                        result_str = json.dumps(tc.get('result', {}), indent=4)
                        if len(result_str) > 1000:
                            print(f"  Result (truncated): {result_str[:1000]}...")
                        else:
                            print(f"  Result: {result_str}")
                    log.section("END TOOL CALLS")
                else:
                    log.info("Model answered directly without tool calls")
            
            log.agent("run_complete", steps=steps, tool_calls=len(tool_calls))
            
            # Log the final response
            content = result.get("content", "")
            log_model_response(content, "FINAL ANSWER")
            
            elapsed = time.time() - start_time
            log.response(200, time=f"{elapsed:.2f}s", steps=steps, tool_calls=len(tool_calls))
            
            # Audit: Response generated
            if AUDIT_AVAILABLE:
                audit.chat("response", user=user.username, ip=client_ip,
                          session=session_id, model=msg_request.model,
                          tokens_out=len(content.split()),  # Rough word count
                          latency_ms=int(elapsed * 1000), steps=steps,
                          tool_calls=len(tool_calls),
                          details=content[:200] + ("..." if len(content) > 200 else ""))
                
                # Audit: Each tool call
                for tc in tool_calls:
                    audit.agent("tool_call", user=user.username, session=session_id,
                               tool=tc.get("name", "unknown"),
                               details=json.dumps(tc.get("arguments", {}))[:200])
            
            return SendMessageResponse(
                content=content,
                tool_calls=tool_calls,
                steps=steps
            )
            
        except ImportError as e:
            # Agent modules not available - this is a code issue, not runtime
            log.error(f"Agent modules not available: {e}")
            if is_debug_mode():
                import traceback
                traceback.print_exc()
            return SendMessageResponse(
                content=f"Agent system not available: {e}. Check core/agent/ and core/workspace/ modules.",
                tool_calls=[],
                steps=0
            )
        except Exception as e:
            log.error("Agent execution failed", e)
            if is_debug_mode():
                import traceback
                traceback.print_exc()
            return SendMessageResponse(
                content=f"Error in agent execution: {str(e)}",
                tool_calls=[],
                steps=0
            )
        
    except Exception as e:
        log.error("Chat request failed", e)
        if is_debug_mode():
            import traceback
            traceback.print_exc()
        return SendMessageResponse(
            content=f"Error generating response: {str(e)}",
            tool_calls=[],
            steps=0
        )


@router.post("/clear")
async def clear_history(session_id: str = "default", user: User = Depends(require_user)):
    """Clear chat history for a session - requires authentication."""
    log.request("POST", "/clear", session=session_id, user=user.username)
    if session_id in _chat_sessions:
        _chat_sessions[session_id] = {"messages": [], "history": []}
    log.response(200)
    return {"success": True}
