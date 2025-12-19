"""
MyPT Web Application - Self-Contained Offline GUI

A FastAPI-based web interface for MyPT that can run as:
- Local agent (localhost)
- Department server (network accessible)

All resources are bundled locally - no external CDN dependencies.

Usage:
    # Local mode
    python -m webapp.main
    
    # Or via entry point (after pip install -e .)
    mypt-webapp
    
    # Server mode (accessible on network)
    mypt-webapp --host 0.0.0.0 --port 8000
    
    # Debug mode (verbose logging) - shows all prompts and responses
    mypt-webapp --debug
    
    # Or via environment variable:
    MYPT_DEBUG=1 mypt-webapp
    
    # Toggle debug at runtime via API:
    POST /api/debug/toggle

Debug mode will show:
- Full prompts being sent to the model
- Complete model responses  
- Tool call arguments and results
- RAG retrieval details
- All communication between User → RAG → Workspace → Model
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.logging_config import setup_logging, get_logger, set_debug_mode, is_debug_mode
from webapp.routers import chat, training, workspace
from webapp.auth import (
    User, authenticate_user, require_user, require_admin, optional_user,
    create_login_response, create_logout_response, get_current_user
)
from core.banner import print_banner, ROBOT_HEAD

# Paths
WEBAPP_DIR = Path(__file__).parent
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"

# Initialize logger
log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    log.info("=" * 60)
    log.info("  MyPT Web Application Starting...")
    log.info("=" * 60)
    log.info(f"  Static files: {STATIC_DIR}")
    log.info(f"  Templates:    {TEMPLATES_DIR}")
    log.info(f"  Debug mode:   {'ON' if is_debug_mode() else 'OFF'}")
    log.info("=" * 60)
    yield
    log.info("MyPT Web Application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MyPT",
    description="Offline GPT Training & Agentic RAG Pipeline",
    version="0.2.0",
    lifespan=lifespan,
)

# Mount static files (CSS, JS - all bundled locally)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(workspace.router, prefix="/api/workspace", tags=["workspace"])


# ============================================================================
# Authentication Routes
# ============================================================================

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/", error: str = None):
    """Login page."""
    # If already logged in, redirect
    user = await get_current_user(request)
    if user:
        return RedirectResponse(url=next, status_code=302)
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "next_url": next,
        "error": error
    })


@app.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/")
):
    """Handle login form submission."""
    user = authenticate_user(username, password)
    
    if not user:
        log.warning(f"Failed login attempt for user: {username}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "next_url": next,
            "error": "Invalid username or password"
        })
    
    log.info(f"User logged in: {username} (role: {user.role})")
    return create_login_response(user, redirect_url=next)


@app.get("/logout")
async def logout(request: Request):
    """Log out the current user."""
    user = await get_current_user(request)
    if user:
        log.info(f"User logged out: {user.username}")
    return create_logout_response()


# ============================================================================
# Page Routes (Protected)
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: User = Depends(require_user)):
    """Redirect to chat page."""
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: User = Depends(require_user)):
    """Chat/RAG interface page - requires user or admin."""
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request, user: User = Depends(require_admin)):
    """Training pipeline page - requires admin."""
    return templates.TemplateResponse("training.html", {"request": request, "user": user})


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring (public)."""
    return {"status": "healthy", "version": "0.2.0", "debug": is_debug_mode()}


@app.get("/api/debug/status")
async def debug_status(user: User = Depends(require_admin)):
    """Get debug mode status and logging info - admin only."""
    return {
        "debug_mode": is_debug_mode(),
        "static_dir": str(STATIC_DIR),
        "templates_dir": str(TEMPLATES_DIR),
        "project_root": str(PROJECT_ROOT),
        "checkpoints_dir": str(PROJECT_ROOT / "checkpoints"),
        "workspace_dir": str(PROJECT_ROOT / "workspace"),
        "user": user.username,
    }


@app.post("/api/debug/toggle")
async def toggle_debug(user: User = Depends(require_admin)):
    """Toggle debug mode at runtime - admin only."""
    new_state = not is_debug_mode()
    set_debug_mode(new_state)
    log.info(f"Debug mode {'enabled' if new_state else 'disabled'} by {user.username}")
    return {"debug_mode": new_state}


@app.get("/api/auth/me")
async def get_me(user: User = Depends(require_user)):
    """Get current user info."""
    return {"username": user.username, "role": user.role, "is_admin": user.is_admin()}


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for mypt-webapp command."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MyPT Web Application - Offline GPT Pipeline GUI"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for local, use 0.0.0.0 for network)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging (also: set MYPT_DEBUG=1)"
    )
    
    args = parser.parse_args()
    
    # Also check environment variable for debug mode
    env_debug = os.environ.get("MYPT_DEBUG", "").lower() in ("1", "true", "yes", "on")
    args.debug = args.debug or env_debug
    
    # Setup logging
    setup_logging(debug=args.debug)
    log = get_logger("main")
    
    # Print banner
    print_banner("MyPT Web Application", "Offline GPT Training & Agentic RAG Pipeline")
    
    if args.host == "127.0.0.1":
        print(f"  Mode:   Local Agent")
        print(f"  URL:    http://localhost:{args.port}")
    else:
        print(f"  Mode:   Department Server")
        print(f"  URL:    http://{args.host}:{args.port}")
    
    if args.debug:
        print(f"  Debug:  ENABLED (verbose logging)")
    
    print(f"  Auth:   ENABLED (user/admin roles)")
    print(f"          Default: admin:admin, user:user")
    print("=" * 60 + "\n")
    
    # Set log level for uvicorn
    log_level = "debug" if args.debug else "info"
    
    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
