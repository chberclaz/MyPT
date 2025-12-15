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
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.routers import chat, training, workspace

# Paths
WEBAPP_DIR = Path(__file__).parent
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("=" * 60)
    print("  MyPT Web Application Starting...")
    print("=" * 60)
    print(f"  Static files: {STATIC_DIR}")
    print(f"  Templates:    {TEMPLATES_DIR}")
    print("=" * 60)
    yield
    print("\nMyPT Web Application shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MyPT",
    description="Offline GPT Training & RAG Pipeline",
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
# Page Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Redirect to chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat/RAG interface page."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """Training pipeline page."""
    return templates.TemplateResponse("training.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "0.2.0"}


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
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MyPT Web Application")
    print("  Offline GPT Training & RAG Pipeline")
    print("=" * 60)
    
    if args.host == "127.0.0.1":
        print(f"  Mode: Local Agent")
        print(f"  URL:  http://localhost:{args.port}")
    else:
        print(f"  Mode: Department Server")
        print(f"  URL:  http://{args.host}:{args.port}")
    
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

