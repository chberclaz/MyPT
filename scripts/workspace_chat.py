#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive workspace agent chat CLI.

Full agentic RAG demo: model can call workspace tools to search,
read, and summarize documents.

Usage:
    python scripts/workspace_chat.py --model_name my_agent --workspace_dir workspace/ --index_dir workspace/index/latest
    
Commands during chat:
    /reload     - Reload the workspace index
    /docs       - List documents in workspace
    /tools      - Show available tools
    /history    - Show conversation history
    /clear      - Clear conversation history
    /verbose    - Toggle verbose mode (show tool calls)
    /quit       - Exit chat
"""

import argparse
import os
import sys
import json

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_model
from core.workspace import WorkspaceEngine, WorkspaceTools
from core.agent import AgentController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive workspace agent chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of trained model in checkpoints/")
    parser.add_argument("--workspace_dir", type=str, default="workspace/",
                        help="Workspace directory (default: workspace/)")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="RAG index directory (default: workspace/index/latest)")
    
    parser.add_argument("--max_steps", type=int, default=5,
                        help="Maximum tool execution steps (default: 5)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show tool calls and results")
    
    parser.add_argument("--system", type=str, default=None,
                        help="Custom system prompt")
    
    return parser.parse_args()


def print_tool_calls(tool_calls: list):
    """Print tool calls nicely."""
    if not tool_calls:
        return
    
    print("\n  [Tools used]")
    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        result = tc.get("result", {})
        
        args_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in args.items())
        print(f"    {i}. {name}({args_str})")
        
        if "error" in result:
            print(f"       Error: {result['error']}")
        elif "documents" in result:
            print(f"       Found {len(result['documents'])} documents")
        elif "summary" in result:
            print(f"       Summary: {result['summary'][:100]}...")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MyPT Workspace Agent")
    print("=" * 60)
    
    # Determine index directory
    index_dir = args.index_dir
    if index_dir is None:
        index_dir = os.path.join(args.workspace_dir, "index", "latest")
    
    # Load model
    print(f"\nLoading model: {args.model_name}...")
    try:
        model = load_model(args.model_name)
        print(f"  [OK] Model loaded")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    # Initialize workspace
    print(f"Loading workspace: {args.workspace_dir}...")
    try:
        engine = WorkspaceEngine(args.workspace_dir, index_dir)
        print(f"  [OK] Workspace loaded: {engine.num_docs} documents")
        if engine.has_index:
            print(f"  [OK] Index loaded: {engine.num_chunks} chunks")
        else:
            print(f"  [WARN] No index loaded. Run build_rag_index.py first.")
    except Exception as e:
        print(f"  [ERROR] Failed to load workspace: {e}")
        sys.exit(1)
    
    # Create tools and controller
    tools = WorkspaceTools(engine, model=model)
    controller = AgentController(
        model=model,
        tools=tools,
        system_prompt=args.system,
    )
    
    # State
    history = []
    verbose = args.verbose
    
    print("\n" + "-" * 60)
    print("Ready! Type your question or /help for commands.")
    print("-" * 60 + "\n")
    
    while True:
        try:
            user_input = input("user> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
            
            if cmd in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            elif cmd == "reload":
                print("Reloading workspace...")
                try:
                    engine.refresh()
                    engine.reload_index()
                    print(f"  [OK] Reloaded: {engine.num_docs} docs, {engine.num_chunks} chunks")
                except Exception as e:
                    print(f"  [ERROR] {e}")
            
            elif cmd == "docs":
                docs = engine.list_docs()
                if docs:
                    print(f"\nDocuments ({len(docs)}):")
                    for d in docs:
                        print(f"  - {d.title} ({d.doc_id[:8]}...)")
                else:
                    print("No documents in workspace.")
            
            elif cmd == "tools":
                print("\nAvailable tools:")
                for name in tools.list_tools():
                    print(f"  - {name}")
            
            elif cmd == "history":
                print(f"\nConversation history ({len(history)} messages):")
                for msg in history:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")[:100]
                    print(f"  [{role}] {content}...")
            
            elif cmd == "clear":
                history = []
                print("Conversation history cleared.")
            
            elif cmd == "verbose":
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            
            elif cmd == "help":
                print("""
Commands:
  /reload   - Reload workspace and index
  /docs     - List documents
  /tools    - Show available tools
  /history  - Show conversation history
  /clear    - Clear history
  /verbose  - Toggle verbose mode
  /quit     - Exit
                """)
            
            else:
                print(f"Unknown command: /{cmd}")
            
            continue
        
        # Add user message to history
        history.append({"role": "user", "content": user_input})
        
        # Run agent
        try:
            result = controller.run(
                history,
                max_steps=args.max_steps,
                max_new_tokens=args.max_tokens,
                verbose=verbose,
            )
            
            # Show tool calls if verbose
            if verbose and result.get("tool_calls"):
                print_tool_calls(result["tool_calls"])
            
            # Show answer
            answer = result.get("content", "")
            print(f"\nassistant> {answer}")
            
            # Show step count
            steps = result.get("steps", 0)
            tool_count = len(result.get("tool_calls", []))
            if tool_count > 0:
                print(f"\n  ({tool_count} tool calls, {steps} steps)")
            
            # Add to history
            history.append({"role": "assistant", "content": answer})
            
            print()
        
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            import traceback
            if verbose:
                traceback.print_exc()


if __name__ == "__main__":
    main()

