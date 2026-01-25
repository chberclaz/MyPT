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
    
    parser.add_argument("--mode", type=str, default="agentic",
                        choices=["conversation", "agentic"],
                        help="Chat mode: 'conversation' (no tools) or 'agentic' (with tools)")
    
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
    
    from core.banner import print_banner
    from core.system_prompts import CONVERSATION_SYSTEM_PROMPT, DEFAULT_AGENTIC_PROMPT
    from core.special_tokens import SPECIAL_TOKEN_STRINGS
    
    mode = args.mode
    subtitle = "Agentic RAG Chat Interface" if mode == "agentic" else "Conversation Mode"
    print_banner("MyPT Workspace Agent", subtitle)
    
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
    
    # Initialize workspace (only needed for agentic mode)
    engine = None
    tools = None
    controller = None
    
    if mode == "agentic":
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
        system_prompt = args.system or DEFAULT_AGENTIC_PROMPT
        controller = AgentController(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
    else:
        print(f"  [OK] Conversation mode (no workspace tools)")
    
    # State
    history = []
    verbose = args.verbose
    
    # Get special tokens for conversation mode
    SYSTEM_OPEN = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
    SYSTEM_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_system_close"]
    USER_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
    USER_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_close"]
    ASSISTANT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
    ASSISTANT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]
    
    print("\n" + "-" * 60)
    print(f"Mode: {mode.upper()}")
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
                if mode == "agentic" and engine:
                    print("Reloading workspace...")
                    try:
                        engine.refresh()
                        engine.reload_index()
                        print(f"  [OK] Reloaded: {engine.num_docs} docs, {engine.num_chunks} chunks")
                    except Exception as e:
                        print(f"  [ERROR] {e}")
                else:
                    print("Reload not available in conversation mode.")
            
            elif cmd == "docs":
                if mode == "agentic" and engine:
                    docs = engine.list_docs()
                    if docs:
                        print(f"\nDocuments ({len(docs)}):")
                        for d in docs:
                            print(f"  - {d.title} ({d.doc_id[:8]}...)")
                    else:
                        print("No documents in workspace.")
                else:
                    print("Document listing not available in conversation mode.")
            
            elif cmd == "tools":
                if mode == "agentic" and tools:
                    print("\nAvailable tools:")
                    for name in tools.list_tools():
                        print(f"  - {name}")
                else:
                    print("\nNo tools available in conversation mode.")
            
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
        
        # Generate response based on mode
        try:
            if mode == "agentic":
                # Use AgentController with tools
                result = controller.run(
                    history,
                    max_steps=args.max_steps,
                    max_new_tokens=args.max_tokens,
                    verbose=verbose,
                )
                
                # Show tool calls if verbose
                if verbose and result.get("tool_calls"):
                    print_tool_calls(result["tool_calls"])
                
                answer = result.get("content", "")
                steps = result.get("steps", 0)
                tool_count = len(result.get("tool_calls", []))
                
            else:
                # Conversation mode - simple generation
                system_prompt = args.system or CONVERSATION_SYSTEM_PROMPT
                
                # Build prompt
                parts = [f"{SYSTEM_OPEN}{system_prompt}{SYSTEM_CLOSE}"]
                for msg in history:
                    if msg["role"] == "user":
                        parts.append(f"{USER_OPEN}{msg['content']}{USER_CLOSE}")
                    elif msg["role"] == "assistant":
                        parts.append(f"{ASSISTANT_OPEN}{msg['content']}{ASSISTANT_CLOSE}")
                parts.append(ASSISTANT_OPEN)
                
                prompt = "\n".join(parts)
                
                if verbose:
                    print(f"\n[Prompt: {len(prompt)} chars]")
                
                # Generate
                output = model.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=0.7,
                    top_k=40,
                    repetition_penalty=1.1
                )
                
                # Extract response
                if prompt in output:
                    answer = output[len(prompt):]
                else:
                    answer = output
                
                if ASSISTANT_CLOSE in answer:
                    answer = answer.split(ASSISTANT_CLOSE)[0]
                
                answer = answer.strip()
                steps = 1
                tool_count = 0
            
            # Show answer
            print(f"\nassistant> {answer}")
            
            # Show step count
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



