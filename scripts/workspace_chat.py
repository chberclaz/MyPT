#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive workspace agent chat CLI (PyTorch GOLD or GGUF).

    python scripts/workspace_chat.py --model_name phase6_3_ground_gold
    python scripts/workspace_chat.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf
    python scripts/workspace_chat.py --list-models
    python scripts/workspace_chat.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf --query "Which documents exist?"

Commands during chat:
    /reload /docs /tools /history /clear /verbose /quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent.workspace_ask import workspace_ask
from core.inference.runtime import (
    close_runtime_model,
    list_runtime_models,
    load_runtime_model,
    runtime_backend,
)
from core.workspace import WorkspaceEngine, WorkspaceTools
from core.agent import AgentController
from core.agent.workspace_ask import _conversation_generate
from core.system_prompts import CHAT_SYSTEM_PROMPT, DEFAULT_AGENTIC_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive or one-shot workspace agent chat (PyTorch or GGUF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --list-models\n"
            "  %(prog)s --model_name phase6_3_ground_gold\n"
            "  %(prog)s --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf\n"
            "  %(prog)s --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf "
            "--query \"List the documents\" --json\n"
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Checkpoint folder, dir/file.gguf picker id, or path to a .gguf",
    )
    parser.add_argument("--workspace_dir", type=str, default="workspace/",
                        help="Workspace directory (default: workspace/)")
    parser.add_argument("--index_dir", type=str, default=None,
                        help="RAG index directory (default: workspace/index/latest)")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="agentic",
        choices=["conversation", "agentic"],
    )
    parser.add_argument("--list-models", action="store_true",
                        help="Print available PyTorch folders and GGUF picker ids")
    parser.add_argument("--query", type=str, default=None,
                        help="One-shot user message (no interactive loop). Use - to read stdin.")
    parser.add_argument("--json", action="store_true",
                        help="With --query: write one JSON object to stdout")
    parser.add_argument("--ngl", type=int, default=99,
                        help="llama.cpp GPU layers for GGUF (0 = CPU)")
    parser.add_argument("--port", type=int, default=None,
                        help="llama-server port (default: first free from 8765)")
    parser.add_argument("--n_ctx", type=int, default=4096)
    return parser.parse_args()


def print_tool_calls(tool_calls: list):
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


def _read_query(raw: str) -> str:
    if raw == "-":
        return sys.stdin.read().strip()
    return raw.strip()


def main():
    args = parse_args()

    if args.list_models:
        names = list_runtime_models()
        if not names:
            print("No models found under checkpoints/ or export/artifacts/", file=sys.stderr)
            sys.exit(1)
        for n in names:
            print(n)
        return

    if not args.model_name:
        print("error: --model_name is required (or pass --list-models)", file=sys.stderr)
        sys.exit(2)

    if args.query is not None:
        query = _read_query(args.query)
        if not query:
            print("error: empty --query", file=sys.stderr)
            sys.exit(2)
        result = workspace_ask(
            args.model_name,
            query,
            workspace_dir=args.workspace_dir,
            index_dir=args.index_dir,
            mode=args.mode,
            max_steps=args.max_steps,
            max_new_tokens=args.max_tokens,
            system=args.system,
            verbose=args.verbose and not args.json,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.ngl,
            port=args.port,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False))
        else:
            if args.verbose and result.get("tool_calls"):
                print_tool_calls(result["tool_calls"])
            print(result.get("content") or "")
            if result.get("error"):
                print(f"[ERROR] {result['error']}", file=sys.stderr)
        sys.exit(0 if result.get("ok") else 1)

    from core.banner import print_banner

    mode = args.mode
    subtitle = "Agentic RAG Chat Interface" if mode == "agentic" else "Conversation Mode"
    print_banner("MyPT Workspace Agent", subtitle)

    index_dir = args.index_dir
    if index_dir is None:
        index_dir = os.path.join(args.workspace_dir, "index", "latest")

    print(f"\nLoading model: {args.model_name}...")
    try:
        model = load_runtime_model(
            args.model_name,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.ngl,
            port=args.port,
        )
        print(f"  [OK] Model loaded ({runtime_backend(model)})")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    engine = None
    tools = None
    controller = None

    try:
        if mode == "agentic":
            print(f"Loading workspace: {args.workspace_dir}...")
            try:
                engine = WorkspaceEngine(args.workspace_dir, index_dir)
                print(f"  [OK] Workspace loaded: {engine.num_docs} documents")
                if engine.has_index:
                    print(f"  [OK] Index loaded: {engine.num_chunks} chunks")
                else:
                    print("  [WARN] No index loaded. Run build_rag_index.py first.")
            except Exception as e:
                print(f"  [ERROR] Failed to load workspace: {e}", file=sys.stderr)
                sys.exit(1)
            tools = WorkspaceTools(engine, model=model)
            controller = AgentController(
                model=model,
                tools=tools,
                system_prompt=args.system or DEFAULT_AGENTIC_PROMPT,
            )
        else:
            print("  [OK] Conversation mode (no workspace tools)")

        history = []
        verbose = args.verbose

        print("\n" + "-" * 60)
        print(f"Mode: {mode.upper()}  backend: {runtime_backend(model)}")
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

            if user_input.startswith("/"):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()

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

            history.append({"role": "user", "content": user_input})
            try:
                if mode == "agentic":
                    result = controller.run(
                        history,
                        max_steps=args.max_steps,
                        max_new_tokens=args.max_tokens,
                        verbose=verbose,
                    )
                    if verbose and result.get("tool_calls"):
                        print_tool_calls(result["tool_calls"])
                    answer = result.get("content", "")
                    steps = result.get("steps", 0)
                    tool_count = len(result.get("tool_calls", []))
                else:
                    answer = _conversation_generate(
                        model,
                        history,
                        args.system or CHAT_SYSTEM_PROMPT,
                        args.max_tokens,
                    )
                    steps = 1
                    tool_count = 0

                print(f"\nassistant> {answer}")
                if tool_count > 0:
                    print(f"\n  ({tool_count} tool calls, {steps} steps)")
                history.append({"role": "assistant", "content": answer})
                print()
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                if verbose:
                    import traceback
                    traceback.print_exc()
    finally:
        close_runtime_model(model)


if __name__ == "__main__":
    main()
