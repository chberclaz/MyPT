#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot workspace RAG for other applications.

Always prints one JSON object to stdout (logs on stderr). Exit 0 if ok.

    python scripts/workspace_rag.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf --query "What is in the handbook?"
    echo "Who signed the contract?" | python scripts/workspace_rag.py --model_name phase6_3_ground_gold --query -

Python:

    from core.agent import workspace_ask
    print(workspace_ask("phase6_3_ground_gold/mypt-q4_k_m.gguf", "List documents"))
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
from core.inference.runtime import list_runtime_models


def parse_args():
    p = argparse.ArgumentParser(description="One-shot MyPT workspace RAG (JSON stdout)")
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--query", type=str, required=False, help="User question; - reads stdin")
    p.add_argument("--workspace_dir", type=str, default="workspace/")
    p.add_argument("--index_dir", type=str, default=None)
    p.add_argument("--mode", type=str, default="agentic", choices=["conversation", "agentic"])
    p.add_argument("--max_steps", type=int, default=5)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--system", type=str, default=None)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--n_ctx", type=int, default=4096)
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list_models:
        for n in list_runtime_models():
            print(n)
        return
    if not args.model_name:
        print(json.dumps({"ok": False, "error": "--model_name required"}))
        sys.exit(2)
    if args.query == "-":
        query = sys.stdin.read()
    elif args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read()
    else:
        print(json.dumps({"ok": False, "error": "--query required (or pipe stdin)"}))
        sys.exit(2)
    query = (query or "").strip()
    if not query:
        print(json.dumps({"ok": False, "error": "empty query"}))
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
        verbose=args.verbose,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.ngl,
        port=args.port,
    )
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
