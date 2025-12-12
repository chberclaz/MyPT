#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive RAG chat CLI.

Loads a trained model and retriever, then provides an interactive
chat interface with retrieval-augmented generation.

Usage:
    python scripts/rag_chat.py --model_name my_model --index_dir workspace/index/latest
    
    # With custom settings
    python scripts/rag_chat.py --model_name my_model --index_dir workspace/index/latest \
        --top_k 3 --max_tokens 512 --system "You are a helpful assistant."

Commands during chat:
    /reload     - Reload the index (after adding new documents)
    /sources    - Show sources from last answer
    /system X   - Set system prompt to X
    /topk N     - Set top_k to N
    /quit       - Exit chat
"""

import argparse
import os
import sys

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import load_model
from core.rag import Retriever, RAGPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive RAG chat with your trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of trained model in checkpoints/")
    parser.add_argument("--index_dir", type=str, required=True,
                        help="Path to RAG index directory")
    
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate (default: 256)")
    parser.add_argument("--system", type=str, default="",
                        help="System prompt (optional)")
    parser.add_argument("--show_sources", action="store_true",
                        help="Always show sources with answers")
    parser.add_argument("--min_score", type=float, default=0.0,
                        help="Minimum similarity score for retrieval (0-1)")
    
    return parser.parse_args()


def print_sources(sources: list):
    """Print source information nicely."""
    if not sources:
        print("  (No sources retrieved)")
        return
    
    print("\n📚 Sources:")
    for i, src in enumerate(sources, 1):
        score_bar = "█" * int(src['score'] * 10) + "░" * (10 - int(src['score'] * 10))
        print(f"  [{i}] {src['filename']}")
        print(f"      Score: {score_bar} {src['score']:.2f}")
        print(f"      Preview: {src['text_preview'][:100]}...")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🤖 MyPT RAG Chat")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {args.model_name}...")
    try:
        model = load_model(args.model_name)
        print(f"  ✅ Model loaded")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Load retriever
    print(f"Loading index: {args.index_dir}...")
    try:
        retriever = Retriever(args.index_dir)
        print(f"  ✅ Index loaded ({retriever.num_chunks} chunks)")
    except Exception as e:
        print(f"  ❌ Failed to load index: {e}")
        sys.exit(1)
    
    # Create pipeline
    pipeline = RAGPipeline(
        model=model,
        retriever=retriever,
        default_system=args.system,
        default_top_k=args.top_k,
    )
    
    # State
    current_system = args.system
    current_top_k = args.top_k
    last_sources = []
    show_sources = args.show_sources
    
    print("\n" + "-" * 60)
    print("Ready! Type your question or /help for commands.")
    print("-" * 60 + "\n")
    
    while True:
        try:
            user_input = input("user> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
            
            if cmd in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            
            elif cmd == "reload":
                print("Reloading index...")
                try:
                    pipeline.reload_index()
                    print(f"  ✅ Reloaded ({retriever.num_chunks} chunks)")
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
            
            elif cmd == "sources":
                print_sources(last_sources)
            
            elif cmd == "system":
                current_system = cmd_arg
                pipeline.default_system = current_system
                print(f"System prompt set to: {current_system or '(empty)'}")
            
            elif cmd == "topk":
                try:
                    current_top_k = int(cmd_arg)
                    pipeline.default_top_k = current_top_k
                    print(f"top_k set to: {current_top_k}")
                except ValueError:
                    print(f"Invalid number: {cmd_arg}")
            
            elif cmd == "show_sources":
                show_sources = not show_sources
                print(f"Show sources: {'ON' if show_sources else 'OFF'}")
            
            elif cmd == "help":
                print("""
Commands:
  /reload       - Reload the index after adding documents
  /sources      - Show sources from the last answer
  /system TEXT  - Set system prompt
  /topk N       - Set number of chunks to retrieve
  /show_sources - Toggle always showing sources
  /quit         - Exit chat
                """)
            
            else:
                print(f"Unknown command: /{cmd}")
            
            continue
        
        # Regular question - get RAG answer
        try:
            result = pipeline.answer_with_sources(
                question=user_input,
                top_k=current_top_k,
                max_new_tokens=args.max_tokens,
            )
            
            last_sources = result['sources']
            
            print(f"\nassistant> {result['answer']}")
            
            if show_sources:
                print_sources(last_sources)
            else:
                print(f"\n  ({result['num_sources']} sources used, /sources to view)")
            
            print()
        
        except Exception as e:
            print(f"\n❌ Error generating response: {e}\n")


if __name__ == "__main__":
    main()

