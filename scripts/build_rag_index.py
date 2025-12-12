#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build RAG embedding index from documents.

Walks a directory, chunks documents, embeds them, and saves to an index.

Usage:
    python scripts/build_rag_index.py --docs_dir workspace/docs --out_dir workspace/index/latest
    
    # Custom chunking
    python scripts/build_rag_index.py --docs_dir workspace/docs --out_dir workspace/index/v1 \
        --chunk_size 800 --chunk_overlap 100
    
    # With embedder settings
    python scripts/build_rag_index.py --docs_dir workspace/docs --out_dir workspace/index/v1 \
        --embed_dim 512

Output:
    out_dir/
        embeddings.npy      # (N, D) normalized vectors
        meta.jsonl          # one JSON per chunk with text + source
        config.json         # index configuration
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document import DocumentLoader, TextChunker
from core.embeddings import LocalEmbedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build RAG embedding index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_rag_index.py --docs_dir workspace/docs --out_dir workspace/index/latest
    python scripts/build_rag_index.py --docs_dir data/knowledge --out_dir workspace/index/knowledge --chunk_size 1500
        """
    )
    
    # Required
    parser.add_argument("--docs_dir", type=str, required=True,
                        help="Directory containing documents to index")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for index files")
    
    # Chunking options
    parser.add_argument("--chunk_size", type=int, default=1200,
                        help="Target chunk size in characters (default: 1200)")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between chunks (default: 200)")
    parser.add_argument("--min_chunk_size", type=int, default=50,
                        help="Minimum chunk size to keep (default: 50)")
    
    # Embedding options
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension (default: 256)")
    parser.add_argument("--ngram_min", type=int, default=2,
                        help="Minimum n-gram size (default: 2)")
    parser.add_argument("--ngram_max", type=int, default=5,
                        help="Maximum n-gram size (default: 5)")
    
    # Other options
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Search subdirectories (default: True)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false",
                        help="Don't search subdirectories")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("RAG Index Builder")
    print("=" * 60)
    
    # Validate input directory
    if not os.path.isdir(args.docs_dir):
        print(f"Error: Documents directory not found: {args.docs_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize components
    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
    )
    embedder = LocalEmbedder(
        dim=args.embed_dim,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    
    print(f"\nConfiguration:")
    print(f"  Documents: {args.docs_dir}")
    print(f"  Output: {args.out_dir}")
    print(f"  Chunk size: {args.chunk_size} chars (overlap: {args.chunk_overlap})")
    print(f"  Embedding dim: {args.embed_dim}")
    print()
    
    # Load documents
    print("Loading documents...")
    start_time = time.time()
    
    documents = loader.load_directory(args.docs_dir, recursive=args.recursive)
    
    if not documents:
        print(f"No supported documents found in {args.docs_dir}")
        print(f"Supported formats: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS)}")
        sys.exit(1)
    
    load_time = time.time() - start_time
    total_chars = sum(doc.num_chars for doc in documents)
    print(f"  Loaded {len(documents)} documents ({total_chars:,} chars) in {load_time:.2f}s")
    
    if args.verbose:
        for doc in documents:
            print(f"    - {doc.filename}: {doc.num_chars:,} chars")
    
    # Chunk documents
    print("\nChunking documents...")
    start_time = time.time()
    
    chunks = chunker.chunk_documents(documents)
    
    chunk_time = time.time() - start_time
    print(f"  Created {len(chunks)} chunks in {chunk_time:.2f}s")
    
    if args.verbose and len(chunks) <= 20:
        for chunk in chunks:
            src = chunk.source
            print(f"    [{chunk.chunk_id}] {src['filename']} (lines {src['start_line']}-{src['end_line']}): {chunk.num_chars} chars")
    
    # Embed chunks
    print("\nEmbedding chunks...")
    start_time = time.time()
    
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode_batch(texts)
    
    embed_time = time.time() - start_time
    chunks_per_sec = len(chunks) / embed_time if embed_time > 0 else 0
    print(f"  Embedded {len(chunks)} chunks in {embed_time:.2f}s ({chunks_per_sec:.0f} chunks/sec)")
    
    # Save index
    print("\nSaving index...")
    
    # 1. Save embeddings
    embeddings_path = os.path.join(args.out_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings.astype(np.float32))
    print(f"  Saved embeddings: {embeddings_path} (shape: {embeddings.shape})")
    
    # 2. Save metadata
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    with open(meta_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
    print(f"  Saved metadata: {meta_path} ({len(chunks)} entries)")
    
    # 3. Save configuration
    config = {
        "version": "1.0",
        "num_chunks": len(chunks),
        "num_documents": len(documents),
        "documents": [doc.filename for doc in documents],
        "chunker": chunker.get_config(),
        "embedder": embedder.get_config(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config: {config_path}")
    
    # Summary
    total_time = load_time + chunk_time + embed_time
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Index built successfully!")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Output: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

