#!/usr/bin/env python
"""
Append additional data to an existing sharded dataset.

This script adds new source data to an already prepared dataset,
creating additional shards and updating metadata.

Usage:
    python scripts/append_to_dataset.py \
        --dataset_dir data/multilingual_1.5B_wiki90 \
        --source opensub:data/raw/opensubtitles_de_en/opensubtitles_de_en.tsv \
        --target_tokens 75000000 \
        --split_tsv

This will:
1. Load existing dataset metadata
2. Tokenize and write new shards for the additional source
3. Distribute new shards to train/val based on existing ratio
4. Update metadata with combined totals
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import List, Iterator

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import GPTConfig, Tokenizer


def normalize(text: str) -> str:
    """Normalize text: lowercase and collapse whitespace."""
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def good_line(line: str, min_len: int = 20, max_len: int = 2000, min_words: int = 3) -> bool:
    """Filter out lines that are too short, too long, or have too few words."""
    if len(line) < min_len:
        return False
    if len(line) > max_len:
        return False
    if line.count(" ") < min_words:
        return False
    return True


def stream_lines(
    file_paths: List[str],
    normalize_text: bool = True,
    filter_lines: bool = True,
    split_tsv: bool = False,
) -> Iterator[str]:
    """Yield cleaned lines from source files."""
    for path in file_paths:
        print(f"[INFO] Reading {path}...")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    if split_tsv and "\t" in raw:
                        fields = [p.strip() for p in raw.split("\t") if p.strip()]
                    else:
                        fields = [raw.strip()]
                    
                    for field in fields:
                        if not field:
                            continue
                        
                        if normalize_text:
                            line = normalize(field)
                        else:
                            line = field.strip()
                        
                        if filter_lines and not good_line(line):
                            continue
                        
                        yield line
        except Exception as e:
            print(f"[ERROR] Reading {path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Append additional data to an existing sharded dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Existing dataset directory with train/val subdirs and metadata"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source spec: NAME:path1,path2 (e.g., opensub:data/opensub.tsv)"
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        required=True,
        help="Number of tokens to add from this source"
    )
    parser.add_argument(
        "--tokenization",
        type=str,
        default="gpt2",
        choices=["gpt2", "char"],
        help="Tokenizer type (default: gpt2)"
    )
    parser.add_argument(
        "--tokens_per_shard",
        type=int,
        default=10_000_000,
        help="Tokens per shard (default: 10M)"
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of new shards for validation (default: 0.05)"
    )
    parser.add_argument(
        "--split_tsv",
        action="store_true",
        help="Split TSV lines into separate segments"
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip text normalization"
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip line filtering"
    )
    
    args = parser.parse_args()
    
    # Parse source
    if ":" not in args.source:
        print(f"[ERROR] Invalid source format. Use NAME:path1,path2")
        sys.exit(1)
    
    source_name, paths_str = args.source.split(":", 1)
    source_files = [p.strip() for p in paths_str.split(",") if p.strip()]
    
    # Verify files exist
    missing = [f for f in source_files if not os.path.exists(f)]
    if missing:
        print(f"[ERROR] Files not found: {missing}")
        sys.exit(1)
    
    # Check dataset directory
    train_dir = os.path.join(args.dataset_dir, "train")
    val_dir = os.path.join(args.dataset_dir, "val")
    meta_path = os.path.join(args.dataset_dir, "dataset_metadata.json")
    
    if not os.path.exists(train_dir):
        print(f"[ERROR] Train directory not found: {train_dir}")
        sys.exit(1)
    
    # Load existing metadata if present
    existing_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            existing_meta = json.load(f)
        print(f"[INFO] Loaded existing metadata: {existing_meta.get('total_tokens', 0):,} tokens")
    
    # Find next shard index
    existing_shards = sorted([f for f in os.listdir(train_dir) if f.endswith('.bin')])
    existing_shards += sorted([f for f in os.listdir(val_dir) if f.endswith('.bin')])
    
    if existing_shards:
        last_shard = max(existing_shards)
        # Extract shard number from name like shard_00045.bin
        last_idx = int(last_shard.replace('shard_', '').replace('.bin', ''))
        next_shard_idx = last_idx + 1
    else:
        next_shard_idx = 0
    
    print()
    print("=" * 60)
    print("  Append to Dataset")
    print("=" * 60)
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Source: {source_name}")
    print(f"  Files: {source_files}")
    print(f"  Target tokens: {args.target_tokens:,}")
    print(f"  Starting shard index: {next_shard_idx}")
    print()
    
    # Initialize tokenizer
    cfg = GPTConfig()
    tokenizer = Tokenizer(cfg, args.tokenization)
    if args.tokenization == "gpt2":
        cfg.vocab_size = 50304
    
    # Process and write shards
    shard_tokens: List[int] = []
    shard_idx = next_shard_idx
    total_tokens_written = 0
    remaining = args.target_tokens
    new_shard_paths: List[str] = []
    
    def flush_shard():
        nonlocal shard_tokens, shard_idx, total_tokens_written
        if not shard_tokens:
            return
        
        arr = np.array(shard_tokens, dtype=np.uint32)
        shard_path = os.path.join(args.dataset_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(shard_path)
        new_shard_paths.append(shard_path)
        
        total_tokens_written += len(shard_tokens)
        size_mb = arr.nbytes / (1024**2)
        print(f"[SHARD] {shard_idx:05d}: {len(shard_tokens):,} tokens ({size_mb:.1f} MB)")
        
        shard_tokens = []
        shard_idx += 1
    
    print(f"[INFO] Processing {source_name}...")
    
    for line in stream_lines(
        source_files,
        normalize_text=not args.no_normalize,
        filter_lines=not args.no_filter,
        split_tsv=args.split_tsv,
    ):
        if remaining <= 0:
            break
        
        ids = tokenizer.encode(line + "\n")
        if len(ids) <= remaining:
            to_take = ids
        else:
            to_take = ids[:remaining]
        
        remaining -= len(to_take)
        shard_tokens.extend(to_take)
        
        if len(shard_tokens) >= args.tokens_per_shard:
            flush_shard()
    
    # Flush final shard
    if shard_tokens:
        arr = np.array(shard_tokens, dtype=np.uint32)
        shard_path = os.path.join(args.dataset_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(shard_path)
        new_shard_paths.append(shard_path)
        total_tokens_written += len(shard_tokens)
        print(f"[SHARD] {shard_idx:05d}: {len(shard_tokens):,} tokens (final)")
    
    # Distribute new shards to train/val
    num_new_shards = len(new_shard_paths)
    num_val = max(1, int(round(num_new_shards * args.val_fraction)))
    num_val = min(num_val, num_new_shards - 1) if num_new_shards > 1 else 0
    
    new_val_shards = new_shard_paths[-num_val:] if num_val > 0 else []
    new_train_shards = new_shard_paths[:-num_val] if num_val > 0 else new_shard_paths
    
    for p in new_train_shards:
        fname = os.path.basename(p)
        shutil.move(p, os.path.join(train_dir, fname))
    for p in new_val_shards:
        fname = os.path.basename(p)
        shutil.move(p, os.path.join(val_dir, fname))
    
    # Update metadata
    old_total = existing_meta.get('total_tokens', 0)
    old_train_shards = existing_meta.get('train_shards', 0)
    old_val_shards = existing_meta.get('val_shards', 0)
    
    updated_meta = existing_meta.copy()
    updated_meta['total_tokens'] = old_total + total_tokens_written
    updated_meta['total_tokens_written'] = updated_meta['total_tokens']
    updated_meta['train_shards'] = old_train_shards + len(new_train_shards)
    updated_meta['val_shards'] = old_val_shards + len(new_val_shards)
    updated_meta['total_shards'] = updated_meta['train_shards'] + updated_meta['val_shards']
    
    # Track appended sources
    if 'appended_sources' not in updated_meta:
        updated_meta['appended_sources'] = {}
    updated_meta['appended_sources'][source_name] = {
        'tokens': total_tokens_written,
        'shards': num_new_shards,
        'files': source_files,
    }
    
    with open(meta_path, 'w') as f:
        json.dump(updated_meta, f, indent=2)
    
    # Ensure tokenizer_state.json exists (for train.py compatibility)
    tokenizer_state_path = os.path.join(args.dataset_dir, "tokenizer_state.json")
    if not os.path.exists(tokenizer_state_path):
        tokenizer_state = {
            "token_kind": args.tokenization,
            "chars": None if args.tokenization == "gpt2" else tokenizer.chars,
            "base_vocab_size": 50257 if args.tokenization == "gpt2" else len(tokenizer.chars),
        }
        with open(tokenizer_state_path, 'w') as f:
            json.dump(tokenizer_state, f, indent=2)
        print(f"  Created tokenizer_state.json ({args.tokenization})")
    
    print()
    print("=" * 60)
    print("  APPEND COMPLETE")
    print("=" * 60)
    print(f"  New tokens added: {total_tokens_written:,}")
    print(f"  New shards: {num_new_shards} (train: {len(new_train_shards)}, val: {len(new_val_shards)})")
    print()
    print(f"  Updated totals:")
    print(f"    Total tokens: {old_total:,} → {updated_meta['total_tokens']:,}")
    print(f"    Train shards: {old_train_shards} → {updated_meta['train_shards']}")
    print(f"    Val shards: {old_val_shards} → {updated_meta['val_shards']}")
    print()


if __name__ == "__main__":
    main()

