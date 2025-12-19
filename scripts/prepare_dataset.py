"""
Prepare a sharded, pre-tokenized dataset for large-scale training.

This script:
- Streams text from multiple input files (low memory usage)
- Cleans and normalizes text
- Tokenizes incrementally
- Writes binary shards (e.g., 10M tokens per shard)
- Splits into train/val directories

Usage:
    # Basic usage
    python scripts/prepare_dataset.py --input_files data/*.txt --out_dir data/my_dataset
    
    # With multiple sources
    python scripts/prepare_dataset.py \
        --input_files wiki_en.txt gutenberg.txt europarl.txt \
        --out_dir data/large_corpus \
        --tokenization gpt2 \
        --tokens_per_shard 10000000 \
        --val_fraction 0.1
    
    # Character-level tokenization
    python scripts/prepare_dataset.py \
        --input_files input.txt \
        --out_dir data/char_dataset \
        --tokenization char
"""

import argparse
import os
import hashlib
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import GPTConfig, Tokenizer


def normalize(text: str) -> str:
    """Normalize text: lowercase and collapse whitespace."""
    text = text.lower()
    text = " ".join(text.split())  # collapse whitespace
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


def stream_clean_lines(paths, normalize_text=True, filter_lines=True):
    """
    Yield cleaned lines from a list of text files.
    
    Args:
        paths: List of file paths to read
        normalize_text: Whether to normalize (lowercase, collapse whitespace)
        filter_lines: Whether to filter out bad lines
    """
    total_lines = 0
    kept_lines = 0
    
    for path in paths:
        print(f"Reading {path}...")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    total_lines += 1
                    
                    if normalize_text:
                        line = normalize(raw)
                    else:
                        line = raw.strip()
                    
                    if filter_lines:
                        if not good_line(line):
                            continue
                    
                    kept_lines += 1
                    yield line
                    
                    # Progress update
                    if total_lines % 100000 == 0:
                        print(f"  Processed {total_lines:,} lines, kept {kept_lines:,}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    
    print(f"Total: processed {total_lines:,} lines, kept {kept_lines:,} lines")


def dedup_lines(lines, verbose=True):
    """
    Exact deduplication by MD5 hash.
    
    Note: For very large corpora (100M+ lines), you might want to:
    - Skip this step (dedup can use a lot of RAM)
    - Use bloom filters
    - Shard the dedup process
    """
    seen = set()
    kept = 0
    dropped = 0
    
    for l in lines:
        h = hashlib.md5(l.encode("utf-8")).hexdigest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        kept += 1
        yield l
        
        if verbose and (kept + dropped) % 100000 == 0:
            dup_rate = dropped / (kept + dropped) * 100 if (kept + dropped) > 0 else 0
            print(f"  Dedup: {kept:,} kept, {dropped:,} dropped ({dup_rate:.1f}% duplicates)")
    
    if verbose:
        total = kept + dropped
        dup_rate = dropped / total * 100 if total > 0 else 0
        print(f"Deduplication complete: {kept:,} kept, {dropped:,} dropped ({dup_rate:.1f}% duplicates)")


def main():
    from core.banner import print_banner
    print_banner("MyPT Dataset", "Sharded Dataset Preparer")
    
    parser = argparse.ArgumentParser(
        description="Prepare sharded dataset for large-scale training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/prepare_dataset.py --input_files data.txt --out_dir data/my_dataset
  
  # Multiple sources
  python scripts/prepare_dataset.py \\
      --input_files wiki.txt books.txt news.txt \\
      --out_dir data/large_corpus \\
      --tokens_per_shard 10000000
  
  # Character-level
  python scripts/prepare_dataset.py \\
      --input_files input.txt \\
      --out_dir data/char_dataset \\
      --tokenization char
        """
    )
    
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="List of text files to use for dataset",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for .bin shards and metadata",
    )
    parser.add_argument(
        "--tokenization",
        type=str,
        default="gpt2",
        choices=["gpt2", "char"],
        help="Tokenizer type (default: gpt2)",
    )
    parser.add_argument(
        "--tokens_per_shard",
        type=int,
        default=10_000_000,
        help="Number of tokens per .bin shard (default: 10M)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of shards for validation (default: 0.1)",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip text normalization (keep original case and whitespace)",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip line filtering (keep all lines)",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Skip deduplication (faster but may have duplicates)",
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print("=" * 70)
    print("MyPT Dataset Preparation")
    print("=" * 70)
    print(f"Input files: {len(args.input_files)}")
    for f in args.input_files:
        print(f"  - {f}")
    print(f"Output directory: {args.out_dir}")
    print(f"Tokenization: {args.tokenization}")
    print(f"Tokens per shard: {args.tokens_per_shard:,}")
    print(f"Validation fraction: {args.val_fraction}")
    print(f"Normalize text: {not args.no_normalize}")
    print(f"Filter lines: {not args.no_filter}")
    print(f"Deduplicate: {not args.no_dedup}")
    print("=" * 70)
    print()
    
    # Build tokenizer
    print("Initializing tokenizer...")
    cfg = GPTConfig()
    tokenizer = Tokenizer(cfg, args.tokenization)
    
    # For char-level, build vocab from sample
    if args.tokenization == "char":
        print("Building character vocabulary from input...")
        sample_chars = []
        for i, line in enumerate(stream_clean_lines(
            args.input_files,
            normalize_text=not args.no_normalize,
            filter_lines=not args.no_filter
        )):
            sample_chars.append(line)
            if i >= 100_000:
                break
        
        vocab_text = "\n".join(sample_chars)
        tokenizer.build_char_vocab(vocab_text)
        cfg.vocab_size = len(tokenizer.chars)
        print(f"Character vocabulary size: {cfg.vocab_size}")
    else:
        # GPT-2 BPE vocab is fixed
        cfg.vocab_size = 50304
        print(f"GPT-2 vocabulary size: {cfg.vocab_size}")
    
    # Save tokenizer state
    print("Saving tokenizer state...")
    import json
    try:
        tok_state = tokenizer.get_state()
        tok_state_path = os.path.join(args.out_dir, "tokenizer_state.json")
        with open(tok_state_path, "w", encoding="utf-8") as f:
            json.dump(tok_state, f, indent=2)
        print(f"Saved tokenizer state to {tok_state_path}")
    except Exception as e:
        print(f"Warning: Could not save tokenizer state: {e}")
    
    print()
    print("=" * 70)
    print("Processing text and creating shards...")
    print("=" * 70)
    print()
    
    # Stream, clean, optionally dedup, tokenize, and shard
    shard_tokens = []
    shard_idx = 0
    total_tokens = 0
    all_shard_paths = []
    
    # Build pipeline
    lines = stream_clean_lines(
        args.input_files,
        normalize_text=not args.no_normalize,
        filter_lines=not args.no_filter
    )
    
    if not args.no_dedup:
        lines = dedup_lines(lines)
    
    # Process lines
    for line in lines:
        # Add newline to preserve sentence boundaries
        ids = tokenizer.encode(line + "\n")
        
        # Fill shards up to tokens_per_shard
        pos = 0
        while pos < len(ids):
            remaining = args.tokens_per_shard - len(shard_tokens)
            take = min(remaining, len(ids) - pos)
            shard_tokens.extend(ids[pos : pos + take])
            pos += take
            
            # Write shard when full
            if len(shard_tokens) >= args.tokens_per_shard:
                arr = np.array(shard_tokens, dtype=np.uint32)
                shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.bin")
                arr.tofile(shard_path)
                all_shard_paths.append(shard_path)
                
                total_tokens += len(shard_tokens)
                size_mb = arr.nbytes / (1024 ** 2)
                print(
                    f"Shard {shard_idx:05d}: {len(shard_tokens):,} tokens "
                    f"({size_mb:.1f} MB) | Total: {total_tokens:,} tokens"
                )
                shard_tokens = []
                shard_idx += 1
    
    # Flush final partial shard
    if shard_tokens:
        arr = np.array(shard_tokens, dtype=np.uint32)
        shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(shard_path)
        all_shard_paths.append(shard_path)
        total_tokens += len(shard_tokens)
        size_mb = arr.nbytes / (1024 ** 2)
        print(
            f"Shard {shard_idx:05d}: {len(shard_tokens):,} tokens "
            f"({size_mb:.1f} MB) | Total: {total_tokens:,} tokens (final)"
        )
    
    print()
    print("=" * 70)
    print("Splitting into train/val...")
    print("=" * 70)
    
    # Train/val split at shard level
    num_val = max(1, int(len(all_shard_paths) * args.val_fraction))
    val_shards = all_shard_paths[-num_val:]
    train_shards = all_shard_paths[:-num_val]
    
    # Move shards into train/val subdirectories
    import shutil
    for p in train_shards:
        fname = os.path.basename(p)
        shutil.move(p, os.path.join(train_dir, fname))
    for p in val_shards:
        fname = os.path.basename(p)
        shutil.move(p, os.path.join(val_dir, fname))
    
    # Save dataset metadata
    metadata = {
        "total_tokens": total_tokens,
        "total_shards": len(all_shard_paths),
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "tokens_per_shard": args.tokens_per_shard,
        "tokenization": args.tokenization,
        "vocab_size": cfg.vocab_size,
        "input_files": args.input_files,
    }
    
    metadata_path = os.path.join(args.out_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total shards: {len(all_shard_paths)}")
    print(f"Train shards: {len(train_shards)} in {train_dir}")
    print(f"Val shards: {len(val_shards)} in {val_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print()
    print("To train with this dataset:")
    print(f"  python train.py --dataset_dir {args.out_dir} --model_name my_model")
    print("=" * 70)


if __name__ == "__main__":
    main()

