#!/usr/bin/env python
"""
Prepare a sharded, *weighted* dataset from multiple sources.

Features:
- Multiple named sources (e.g. wiki_en, wiki_de, opensub, europarl, news)
- Per-source target weights (e.g. 0.3, 0.3, 0.25, 0.10, 0.05)
- Global token budget (e.g. 4e8 tokens)
- Two-pass processing:
    1) Count tokens per source
    2) Take tokens from each source up to target, write .bin shards
- Supports GPT-2 BPE or char-level Tokenizer
- Optional TSV splitting for parallel corpora
- Optional REPEATING of small sources to meet weight targets (--repeat)

Usage example:

python scripts/prepare_weighted_dataset.py \
  --source wiki_en:data/wikipedia_en/*.txt \
  --source wiki_de:data/wikipedia_de/*.txt \
  --source opensub:data/opensubtitles_de_en/*.txt \
  --source europarl:data/europarl_de_en/*.tsv \
  --source news:data/news_commentary_de_en/*.tsv \
  --weight wiki_en:0.30 \
  --weight wiki_de:0.30 \
  --weight opensub:0.25 \
  --weight europarl:0.10 \
  --weight news:0.05 \
  --total_tokens 450000000 \
  --tokenization gpt2 \
  --tokens_per_shard 10000000 \
  --val_fraction 0.05 \
  --split_tsv \
  --repeat \
  --out_dir data/base_phase1_weighted
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import GPTConfig, Tokenizer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


@dataclass
class SourceSpec:
    name: str
    patterns: List[str]
    files: List[str]


def parse_sources(source_args: List[str]) -> Dict[str, SourceSpec]:
    """
    Parse --source arguments of the form:
        NAME:path1,path2,glob3
    and expand globs into file lists.
    """
    sources: Dict[str, SourceSpec] = {}
    for arg in source_args:
        if ":" not in arg:
            raise ValueError(
                f"Invalid --source '{arg}'. Expected format NAME:path1,path2,glob3"
            )
        name, paths_str = arg.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Source name is empty in '{arg}'")

        patterns = [p.strip() for p in paths_str.split(",") if p.strip()]
        files: List[str] = []
        for pat in patterns:
            matched = glob.glob(pat)
            if matched:
                files.extend(sorted(matched))
            elif os.path.isfile(pat):
                files.append(pat)
            else:
                print(f"[WARN] Pattern/file '{pat}' for source '{name}' matched nothing.")

        if not files:
            print(f"[WARN] Source '{name}' has no files after expansion.")

        sources[name] = SourceSpec(name=name, patterns=patterns, files=files)

    if not sources:
        raise ValueError("No sources parsed. Provide at least one --source.")

    return sources


def parse_weights(weight_args: List[str], sources: Dict[str, SourceSpec]) -> Dict[str, float]:
    """
    Parse --weight arguments of the form:
        NAME:0.3
    and normalize them to sum to 1.
    """
    weights: Dict[str, float] = {}
    for arg in weight_args:
        if ":" not in arg:
            raise ValueError(
                f"Invalid --weight '{arg}'. Expected format NAME:weight"
            )
        name, w_str = arg.split(":", 1)
        name = name.strip()
        if name not in sources:
            raise ValueError(
                f"Weight given for unknown source '{name}'. "
                f"Known sources: {list(sources.keys())}"
            )
        w = float(w_str)
        if w <= 0:
            raise ValueError(f"Weight for source '{name}' must be > 0, got {w}.")
        weights[name] = w

    # Any source without explicit weight: default to 1
    for name in sources.keys():
        if name not in weights:
            weights[name] = 1.0
            print(f"[INFO] No weight provided for '{name}', defaulting to 1.0")

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Sum of weights must be > 0.")
    for name in weights:
        weights[name] /= total

    print("Normalized source weights:")
    for name, w in weights.items():
        print(f"  - {name}: {w:.4f}")
    return weights


def stream_lines_for_source(
    source: SourceSpec,
    normalize_text: bool,
    filter_lines: bool,
    split_tsv: bool,
) -> Iterator[str]:
    """
    Yield cleaned lines for a single source.

    If split_tsv=True and a line contains tabs, each field is treated as a separate segment.
    """
    total_raw = 0
    yielded = 0
    for path in source.files:
        print(f"[{source.name}] Reading {path}...")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    total_raw += 1

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

                        yielded += 1
                        yield line

                        if yielded % 200000 == 0:
                            print(
                                f"[{source.name}] ... yielded {yielded:,} segments "
                                f"(from {total_raw:,} raw lines)"
                            )
        except Exception as e:
            print(f"[{source.name}] Error reading {path}: {e}")
            continue

    print(f"[{source.name}] Total raw lines: {total_raw:,}, yielded segments: {yielded:,}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_tokenizer(tokenization: str, sources: Dict[str, SourceSpec],
                    normalize_text: bool, filter_lines: bool, split_tsv: bool) -> Tuple[GPTConfig, Tokenizer]:
    """
    Build GPTConfig + Tokenizer.
    For char-level, builds vocab from a sample of the data.
    """
    print("Initializing tokenizer...")
    cfg = GPTConfig()
    tokenizer = Tokenizer(cfg, tokenization)

    if tokenization == "char":
        print("Building character vocabulary from sample of all sources...")
        sample_segments: List[str] = []
        max_segments = 100_000

        for source in sources.values():
            for line in stream_lines_for_source(
                source,
                normalize_text=normalize_text,
                filter_lines=filter_lines,
                split_tsv=split_tsv,
            ):
                sample_segments.append(line)
                if len(sample_segments) >= max_segments:
                    break
            if len(sample_segments) >= max_segments:
                break

        vocab_text = "\n".join(sample_segments)
        tokenizer.build_char_vocab(vocab_text)
        cfg.vocab_size = len(tokenizer.chars)
        print(f"Character vocabulary size: {cfg.vocab_size}")
    else:
        # GPT-2 BPE vocab is fixed in your implementation
        cfg.vocab_size = 50304
        print(f"GPT-2 vocabulary size: {cfg.vocab_size}")

    return cfg, tokenizer


def pass1_count_tokens(
    sources: Dict[str, SourceSpec],
    tokenizer: Tokenizer,
    tokenization: str,
    normalize_text: bool,
    filter_lines: bool,
    split_tsv: bool,
) -> Dict[str, int]:
    """
    First pass: count how many tokens are available per source.
    """
    print("\n" + "=" * 70)
    print("PASS 1: Counting tokens per source")
    print("=" * 70)

    token_counts: Dict[str, int] = {name: 0 for name in sources.keys()}

    total_tokens_global = 0
    for name, source in sources.items():
        count = 0
        for line in stream_lines_for_source(
            source,
            normalize_text=normalize_text,
            filter_lines=filter_lines,
            split_tsv=split_tsv,
        ):
            ids = tokenizer.encode(line + "\n")
            n = len(ids)
            count += n
            total_tokens_global += n

        token_counts[name] = count
        print(f"[PASS1] Source '{name}' total tokens: {count:,}")

    print(f"[PASS1] Global total tokens across all sources: {total_tokens_global:,}")
    return token_counts


def compute_target_tokens(
    token_counts: Dict[str, int],
    weights: Dict[str, float],
    total_tokens_target: int,
    allow_repeat: bool = False,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given available tokens per source, weights and a global target, compute
    how many tokens we will actually take from each source.
    
    Returns:
        (final_targets, repeat_counts) - target tokens per source and how many
        times to repeat each source (1 = no repeat, 2 = repeat once, etc.)
    """
    print("\n" + "=" * 70)
    print("Computing target tokens per source")
    print("=" * 70)

    names = list(token_counts.keys())

    # Ideal allocation based on weights
    ideal = {name: int(round(weights[name] * total_tokens_target)) for name in names}

    if allow_repeat:
        # With repeat enabled, we can meet ideal targets by repeating small sources
        print("[REPEAT MODE] Small sources will be repeated to meet weight targets.")
        
        final_targets = {}
        repeat_counts = {}
        
        for name in names:
            avail = token_counts[name]
            tgt = ideal[name]
            
            if avail <= 0:
                final_targets[name] = 0
                repeat_counts[name] = 0
                print(f"[TARGET] {name}: SKIPPED (no data)")
                continue
            
            if avail >= tgt:
                # Enough data, no repeat needed
                final_targets[name] = tgt
                repeat_counts[name] = 1
                print(
                    f"[TARGET] {name}: target={tgt:,} tokens, "
                    f"available={avail:,}, repeat=1x"
                )
            else:
                # Need to repeat - calculate how many times
                repeats_needed = (tgt + avail - 1) // avail  # ceiling division
                final_targets[name] = tgt
                repeat_counts[name] = repeats_needed
                print(
                    f"[TARGET] {name}: target={tgt:,} tokens, "
                    f"available={avail:,}, repeat={repeats_needed}x "
                    f"(upscaled from {avail:,} → ~{avail * repeats_needed:,})"
                )
        
        print(f"\n[INFO] Final target token sum: {sum(final_targets.values()):,}")
        
    else:
        # Original behavior: clip by availability
        clipped = {}
        for name in names:
            avail = token_counts[name]
            tgt = ideal[name]
            clipped[name] = min(avail, tgt)
            print(
                f"[TARGET] {name}: ideal={tgt:,} tokens, "
                f"available={avail:,}, clipped={clipped[name]:,}"
            )

        sum_clipped = sum(clipped.values())
        if sum_clipped == 0:
            raise ValueError("All sources have zero tokens after clipping. Check inputs.")

        if sum_clipped < total_tokens_target:
            print(
                f"[WARN] Cannot reach requested total_tokens={total_tokens_target:,}. "
                f"Only {sum_clipped:,} tokens available across all sources."
            )
            print(
                f"[HINT] Use --repeat to automatically repeat small sources to meet targets."
            )
            # Just use what we have
            final_targets = clipped
        else:
            # Re-normalize clipped counts down to exactly total_tokens_target (approximately)
            scale = total_tokens_target / sum_clipped
            final_targets = {}
            running_sum = 0
            print(f"[INFO] Scaling clipped counts by {scale:.6f} to match total_tokens_target.")

            # Distribute with rounding; last source gets the remainder
            for i, name in enumerate(names):
                if i < len(names) - 1:
                    v = int(round(clipped[name] * scale))
                    v = min(v, clipped[name])
                    final_targets[name] = v
                    running_sum += v
                else:
                    # last source: ensure exact total
                    v = total_tokens_target - running_sum
                    v = min(v, clipped[name])
                    final_targets[name] = v
                    running_sum += v

            print(f"[INFO] Final target token sum: {running_sum:,}")
        
        # No repeating
        repeat_counts = {name: 1 for name in names}

    print("\nFinal per-source targets:")
    for name in names:
        repeat_info = f", repeat={repeat_counts[name]}x" if repeat_counts[name] > 1 else ""
        print(
            f"  - {name}: target={final_targets[name]:,} / available={token_counts[name]:,} "
            f"(weight ~{weights[name]:.4f}{repeat_info})"
        )

    return final_targets, repeat_counts


def pass2_write_shards(
    sources: Dict[str, SourceSpec],
    tokenizer: Tokenizer,
    tokenization: str,
    normalize_text: bool,
    filter_lines: bool,
    split_tsv: bool,
    targets: Dict[str, int],
    repeat_counts: Dict[str, int],
    out_dir: str,
    tokens_per_shard: int,
    val_fraction: float,
) -> None:
    """
    Second pass: take up to target tokens per source, write into .bin shards,
    then split shards into train/val.
    
    Supports repeating sources multiple times if repeat_counts[name] > 1.
    """
    print("\n" + "=" * 70)
    print("PASS 2: Sampling tokens and writing shards")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    shard_tokens: List[int] = []
    shard_idx = 0
    total_tokens_written = 0
    all_shard_paths: List[str] = []

    def flush_shard():
        """Write current shard buffer to disk."""
        nonlocal shard_tokens, shard_idx, total_tokens_written
        if not shard_tokens:
            return
        arr = np.array(shard_tokens, dtype=np.uint32)
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(shard_path)
        all_shard_paths.append(shard_path)

        total_tokens_written += len(shard_tokens)
        size_mb = arr.nbytes / (1024 ** 2)
        print(
            f"[SHARD] {shard_idx:05d}: {len(shard_tokens):,} tokens "
            f"({size_mb:.1f} MB) | Total written: {total_tokens_written:,}"
        )

        shard_tokens = []
        shard_idx += 1

    def add_tokens(ids: List[int]):
        """Add tokens to the current shard, flushing when full."""
        nonlocal shard_tokens
        pos = 0
        while pos < len(ids):
            space_left = tokens_per_shard - len(shard_tokens)
            take = min(space_left, len(ids) - pos)
            shard_tokens.extend(ids[pos : pos + take])
            pos += take

            if len(shard_tokens) >= tokens_per_shard:
                flush_shard()

    # We'll process sources in a fixed order; for more mixing you can shuffle source order
    for name, source in sources.items():
        remaining = targets.get(name, 0)
        if remaining <= 0:
            print(f"[PASS2] Skipping source '{name}' (target 0 tokens).")
            continue

        max_repeats = repeat_counts.get(name, 1)
        repeat_info = f" (max {max_repeats}x repeats)" if max_repeats > 1 else ""
        print(f"[PASS2] Processing source '{name}' with target {remaining:,} tokens{repeat_info}...")

        current_repeat = 0
        while remaining > 0 and current_repeat < max_repeats:
            current_repeat += 1
            if max_repeats > 1:
                print(f"[PASS2] '{name}' - repeat cycle {current_repeat}/{max_repeats}...")
            
            for line in stream_lines_for_source(
                source,
                normalize_text=normalize_text,
                filter_lines=filter_lines,
                split_tsv=split_tsv,
            ):
                if remaining <= 0:
                    break

                ids = tokenizer.encode(line + "\n")
                if len(ids) <= remaining:
                    to_take = ids
                else:
                    to_take = ids[:remaining]

                remaining -= len(to_take)
                add_tokens(to_take)

                if remaining <= 0:
                    print(f"[PASS2] Reached target for '{name}' after {current_repeat} cycle(s).")
                    break

        if remaining > 0:
            print(
                f"[WARN] Source '{name}' ended with {remaining:,} tokens still "
                f"desired after {current_repeat} cycle(s). "
                f"You may not have enough data or strong filtering removed many lines."
            )

    # flush final shard
    if shard_tokens:
        arr = np.array(shard_tokens, dtype=np.uint32)
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(shard_path)
        all_shard_paths.append(shard_path)

        total_tokens_written += len(shard_tokens)
        size_mb = arr.nbytes / (1024 ** 2)
        print(
            f"[SHARD] {shard_idx:05d}: {len(shard_tokens):,} tokens "
            f"({size_mb:.1f} MB) | Total written: {total_tokens_written:,} (final)"
        )

    # Train/val split at shard level
    print("\n" + "=" * 70)
    print("Splitting shards into train/val...")
    print("=" * 70)

    num_shards = len(all_shard_paths)
    if num_shards == 0:
        raise RuntimeError("No shards were written. Check your targets and input data.")

    num_val = max(1, int(round(num_shards * val_fraction)))
    num_val = min(num_val, num_shards - 1)  # keep at least 1 train shard
    val_shards = all_shard_paths[-num_val:]
    train_shards = all_shard_paths[:-num_val]

    import shutil
    for p in train_shards:
        fname = os.path.basename(p)
        shutil.move(p, os.path.join(train_dir, fname))
    for p in val_shards:
        fname = os.path.basename(p)
        shutil.move(p, os.path.join(val_dir, fname))

    # Save metadata
    metadata = {
        "total_tokens_written": total_tokens_written,
        "total_tokens": total_tokens_written,
        "total_shards": num_shards,
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "tokens_per_shard": tokens_per_shard,
        "val_fraction": val_fraction,
        "sources": {name: [f for f in src.files] for name, src in sources.items()},
        "targets": targets,
        "repeat_counts": repeat_counts,
    }
    meta_path = os.path.join(out_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SUMMARY] Total tokens written: {total_tokens_written:,}")
    print(f"[SUMMARY] Total shards: {num_shards}")
    print(f"[SUMMARY] Train shards: {len(train_shards)} in {train_dir}")
    print(f"[SUMMARY] Val shards:   {len(val_shards)} in {val_dir}")
    print(f"[SUMMARY] Metadata saved to: {meta_path}")
    print("\nTo train with this dataset:")
    print(f"  python train.py --dataset_dir {out_dir} --model_name my_model")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a weighted multi-source dataset for MyPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source spec of the form NAME:path1,path2,glob3. Can be repeated.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        help="Source weight of the form NAME:weight. "
             "Sources without explicit weights default to 1.0.",
    )
    parser.add_argument(
        "--total_tokens",
        type=int,
        required=True,
        help="Total number of tokens to aim for across all sources.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for shards and metadata.",
    )
    parser.add_argument(
        "--tokenization",
        type=str,
        default="gpt2",
        choices=["gpt2", "char"],
        help="Tokenizer type (default: gpt2).",
    )
    parser.add_argument(
        "--tokens_per_shard",
        type=int,
        default=10_000_000,
        help="Number of tokens per .bin shard (default: 10M).",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of shards for validation (default: 0.1).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip text normalization (keep original case and whitespace).",
    )
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip line filtering (keep all lines).",
    )
    parser.add_argument(
        "--split_tsv",
        action="store_true",
        help="If set, split TSV lines (with '\\t') into separate segments.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Repeat small sources to meet their weight targets (upscaling).",
    )
    parser.add_argument(
        "--max_repeat",
        type=int,
        default=100,
        help="Maximum times a source can be repeated (default: 100, prevents infinite loops).",
    )

    args = parser.parse_args()

    sources = parse_sources(args.source)
    weights = parse_weights(args.weight, sources)

    normalize_text = not args.no_normalize
    filter_lines = not args.no_filter

    from core.banner import print_banner
    print_banner("MyPT Weighted Dataset", "Sharded Multi-Source Data Preparation")
    print(f"Sources: {len(sources)}")
    for name, spec in sources.items():
        print(f"  - {name}: {len(spec.files)} files")
    print(f"Total requested tokens: {args.total_tokens:,}")
    print(f"Tokenization: {args.tokenization}")
    print(f"Tokens per shard: {args.tokens_per_shard:,}")
    print(f"Validation fraction: {args.val_fraction}")
    print(f"Normalize text: {normalize_text}")
    print(f"Filter lines: {filter_lines}")
    print(f"Split TSV lines: {args.split_tsv}")
    print(f"Repeat small sources: {args.repeat}" + (f" (max {args.max_repeat}x)" if args.repeat else ""))
    print(f"Output directory: {args.out_dir}")
    print("=" * 70)

    # Build tokenizer
    cfg, tokenizer = build_tokenizer(
        tokenization=args.tokenization,
        sources=sources,
        normalize_text=normalize_text,
        filter_lines=filter_lines,
        split_tsv=args.split_tsv,
    )

    # Pass 1: count tokens
    token_counts = pass1_count_tokens(
        sources=sources,
        tokenizer=tokenizer,
        tokenization=args.tokenization,
        normalize_text=normalize_text,
        filter_lines=filter_lines,
        split_tsv=args.split_tsv,
    )

    # Compute targets per source
    targets, repeat_counts = compute_target_tokens(
        token_counts=token_counts,
        weights=weights,
        total_tokens_target=args.total_tokens,
        allow_repeat=args.repeat,
    )
    
    # Cap repeat counts by max_repeat
    if args.repeat:
        for name in repeat_counts:
            if repeat_counts[name] > args.max_repeat:
                print(f"[WARN] Capping {name} repeats from {repeat_counts[name]} to {args.max_repeat}")
                repeat_counts[name] = args.max_repeat

    # Pass 2: write shards
    pass2_write_shards(
        sources=sources,
        tokenizer=tokenizer,
        tokenization=args.tokenization,
        normalize_text=normalize_text,
        filter_lines=filter_lines,
        split_tsv=args.split_tsv,
        targets=targets,
        repeat_counts=repeat_counts,
        out_dir=args.out_dir,
        tokens_per_shard=args.tokens_per_shard,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
