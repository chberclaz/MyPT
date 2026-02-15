#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Pre-training Replay Buffer for SFT Phases.

Samples diverse text from pre-training sources and wraps them as
simple continuation episodes (no special tags, full loss_mask=1).
Mixing 1-5% of this data into each SFT phase prevents catastrophic
forgetting of base knowledge (math, facts, code, German).

Reference: "Scaling Laws for Forgetting during Finetuning with
Pretraining Data Injection" (openreview.net/forum?id=vWMij23BmQ)

The output JSONL has a special format flag so prepare_chat_sft.py
can emit full-loss (mask=1 everywhere) instead of the usual
assistant-only masking.

Output format:
    {"_replay": true, "text": "raw pre-training text passage..."}

Usage:
    # Generate 2000 replay episodes from pre-training shards
    python scripts/sft/generate_pretrain_replay.py \\
        --shard_dirs data/unified_tokenized/fineweb_edu \\
                     data/unified_tokenized/stackexchange_qa \\
                     data/unified_tokenized/code_python \\
                     data/multilingual_1.5B_wiki90 \\
        --output data/sft_replay/pretrain_replay.jsonl \\
        --num_episodes 2000 --max_tokens 256

    # Then mix into an SFT phase (5% replay):
    python scripts/sft/mix_sft_jsonl.py \\
        --inputs data/sft_replay/pretrain_replay.jsonl:1.0 \\
                 data/sft_phase3_intermediate/episodes.jsonl:19.0 \\
        --output data/sft_phase3_intermediate/mixed_with_replay.jsonl \\
        --shuffle
"""

import argparse
import json
import os
import random
import struct
import sys
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.tokenizer import Tokenizer
from core.model import GPTConfig


def find_token_shards(shard_dir: str) -> List[Path]:
    """Find all tokens.bin files in a shard directory (handles multi-shard)."""
    root = Path(shard_dir)
    shards = []
    
    # Direct tokens.bin
    direct = root / "tokens.bin"
    if direct.exists():
        shards.append(direct)
    
    # Multi-shard: shard_XXXXX/tokens.bin
    for subdir in sorted(root.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("shard_"):
            t = subdir / "tokens.bin"
            if t.exists():
                shards.append(t)
    
    # Also check train/ subdirectory
    train_dir = root / "train"
    if train_dir.is_dir():
        t = train_dir / "tokens.bin"
        if t.exists():
            shards.append(t)
        for subdir in sorted(train_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("shard_"):
                t = subdir / "tokens.bin"
                if t.exists():
                    shards.append(t)
    
    return shards


def sample_passage_from_shard(shard_path: Path, max_tokens: int, rng: random.Random) -> Optional[List[int]]:
    """Sample a random passage of up to max_tokens from a token shard."""
    file_size = shard_path.stat().st_size
    
    # Each token is uint16 (2 bytes) or uint32 (4 bytes)
    # Try to detect format from file size and common vocab sizes
    total_tokens_u16 = file_size // 2
    total_tokens_u32 = file_size // 4
    
    # GPT-2 vocab is 50304, which fits in uint16 (max 65535)
    # But some shards may use uint32. Try uint16 first.
    if total_tokens_u16 < max_tokens:
        return None
    
    # Sample a random start position
    max_start = total_tokens_u16 - max_tokens
    if max_start <= 0:
        return None
    
    start = rng.randint(0, max_start)
    
    try:
        with open(shard_path, 'rb') as f:
            f.seek(start * 2)  # uint16 = 2 bytes
            raw = f.read(max_tokens * 2)
            tokens = list(struct.unpack(f'<{max_tokens}H', raw))
            
            # Validate: all tokens should be < vocab_size
            if any(t >= 50304 for t in tokens):
                # Probably uint32 format, try again
                f.seek(start * 4)
                raw = f.read(max_tokens * 4)
                if len(raw) < max_tokens * 4:
                    return None
                tokens = list(struct.unpack(f'<{max_tokens}I', raw))
                if any(t >= 50304 for t in tokens):
                    return None
            
            return tokens
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-training replay buffer for SFT phases"
    )
    parser.add_argument("--shard_dirs", nargs="+", required=True,
                        help="Pre-training tokenized shard directories to sample from")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--num_episodes", type=int, default=2000,
                        help="Number of replay episodes to generate (default: 2000)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens per replay passage (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--source_weights", nargs="+", type=float, default=None,
                        help="Sampling weights per shard_dir (default: uniform)")
    
    args = parser.parse_args()
    rng = random.Random(args.seed)
    
    # Discover all shards
    print("=" * 60)
    print("  PRE-TRAINING REPLAY BUFFER GENERATOR")
    print("=" * 60)
    
    source_shards = {}  # dir_name -> [shard_paths]
    for shard_dir in args.shard_dirs:
        shards = find_token_shards(shard_dir)
        if not shards:
            print(f"  WARNING: No token shards found in {shard_dir}")
            continue
        source_shards[shard_dir] = shards
        print(f"  {shard_dir}: {len(shards)} shard(s)")
    
    if not source_shards:
        print("ERROR: No valid shard directories found.")
        sys.exit(1)
    
    # Build weighted source list
    source_dirs = list(source_shards.keys())
    if args.source_weights and len(args.source_weights) == len(source_dirs):
        weights = args.source_weights
    else:
        weights = [1.0] * len(source_dirs)
    
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    
    print(f"\n  Generating {args.num_episodes} episodes, max {args.max_tokens} tokens each")
    print(f"  Source weights: {dict(zip([Path(d).name for d in source_dirs], weights))}")
    
    # Initialize tokenizer for decoding
    config = GPTConfig()
    tokenizer = Tokenizer(config, "gpt2")
    
    # Generate episodes
    os.makedirs(Path(args.output).parent, exist_ok=True)
    
    episodes = []
    source_counts = {d: 0 for d in source_dirs}
    attempts = 0
    max_attempts = args.num_episodes * 5
    
    while len(episodes) < args.num_episodes and attempts < max_attempts:
        attempts += 1
        
        # Pick a source directory
        src_dir = rng.choices(source_dirs, weights=weights, k=1)[0]
        shards = source_shards[src_dir]
        shard = rng.choice(shards)
        
        # Sample a passage
        tokens = sample_passage_from_shard(shard, args.max_tokens, rng)
        if tokens is None:
            continue
        
        # Decode to text
        try:
            text = tokenizer.decode(tokens)
        except Exception:
            continue
        
        # Basic quality filter: skip very short or empty passages
        stripped = text.strip()
        if len(stripped) < 20:
            continue
        
        # Skip passages that are mostly special characters or whitespace
        alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
        if alpha_ratio < 0.3:
            continue
        
        episode = {
            "_replay": True,
            "text": stripped,
            "_meta": {
                "source": Path(src_dir).name,
                "tokens": len(tokens),
            }
        }
        episodes.append(episode)
        source_counts[src_dir] += 1
    
    # Shuffle
    rng.shuffle(episodes)
    
    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + '\n')
    
    print(f"\n  Generated {len(episodes)} replay episodes")
    print(f"  Source breakdown:")
    for d, count in source_counts.items():
        print(f"    {Path(d).name}: {count} ({count/max(len(episodes),1)*100:.1f}%)")
    print(f"\n  Output: {args.output}")
    print(f"  Mix into SFT phases with mix_sft_jsonl.py (recommended: 5% ratio)")
    print()


if __name__ == "__main__":
    main()
