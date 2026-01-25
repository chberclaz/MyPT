#!/usr/bin/env python3
"""
Inspect SFT episode-indexed dataset.

Shows episodes with:
- Raw token IDs
- Decoded text
- Loss mask alignment (which tokens are trained on)

Usage:
    python scripts/inspect_sft_dataset.py --dataset data/sft_phase3a_pure --episode 0
    python scripts/inspect_sft_dataset.py --dataset data/sft_phase3a_pure --episode 0 --show_all
    python scripts/inspect_sft_dataset.py --dataset data/sft_phase3a_pure --stats
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tokenizer import Tokenizer
from core.special_tokens import SPECIAL_TOKEN_STRINGS
from core.model import GPTConfig


def load_episode_indexed_dataset(dataset_dir: str, split: str = "train"):
    """Load tokens, mask, and episode index from dataset."""
    split_dir = Path(dataset_dir) / split
    
    tokens_path = split_dir / "tokens.bin"
    mask_path = split_dir / "mask.bin"
    index_path = split_dir / "episodes.idx"
    
    if not tokens_path.exists():
        raise FileNotFoundError(f"tokens.bin not found in {split_dir}")
    
    # Load tokens (uint32)
    tokens = np.fromfile(tokens_path, dtype=np.uint32)
    
    # Load mask (uint8) - optional
    if mask_path.exists():
        mask = np.fromfile(mask_path, dtype=np.uint8)
    else:
        mask = None
        print(f"Warning: mask.bin not found in {split_dir}")
    
    # Load episode index (uint64 pairs: start, length)
    if index_path.exists():
        index_data = np.fromfile(index_path, dtype=np.uint64)
        episodes = index_data.reshape(-1, 2)  # (num_episodes, 2) - [start, length]
    else:
        raise FileNotFoundError(f"episodes.idx not found in {split_dir}")
    
    return tokens, mask, episodes


def get_episode(tokens, mask, episodes, episode_idx):
    """Extract a single episode's tokens and mask."""
    if episode_idx >= len(episodes):
        raise IndexError(f"Episode {episode_idx} out of range (max: {len(episodes)-1})")
    
    start, length = episodes[episode_idx]
    start, length = int(start), int(length)
    
    ep_tokens = tokens[start:start+length]
    ep_mask = mask[start:start+length] if mask is not None else None
    
    return ep_tokens, ep_mask


def create_tokenizer():
    """Create tokenizer with special tokens registered."""
    # Create a minimal config for tokenizer initialization
    config = GPTConfig(vocab_size=50304)
    tokenizer = Tokenizer(config, kind="gpt2")
    # Special tokens are auto-registered in __init__ for gpt2
    return tokenizer


def display_episode(ep_tokens, ep_mask, tokenizer, show_all=False):
    """Display episode with token-by-token breakdown."""
    print(f"\n{'='*80}")
    print(f"Episode: {len(ep_tokens)} tokens")
    print(f"{'='*80}\n")
    
    # Decode full text
    full_text = tokenizer.decode(ep_tokens.tolist())
    print("=== FULL TEXT ===")
    print(full_text)
    print()
    
    # Mask statistics
    if ep_mask is not None:
        mask_1_count = np.sum(ep_mask == 1)
        mask_0_count = np.sum(ep_mask == 0)
        print(f"=== MASK STATS ===")
        print(f"  mask=1 (trained): {mask_1_count} tokens ({100*mask_1_count/len(ep_mask):.1f}%)")
        print(f"  mask=0 (context): {mask_0_count} tokens ({100*mask_0_count/len(ep_mask):.1f}%)")
        print()
    
    if show_all:
        print("=== TOKEN-BY-TOKEN ===")
        print(f"{'Pos':>5} | {'Token ID':>8} | {'Mask':>4} | Decoded")
        print("-" * 80)
        
        for i, tok_id in enumerate(ep_tokens):
            tok_str = tokenizer.decode([int(tok_id)])
            # Escape special characters for display
            tok_display = repr(tok_str)[1:-1]  # Remove quotes from repr
            mask_val = ep_mask[i] if ep_mask is not None else "?"
            
            # Highlight mask=1 tokens
            marker = ">>>" if mask_val == 1 else "   "
            print(f"{marker} {i:>5} | {tok_id:>8} | {mask_val:>4} | {tok_display}")
    
    # Show mask transitions (where mask changes from 0 to 1 and back)
    if ep_mask is not None:
        print("\n=== MASK TRANSITIONS ===")
        print("(Shows where training signal starts/stops)\n")
        
        prev_mask = 0
        for i, (tok_id, m) in enumerate(zip(ep_tokens, ep_mask)):
            if m != prev_mask:
                tok_str = tokenizer.decode([int(tok_id)])
                direction = "START training >>>" if m == 1 else "<<< END training"
                print(f"  Position {i}: {direction}")
                print(f"    Token: {tok_id} = {repr(tok_str)}")
                
                # Show context around transition
                ctx_start = max(0, i-2)
                ctx_end = min(len(ep_tokens), i+3)
                ctx_tokens = ep_tokens[ctx_start:ctx_end].tolist()
                ctx_text = tokenizer.decode(ctx_tokens)
                print(f"    Context: ...{repr(ctx_text)}...")
                print()
                
            prev_mask = m


def show_stats(dataset_dir: str):
    """Show overall dataset statistics."""
    for split in ["train", "val"]:
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            continue
            
        print(f"\n=== {split.upper()} SPLIT ===")
        
        try:
            tokens, mask, episodes = load_episode_indexed_dataset(dataset_dir, split)
            
            print(f"  Total tokens: {len(tokens):,}")
            print(f"  Total episodes: {len(episodes):,}")
            
            # Episode length stats
            lengths = episodes[:, 1].astype(int)
            print(f"  Episode lengths:")
            print(f"    Min: {lengths.min()}")
            print(f"    Max: {lengths.max()}")
            print(f"    Mean: {lengths.mean():.1f}")
            print(f"    Median: {np.median(lengths):.1f}")
            
            if mask is not None:
                mask_1_pct = 100 * np.sum(mask == 1) / len(mask)
                print(f"  Mask=1 (trained tokens): {mask_1_pct:.1f}%")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect SFT episode-indexed dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"],
                        help="Which split to inspect (default: train)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to display (default: 0)")
    parser.add_argument("--show_all", action="store_true",
                        help="Show all tokens with their masks")
    parser.add_argument("--stats", action="store_true",
                        help="Show dataset statistics only")
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats(args.dataset)
        return
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset}/{args.split}")
    tokens, mask, episodes = load_episode_indexed_dataset(args.dataset, args.split)
    
    print(f"Loaded {len(episodes)} episodes, {len(tokens):,} total tokens")
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Get and display episode
    ep_tokens, ep_mask = get_episode(tokens, mask, episodes, args.episode)
    display_episode(ep_tokens, ep_mask, tokenizer, show_all=args.show_all)


if __name__ == "__main__":
    main()
