#!/usr/bin/env python3
"""
Inspect SFT dataset: show statistics, packing info, and sample decoded sequences.

Usage:
    python scripts/inspect_sft_dataset.py --dataset_dir data/sft_prepared
    
    # Show decoded samples
    python scripts/inspect_sft_dataset.py --dataset_dir data/sft_prepared --show_samples 3
"""

import argparse
import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tokenizer import Tokenizer
from core.model import GPTConfig


def main():
    parser = argparse.ArgumentParser(description="Inspect SFT dataset")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to prepared SFT dataset")
    parser.add_argument("--show_samples", type=int, default=0,
                        help="Number of samples to decode and display")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Which split to inspect")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  SFT DATASET INSPECTOR")
    print("=" * 70)
    
    # Load metadata
    metadata_path = os.path.join(args.dataset_dir, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found: {metadata_path}")
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\nDataset: {args.dataset_dir}")
    print(f"Schema: {metadata.get('schema', 'unknown')}")
    print()
    
    # Basic stats
    print("=" * 40)
    print("  BASIC STATISTICS")
    print("=" * 40)
    print(f"  Conversations (original): {metadata.get('num_conversations', '?')}")
    print(f"  Train episodes/sequences: {metadata.get('num_train_episodes', '?')}")
    print(f"  Val episodes/sequences:   {metadata.get('num_val_episodes', '?')}")
    print(f"  Train tokens:             {metadata.get('num_train_tokens', 0):,}")
    print(f"  Val tokens:               {metadata.get('num_val_tokens', 0):,}")
    print(f"  Train mask ratio:         {metadata.get('train_mask_ratio', 0):.1%}")
    print(f"  Val mask ratio:           {metadata.get('val_mask_ratio', 0):.1%}")
    print()
    
    # Supervised token stats
    print("=" * 40)
    print("  SUPERVISED TOKENS (loss_mask=1)")
    print("=" * 40)
    train_mask_tokens = metadata.get('num_train_mask_tokens', 0)
    val_mask_tokens = metadata.get('num_val_mask_tokens', 0)
    print(f"  Train supervised tokens:  {train_mask_tokens:,}")
    print(f"  Val supervised tokens:    {val_mask_tokens:,}")
    print()
    
    # Packing info
    print("=" * 40)
    print("  PACKING INFO")
    print("=" * 40)
    packing_enabled = metadata.get('packing_enabled', False)
    print(f"  Packing enabled:          {packing_enabled}")
    
    if packing_enabled:
        print(f"  Pack block size:          {metadata.get('pack_block_size', '?')}")
        print(f"  Pack by field:            {metadata.get('pack_by_field', 'none')}")
        print(f"  Original train episodes:  {metadata.get('original_train_episodes', '?')}")
        print(f"  Original val episodes:    {metadata.get('original_val_episodes', '?')}")
        print(f"  Avg episode length:       {metadata.get('avg_episode_len', 0):.1f} tokens")
        print(f"  Original mask ratio:      {metadata.get('original_train_mask_ratio', 0):.1%}")
        print(f"  Avg episodes per pack:    {metadata.get('avg_episodes_per_pack', 0):.1f}")
        print(f"  Non-pad ratio:            {metadata.get('nonpad_ratio', 0):.1%}")
        # Training efficiency
        efficiency = metadata.get('training_efficiency_gain', metadata.get('packing_efficiency', 1.0))
        unpacked_eff = metadata.get('unpacked_effective_mask', 0)
        packed_eff = metadata.get('packed_effective_mask', 0)
        print(f"  ⚡ Training efficiency:   {efficiency:.1f}x more supervised tokens/step")
        if unpacked_eff > 0 and packed_eff > 0:
            print(f"     (Without packing: {unpacked_eff:.1%} supervised → With packing: {packed_eff:.1%})")
    print()
    
    # Provenance
    print("=" * 40)
    print("  PROVENANCE")
    print("=" * 40)
    print(f"  Prepare mode:             {metadata.get('prepare_mode', '?')}")
    print(f"  Source train file:        {metadata.get('source_train_file', '?')}")
    if metadata.get('source_val_file'):
        print(f"  Source val file:          {metadata.get('source_val_file')}")
    print()
    
    # Show samples if requested
    if args.show_samples > 0:
        print("=" * 70)
        print(f"  SAMPLE DECODED SEQUENCES ({args.split})")
        print("=" * 70)
        
        # Load tokenizer
        tokenizer_path = os.path.join(args.dataset_dir, "tokenizer_state.json")
        config = GPTConfig(vocab_size=metadata.get('vocab_size', 50304))
        tokenizer = Tokenizer(config, metadata.get('tokenization', 'gpt2'))
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                tokenizer_state = json.load(f)
            if 'special_tokens' in tokenizer_state:
                tokenizer.special_tokens = tokenizer_state['special_tokens']
        
        # Load tokens and mask
        split_dir = os.path.join(args.dataset_dir, args.split)
        
        # Check for single-file or shard format
        if os.path.exists(os.path.join(split_dir, "tokens.bin")):
            tokens_path = os.path.join(split_dir, "tokens.bin")
            mask_path = os.path.join(split_dir, "mask.bin")
            episodes_path = os.path.join(split_dir, "episodes.idx")
        else:
            # Try first shard
            shard_dir = os.path.join(split_dir, "shard_00000")
            tokens_path = os.path.join(shard_dir, "tokens.bin")
            mask_path = os.path.join(shard_dir, "mask.bin")
            episodes_path = os.path.join(shard_dir, "episodes.idx")
        
        if not os.path.exists(tokens_path):
            print(f"  Error: tokens.bin not found")
            return
        
        tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
        mask = np.memmap(mask_path, dtype=np.uint8, mode='r') if os.path.exists(mask_path) else None
        episodes = np.memmap(episodes_path, dtype=np.uint64, mode='r').reshape(-1, 2)
        
        n_samples = min(args.show_samples, len(episodes))
        
        for i in range(n_samples):
            start, length = int(episodes[i][0]), int(episodes[i][1])
            ep_tokens = tokens[start:start+length]
            ep_mask = mask[start:start+length] if mask is not None else None
            
            print(f"\n{'─'*70}")
            print(f"Sequence {i+1}/{n_samples} (tokens {start}:{start+length}, length={length})")
            print(f"{'─'*70}")
            
            # Decode
            try:
                text = tokenizer.decode(ep_tokens.tolist())
            except:
                text = f"[Decode error - raw tokens: {ep_tokens[:50].tolist()}...]"
            
            # Show text (truncated)
            max_chars = 500
            if len(text) > max_chars:
                print(f"{text[:max_chars]}...")
                print(f"  [...truncated, total {len(text)} chars]")
            else:
                print(text)
            
            # Show mask stats
            if ep_mask is not None:
                mask_sum = int(ep_mask.sum())
                mask_ratio = mask_sum / length if length > 0 else 0
                print(f"\n  Mask: {mask_sum}/{length} tokens ({mask_ratio:.1%})")
            
            # Show token boundaries (first/last few)
            print(f"  First 10 tokens: {ep_tokens[:10].tolist()}")
            print(f"  Last 10 tokens:  {ep_tokens[-10:].tolist()}")
    
    # Training recommendations
    print("\n" + "=" * 70)
    print("  TRAINING RECOMMENDATIONS")
    print("=" * 70)
    
    train_episodes = metadata.get('num_train_episodes', 0)
    train_mask_tokens = metadata.get('num_train_mask_tokens', 0)
    batch_size = 16  # Assume default
    
    if packing_enabled:
        block_size = metadata.get('pack_block_size', 1024)
        tokens_per_step = batch_size * block_size
        mask_per_step = int(tokens_per_step * metadata.get('train_mask_ratio', 0.5))
        
        if train_mask_tokens > 0:
            iters_2x = int((2.0 * train_mask_tokens) / mask_per_step)
            iters_3x = int((3.0 * train_mask_tokens) / mask_per_step)
            iters_35x = int((3.5 * train_mask_tokens) / mask_per_step)
            
            print(f"\n  Masked-token budget approach (recommended for packed data):")
            print(f"    Train supervised tokens: {train_mask_tokens:,}")
            print(f"    Est. supervised/step:    ~{mask_per_step:,}")
            print(f"    2.0x mask passes:        max_iters = {iters_2x}")
            print(f"    3.0x mask passes:        max_iters = {iters_3x}")
            print(f"    3.5x mask passes:        max_iters = {iters_35x} (recommended)")
    else:
        batches_per_epoch = train_episodes // batch_size if train_episodes > 0 else 0
        
        if batches_per_epoch > 0:
            iters_2x = int(batches_per_epoch * 2.0)
            iters_3x = int(batches_per_epoch * 3.0)
            iters_35x = int(batches_per_epoch * 3.5)
            
            print(f"\n  Episode-based approach (unpacked data):")
            print(f"    Train episodes:          {train_episodes}")
            print(f"    Batches per epoch:       {batches_per_epoch}")
            print(f"    2.0x coverage:           max_iters = {iters_2x}")
            print(f"    3.0x coverage:           max_iters = {iters_3x}")
            print(f"    3.5x coverage:           max_iters = {iters_35x} (recommended)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
