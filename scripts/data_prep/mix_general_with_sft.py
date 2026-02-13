#!/usr/bin/env python3
"""
Mix General Token-Stream Data with SFT Episode Dataset

Converts general token-stream shards into pseudo-episodes and merges them
with an SFT episode-indexed dataset. This enables replay during SFT training
to prevent catastrophic forgetting.

General data is chunked into fixed-size episodes with mask=1 (train on all tokens).
SFT data keeps its original episode boundaries and loss masks.

Usage:
    python scripts/mix_general_with_sft.py \
        --sft_dir data/sft_conversation_goldset_prepared \
        --general_dir data/domain_v8_prepared \
        --output_dir data/sft_mixed_prepared \
        --general_ratio 0.2 \
        --chunk_size 512

Output structure (episode-indexed format):
    output_dir/
        train/tokens.bin      # Combined SFT + general tokens
        train/mask.bin        # Combined masks (SFT selective, general all-1s)
        train/episodes.idx    # Combined episode index
        val/...               # SFT val only (for proper SFT evaluation)
        dataset_metadata.json
        tokenizer_state.json
"""

import argparse
import json
import sys
import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

from core.banner import print_banner


def load_episode_dataset(dataset_dir: Path) -> dict:
    """Load an episode-indexed SFT dataset."""
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    # Check for shard subdirectories or direct files
    shard_dirs = sorted(train_dir.glob("shard_*"))
    
    if shard_dirs:
        # Multi-shard format
        tokens_list = []
        masks_list = []
        episodes_list = []
        offset = 0
        
        for shard_dir in shard_dirs:
            tokens = np.memmap(shard_dir / "tokens.bin", dtype=np.uint32, mode='r')
            mask = np.memmap(shard_dir / "mask.bin", dtype=np.uint8, mode='r')
            episodes = np.memmap(shard_dir / "episodes.idx", dtype=np.uint64, mode='r').reshape(-1, 2)
            
            # Adjust episode offsets
            adjusted_episodes = episodes.copy()
            adjusted_episodes[:, 0] += offset
            
            tokens_list.append(np.array(tokens))
            masks_list.append(np.array(mask))
            episodes_list.append(adjusted_episodes)
            
            offset += len(tokens)
        
        train_tokens = np.concatenate(tokens_list)
        train_mask = np.concatenate(masks_list)
        train_episodes = np.concatenate(episodes_list)
    else:
        # Single-file format
        train_tokens = np.array(np.memmap(train_dir / "tokens.bin", dtype=np.uint32, mode='r'))
        train_mask = np.array(np.memmap(train_dir / "mask.bin", dtype=np.uint8, mode='r'))
        train_episodes = np.array(np.memmap(train_dir / "episodes.idx", dtype=np.uint64, mode='r').reshape(-1, 2))
    
    # Load val similarly
    val_tokens = None
    val_mask = None
    val_episodes = None
    
    if val_dir.exists():
        val_shard_dirs = sorted(val_dir.glob("shard_*"))
        if val_shard_dirs:
            tokens_list = []
            masks_list = []
            episodes_list = []
            offset = 0
            
            for shard_dir in val_shard_dirs:
                tokens = np.memmap(shard_dir / "tokens.bin", dtype=np.uint32, mode='r')
                mask = np.memmap(shard_dir / "mask.bin", dtype=np.uint8, mode='r')
                episodes = np.memmap(shard_dir / "episodes.idx", dtype=np.uint64, mode='r').reshape(-1, 2)
                
                adjusted_episodes = episodes.copy()
                adjusted_episodes[:, 0] += offset
                
                tokens_list.append(np.array(tokens))
                masks_list.append(np.array(mask))
                episodes_list.append(adjusted_episodes)
                
                offset += len(tokens)
            
            val_tokens = np.concatenate(tokens_list)
            val_mask = np.concatenate(masks_list)
            val_episodes = np.concatenate(episodes_list)
        elif (val_dir / "tokens.bin").exists():
            val_tokens = np.array(np.memmap(val_dir / "tokens.bin", dtype=np.uint32, mode='r'))
            val_mask = np.array(np.memmap(val_dir / "mask.bin", dtype=np.uint8, mode='r'))
            val_episodes = np.array(np.memmap(val_dir / "episodes.idx", dtype=np.uint64, mode='r').reshape(-1, 2))
    
    # Load metadata
    metadata = {}
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return {
        'train_tokens': train_tokens,
        'train_mask': train_mask,
        'train_episodes': train_episodes,
        'val_tokens': val_tokens,
        'val_mask': val_mask,
        'val_episodes': val_episodes,
        'metadata': metadata,
    }


def load_general_shards(dataset_dir: Path) -> np.ndarray:
    """Load token-stream shards from a general dataset."""
    train_dir = dataset_dir / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"No train directory found: {train_dir}")
    
    # Find all .bin files (excluding mask files)
    shard_files = sorted([f for f in train_dir.glob("*.bin") 
                          if not f.name.endswith("_mask.bin")])
    
    if not shard_files:
        raise FileNotFoundError(f"No .bin shard files found in {train_dir}")
    
    print(f"    Found {len(shard_files)} general shards")
    
    # Load and concatenate all shards
    all_tokens = []
    for shard_path in shard_files:
        tokens = np.memmap(shard_path, dtype=np.uint32, mode='r')
        all_tokens.append(np.array(tokens))
    
    return np.concatenate(all_tokens)


def chunk_into_episodes(tokens: np.ndarray, chunk_size: int) -> tuple:
    """
    Convert continuous token stream into fixed-size episodes.
    
    Returns:
        (tokens, mask, episodes) where:
        - tokens: same as input (may be trimmed to fit chunks)
        - mask: all 1s (train on everything)
        - episodes: (start, length) pairs for each chunk
    """
    n_tokens = len(tokens)
    n_episodes = n_tokens // chunk_size
    
    # Trim to exact multiple of chunk_size
    trimmed_len = n_episodes * chunk_size
    tokens = tokens[:trimmed_len]
    
    # Create mask (all 1s - train on everything for replay data)
    mask = np.ones(trimmed_len, dtype=np.uint8)
    
    # Create episode index
    episodes = np.zeros((n_episodes, 2), dtype=np.uint64)
    for i in range(n_episodes):
        episodes[i, 0] = i * chunk_size  # start
        episodes[i, 1] = chunk_size       # length
    
    return tokens, mask, episodes


def main():
    parser = argparse.ArgumentParser(
        description="Mix general token-stream data with SFT episode dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--sft_dir", type=str, required=True,
                        help="SFT episode-indexed dataset directory")
    parser.add_argument("--general_dir", type=str, required=True,
                        help="General token-stream dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for mixed dataset")
    
    parser.add_argument("--general_ratio", type=float, default=0.2,
                        help="Ratio of general data (0.2 = 20%% general, 80%% SFT)")
    parser.add_argument("--general_tokens", type=int, default=0,
                        help="Exact general tokens to add (overrides ratio if > 0)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Episode size for general data chunks (default: 512)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle combined episodes (recommended)")
    
    args = parser.parse_args()
    
    print_banner("MyPT", "SFT + General Data Mixer")
    
    print(f"\n  SFT dataset:     {args.sft_dir}")
    print(f"  General dataset: {args.general_dir}")
    print(f"  Output:          {args.output_dir}")
    print(f"  General ratio:   {args.general_ratio:.0%}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  Shuffle:         {args.shuffle}")
    
    np.random.seed(args.seed)
    
    sft_dir = Path(args.sft_dir)
    general_dir = Path(args.general_dir)
    output_dir = Path(args.output_dir)
    
    # ========================================
    # Load SFT dataset
    # ========================================
    print("\n" + "=" * 60)
    print("  Loading SFT Episode Dataset")
    print("=" * 60)
    
    sft_data = load_episode_dataset(sft_dir)
    sft_tokens = sft_data['train_tokens']
    sft_mask = sft_data['train_mask']
    sft_episodes = sft_data['train_episodes']
    
    sft_token_count = len(sft_tokens)
    sft_episode_count = len(sft_episodes)
    sft_mask_ratio = sft_mask.sum() / len(sft_mask) if len(sft_mask) > 0 else 0
    
    print(f"    SFT tokens:   {sft_token_count:,}")
    print(f"    SFT episodes: {sft_episode_count}")
    print(f"    SFT mask ratio: {sft_mask_ratio:.1%} (assistant tokens)")
    
    # ========================================
    # Load General dataset
    # ========================================
    print("\n" + "=" * 60)
    print("  Loading General Token-Stream Dataset")
    print("=" * 60)
    
    general_tokens_all = load_general_shards(general_dir)
    print(f"    Total general tokens available: {len(general_tokens_all):,}")
    
    # Calculate how many general tokens to use
    if args.general_tokens > 0:
        target_general_tokens = args.general_tokens
    else:
        # SFT is (1 - ratio), so general = SFT * ratio / (1 - ratio)
        target_general_tokens = int(sft_token_count * args.general_ratio / (1 - args.general_ratio))
    
    # Limit to available tokens
    actual_general_tokens = min(target_general_tokens, len(general_tokens_all))
    actual_general_tokens = (actual_general_tokens // args.chunk_size) * args.chunk_size  # Round to chunk size
    
    print(f"    Target general tokens: {target_general_tokens:,}")
    print(f"    Using general tokens:  {actual_general_tokens:,}")
    
    # Sample tokens from general dataset
    if actual_general_tokens < len(general_tokens_all):
        # Random sample
        start_idx = np.random.randint(0, len(general_tokens_all) - actual_general_tokens)
        general_tokens = general_tokens_all[start_idx:start_idx + actual_general_tokens]
    else:
        general_tokens = general_tokens_all[:actual_general_tokens]
    
    # ========================================
    # Convert general to episodes
    # ========================================
    print("\n" + "=" * 60)
    print("  Converting General Data to Episodes")
    print("=" * 60)
    
    gen_tokens, gen_mask, gen_episodes = chunk_into_episodes(general_tokens, args.chunk_size)
    gen_episode_count = len(gen_episodes)
    
    print(f"    General episodes created: {gen_episode_count}")
    print(f"    Episode size: {args.chunk_size} tokens each")
    print(f"    General mask: 100% (train on all tokens)")
    
    # ========================================
    # Merge datasets
    # ========================================
    print("\n" + "=" * 60)
    print("  Merging Datasets")
    print("=" * 60)
    
    # Concatenate tokens and masks
    combined_tokens = np.concatenate([sft_tokens, gen_tokens])
    combined_mask = np.concatenate([sft_mask, gen_mask])
    
    # Adjust general episode offsets (they come after SFT tokens)
    gen_episodes_adjusted = gen_episodes.copy()
    gen_episodes_adjusted[:, 0] += len(sft_tokens)
    
    # Concatenate episodes
    combined_episodes = np.concatenate([sft_episodes, gen_episodes_adjusted])
    
    total_episodes = len(combined_episodes)
    total_tokens = len(combined_tokens)
    actual_ratio = len(gen_tokens) / total_tokens
    
    print(f"    Combined tokens:   {total_tokens:,}")
    print(f"    Combined episodes: {total_episodes}")
    print(f"    SFT episodes:      {sft_episode_count} ({sft_episode_count/total_episodes:.1%})")
    print(f"    General episodes:  {gen_episode_count} ({gen_episode_count/total_episodes:.1%})")
    print(f"    Actual token ratio: {1-actual_ratio:.1%} SFT, {actual_ratio:.1%} general")
    
    # ========================================
    # Optionally shuffle episodes
    # ========================================
    if args.shuffle:
        print("\n  Shuffling episode order...")
        # We need to reorder tokens/mask based on shuffled episodes
        perm = np.random.permutation(total_episodes)
        
        # Build new token/mask arrays in shuffled episode order
        new_tokens = []
        new_mask = []
        new_episodes = []
        offset = 0
        
        for idx in perm:
            start, length = combined_episodes[idx]
            start, length = int(start), int(length)
            
            new_tokens.append(combined_tokens[start:start+length])
            new_mask.append(combined_mask[start:start+length])
            new_episodes.append((offset, length))
            offset += length
        
        combined_tokens = np.concatenate(new_tokens)
        combined_mask = np.concatenate(new_mask)
        combined_episodes = np.array(new_episodes, dtype=np.uint64)
        
        print(f"    Shuffled {total_episodes} episodes")
    
    # ========================================
    # Write output
    # ========================================
    print("\n" + "=" * 60)
    print("  Writing Output Dataset")
    print("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train"
    val_out = output_dir / "val"
    train_out.mkdir(exist_ok=True)
    val_out.mkdir(exist_ok=True)
    
    # Write train files
    combined_tokens.astype(np.uint32).tofile(train_out / "tokens.bin")
    combined_mask.astype(np.uint8).tofile(train_out / "mask.bin")
    combined_episodes.astype(np.uint64).tofile(train_out / "episodes.idx")
    print(f"    Written train/tokens.bin ({len(combined_tokens):,} tokens)")
    print(f"    Written train/mask.bin")
    print(f"    Written train/episodes.idx ({total_episodes} episodes)")
    
    # Copy SFT val data (keep SFT-only for proper evaluation)
    if sft_data['val_tokens'] is not None:
        sft_data['val_tokens'].astype(np.uint32).tofile(val_out / "tokens.bin")
        sft_data['val_mask'].astype(np.uint8).tofile(val_out / "mask.bin")
        sft_data['val_episodes'].astype(np.uint64).tofile(val_out / "episodes.idx")
        val_episodes_count = len(sft_data['val_episodes'])
        val_tokens_count = len(sft_data['val_tokens'])
        print(f"    Written val/ (SFT-only: {val_episodes_count} episodes, {val_tokens_count:,} tokens)")
    
    # Copy tokenizer state from SFT
    tokenizer_src = sft_dir / "tokenizer_state.json"
    if tokenizer_src.exists():
        import shutil
        shutil.copy2(tokenizer_src, output_dir / "tokenizer_state.json")
        print(f"    Copied tokenizer_state.json")
    
    # Write metadata
    metadata = {
        "schema": "episode_indexed_sft_v1",
        "has_loss_mask": True,
        "num_train_episodes": total_episodes,
        "num_val_episodes": len(sft_data['val_episodes']) if sft_data['val_episodes'] is not None else 0,
        "num_train_tokens": int(total_tokens),
        "num_val_tokens": int(len(sft_data['val_tokens'])) if sft_data['val_tokens'] is not None else 0,
        "train_mask_ratio": float(combined_mask.sum() / len(combined_mask)),
        "mix_info": {
            "sft_source": str(sft_dir),
            "sft_tokens": int(sft_token_count),
            "sft_episodes": int(sft_episode_count),
            "sft_mask_ratio": float(sft_mask_ratio),
            "general_source": str(general_dir),
            "general_tokens": int(len(gen_tokens)),
            "general_episodes": int(gen_episode_count),
            "general_chunk_size": args.chunk_size,
            "general_ratio_target": args.general_ratio,
            "general_ratio_actual": float(actual_ratio),
            "shuffled": args.shuffle,
            "seed": args.seed,
            "mixed_at": datetime.now().isoformat(),
        },
        # Preserve original SFT metadata
        "original_sft_metadata": sft_data['metadata'],
    }
    
    with open(output_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"    Written dataset_metadata.json")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"\n  Train set:")
    print(f"    Total tokens:   {total_tokens:,}")
    print(f"    Total episodes: {total_episodes}")
    print(f"    SFT ratio:      {1-actual_ratio:.1%} ({sft_episode_count} episodes)")
    print(f"    General ratio:  {actual_ratio:.1%} ({gen_episode_count} episodes)")
    print(f"    Mask ratio:     {combined_mask.sum()/len(combined_mask):.1%}")
    
    if sft_data['val_tokens'] is not None:
        print(f"\n  Val set (SFT-only for proper evaluation):")
        print(f"    Tokens:   {len(sft_data['val_tokens']):,}")
        print(f"    Episodes: {len(sft_data['val_episodes'])}")
    
    print(f"\n  To train with this dataset:")
    print(f"    python train.py \\")
    print(f"      --config_file configs/sft1/750M_1024_chat_sft_phase3a.json \\")
    print(f"      --dataset_dir {output_dir} \\")
    print(f"      --init_from_model checkpoints/domain_v8 \\")
    print(f"      --model_name phase3a_sft_mixed")
    
    print("\n  Done!\n")


if __name__ == "__main__":
    main()

