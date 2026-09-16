#!/usr/bin/env python
"""
Mix Tokenized Datasets for Domain Adaptation

This script mixes pre-tokenized datasets by copying shards from one or two "replay"
datasets (e.g. Wikipedia + domain corpus) into a domain-specific dataset. This helps
prevent catastrophic forgetting during domain adaptation or continued pre-training.

Usage:
    # Single replay source
    python scripts/mix_tokenized_datasets.py \
        --domain_dir data/domain_161M_corpus_tokenized \
        --replay_dir data/multilingual_1.5B_wiki90 \
        --output_dir data/domain_mixed \
        --replay_ratio 0.3

    # Dual replay sources (e.g. Phase 1.5 with wiki + domain replay)
    python scripts/mix_tokenized_datasets.py \
        --domain_dir data/phase1_5_induction_raw \
        --replay_dir data/multilingual_1.5B_wiki90 \
        --replay2_dir data/domain_161M_corpus_tokenized \
        --output_dir data/phase1_5_mixed \
        --replay_ratio 0.10 \
        --replay2_ratio 0.03

The script will:
1. Copy all shards from domain_dir to output_dir
2. Add replay shards from replay_dir to achieve the target ratio
3. Optionally add replay2 shards from replay2_dir at a separate ratio
4. Update metadata with combined information
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.banner import print_banner


def get_shard_info(dataset_dir: Path) -> dict:
    """Read dataset metadata and count shards."""
    metadata_path = dataset_dir / "dataset_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No dataset_metadata.json found in {dataset_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    train_shards = sorted(train_dir.glob("*.bin")) if train_dir.exists() else []
    val_shards = sorted(val_dir.glob("*.bin")) if val_dir.exists() else []
    
    return {
        "metadata": metadata,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "total_tokens": metadata.get("total_tokens", 0),
        "tokens_per_shard": metadata.get("tokens_per_shard", 10_000_000),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mix tokenized datasets for domain adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--domain_dir", type=str, required=True,
                        help="Domain dataset directory (will be copied)")
    parser.add_argument("--replay_dir", type=str, required=True,
                        help="Replay (general) dataset directory")
    parser.add_argument("--replay2_dir", type=str, default=None,
                        help="Optional second replay dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for mixed dataset")
    parser.add_argument("--replay_ratio", type=float, default=0.3,
                        help="Ratio of replay data (0.3 = 30%% replay, 70%% domain)")
    parser.add_argument("--replay2_ratio", type=float, default=0.0,
                        help="Ratio of second replay data (0.03 = 3%%)")
    parser.add_argument("--replay_tokens", type=int, default=0,
                        help="Exact replay tokens to add (overrides ratio if > 0)")
    parser.add_argument("--replay2_tokens", type=int, default=0,
                        help="Exact replay2 tokens to add (overrides replay2_ratio if > 0)")
    parser.add_argument("--include_replay_val", action="store_true",
                        help="Also mix replay val shards (default: domain-only val)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shard selection")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner("MyPT", "Tokenized Dataset Mixer")
    
    print(f"\n  Domain dataset:  {args.domain_dir}")
    print(f"  Replay dataset:  {args.replay_dir}")
    if args.replay2_dir:
        print(f"  Replay2 dataset: {args.replay2_dir}")
    print(f"  Output:          {args.output_dir}")
    print(f"  Replay ratio:    {args.replay_ratio:.0%}")
    if args.replay2_dir:
        print(f"  Replay2 ratio:   {args.replay2_ratio:.0%}")
    print(f"  Seed:            {args.seed}")
    
    # Set seed
    random.seed(args.seed)
    
    # Load dataset info
    domain_dir = Path(args.domain_dir)
    replay_dir = Path(args.replay_dir)
    replay2_dir = Path(args.replay2_dir) if args.replay2_dir else None
    output_dir = Path(args.output_dir)
    
    print("\n" + "=" * 60)
    print("  Loading Dataset Information")
    print("=" * 60)
    
    domain_info = get_shard_info(domain_dir)
    replay_info = get_shard_info(replay_dir)
    replay2_info = get_shard_info(replay2_dir) if replay2_dir else None
    
    domain_tokens = domain_info["total_tokens"]
    tokens_per_shard = domain_info["tokens_per_shard"]
    
    print(f"\n  Domain:  {domain_tokens:,} tokens, {len(domain_info['train_shards'])} train + {len(domain_info['val_shards'])} val shards")
    print(f"  Replay:  {replay_info['total_tokens']:,} tokens, {len(replay_info['train_shards'])} train + {len(replay_info['val_shards'])} val shards")
    if replay2_info:
        print(f"  Replay2: {replay2_info['total_tokens']:,} tokens, {len(replay2_info['train_shards'])} train + {len(replay2_info['val_shards'])} val shards")
    
    # Calculate total replay ratio (sum of both replays)
    total_replay_ratio = args.replay_ratio + (args.replay2_ratio if replay2_dir else 0)
    
    # Calculate replay shards needed
    if args.replay_tokens > 0:
        replay_tokens_target = args.replay_tokens
    else:
        # domain_tokens is (1 - total_replay_ratio) of total
        replay_tokens_target = int(domain_tokens * args.replay_ratio / (1 - total_replay_ratio))
    
    replay_shards_needed = max(1, replay_tokens_target // tokens_per_shard)
    
    # Limit to available shards
    available_replay_train = len(replay_info['train_shards'])
    if replay_shards_needed > available_replay_train:
        print(f"\n  [WARN] Requested {replay_shards_needed} replay shards but only {available_replay_train} available")
        replay_shards_needed = available_replay_train
    
    actual_replay_tokens = replay_shards_needed * tokens_per_shard
    
    # Calculate replay2 shards if applicable
    replay2_shards_needed = 0
    actual_replay2_tokens = 0
    if replay2_dir and replay2_info and args.replay2_ratio > 0:
        if args.replay2_tokens > 0:
            replay2_tokens_target = args.replay2_tokens
        else:
            replay2_tokens_target = int(domain_tokens * args.replay2_ratio / (1 - total_replay_ratio))
        
        replay2_shards_needed = max(1, replay2_tokens_target // tokens_per_shard)
        
        available_replay2_train = len(replay2_info['train_shards'])
        if replay2_shards_needed > available_replay2_train:
            print(f"\n  [WARN] Requested {replay2_shards_needed} replay2 shards but only {available_replay2_train} available")
            replay2_shards_needed = available_replay2_train
        
        actual_replay2_tokens = replay2_shards_needed * tokens_per_shard
    
    total_tokens = domain_tokens + actual_replay_tokens + actual_replay2_tokens
    actual_ratio = actual_replay_tokens / total_tokens
    actual_ratio2 = actual_replay2_tokens / total_tokens if actual_replay2_tokens > 0 else 0
    domain_ratio = domain_tokens / total_tokens
    
    print(f"\n  Replay plan:  {replay_shards_needed} shards (~{actual_replay_tokens:,} tokens)")
    if replay2_shards_needed > 0:
        print(f"  Replay2 plan: {replay2_shards_needed} shards (~{actual_replay2_tokens:,} tokens)")
    print(f"  Total mixed:  {total_tokens:,} tokens ({domain_ratio:.1%} domain, {actual_ratio:.1%} replay" + 
          (f", {actual_ratio2:.1%} replay2)" if actual_ratio2 > 0 else ")"))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Copying Domain Shards")
    print("=" * 60)
    
    # Copy domain train shards
    domain_train_copied = 0
    for shard in domain_info['train_shards']:
        target = output_dir / "train" / shard.name
        shutil.copy2(shard, target)
        domain_train_copied += 1
    print(f"  Copied {domain_train_copied} domain train shards")
    
    # Copy domain val shards
    domain_val_copied = 0
    for shard in domain_info['val_shards']:
        target = output_dir / "val" / shard.name
        shutil.copy2(shard, target)
        domain_val_copied += 1
    print(f"  Copied {domain_val_copied} domain val shards")
    
    print("\n" + "=" * 60)
    print("  Adding Replay Shards")
    print("=" * 60)
    
    # --- Replay 1 ---
    replay_train_selected = random.sample(replay_info['train_shards'], replay_shards_needed)
    
    # Find highest existing shard number
    existing_train = list((output_dir / "train").glob("*.bin"))
    if existing_train:
        max_shard_num = max(int(s.stem.split('_')[-1]) for s in existing_train)
    else:
        max_shard_num = -1
    
    # Copy replay shards with new numbering
    replay_train_copied = 0
    for i, shard in enumerate(replay_train_selected):
        new_num = max_shard_num + 1 + i
        target = output_dir / "train" / f"shard_{new_num:05d}.bin"
        shutil.copy2(shard, target)
        replay_train_copied += 1
    print(f"  Added {replay_train_copied} replay train shards")
    
    # --- Replay 2 ---
    replay2_train_copied = 0
    if replay2_info and replay2_shards_needed > 0:
        replay2_train_selected = random.sample(replay2_info['train_shards'], replay2_shards_needed)
        
        # Update max shard number after replay 1
        existing_train = list((output_dir / "train").glob("*.bin"))
        if existing_train:
            max_shard_num = max(int(s.stem.split('_')[-1]) for s in existing_train)
        else:
            max_shard_num = -1
        
        for i, shard in enumerate(replay2_train_selected):
            new_num = max_shard_num + 1 + i
            target = output_dir / "train" / f"shard_{new_num:05d}.bin"
            shutil.copy2(shard, target)
            replay2_train_copied += 1
        print(f"  Added {replay2_train_copied} replay2 train shards")
    
    # Optionally copy replay val shards
    replay_val_copied = 0
    replay2_val_copied = 0
    if args.include_replay_val and replay_info['val_shards']:
        # Add a few replay val shards proportionally
        replay_val_needed = max(1, int(len(replay_info['val_shards']) * args.replay_ratio))
        replay_val_selected = random.sample(replay_info['val_shards'], 
                                            min(replay_val_needed, len(replay_info['val_shards'])))
        
        existing_val = list((output_dir / "val").glob("*.bin"))
        if existing_val:
            max_val_num = max(int(s.stem.split('_')[-1]) for s in existing_val)
        else:
            max_val_num = -1
        
        for i, shard in enumerate(replay_val_selected):
            new_num = max_val_num + 1 + i
            target = output_dir / "val" / f"shard_{new_num:05d}.bin"
            shutil.copy2(shard, target)
            replay_val_copied += 1
        print(f"  Added {replay_val_copied} replay val shards")
        
        # Also add replay2 val shards if requested
        if replay2_info and args.replay2_ratio > 0 and replay2_info['val_shards']:
            replay2_val_needed = max(1, int(len(replay2_info['val_shards']) * args.replay2_ratio))
            replay2_val_selected = random.sample(replay2_info['val_shards'],
                                                 min(replay2_val_needed, len(replay2_info['val_shards'])))
            
            existing_val = list((output_dir / "val").glob("*.bin"))
            if existing_val:
                max_val_num = max(int(s.stem.split('_')[-1]) for s in existing_val)
            else:
                max_val_num = -1
            
            for i, shard in enumerate(replay2_val_selected):
                new_num = max_val_num + 1 + i
                target = output_dir / "val" / f"shard_{new_num:05d}.bin"
                shutil.copy2(shard, target)
                replay2_val_copied += 1
            print(f"  Added {replay2_val_copied} replay2 val shards")
    else:
        print("  Keeping domain-only val (recommended for domain eval)")
    
    # Copy tokenizer state
    tokenizer_state = domain_dir / "tokenizer_state.json"
    if tokenizer_state.exists():
        shutil.copy2(tokenizer_state, output_dir / "tokenizer_state.json")
        print("  Copied tokenizer_state.json")
    
    print("\n" + "=" * 60)
    print("  Writing Metadata")
    print("=" * 60)
    
    # Create combined metadata
    final_train_shards = len(list((output_dir / "train").glob("*.bin")))
    final_val_shards = len(list((output_dir / "val").glob("*.bin")))
    final_total_tokens = (final_train_shards + final_val_shards) * tokens_per_shard
    
    mix_info = {
        "domain_source": str(domain_dir),
        "domain_tokens": domain_tokens,
        "domain_train_shards": domain_train_copied,
        "domain_val_shards": domain_val_copied,
        "replay_source": str(replay_dir),
        "replay_tokens": actual_replay_tokens,
        "replay_train_shards": replay_train_copied,
        "replay_val_shards": replay_val_copied,
        "replay_ratio_target": args.replay_ratio,
        "replay_ratio_actual": actual_ratio,
        "seed": args.seed,
        "mixed_at": datetime.now().isoformat(),
    }
    
    if replay2_dir and actual_replay2_tokens > 0:
        mix_info.update({
            "replay2_source": str(replay2_dir),
            "replay2_tokens": actual_replay2_tokens,
            "replay2_train_shards": replay2_train_copied,
            "replay2_val_shards": replay2_val_copied,
            "replay2_ratio_target": args.replay2_ratio,
            "replay2_ratio_actual": actual_ratio2,
        })
    
    metadata = {
        "total_tokens": final_total_tokens,
        "total_tokens_written": final_total_tokens,
        "total_shards": final_train_shards + final_val_shards,
        "train_shards": final_train_shards,
        "val_shards": final_val_shards,
        "tokens_per_shard": tokens_per_shard,
        "val_fraction": final_val_shards / (final_train_shards + final_val_shards),
        "mix_info": mix_info,
        "sources": {
            **domain_info["metadata"].get("sources", {}),
        },
    }
    
    # Add replay source info
    if "sources" in replay_info["metadata"]:
        for k, v in replay_info["metadata"]["sources"].items():
            metadata["sources"][f"replay_{k}"] = v
    
    # Add replay2 source info
    if replay2_info and "sources" in replay2_info["metadata"]:
        for k, v in replay2_info["metadata"]["sources"].items():
            metadata["sources"][f"replay2_{k}"] = v
    
    with open(output_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Written dataset_metadata.json")
    
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Total tokens:     {final_total_tokens:,}")
    print(f"  Train shards:     {final_train_shards}")
    print(f"  Val shards:       {final_val_shards}")
    print(f"  Domain ratio:     {domain_ratio:.1%}")
    print(f"  Replay ratio:     {actual_ratio:.1%}")
    if actual_ratio2 > 0:
        print(f"  Replay2 ratio:    {actual_ratio2:.1%}")
    
    print(f"\n  To train with this dataset:")
    print(f"    python train.py \\")
    print(f"      --config configs/pretrain/750M_1024_domain_adapt.json \\")
    print(f"      --dataset_dir {output_dir} \\")
    print(f"      --init_from_model checkpoints/your_base_model \\")
    print(f"      --model_name domain_mixed")
    
    print("\n  Done!\n")


if __name__ == "__main__":
    main()

