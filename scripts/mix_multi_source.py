#!/usr/bin/env python
"""
Mix Multiple Tokenized Sources into a Unified Dataset

Takes N pre-tokenized binary shard datasets and combines them into a single
unified dataset at specified proportions. Supports both subsampling (taking
fewer shards) and upsampling (duplicating shards) to hit target token counts.

This is designed for the from-scratch training mix where we need 10 sources
at precise proportions:
- Some sources need subsampling (wiki: 1.5B available, need 0.9B)
- Some sources need upsampling (domain: 161M available, need 400M)
- Most sources are used as-is

Usage:
    python scripts/mix_multi_source.py \
        --config data/sources/unified_from_scratch.json \
        --output_dir data/unified_6B

    # Dry run to check proportions
    python scripts/mix_multi_source.py \
        --config data/sources/unified_from_scratch.json \
        --output_dir data/unified_6B \
        --dry_run
"""

import argparse
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.banner import print_banner


def get_dataset_info(dataset_dir: Path) -> dict:
    """Read dataset metadata and list shards."""
    metadata_path = dataset_dir / "dataset_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"No dataset_metadata.json in {dataset_dir}")
    
    with open(metadata_path, "r") as f:
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
        "train_tokens": len(train_shards) * metadata.get("tokens_per_shard", 10_000_000),
        "val_tokens": len(val_shards) * metadata.get("tokens_per_shard", 10_000_000),
    }


def compute_shard_plan(sources: Dict[str, dict], total_target: int,
                       tokens_per_shard: int, seed: int = 42) -> Dict[str, dict]:
    """
    Compute how many shards to take/duplicate from each source.
    
    Returns a plan dict with:
    - train_shards_needed: number of shards for train
    - val_shards_needed: number of shards for val
    - actual_tokens: tokens that will actually be in the mix
    - method: 'subsample', 'use_all', or 'upsample'
    """
    random.seed(seed)
    plan = {}
    
    for name, src in sources.items():
        target_tokens = src["target_tokens"]
        info = src["info"]
        
        available_train = len(info["train_shards"])
        available_val = len(info["val_shards"])
        
        # How many train shards needed?
        # Reserve ~5% for val (matching original datasets)
        train_target = int(target_tokens * 0.95)
        val_target = int(target_tokens * 0.05)
        
        train_shards_needed = max(1, train_target // tokens_per_shard)
        val_shards_needed = max(1, val_target // tokens_per_shard)
        
        # Determine method
        if train_shards_needed <= available_train:
            method = "subsample" if train_shards_needed < available_train else "use_all"
        else:
            method = "upsample"
        
        # Clamp val shards
        val_shards_needed = min(val_shards_needed, available_val)
        if val_shards_needed == 0 and available_val > 0:
            val_shards_needed = 1
        
        actual_tokens = (train_shards_needed + val_shards_needed) * tokens_per_shard
        
        plan[name] = {
            "target_tokens": target_tokens,
            "actual_tokens": actual_tokens,
            "train_shards_needed": train_shards_needed,
            "val_shards_needed": val_shards_needed,
            "available_train": available_train,
            "available_val": available_val,
            "method": method,
            "upsample_factor": (train_shards_needed / available_train 
                               if available_train > 0 else 0),
        }
    
    return plan


def execute_plan(plan: Dict[str, dict], sources: Dict[str, dict],
                 output_dir: Path, tokens_per_shard: int, seed: int = 42,
                 dry_run: bool = False) -> Dict[str, int]:
    """
    Execute the mixing plan: copy/duplicate shards into output directory.
    
    Returns stats dict.
    """
    random.seed(seed)
    
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    if not dry_run:
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
    
    train_shard_idx = 0
    val_shard_idx = 0
    total_train_tokens = 0
    total_val_tokens = 0
    
    for name, p in plan.items():
        info = sources[name]["info"]
        
        print(f"\n  {name}: {p['method']} "
              f"({p['available_train']} -> {p['train_shards_needed']} train, "
              f"{p['available_val']} -> {p['val_shards_needed']} val)")
        
        # --- Train shards ---
        if p["method"] == "subsample":
            # Random sample without replacement
            selected = random.sample(info["train_shards"], p["train_shards_needed"])
        elif p["method"] == "use_all":
            selected = list(info["train_shards"])
        else:  # upsample
            # Take all shards, then duplicate some to reach target
            full_copies = p["train_shards_needed"] // p["available_train"]
            remainder = p["train_shards_needed"] % p["available_train"]
            
            selected = []
            for _ in range(full_copies):
                selected.extend(info["train_shards"])
            if remainder > 0:
                selected.extend(random.sample(info["train_shards"], remainder))
            
            print(f"    Upsampling: {full_copies}x full + {remainder} extra "
                  f"= {len(selected)} shards")
        
        # Copy train shards
        for shard_path in selected:
            target = train_dir / f"shard_{train_shard_idx:05d}.bin"
            if not dry_run:
                shutil.copy2(shard_path, target)
            train_shard_idx += 1
            total_train_tokens += tokens_per_shard
        
        # --- Val shards ---
        if p["val_shards_needed"] > 0 and info["val_shards"]:
            val_selected = info["val_shards"][:p["val_shards_needed"]]
            for shard_path in val_selected:
                target = val_dir / f"shard_{val_shard_idx:05d}.bin"
                if not dry_run:
                    shutil.copy2(shard_path, target)
                val_shard_idx += 1
                total_val_tokens += tokens_per_shard
        
        print(f"    Copied: {len(selected)} train + "
              f"{min(p['val_shards_needed'], len(info.get('val_shards', [])))} val shards")
    
    return {
        "train_shards": train_shard_idx,
        "val_shards": val_shard_idx,
        "total_train_tokens": total_train_tokens,
        "total_val_tokens": total_val_tokens,
        "total_tokens": total_train_tokens + total_val_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mix multiple tokenized sources into a unified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config file with source definitions and targets")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for unified dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shard selection")
    parser.add_argument("--dry_run", action="store_true",
                        help="Plan only, don't copy files")
    
    args = parser.parse_args()
    
    print_banner("MyPT", "Multi-Source Dataset Mixer")
    
    # Load config
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    output_dir = Path(args.output_dir)
    
    print(f"\n  Config: {config_path}")
    print(f"  Output: {output_dir}")
    print(f"  Seed:   {args.seed}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    
    # Load source info
    print("\n" + "=" * 70)
    print("  Loading Source Information")
    print("=" * 70)
    
    sources = {}
    total_target = 0
    
    for name, src_config in config["sources"].items():
        src_dir = Path(src_config["directory"])
        target_tokens = src_config["target_tokens"]
        
        if not src_dir.exists():
            print(f"\n  [WARN] Source directory not found: {src_dir}")
            print(f"         Skipping {name}")
            continue
        
        try:
            info = get_dataset_info(src_dir)
        except FileNotFoundError as e:
            print(f"\n  [WARN] {e}")
            print(f"         Skipping {name}")
            continue
        
        sources[name] = {
            "config": src_config,
            "info": info,
            "target_tokens": target_tokens,
        }
        total_target += target_tokens
        
        print(f"\n  {name}:")
        print(f"    Directory:   {src_dir}")
        print(f"    Available:   {info['total_tokens']:,} tokens "
              f"({len(info['train_shards'])} train + {len(info['val_shards'])} val shards)")
        print(f"    Target:      {target_tokens:,} tokens "
              f"({target_tokens / total_target * 100 if total_target > 0 else 0:.1f}%)")
    
    if not sources:
        print("\n  ERROR: No valid sources found!")
        sys.exit(1)
    
    tokens_per_shard = config.get("tokens_per_shard", 10_000_000)
    
    print(f"\n  Total target: {total_target:,} tokens")
    print(f"  Tokens/shard: {tokens_per_shard:,}")
    
    # Compute plan
    print("\n" + "=" * 70)
    print("  Mixing Plan")
    print("=" * 70)
    
    plan = compute_shard_plan(sources, total_target, tokens_per_shard, args.seed)
    
    actual_total = 0
    for name, p in plan.items():
        pct = p["actual_tokens"] / sum(pp["actual_tokens"] for pp in plan.values()) * 100
        print(f"\n  {name}:")
        print(f"    Method:    {p['method']}"
              + (f" ({p['upsample_factor']:.1f}x)" if p['method'] == 'upsample' else ""))
        print(f"    Target:    {p['target_tokens']:,} tokens")
        print(f"    Actual:    {p['actual_tokens']:,} tokens ({pct:.1f}%)")
        print(f"    Train:     {p['train_shards_needed']} shards "
              f"(from {p['available_train']} available)")
        print(f"    Val:       {p['val_shards_needed']} shards "
              f"(from {p['available_val']} available)")
        actual_total += p["actual_tokens"]
    
    print(f"\n  Actual total: {actual_total:,} tokens")
    
    if args.dry_run:
        print("\n  *** DRY RUN - no files copied ***")
        print("\n  Done!\n")
        return
    
    # Execute
    print("\n" + "=" * 70)
    print("  Executing Mix")
    print("=" * 70)
    
    stats = execute_plan(plan, sources, output_dir, tokens_per_shard, args.seed)
    
    # Copy tokenizer state from first source
    for name, src in sources.items():
        tok_state = Path(src["config"]["directory"]) / "tokenizer_state.json"
        if tok_state.exists():
            shutil.copy2(tok_state, output_dir / "tokenizer_state.json")
            print(f"\n  Tokenizer state copied from {name}")
            break
    
    # Write metadata
    metadata = {
        "total_tokens": stats["total_tokens"],
        "total_tokens_written": stats["total_tokens"],
        "total_shards": stats["train_shards"] + stats["val_shards"],
        "train_shards": stats["train_shards"],
        "val_shards": stats["val_shards"],
        "tokens_per_shard": tokens_per_shard,
        "val_fraction": stats["total_val_tokens"] / stats["total_tokens"] if stats["total_tokens"] > 0 else 0,
        "mix_info": {
            "config_file": str(config_path),
            "seed": args.seed,
            "mixed_at": datetime.now().isoformat(),
            "sources": {
                name: {
                    "directory": sources[name]["config"]["directory"],
                    "target_tokens": plan[name]["target_tokens"],
                    "actual_tokens": plan[name]["actual_tokens"],
                    "method": plan[name]["method"],
                    "train_shards": plan[name]["train_shards_needed"],
                    "val_shards": plan[name]["val_shards_needed"],
                }
                for name in plan
            },
        },
    }
    
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Total tokens:     {stats['total_tokens']:,}")
    print(f"  Train shards:     {stats['train_shards']}")
    print(f"  Val shards:       {stats['val_shards']}")
    
    print(f"\n  Composition:")
    for name, p in plan.items():
        pct = p["actual_tokens"] / stats["total_tokens"] * 100
        print(f"    {name:30s} {p['actual_tokens']:>15,} tokens  ({pct:5.1f}%)")
    
    print(f"\n  Done!\n")


if __name__ == "__main__":
    main()
