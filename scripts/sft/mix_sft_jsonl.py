#!/usr/bin/env python3
"""
Mix SFT JSONL Datasets with Sampling Ratios

Combines multiple SFT JSONL files with configurable sampling ratios.
Use this for replay strategies where you want X% of previous stage data
mixed with 100% of current stage data.

Usage:
    # Mix 20% of Run1 with 100% of Run2
    python scripts/mix_sft_jsonl.py \
        --inputs data/sft_format_lock/mypt_format_lock_v1.jsonl:0.2 \
                 data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl:1.0 \
        --output data/sft_run2_mixed/mypt_run2_with_replay.jsonl
    
    # Mix multiple sources with different ratios
    python scripts/mix_sft_jsonl.py \
        --inputs run1.jsonl:0.1 run2.jsonl:0.2 run3.jsonl:1.0 \
        --output mixed.jsonl \
        --shuffle

Input format:
    path/to/file.jsonl:RATIO
    
    RATIO is a float:
    - 1.0 = use 100% of episodes
    - 0.2 = randomly sample 20% of episodes
    - 2.0 = duplicate all episodes 2x (for upweighting)

Output:
    Combined JSONL file ready for prepare_chat_sft.py
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.banner import print_banner
from core.dataset_lineage import iso_now, merge_lineage, write_lineage_sidecar


def parse_input_spec(spec: str) -> tuple:
    """Parse 'path:ratio' specification."""
    if ':' not in spec:
        # Default to 1.0 if no ratio specified
        return spec, 1.0
    
    parts = spec.rsplit(':', 1)
    path = parts[0]
    try:
        ratio = float(parts[1])
    except ValueError:
        raise ValueError(f"Invalid ratio in '{spec}'. Expected format: path/to/file.jsonl:0.2")
    
    return path, ratio


def load_jsonl(filepath: Path) -> list:
    """Load all episodes from a JSONL file."""
    episodes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
                episodes.append(episode)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping invalid JSON at line {line_num}: {e}")
    return episodes


def sample_episodes(episodes: list, ratio: float, seed: int) -> list:
    """Sample episodes according to ratio."""
    if ratio >= 1.0:
        if ratio == 1.0:
            return episodes.copy()
        else:
            # ratio > 1.0 means duplication
            result = []
            full_copies = int(ratio)
            partial = ratio - full_copies
            
            for _ in range(full_copies):
                result.extend(episodes)
            
            if partial > 0:
                rng = random.Random(seed)
                n_extra = int(len(episodes) * partial)
                result.extend(rng.sample(episodes, n_extra))
            
            return result
    else:
        # ratio < 1.0 means sampling
        rng = random.Random(seed)
        n_samples = max(1, int(len(episodes) * ratio))
        return rng.sample(episodes, n_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Mix SFT JSONL datasets with sampling ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 20% replay from Run1 + 100% of Run2
    python scripts/mix_sft_jsonl.py \\
        --inputs run1.jsonl:0.2 run2.jsonl:1.0 \\
        --output mixed.jsonl
    
    # Multiple replay sources
    python scripts/mix_sft_jsonl.py \\
        --inputs run1.jsonl:0.1 run2.jsonl:0.2 run3.jsonl:1.0 \\
        --output mixed.jsonl --shuffle
"""
    )
    
    parser.add_argument("--inputs", nargs='+', required=True,
                        help="Input files. Can include ratios inline (data.jsonl:0.2) or use --weights")
    parser.add_argument("--weights", nargs='+', type=float, default=None,
                        help="Explicit weights for each input (e.g., --weights 0.7 0.2 0.1). "
                             "If provided, must match number of inputs. Overrides inline ratios.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle combined episodes (recommended)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and shuffling")
    
    args = parser.parse_args()
    
    print_banner("MyPT", "SFT JSONL Mixer")
    
    print(f"\n  Output: {args.output}")
    print(f"  Shuffle: {args.shuffle}")
    print(f"  Seed: {args.seed}")
    
    # Parse input specifications
    input_specs = []
    
    # Check if --weights is provided
    if args.weights is not None:
        if len(args.weights) != len(args.inputs):
            print(f"\n  ERROR: --weights has {len(args.weights)} values but --inputs has {len(args.inputs)} files")
            print("         These must match.")
            sys.exit(1)
        # Use explicit weights, strip any inline ratios from paths
        for i, spec in enumerate(args.inputs):
            path = spec.rsplit(':', 1)[0] if ':' in spec and spec.rsplit(':', 1)[1].replace('.', '').isdigit() else spec
            input_specs.append((Path(path), args.weights[i]))
    else:
        # Parse inline ratios
        for spec in args.inputs:
            path, ratio = parse_input_spec(spec)
            input_specs.append((Path(path), ratio))
    
    print(f"\n  Inputs ({len(input_specs)}):")
    for path, ratio in input_specs:
        print(f"    - {path} @ {ratio:.0%}")
    
    # Load and sample from each input
    print("\n" + "=" * 60)
    print("  Loading and Sampling")
    print("=" * 60)
    
    all_episodes = []
    source_stats = []
    
    for path, ratio in input_specs:
        if not path.exists():
            # Try relative to project root
            path = PROJECT_ROOT / path
        
        if not path.exists():
            print(f"\n  ERROR: File not found: {path}")
            sys.exit(1)
        
        print(f"\n  Loading: {path}")
        episodes = load_jsonl(path)
        original_count = len(episodes)
        print(f"    Original episodes: {original_count}")
        
        sampled = sample_episodes(episodes, ratio, args.seed)
        sampled_count = len(sampled)
        print(f"    After {ratio:.0%} sampling: {sampled_count}")
        
        # Tag episodes with source for debugging
        source_name = path.stem
        for ep in sampled:
            if 'mix_source' not in ep:
                ep['mix_source'] = source_name
        
        all_episodes.extend(sampled)
        source_stats.append({
            'source': str(path),
            'original': original_count,
            'ratio': ratio,
            'sampled': sampled_count,
        })
    
    total_episodes = len(all_episodes)
    print(f"\n  Total combined: {total_episodes} episodes")
    
    # Shuffle if requested
    if args.shuffle:
        print("\n  Shuffling episodes...")
        rng = random.Random(args.seed + 1)  # Different seed for shuffle
        rng.shuffle(all_episodes)
    
    # Write output
    print("\n" + "=" * 60)
    print("  Writing Output")
    print("=" * 60)
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for episode in all_episodes:
            json_line = json.dumps(episode, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"  Written: {output_path}")
    print(f"  Episodes: {total_episodes}")
    
    # Write mix metadata alongside
    lineage_inputs = []
    for stat in source_stats:
        lineage_inputs.append({
            "path": stat["source"],
            "sampled_rows": stat["sampled"],
            "effective_ratio": (stat["sampled"] / total_episodes) if total_episodes > 0 else 0.0,
        })
    lineage = merge_lineage(
        inputs=lineage_inputs,
        output_rows=total_episodes,
        creation_context={
            "timestamp": iso_now(),
            "script": "scripts/sft/mix_sft_jsonl.py",
            "args": {
                "inputs": args.inputs,
                "weights": args.weights,
                "output": str(output_path),
                "shuffle": args.shuffle,
                "seed": args.seed,
            },
        },
    )

    metadata = {
        'created_at': datetime.now().isoformat(),
        'seed': args.seed,
        'shuffled': args.shuffle,
        'total_episodes': total_episodes,
        'sources': source_stats,
        'lineage': lineage,
    }
    
    metadata_path = output_path.with_suffix('.mix_meta.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    lineage_path = write_lineage_sidecar(output_path, lineage)
    print(f"  Lineage: {lineage_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Total episodes: {total_episodes}")
    print(f"\n  Composition:")
    for stat in source_stats:
        pct = stat['sampled'] / total_episodes * 100
        print(f"    {Path(stat['source']).stem}: {stat['sampled']} ({pct:.1f}%)")
    
    print(f"\n  Next step:")
    print(f"    python scripts/prepare_chat_sft.py \\")
    print(f"        --input {output_path} \\")
    print(f"        --output_dir {output_path.parent / (output_path.stem + '_prepared')} \\")
    print(f"        --val_split 0.1")
    
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
