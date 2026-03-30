#!/usr/bin/env python3
"""
Mix SFT JSONL Datasets with Sampling Ratios

Combines multiple SFT JSONL files in two different semantics (see --output_blend):

1) **Source-relative (default)** — `path:RATIO` or `--weights` with `--output_blend` off:
   RATIO is a fraction **of that file's row count** (not of the output):
   - 1.0 = take all episodes from that file
   - 0.2 = sample 20% **of that source**
   - 2.0 = duplicate (upsample) that source
   Output size is the **sum** of per-source draws; composition depends on file sizes.

2) **Output-relative** — `--output_blend` + `--target_size N` + `--weights w1 w2 ...`:
   Weights are **only relative** and normalized so each source contributes a fixed
   **share of N output rows** (e.g. weights 1 0.3 0.15 ≈ 68.9% / 20.7% / 10.3% of N).
   Use this when you want "20% of the mix is source B", not "20% of B's rows".

Output:
    Combined JSONL file ready for prepare_chat_sft.py
"""

import argparse
import json
import random
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.banner import print_banner
from core.dataset_lineage import iso_now, merge_lineage, write_lineage_sidecar


def get_user_message(conv: Dict[str, Any]) -> str:
    """Extract user message from conversation."""
    for msg in conv.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_payload_from_text(user_msg: str) -> Optional[str]:
    """Extract payload from user message text using same heuristics as prepare_chat_sft.py."""
    quote_match = re.search(r'"([^"]+)"', user_msg)
    if quote_match:
        return quote_match.group(1)
    colon_match = re.search(r':\s*(.+)$', user_msg)
    if colon_match:
        return colon_match.group(1).strip()
    return None


def extract_payload_from_conv(conv: Dict[str, Any]) -> Optional[str]:
    """Extract payload from a conversation."""
    meta = conv.get("_meta", {})
    if "payload" in meta:
        return meta["payload"]
    return extract_payload_from_text(get_user_message(conv))


def get_template_signature(user_msg: str, payload: Optional[str]) -> str:
    """Replace payload with {PAYLOAD} to get template signature."""
    if payload and payload in user_msg:
        sig = user_msg.replace(f'"{payload}"', '"{PAYLOAD}"')
        sig = sig.replace(payload, "{PAYLOAD}")
        return " ".join(sig.split())
    return " ".join(user_msg.split())


def get_pair_key(conv: Dict[str, Any], payload: Optional[str]) -> Optional[str]:
    """Extract (operator,payload) pair key."""
    if not payload:
        return None
    op = str(conv.get("_meta", {}).get("operator", "")).strip().upper()
    if not op:
        return None
    return f"{op}::{payload}"


def build_disjoint_sets(convs: List[Dict[str, Any]], keys: Set[str]) -> Dict[str, Set[str]]:
    payloads: Set[str] = set()
    templates: Set[str] = set()
    pairs: Set[str] = set()
    for conv in convs:
        payload = extract_payload_from_conv(conv)
        user_msg = get_user_message(conv)
        if "payload" in keys and payload:
            payloads.add(payload)
        if "template" in keys:
            templates.add(get_template_signature(user_msg, payload))
        if "pair" in keys:
            pair = get_pair_key(conv, payload)
            if pair:
                pairs.add(pair)
    return {"payload": payloads, "template": templates, "pair": pairs}


def filter_disjoint_candidates(
    candidates: List[Dict[str, Any]],
    train_sets: Dict[str, Set[str]],
    keys: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    stats = {"payload_overlap": 0, "template_overlap": 0, "pair_overlap": 0}
    for conv in candidates:
        payload = extract_payload_from_conv(conv)
        user_msg = get_user_message(conv)
        template = get_template_signature(user_msg, payload)
        pair = get_pair_key(conv, payload)
        blocked = False
        if "payload" in keys and payload and payload in train_sets["payload"]:
            stats["payload_overlap"] += 1
            blocked = True
        if not blocked and "template" in keys and template in train_sets["template"]:
            stats["template_overlap"] += 1
            blocked = True
        if not blocked and "pair" in keys and pair and pair in train_sets["pair"]:
            stats["pair_overlap"] += 1
            blocked = True
        if not blocked:
            kept.append(conv)
    return kept, stats


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


def allocate_output_counts(target_size: int, weights: List[float]) -> List[int]:
    """
    Split target_size across sources proportionally to weights (normalized).
    Adjusts for rounding so counts sum exactly to target_size.
    """
    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if not weights or any(w < 0 for w in weights):
        raise ValueError("weights must be non-empty and non-negative")
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("sum(weights) must be positive")
    normed = [w / wsum for w in weights]
    raw = [target_size * p for p in normed]
    counts = [int(x) for x in raw]
    drift = target_size - sum(counts)
    if drift > 0:
        fracs = sorted(enumerate([raw[i] - counts[i] for i in range(len(weights))]), key=lambda t: -t[1])
        for j in range(drift):
            counts[fracs[j % len(fracs)][0]] += 1
    return counts


def sample_row_target(episodes: list, n: int, seed: int) -> list:
    """
    Draw exactly n rows from episodes. If n > len(episodes), upsample with replacement.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    if n <= len(episodes):
        return rng.sample(episodes, n)
    out = episodes.copy()
    rng.shuffle(out)
    deficit = n - len(out)
    out.extend(rng.choices(episodes, k=deficit))
    return out


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
    
    # Multiple replay sources (each ratio is fraction OF THAT FILE)
    python scripts/mix_sft_jsonl.py \\
        --inputs run1.jsonl:0.1 run2.jsonl:0.2 run3.jsonl:1.0 \\
        --output mixed.jsonl --shuffle

    # Target-sized mix: weights are shares of OUTPUT (normalized), not % of each file
    python scripts/mix_sft_jsonl.py \\
        --output_blend --target_size 100000 \\
        --inputs data/a.jsonl data/b.jsonl data/c.jsonl \\
        --weights 1.0 0.3 0.15 \\
        --output mixed.jsonl --shuffle
"""
    )
    
    parser.add_argument("--inputs", nargs='+', required=True,
                        help="Input files. Can include ratios inline (data.jsonl:0.2) or use --weights")
    parser.add_argument("--weights", nargs='+', type=float, default=None,
                        help="Per-input weights. "
                             "Default mode: same as inline ratios — fraction of EACH SOURCE sampled. "
                             "With --output_blend: relative share of --target_size output rows (normalized).")
    parser.add_argument(
        "--output_blend",
        action="store_true",
        help="Interpret --weights as output composition: each source contributes a fixed fraction "
             "of --target_size rows (weights normalized). Requires --target_size. "
             "Inputs should be plain paths (ratios in path:ratio are ignored; use --weights only).",
    )
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle combined episodes (recommended)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling and shuffling")
    parser.add_argument("--exclude_from_train", type=str, default=None,
                        help="Path to train JSONL. If set, drop candidates overlapping train on selected keys.")
    parser.add_argument("--disjoint_keys", type=str, default="payload,template",
                        help="Comma-separated overlap keys: payload,template,pair")
    parser.add_argument("--target_size", type=int, default=None,
                        help="Optional final cap after mixing/filtering. Deterministic sample if larger.")
    
    args = parser.parse_args()

    if args.output_blend:
        if args.weights is None or len(args.weights) != len(args.inputs):
            print("\n  ERROR: --output_blend requires --weights with one float per --inputs")
            sys.exit(1)
        if args.target_size is None or args.target_size <= 0:
            print("\n  ERROR: --output_blend requires a positive --target_size (total output rows)")
            sys.exit(1)
    
    print_banner("MyPT", "SFT JSONL Mixer")
    
    print(f"\n  Output: {args.output}")
    print(f"  Shuffle: {args.shuffle}")
    print(f"  Seed: {args.seed}")
    if args.exclude_from_train:
        print(f"  Disjoint against train: {args.exclude_from_train}")
        print(f"  Disjoint keys: {args.disjoint_keys}")
    if args.target_size is not None:
        print(f"  Target size: {args.target_size}")
    
    # Parse input specifications -> list of (Path, ratio_or_weight)
    input_specs = []
    
    if args.output_blend:
        counts_preview = allocate_output_counts(args.target_size, list(args.weights))
        for i, spec in enumerate(args.inputs):
            path, _ = parse_input_spec(spec)
            input_specs.append((Path(path), float(counts_preview[i])))
        print(f"\n  Mode: OUTPUT_BLEND (each source target row count -> second argument is integer n)")
    elif args.weights is not None:
        if len(args.weights) != len(args.inputs):
            print(f"\n  ERROR: --weights has {len(args.weights)} values but --inputs has {len(args.inputs)} files")
            print("         These must match.")
            sys.exit(1)
        # Use explicit weights as *source* sampling fractions (see sample_episodes)
        for i, spec in enumerate(args.inputs):
            path = spec.rsplit(':', 1)[0] if ':' in spec and spec.rsplit(':', 1)[1].replace('.', '').isdigit() else spec
            input_specs.append((Path(path), args.weights[i]))
    else:
        for spec in args.inputs:
            path, ratio = parse_input_spec(spec)
            input_specs.append((Path(path), ratio))
    
    print(f"\n  Inputs ({len(input_specs)}):")
    if args.output_blend:
        wsum = sum(args.weights)
        for (path, nrows), w in zip(input_specs, args.weights):
            pct = 100.0 * (w / wsum) if wsum > 0 else 0
            print(f"    - {path} -> {int(nrows):,} rows ({pct:.1f}% of output weight)")
    else:
        for path, ratio in input_specs:
            print(f"    - {path} @ source_sample={ratio} ({'fraction of file' if ratio < 1.0 else 'multiplier'})")
    
    # Load and sample from each input
    print("\n" + "=" * 60)
    print("  Loading and Sampling")
    print("=" * 60)
    
    all_episodes = []
    source_stats = []
    
    for idx, (path, ratio) in enumerate(input_specs):
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
        
        if args.output_blend:
            n_target = int(ratio)
            sampled = sample_row_target(episodes, n_target, args.seed + idx * 17)
            sampled_count = len(sampled)
            print(f"    Drawn rows for output blend: {sampled_count} (quota {n_target})")
            if n_target > original_count:
                print(
                    f"    [INFO] Quota {n_target} > source size {original_count}; "
                    f"upsampled with replacement to hit blend target."
                )
        else:
            sampled = sample_episodes(episodes, ratio, args.seed)
            sampled_count = len(sampled)
            print(f"    After source_fraction={ratio} sampling: {sampled_count}")
        
        # Tag episodes with source for debugging
        source_name = path.stem
        for ep in sampled:
            if 'mix_source' not in ep:
                ep['mix_source'] = source_name
        
        all_episodes.extend(sampled)
        stat: Dict[str, Any] = {
            'source': str(path),
            'original': original_count,
            'sampled': sampled_count,
        }
        if args.output_blend:
            stat['blend_weight'] = args.weights[idx]
            stat['output_rows_quota'] = int(ratio)
        else:
            stat['source_sample_ratio'] = ratio
        source_stats.append(stat)
    
    total_episodes = len(all_episodes)
    print(f"\n  Total combined: {total_episodes} episodes")

    overlap_stats = None
    if args.exclude_from_train:
        key_set = {k.strip() for k in args.disjoint_keys.split(",") if k.strip()}
        valid_keys = {"payload", "template", "pair"}
        invalid = key_set - valid_keys
        if invalid:
            print(f"\n  ERROR: Invalid disjoint keys: {sorted(invalid)}. Valid: {sorted(valid_keys)}")
            sys.exit(1)

        train_path = Path(args.exclude_from_train)
        if not train_path.exists():
            train_path = PROJECT_ROOT / train_path
        if not train_path.exists():
            print(f"\n  ERROR: exclude_from_train file not found: {train_path}")
            sys.exit(1)

        print(f"\n  Loading train reference for disjoint filter: {train_path}")
        train_convs = load_jsonl(train_path)
        train_sets = build_disjoint_sets(train_convs, key_set)

        pre_count = len(all_episodes)
        all_episodes, overlap_stats = filter_disjoint_candidates(all_episodes, train_sets, key_set)
        post_count = len(all_episodes)
        total_episodes = post_count
        print(f"  Disjoint filtering: {pre_count} -> {post_count} episodes")
        print(f"    blocked payload overlap:  {overlap_stats['payload_overlap']}")
        print(f"    blocked template overlap: {overlap_stats['template_overlap']}")
        print(f"    blocked pair overlap:     {overlap_stats['pair_overlap']}")

    if args.target_size is not None and len(all_episodes) > args.target_size:
        rng_cap = random.Random(args.seed + 17)
        all_episodes = rng_cap.sample(all_episodes, args.target_size)
        total_episodes = len(all_episodes)
        print(f"  Applied target_size cap: {total_episodes} episodes")
    
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
                "output_blend": args.output_blend,
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
        'exclude_from_train': args.exclude_from_train,
        'disjoint_keys': args.disjoint_keys,
        'target_size': args.target_size,
        'disjoint_filter_stats': overlap_stats,
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
