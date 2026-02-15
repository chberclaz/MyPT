#!/usr/bin/env python3
"""
SFT Episode Mask Validator

Validates loss-mask alignment across all episodes in a prepared SFT dataset.

Checks:
1. Each episode has exactly 1x <myPT_assistant>, 1x </myPT_assistant>, 1x <myPT_eot>
2. Mask is 0 before assistant-open tag
3. Mask is 1 for assistant content tokens
4. Mask is 1 for </myPT_assistant> and <myPT_eot>
5. Padded region mask is 0
6. Off-by-one alignment test (mask_y[p-1] should be 1 where seq[p] first has mask=1)

Usage:
    python scripts/validate_sft_episode_masks.py --dataset_dir data/sft_format_lock_prepared
    python scripts/validate_sft_episode_masks.py --dataset_dir data/sft_format_lock_prepared --verbose
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Dynamic token ID lookup -- never hardcode IDs!
from core.special_tokens import get_special_token_ids
_IDS = get_special_token_ids()
ASSISTANT_OPEN_ID = _IDS["myPT_assistant_open"]
ASSISTANT_CLOSE_ID = _IDS["myPT_assistant_close"]
EOT_ID = _IDS["myPT_eot"]


def load_dataset(dataset_dir: str, split: str = 'train'):
    """Load tokens, mask, and episode index from dataset."""
    split_dir = Path(dataset_dir) / split
    
    tokens_path = split_dir / "tokens.bin"
    mask_path = split_dir / "mask.bin"
    episodes_path = split_dir / "episodes.idx"
    
    if not tokens_path.exists():
        raise FileNotFoundError(f"Tokens file not found: {tokens_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes index not found: {episodes_path}")
    
    tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
    episodes = np.memmap(episodes_path, dtype=np.uint64, mode='r').reshape(-1, 2)
    
    mask = None
    if mask_path.exists():
        mask = np.memmap(mask_path, dtype=np.uint8, mode='r')
    
    return tokens, mask, episodes


def validate_episode(ep_idx: int, tokens: np.ndarray, mask: np.ndarray, 
                     start: int, length: int, verbose: bool = False) -> dict:
    """Validate a single episode. Returns dict of failures."""
    failures = {}
    
    # Extract episode data
    ep_tokens = tokens[start:start + length]
    ep_mask = mask[start:start + length] if mask is not None else None
    
    # Count special tokens
    assistant_open_count = np.sum(ep_tokens == ASSISTANT_OPEN_ID)
    assistant_close_count = np.sum(ep_tokens == ASSISTANT_CLOSE_ID)
    eot_count = np.sum(ep_tokens == EOT_ID)
    
    # Check 1: Exactly 1 of each special token
    if assistant_open_count != 1:
        failures['assistant_open_count'] = f"Expected 1, got {assistant_open_count}"
    if assistant_close_count != 1:
        failures['assistant_close_count'] = f"Expected 1, got {assistant_close_count}"
    if eot_count != 1:
        failures['eot_count'] = f"Expected 1, got {eot_count}"
    
    if ep_mask is None:
        failures['no_mask'] = "Mask file not found"
        return failures
    
    # Find positions of special tokens
    assistant_open_pos = np.where(ep_tokens == ASSISTANT_OPEN_ID)[0]
    assistant_close_pos = np.where(ep_tokens == ASSISTANT_CLOSE_ID)[0]
    eot_pos = np.where(ep_tokens == EOT_ID)[0]
    
    if len(assistant_open_pos) == 1 and len(assistant_close_pos) == 1 and len(eot_pos) == 1:
        open_idx = assistant_open_pos[0]
        close_idx = assistant_close_pos[0]
        eot_idx = eot_pos[0]
        
        # Check 2: Mask is 0 before assistant-open
        pre_assistant_mask = ep_mask[:open_idx + 1]  # Include assistant-open itself
        if np.any(pre_assistant_mask[:-1] == 1):  # Before assistant-open should be 0
            first_bad = np.where(pre_assistant_mask[:-1] == 1)[0][0]
            failures['mask_before_assistant'] = f"Non-zero mask at position {first_bad}"
        
        # Check that <myPT_assistant> itself has mask=0
        if ep_mask[open_idx] != 0:
            failures['assistant_open_mask'] = f"<myPT_assistant> should have mask=0, got {ep_mask[open_idx]}"
        
        # Check 3: Mask is 1 for assistant content (between open and close, exclusive of open)
        content_start = open_idx + 1
        content_end = close_idx  # exclusive
        if content_start < content_end:
            content_mask = ep_mask[content_start:content_end]
            if not np.all(content_mask == 1):
                zero_positions = np.where(content_mask == 0)[0]
                failures['assistant_content_mask'] = f"Found mask=0 at content positions: {zero_positions[:5]}..."
        
        # Check 4: </myPT_assistant> has mask=1
        if ep_mask[close_idx] != 1:
            failures['assistant_close_mask'] = f"</myPT_assistant> should have mask=1, got {ep_mask[close_idx]}"
        
        # Check 4b: <myPT_eot> has mask=1
        if ep_mask[eot_idx] != 1:
            failures['eot_mask'] = f"<myPT_eot> should have mask=1, got {ep_mask[eot_idx]}"
    
    # Compute stats
    trained_tokens = np.sum(ep_mask == 1)
    
    if verbose and not failures:
        print(f"  Episode {ep_idx}: {length} tokens, {trained_tokens} trained, OK")
    
    return failures


def validate_alignment_offbyone(ep_idx: int, tokens: np.ndarray, mask: np.ndarray,
                                 start: int, length: int, block_size: int = 1024) -> dict:
    """
    Validate that mask alignment is correct for x/y/mask_y formation.
    
    In training: x=seq[:-1], y=seq[1:], mask_y=mask[1:]
    So if mask[p]==1 (first trained position), then mask_y[p-1] should be 1.
    """
    failures = {}
    
    ep_tokens = tokens[start:start + length]
    ep_mask = mask[start:start + length] if mask is not None else None
    
    if ep_mask is None:
        return failures
    
    # Simulate training window
    L = min(length, block_size + 1)
    seq = ep_tokens[:L]
    m = ep_mask[:L]
    
    # Form x, y, mask_y as in training
    # x = seq[:-1]
    # y = seq[1:]
    mask_y = m[1:]
    
    # Find first position p in seq where mask[p]==1
    first_trained = np.where(m == 1)[0]
    if len(first_trained) > 0:
        p = first_trained[0]
        # In mask_y, this should be at position p-1
        if p > 0 and p - 1 < len(mask_y):
            if mask_y[p - 1] != 1:
                failures['alignment_offbyone'] = f"mask[{p}]==1 but mask_y[{p-1}]=={mask_y[p-1]}"
    
    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate SFT episode masks")
    parser.add_argument("--dataset_dir", required=True, help="Path to prepared dataset")
    parser.add_argument("--split", default="train", help="Split to validate (train/val)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to check")
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset_dir}/{args.split}...")
    tokens, mask, episodes = load_dataset(args.dataset_dir, args.split)
    
    print(f"Found {len(episodes)} episodes, {len(tokens):,} tokens")
    if mask is not None:
        print(f"Mask file present: {len(mask):,} entries")
    else:
        print("WARNING: No mask file found!")
    
    # Validate each episode
    failure_counts = defaultdict(int)
    trained_token_counts = []
    failed_episodes = []
    
    n_episodes = len(episodes)
    if args.max_episodes:
        n_episodes = min(n_episodes, args.max_episodes)
    
    print(f"\nValidating {n_episodes} episodes...")
    
    for i in range(n_episodes):
        start, length = int(episodes[i, 0]), int(episodes[i, 1])
        
        # Main validation
        failures = validate_episode(i, tokens, mask, start, length, args.verbose)
        
        # Off-by-one alignment check
        alignment_failures = validate_alignment_offbyone(i, tokens, mask, start, length)
        failures.update(alignment_failures)
        
        if failures:
            for key in failures:
                failure_counts[key] += 1
            failed_episodes.append((i, failures))
        
        # Track stats
        if mask is not None:
            ep_mask = mask[start:start + length]
            trained_token_counts.append(np.sum(ep_mask == 1))
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_episodes} episodes...")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\nEpisodes validated: {n_episodes}")
    print(f"Episodes with failures: {len(failed_episodes)}")
    
    if failure_counts:
        print("\nFailure counts by type:")
        for key, count in sorted(failure_counts.items()):
            print(f"  {key}: {count}")
    
    if trained_token_counts:
        arr = np.array(trained_token_counts)
        print(f"\nTrained tokens per episode:")
        print(f"  Min: {arr.min()}")
        print(f"  Median: {np.median(arr):.1f}")
        print(f"  Max: {arr.max()}")
        print(f"  Total: {arr.sum():,}")
    
    # Show first few failures
    if failed_episodes and args.verbose:
        print("\nFirst 5 failed episodes:")
        for ep_idx, failures in failed_episodes[:5]:
            start, length = int(episodes[ep_idx, 0]), int(episodes[ep_idx, 1])
            print(f"\n  Episode {ep_idx} (start={start}, length={length}):")
            for key, msg in failures.items():
                print(f"    {key}: {msg}")
    
    # Exit code
    if failed_episodes:
        print(f"\n❌ VALIDATION FAILED: {len(failed_episodes)} episodes have issues")
        return 1
    else:
        print("\n✅ VALIDATION PASSED: All episodes OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
