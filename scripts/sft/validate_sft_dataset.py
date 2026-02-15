#!/usr/bin/env python3
"""
SFT Dataset Validator

Validates that loss masks are correctly applied in SFT datasets:
1. <myPT_assistant> opening tag should have mask=0
2. Loss mask should only be 1 AFTER the opening tag
3. </myPT_assistant> closing tag should have mask=1
4. <myPT_eot> should have mask=0

Usage:
    python scripts/validate_sft_dataset.py --dataset data/sft_phase3a1_combined_prepared
    python scripts/validate_sft_dataset.py --dataset data/sft_phase3a1_combined_prepared --verbose
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Dynamic token ID lookup -- never hardcode IDs!
from core.special_tokens import get_special_token_ids

SPECIAL_TOKENS = get_special_token_ids()

ASSISTANT_OPEN_ID = SPECIAL_TOKENS["myPT_assistant_open"]
ASSISTANT_CLOSE_ID = SPECIAL_TOKENS["myPT_assistant_close"]
EOT_ID = SPECIAL_TOKENS["myPT_eot"]


def load_episode_dataset(dataset_dir: str, split: str = "train"):
    """Load episode-indexed dataset."""
    split_dir = Path(dataset_dir) / split
    
    # Standard file names
    tokens_path = split_dir / "tokens.bin"
    mask_path = split_dir / "mask.bin"
    index_path = split_dir / "episodes.idx"
    
    if not tokens_path.exists():
        raise ValueError(f"No token files found in {split_dir}")
    
    # Load tokens (uint32)
    tokens = np.fromfile(tokens_path, dtype=np.uint32)
    
    # Load mask (uint8)
    if mask_path.exists():
        masks = np.fromfile(mask_path, dtype=np.uint8)
    else:
        raise ValueError(f"No mask file found in {split_dir}")
    
    # Load episode index (uint64 pairs: start, length)
    if index_path.exists():
        index_data = np.fromfile(index_path, dtype=np.uint64)
        # Reshape to (num_episodes, 2) - [start, length]
        indices = index_data.reshape(-1, 2)
    else:
        raise ValueError(f"No index file found in {split_dir}")
    
    return tokens, masks, indices


def get_episode(tokens, masks, indices, episode_idx):
    """Get tokens and masks for a specific episode."""
    start, length = indices[episode_idx]
    start, length = int(start), int(length)
    return tokens[start:start+length], masks[start:start+length]


def validate_episode(episode_tokens, episode_masks, episode_idx, verbose=False):
    """
    Validate a single episode's loss mask.
    
    Returns:
        dict with validation results
    """
    issues = []
    
    # Find positions of key tokens
    assistant_open_positions = np.where(episode_tokens == ASSISTANT_OPEN_ID)[0]
    assistant_close_positions = np.where(episode_tokens == ASSISTANT_CLOSE_ID)[0]
    eot_positions = np.where(episode_tokens == EOT_ID)[0]
    
    # Check 1: <myPT_assistant> should have mask=0
    for pos in assistant_open_positions:
        if episode_masks[pos] == 1:
            issues.append({
                "type": "ASSISTANT_OPEN_MASKED",
                "position": int(pos),
                "message": f"<myPT_assistant> at position {pos} has mask=1 (should be 0)"
            })
    
    # Check 2: </myPT_assistant> should have mask=1
    for pos in assistant_close_positions:
        if episode_masks[pos] == 0:
            issues.append({
                "type": "ASSISTANT_CLOSE_NOT_MASKED",
                "position": int(pos),
                "message": f"</myPT_assistant> at position {pos} has mask=0 (should be 1)"
            })
    
    # Check 3: <myPT_eot> should have mask=1 (trained for future-proofing)
    for pos in eot_positions:
        if episode_masks[pos] == 0:
            issues.append({
                "type": "EOT_NOT_MASKED",
                "position": int(pos),
                "message": f"<myPT_eot> at position {pos} has mask=0 (should be 1)"
            })
    
    # Check 4: First masked token should be AFTER <myPT_assistant>
    masked_positions = np.where(episode_masks == 1)[0]
    if len(masked_positions) > 0 and len(assistant_open_positions) > 0:
        first_masked = masked_positions[0]
        first_assistant_open = assistant_open_positions[0]
        
        if first_masked <= first_assistant_open:
            issues.append({
                "type": "EARLY_MASK",
                "position": int(first_masked),
                "message": f"First masked token at {first_masked} is before/at <myPT_assistant> at {first_assistant_open}"
            })
        
        # Check that first masked token is immediately after <myPT_assistant>
        expected_first_mask = first_assistant_open + 1
        if first_masked != expected_first_mask:
            # Check if there are tokens between assistant_open and first masked
            if first_masked > expected_first_mask:
                # There are unmaksed tokens between - this might be OK if they're newlines
                pass
            elif episode_tokens[first_masked] == ASSISTANT_OPEN_ID:
                issues.append({
                    "type": "ASSISTANT_OPEN_IS_FIRST_MASKED",
                    "position": int(first_masked),
                    "message": f"First masked token IS <myPT_assistant> (should be content after it)"
                })
    
    # Check 5: No mask=1 before any <myPT_assistant>
    if len(assistant_open_positions) > 0:
        first_assistant = assistant_open_positions[0]
        early_masks = np.where(episode_masks[:first_assistant] == 1)[0]
        if len(early_masks) > 0:
            issues.append({
                "type": "MASK_BEFORE_ASSISTANT",
                "position": int(early_masks[0]),
                "message": f"Found mask=1 at position {early_masks[0]} before first <myPT_assistant> at {first_assistant}"
            })
    
    return {
        "episode_idx": episode_idx,
        "num_tokens": len(episode_tokens),
        "num_masked": int(np.sum(episode_masks)),
        "mask_ratio": float(np.mean(episode_masks)),
        "issues": issues,
        "valid": len(issues) == 0
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SFT dataset loss masks")
    parser.add_argument("--dataset", required=True, help="Path to prepared SFT dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to validate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for each invalid episode")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum episodes to check (default: all)")
    parser.add_argument("--show_valid", action="store_true", help="Also show valid episodes in verbose mode")
    args = parser.parse_args()
    
    print(f"Loading dataset from: {args.dataset}/{args.split}")
    
    try:
        tokens, masks, indices = load_episode_dataset(args.dataset, args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    num_episodes = len(indices)
    print(f"Loaded {num_episodes:,} episodes, {len(tokens):,} total tokens")
    
    # Tokenizer not needed for validation (only token IDs are checked)
    
    # Validate each episode
    max_episodes = args.max_episodes or num_episodes
    
    valid_count = 0
    invalid_count = 0
    issue_counts = {}
    invalid_episodes = []
    
    print(f"\nValidating {min(max_episodes, num_episodes):,} episodes...")
    print("=" * 60)
    
    for i in range(min(max_episodes, num_episodes)):
        ep_tokens, ep_masks = get_episode(tokens, masks, indices, i)
        result = validate_episode(ep_tokens, ep_masks, i, args.verbose)
        
        if result["valid"]:
            valid_count += 1
            if args.verbose and args.show_valid:
                print(f"  [✓] Episode {i}: {result['num_tokens']} tokens, {result['num_masked']} masked ({result['mask_ratio']*100:.1f}%)")
        else:
            invalid_count += 1
            invalid_episodes.append(result)
            
            for issue in result["issues"]:
                issue_type = issue["type"]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            if args.verbose:
                print(f"\n  [✗] Episode {i}: {result['num_tokens']} tokens")
                for issue in result["issues"]:
                    print(f"      - {issue['message']}")
                
                # Show token context
                print(f"      Tokens: {ep_tokens[:20].tolist()}...")
                print(f"      Masks:  {ep_masks[:20].tolist()}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total episodes checked: {valid_count + invalid_count:,}")
    print(f"  Valid episodes:         {valid_count:,} ({100*valid_count/(valid_count+invalid_count):.1f}%)")
    print(f"  Invalid episodes:       {invalid_count:,} ({100*invalid_count/(valid_count+invalid_count):.1f}%)")
    
    if issue_counts:
        print(f"\nIssue breakdown:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue_type}: {count:,}")
    
    if invalid_count == 0:
        print("\n✅ ALL EPISODES VALID - Loss masks are correctly applied!")
        return 0
    else:
        print(f"\n❌ FOUND {invalid_count} INVALID EPISODES")
        print("\nFirst 5 invalid episodes:")
        for result in invalid_episodes[:5]:
            print(f"  Episode {result['episode_idx']}: {[i['type'] for i in result['issues']]}")
        print("\nRun with --verbose to see details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
