#!/usr/bin/env python3
"""
Verify mask alignment in SFT training data.

Shows exactly:
1. Which tokens have mask=1 (model learns to predict these)
2. What the input token is at each masked position
3. What the target token is at each masked position

This helps diagnose if the mask is correctly aligned with assistant responses.

Usage:
    python scripts/verify_mask_alignment.py --dataset data/sft_phase3a --episode 0
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tokenizer import Tokenizer
from core.model import GPTConfig
from core.special_tokens import get_special_token_ids, BASE_VOCAB_SIZE

# Dynamic token ID lookup -- never hardcode IDs!
_IDS = get_special_token_ids()
_ASSISTANT_OPEN_ID  = _IDS["myPT_assistant_open"]
_ASSISTANT_CLOSE_ID = _IDS["myPT_assistant_close"]


def main():
    parser = argparse.ArgumentParser(description="Verify mask alignment")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to inspect")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MASK ALIGNMENT VERIFICATION")
    print("=" * 70)
    
    # Load tokenizer
    config = GPTConfig()
    tokenizer = Tokenizer(config)
    
    # Load data files
    train_dir = os.path.join(args.dataset, "train")
    tokens_path = os.path.join(train_dir, "tokens.bin")
    mask_path = os.path.join(train_dir, "mask.bin")
    episodes_path = os.path.join(train_dir, "episodes.idx")
    
    tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
    mask = np.memmap(mask_path, dtype=np.uint8, mode='r')
    episodes = np.memmap(episodes_path, dtype=np.uint64, mode='r').reshape(-1, 2)
    
    print(f"\nDataset: {args.dataset}")
    print(f"Total episodes: {len(episodes)}")
    print(f"Total tokens: {len(tokens)}")
    
    # Get episode
    ep_idx = args.episode
    start, length = int(episodes[ep_idx][0]), int(episodes[ep_idx][1])
    
    ep_tokens = tokens[start:start + length]
    ep_mask = mask[start:start + length]
    
    print(f"\n" + "=" * 70)
    print(f"EPISODE {ep_idx}: {length} tokens (positions {start} to {start + length - 1})")
    print("=" * 70)
    
    # Show full decoded episode
    print(f"\n--- FULL EPISODE TEXT ---")
    full_text = tokenizer.decode(list(ep_tokens))
    print(full_text)
    print("--- END ---\n")
    
    # Simulate x, y, mask_y alignment (what the data loader does)
    print("=" * 70)
    print("SIMULATED TRAINING ALIGNMENT (x -> y with mask)")
    print("=" * 70)
    print("\nDuring training:")
    print("  x[i] = input token at position i")
    print("  y[i] = target token at position i (what model should predict)")
    print("  mask_y[i] = 1 if we compute loss for predicting y[i]")
    print()
    
    # Create x, y, mask_y like the data loader does
    x = ep_tokens[:-1]  # Input tokens
    y = ep_tokens[1:]   # Target tokens (shifted by 1)
    mask_y = ep_mask[1:]  # Mask aligned with y
    
    print(f"Sequence length: x={len(x)}, y={len(y)}, mask_y={len(mask_y)}")
    print(f"Mask=1 positions: {np.sum(mask_y)} ({100*np.sum(mask_y)/len(mask_y):.1f}%)")
    
    # Show all mask=1 positions
    print("\n" + "-" * 70)
    print("ALL MASK=1 POSITIONS (model learns to predict these targets):")
    print("-" * 70)
    print(f"{'Pos':>4} | {'Input (x)':^30} | {'Target (y)':^30} | Note")
    print("-" * 70)
    
    mask1_positions = np.where(mask_y == 1)[0]
    
    for i, pos in enumerate(mask1_positions):
        x_tok = int(x[pos])
        y_tok = int(y[pos])
        
        x_str = tokenizer.decode([x_tok]).replace('\n', '\\n')
        y_str = tokenizer.decode([y_tok]).replace('\n', '\\n')
        
        # Truncate long strings
        if len(x_str) > 28:
            x_str = x_str[:25] + "..."
        if len(y_str) > 28:
            y_str = y_str[:25] + "..."
        
        # Note special tokens
        note = ""
        if x_tok >= BASE_VOCAB_SIZE:
            note += f"x=special({x_tok}) "
        if y_tok >= BASE_VOCAB_SIZE:
            note += f"y=special({y_tok}) "
        
        print(f"{pos:4d} | {repr(x_str):^30} | {repr(y_str):^30} | {note}")
        
        # Show transitions
        if i == 0:
            print(f"     | {'^ FIRST MASKED POSITION':^62}")
        if i > 0 and pos - mask1_positions[i-1] > 1:
            print(f"     | {'^ GAP - jumped from non-masked':^62}")
    
    print("-" * 70)
    
    # Show mask=0 to mask=1 transitions
    print("\n" + "=" * 70)
    print("MASK TRANSITIONS (0 -> 1)")
    print("=" * 70)
    print("\nThese are the positions where training signal STARTS:")
    
    for i in range(1, len(mask_y)):
        if mask_y[i] == 1 and mask_y[i-1] == 0:
            # Transition from 0 to 1
            x_tok = int(x[i])
            y_tok = int(y[i])
            prev_x = int(x[i-1])
            prev_y = int(y[i-1])
            
            print(f"\n  Position {i}: mask goes 0 -> 1")
            print(f"    Previous: x={repr(tokenizer.decode([prev_x]))} -> y={repr(tokenizer.decode([prev_y]))} (mask=0, no loss)")
            print(f"    Current:  x={repr(tokenizer.decode([x_tok]))} -> y={repr(tokenizer.decode([y_tok]))} (mask=1, LOSS COMPUTED)")
    
    # Show mask=1 to mask=0 transitions
    print("\n" + "=" * 70)
    print("MASK TRANSITIONS (1 -> 0)")
    print("=" * 70)
    print("\nThese are the positions where training signal ENDS:")
    
    for i in range(1, len(mask_y)):
        if mask_y[i] == 0 and mask_y[i-1] == 1:
            # Transition from 1 to 0
            x_tok = int(x[i])
            y_tok = int(y[i])
            prev_x = int(x[i-1])
            prev_y = int(y[i-1])
            
            print(f"\n  Position {i}: mask goes 1 -> 0")
            print(f"    Previous: x={repr(tokenizer.decode([prev_x]))} -> y={repr(tokenizer.decode([prev_y]))} (mask=1, LOSS COMPUTED)")
            print(f"    Current:  x={repr(tokenizer.decode([x_tok]))} -> y={repr(tokenizer.decode([y_tok]))} (mask=0, no loss)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Check if assistant tags are in masked region
    assistant_open_id = _ASSISTANT_OPEN_ID
    assistant_close_id = _ASSISTANT_CLOSE_ID
    
    # Find where assistant tags appear in y (targets)
    assistant_open_in_y = np.where(y == assistant_open_id)[0]
    assistant_close_in_y = np.where(y == assistant_close_id)[0]
    
    print(f"\n<myPT_assistant> ({assistant_open_id}) appears as TARGET at positions: {list(assistant_open_in_y)}")
    for pos in assistant_open_in_y:
        print(f"  Position {pos}: mask_y={mask_y[pos]} (should be 1 to train on generating this tag)")
    
    print(f"\n</myPT_assistant> ({assistant_close_id}) appears as TARGET at positions: {list(assistant_close_in_y)}")
    for pos in assistant_close_in_y:
        print(f"  Position {pos}: mask_y={mask_y[pos]} (should be 1 to train on generating this tag)")
    
    # Check if there's an off-by-one error
    print("\n" + "-" * 70)
    print("CRITICAL CHECK: Is the model learning to generate closing tags?")
    print("-" * 70)
    
    for pos in assistant_close_in_y:
        if mask_y[pos] == 1:
            print(f"  Position {pos}: CORRECT - model learns to predict </myPT_assistant>")
            print(f"    Given input: {repr(tokenizer.decode([int(x[pos])]))}")
            print(f"    Predict target: </myPT_assistant>")
        else:
            print(f"  Position {pos}: BUG! mask=0 means model does NOT learn to predict </myPT_assistant>")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
