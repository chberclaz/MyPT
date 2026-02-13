#!/usr/bin/env python3
"""
Verify that loss is computed on mask=1 positions (not mask=0).

This directly tests the loss computation to ensure we're training
on the RIGHT tokens.

Usage:
    python scripts/verify_loss_mask_direction.py --model domain_v5 --dataset data/sft_phase3a
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT
from core.episode_data_loader import GPTEpisodeDataLoader


def main():
    parser = argparse.ArgumentParser(description="Verify loss mask direction")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    args = parser.parse_args()
    
    print("=" * 70)
    print("LOSS MASK DIRECTION VERIFICATION")
    print("=" * 70)
    print("\nThis test verifies that loss is computed on mask=1 positions.")
    print("If mask is inverted, we'd be training on system/user instead of assistant!\n")
    
    # Load model
    print(f"Loading model '{args.model}'...")
    model, _, _, _ = GPT.load(f"checkpoints/{args.model}")
    model.eval()
    
    # Load data
    print(f"Loading dataset from '{args.dataset}'...")
    loader = GPTEpisodeDataLoader(
        model.config,
        model.tokenizer,
        dataset_dir=args.dataset,
        use_loss_mask=True,
        batch_sampling_mode='epoch'
    )
    
    # Get a batch
    batch = loader.get_batch('train')
    xb, yb, loss_mask = batch
    
    print(f"\nBatch shapes: x={xb.shape}, y={yb.shape}, mask={loss_mask.shape}")
    print(f"Mask statistics: sum={loss_mask.sum().item():.0f}, "
          f"mean={loss_mask.mean().item():.3f}, "
          f"mask=1 positions: {(loss_mask == 1).sum().item()}")
    
    # Forward pass
    print("\n" + "=" * 70)
    print("FORWARD PASS")
    print("=" * 70)
    
    with torch.no_grad():
        logits, _, _ = model(xb)  # No loss computation yet
    
    # Manual loss computation to trace exactly what happens
    print("\n--- Manual Loss Computation ---")
    
    Bt, Tt, Ct = logits.shape
    logits2 = logits.view(Bt * Tt, Ct)
    targets2 = yb.view(Bt * Tt)
    loss_mask2 = loss_mask.view(Bt * Tt)
    
    # Compute per-token loss
    per_token_loss = F.cross_entropy(logits2, targets2, reduction='none')
    
    print(f"\nPer-token loss shape: {per_token_loss.shape}")
    print(f"Per-token loss range: [{per_token_loss.min().item():.4f}, {per_token_loss.max().item():.4f}]")
    
    # Split by mask value
    mask1_indices = (loss_mask2 == 1).nonzero(as_tuple=True)[0]
    mask0_indices = (loss_mask2 == 0).nonzero(as_tuple=True)[0]
    
    loss_on_mask1 = per_token_loss[mask1_indices]
    loss_on_mask0 = per_token_loss[mask0_indices]
    
    print(f"\n--- Loss by Mask Value ---")
    print(f"Mask=1 positions: {len(mask1_indices)}")
    print(f"  Mean loss: {loss_on_mask1.mean().item():.4f}")
    print(f"  Sum loss:  {loss_on_mask1.sum().item():.4f}")
    
    print(f"\nMask=0 positions: {len(mask0_indices)}")
    print(f"  Mean loss: {loss_on_mask0.mean().item():.4f}")
    print(f"  Sum loss:  {loss_on_mask0.sum().item():.4f}")
    
    # Compute final masked loss (what the model actually uses)
    denom = loss_mask2.sum().clamp_min(1.0)
    masked_loss = (per_token_loss * loss_mask2).sum() / denom
    
    print(f"\n--- Final Masked Loss ---")
    print(f"Denominator (sum of mask): {denom.item():.0f}")
    print(f"Final loss: {masked_loss.item():.4f}")
    
    # Verify: masked_loss should equal mean of loss_on_mask1
    expected_loss = loss_on_mask1.mean().item()
    actual_loss = masked_loss.item()
    
    print(f"\n--- VERIFICATION ---")
    print(f"Expected (mean of mask=1 losses): {expected_loss:.6f}")
    print(f"Actual (masked loss formula):     {actual_loss:.6f}")
    
    if abs(expected_loss - actual_loss) < 0.0001:
        print(f"\n✓ CORRECT: Loss is computed on mask=1 positions only!")
    else:
        print(f"\n✗ BUG: Loss values don't match! Something is wrong.")
    
    # Now let's see WHAT tokens are at mask=1 positions
    print("\n" + "=" * 70)
    print("WHAT TOKENS ARE AT MASK=1 POSITIONS?")
    print("=" * 70)
    
    # Take first sequence in batch
    seq_x = xb[0].cpu().numpy()
    seq_y = yb[0].cpu().numpy()
    seq_mask = loss_mask[0].cpu().numpy()
    
    mask1_positions = np.where(seq_mask == 1)[0]
    
    print(f"\nFirst sequence has {len(mask1_positions)} mask=1 positions")
    print(f"\nFirst 20 mask=1 positions (where loss IS computed):")
    print("-" * 70)
    
    for i, pos in enumerate(mask1_positions[:20]):
        x_tok = int(seq_x[pos])
        y_tok = int(seq_y[pos])
        
        x_str = model.tokenizer.decode([x_tok]).replace('\n', '\\n')
        y_str = model.tokenizer.decode([y_tok]).replace('\n', '\\n')
        
        # Truncate
        if len(x_str) > 20: x_str = x_str[:17] + "..."
        if len(y_str) > 20: y_str = y_str[:17] + "..."
        
        loss_val = per_token_loss[pos].item()
        
        note = ""
        if y_tok == 50263: note = "← <myPT_assistant>"
        elif y_tok == 50264: note = "← </myPT_assistant>"
        elif y_tok >= 50257: note = f"← special({y_tok})"
        
        print(f"  [{pos:3d}] x={repr(x_str):22} → y={repr(y_str):22} loss={loss_val:.3f} {note}")
    
    print("\n" + "-" * 70)
    print("First 10 mask=0 positions (where loss is NOT computed):")
    print("-" * 70)
    
    mask0_positions = np.where(seq_mask == 0)[0]
    for i, pos in enumerate(mask0_positions[:10]):
        x_tok = int(seq_x[pos])
        y_tok = int(seq_y[pos])
        
        x_str = model.tokenizer.decode([x_tok]).replace('\n', '\\n')
        y_str = model.tokenizer.decode([y_tok]).replace('\n', '\\n')
        
        if len(x_str) > 20: x_str = x_str[:17] + "..."
        if len(y_str) > 20: y_str = y_str[:17] + "..."
        
        loss_val = per_token_loss[pos].item()
        
        note = ""
        if y_tok == 50257: note = "← <myPT_system>"
        elif y_tok == 50258: note = "← </myPT_system>"
        elif y_tok == 50259: note = "← <myPT_user>"
        elif y_tok == 50260: note = "← </myPT_user>"
        
        print(f"  [{pos:3d}] x={repr(x_str):22} → y={repr(y_str):22} loss={loss_val:.3f} (IGNORED) {note}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Check if assistant tokens are in mask=1 region
    assistant_open_positions = np.where(seq_y == 50263)[0]
    assistant_close_positions = np.where(seq_y == 50264)[0]
    
    print(f"\n<myPT_assistant> (50263) appears as target at positions: {list(assistant_open_positions)}")
    for pos in assistant_open_positions:
        if pos < len(seq_mask):
            print(f"  Position {pos}: mask={int(seq_mask[pos])} → {'TRAINED' if seq_mask[pos]==1 else 'NOT TRAINED'}")
    
    print(f"\n</myPT_assistant> (50264) appears as target at positions: {list(assistant_close_positions)}")
    for pos in assistant_close_positions:
        if pos < len(seq_mask):
            print(f"  Position {pos}: mask={int(seq_mask[pos])} → {'TRAINED' if seq_mask[pos]==1 else 'NOT TRAINED'}")
    
    # Final summary
    print("\n" + "-" * 70)
    if len(mask1_positions) > 0:
        # Check what's predominantly in mask=1
        mask1_targets = seq_y[mask1_positions]
        
        # Count special tokens
        special_count = np.sum(mask1_targets >= 50257)
        regular_count = np.sum(mask1_targets < 50257)
        
        print(f"Mask=1 targets: {special_count} special tokens, {regular_count} regular tokens")
        
        # Check if any system/user tokens are in mask=1 (would be a bug)
        system_user_tokens = {50257, 50258, 50259, 50260}  # system open/close, user open/close
        bug_count = sum(1 for t in mask1_targets if t in system_user_tokens)
        
        if bug_count > 0:
            print(f"\n⚠️  WARNING: {bug_count} system/user tokens found in mask=1 region!")
            print("   This could indicate inverted masking!")
        else:
            print(f"\n✓ No system/user tokens in mask=1 region (correct)")


if __name__ == "__main__":
    main()
