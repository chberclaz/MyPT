#!/usr/bin/env python3
"""
Debug train vs eval parity - check for dropout/mode bugs.

Verifies that:
1. Loss in eval mode vs train mode differs only slightly (due to dropout)
2. Forward pass in eval mode is consistent
3. No exploding differences between modes

Usage:
    python scripts/debug_train_eval_parity.py --model phase3a1_alpha --dataset_dir data/sft_ready/phase3a1_alpha
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import load_model
from core.episode_data_loader import EpisodeSFTDataLoader


def run_parity_check(
    model_name: str,
    dataset_dir: str,
    batch_size: int = 4,
    n_batches: int = 5,
    verbose: bool = False,
) -> bool:
    """Run train/eval parity check.
    
    Returns True if check passes, False otherwise.
    """
    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model(model_name)
    device = next(model.parameters()).device
    
    # Load a few batches from val set
    print(f"\nLoading dataset from: {dataset_dir}")
    data_loader = EpisodeSFTDataLoader(
        dataset_dir=dataset_dir,
        split="val",
        batch_size=batch_size,
        block_size=config.block_size,
        device=device,
        use_loss_mask=True,
    )
    
    print(f"  Episodes: {data_loader.total_episodes}")
    print(f"  Batch size: {batch_size}")
    
    # Collect batches
    batches = []
    for i in range(min(n_batches, data_loader.total_episodes // batch_size)):
        x, y, mask = data_loader.get_batch()
        batches.append((x.clone(), y.clone(), mask.clone() if mask is not None else None))
    
    print(f"  Collected {len(batches)} batches for testing")
    
    # Test 1: Eval mode consistency
    print("\n" + "="*60)
    print("Test 1: Eval Mode Consistency")
    print("="*60)
    
    model.eval()
    eval_losses = []
    
    with torch.no_grad():
        for i, (x, y, mask) in enumerate(batches):
            logits, loss, _ = model(x, targets=y, loss_mask=mask)
            eval_losses.append(loss.item())
            if verbose:
                print(f"  Batch {i}: loss = {loss.item():.4f}")
    
    # Run again to check consistency
    eval_losses_2 = []
    with torch.no_grad():
        for i, (x, y, mask) in enumerate(batches):
            logits, loss, _ = model(x, targets=y, loss_mask=mask)
            eval_losses_2.append(loss.item())
    
    eval_diff = [abs(a - b) for a, b in zip(eval_losses, eval_losses_2)]
    max_eval_diff = max(eval_diff) if eval_diff else 0
    
    print(f"\n  Eval mode losses (run 1): {[f'{l:.4f}' for l in eval_losses]}")
    print(f"  Eval mode losses (run 2): {[f'{l:.4f}' for l in eval_losses_2]}")
    print(f"  Max difference: {max_eval_diff:.6f}")
    
    if max_eval_diff > 1e-5:
        print("  ❌ FAIL: Eval mode is not deterministic!")
        eval_consistent = False
    else:
        print("  ✅ PASS: Eval mode is deterministic")
        eval_consistent = True
    
    # Test 2: Train vs Eval mode difference
    print("\n" + "="*60)
    print("Test 2: Train vs Eval Mode Difference")
    print("="*60)
    
    model.train()
    train_losses = []
    
    with torch.no_grad():  # Still no grad, just checking forward pass
        for i, (x, y, mask) in enumerate(batches):
            logits, loss, _ = model(x, targets=y, loss_mask=mask)
            train_losses.append(loss.item())
            if verbose:
                print(f"  Batch {i}: loss = {loss.item():.4f}")
    
    # Compare train vs eval
    mode_diffs = [abs(t - e) for t, e in zip(train_losses, eval_losses)]
    avg_mode_diff = sum(mode_diffs) / len(mode_diffs) if mode_diffs else 0
    max_mode_diff = max(mode_diffs) if mode_diffs else 0
    
    print(f"\n  Train mode losses: {[f'{l:.4f}' for l in train_losses]}")
    print(f"  Eval mode losses:  {[f'{l:.4f}' for l in eval_losses]}")
    print(f"  Differences:       {[f'{d:.4f}' for d in mode_diffs]}")
    print(f"  Average diff: {avg_mode_diff:.4f}")
    print(f"  Max diff: {max_mode_diff:.4f}")
    
    # Dropout should cause SOME difference, but not huge
    dropout = getattr(config, 'dropout', 0.0)
    print(f"\n  Model dropout: {dropout}")
    
    if dropout > 0 and max_mode_diff < 0.001:
        print("  ⚠️  WARNING: Dropout > 0 but no difference between train/eval")
        print("     This might indicate dropout is not being applied correctly")
        mode_reasonable = True  # Not a hard failure
    elif max_mode_diff > 5.0:
        print("  ❌ FAIL: Huge difference between train/eval modes!")
        print("     This indicates a serious bug in mode handling")
        mode_reasonable = False
    else:
        print("  ✅ PASS: Reasonable difference between train/eval modes")
        mode_reasonable = True
    
    # Test 3: Forward pass parity (eval mode, different cache settings)
    print("\n" + "="*60)
    print("Test 3: Forward Pass Parity (cache settings)")
    print("="*60)
    
    model.eval()
    x, y, mask = batches[0]
    
    # Run with default settings
    with torch.no_grad():
        logits1, loss1, _ = model(x, targets=y, loss_mask=mask)
    
    # Run again
    with torch.no_grad():
        logits2, loss2, _ = model(x, targets=y, loss_mask=mask)
    
    logit_diff = (logits1 - logits2).abs().max().item()
    loss_diff = abs(loss1.item() - loss2.item())
    
    print(f"  Loss 1: {loss1.item():.6f}")
    print(f"  Loss 2: {loss2.item():.6f}")
    print(f"  Loss diff: {loss_diff:.8f}")
    print(f"  Max logit diff: {logit_diff:.8f}")
    
    if logit_diff > 1e-5 or loss_diff > 1e-5:
        print("  ❌ FAIL: Forward pass not deterministic in eval mode!")
        forward_consistent = False
    else:
        print("  ✅ PASS: Forward pass is deterministic")
        forward_consistent = True
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_pass = eval_consistent and mode_reasonable and forward_consistent
    
    print(f"  Eval mode consistent: {'✅' if eval_consistent else '❌'}")
    print(f"  Train/eval difference reasonable: {'✅' if mode_reasonable else '❌'}")
    print(f"  Forward pass deterministic: {'✅' if forward_consistent else '❌'}")
    print(f"\n  OVERALL: {'✅ ALL PASS' if all_pass else '❌ SOME FAILURES'}")
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Debug train vs eval parity")
    parser.add_argument("--model", type=str, required=True, help="Model name to check")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--n_batches", type=int, default=5, help="Number of batches to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    success = run_parity_check(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        verbose=args.verbose,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
