#!/usr/bin/env python3
"""
Trace a single training step to debug gradient flow.

Shows:
1. Which parameters receive gradients
2. Gradient magnitudes for embeddings vs transformer layers
3. Actual weight changes after optimizer.step()

Usage:
    python scripts/trace_training_step.py --model domain_v5_sft_ready --dataset data/sft_phase3a
"""

import argparse
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT
from core.episode_data_loader import GPTEpisodeDataLoader
from core.special_tokens import get_special_token_ids, BASE_VOCAB_SIZE, SPECIAL_TOKEN_STRINGS

# Number of special tokens (dynamic, based on SPECIAL_TOKEN_STRINGS)
_N_SPECIAL = len(SPECIAL_TOKEN_STRINGS)
_SPECIAL_START = BASE_VOCAB_SIZE
_SPECIAL_END = BASE_VOCAB_SIZE + _N_SPECIAL


def main():
    parser = argparse.ArgumentParser(description="Trace a training step")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRAINING STEP TRACE")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading model '{args.model}'...")
    model, tokenizer_state, _, _ = GPT.load(f"checkpoints/{args.model}")
    model.train()
    
    # Store initial weights for comparison
    print("\n2. Storing initial weights...")
    initial_weights = {}
    for name, param in model.named_parameters():
        initial_weights[name] = param.detach().clone()
    
    # Setup optimizer
    print(f"\n3. Setting up optimizer (lr={args.lr})...")
    optimizer = model.configure_optimizer(learning_rate=args.lr, weight_decay=0.1)
    
    # Load data
    print(f"\n4. Loading dataset from '{args.dataset}'...")
    loader = GPTEpisodeDataLoader(
        model.config,
        model.tokenizer,
        dataset_dir=args.dataset,
        use_loss_mask=True,
        batch_sampling_mode='epoch'
    )
    
    # Get a batch
    print("\n5. Getting training batch...")
    batch = loader.get_batch('train')
    xb, yb, loss_mask = batch
    
    print(f"   Input shape: {xb.shape}")
    print(f"   Target shape: {yb.shape}")
    print(f"   Mask shape: {loss_mask.shape}")
    print(f"   Mask sum: {loss_mask.sum().item():.0f} / {loss_mask.numel()} = {100*loss_mask.sum().item()/loss_mask.numel():.1f}%")
    
    # Forward pass
    print("\n6. Forward pass...")
    optimizer.zero_grad()
    _, loss, _ = model(xb, yb, loss_mask=loss_mask)
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    print("\n7. Backward pass (computing gradients)...")
    loss.backward()
    
    # Analyze gradients
    print("\n" + "=" * 60)
    print("GRADIENT ANALYSIS")
    print("=" * 60)
    
    grad_info = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            ratio = grad_norm / (param_norm + 1e-8)
            grad_info.append({
                'name': name,
                'grad_norm': grad_norm,
                'param_norm': param_norm,
                'ratio': ratio,
                'shape': list(param.shape)
            })
    
    # Sort by gradient norm
    grad_info.sort(key=lambda x: x['grad_norm'], reverse=True)
    
    print("\nTop 10 by gradient magnitude:")
    print("-" * 80)
    for i, info in enumerate(grad_info[:10]):
        print(f"  {i+1}. {info['name']}")
        print(f"      grad_norm={info['grad_norm']:.6f}, param_norm={info['param_norm']:.4f}, ratio={info['ratio']:.6f}")
    
    # Check embeddings specifically
    print("\n" + "-" * 60)
    print("EMBEDDING GRADIENTS")
    print("-" * 60)
    
    emb_params = [g for g in grad_info if 'embedding' in g['name'] or 'lm_head' in g['name']]
    for info in emb_params:
        print(f"  {info['name']}")
        print(f"    grad_norm={info['grad_norm']:.6f}, param_norm={info['param_norm']:.4f}")
    
    # Check special token gradients specifically
    print("\n" + "-" * 60)
    print(f"SPECIAL TOKEN EMBEDDING GRADIENTS (IDs {_SPECIAL_START}-{_SPECIAL_END - 1})")
    print("-" * 60)
    
    tok_emb = model.token_embedding_table.weight
    lm_head = model.lm_head.weight
    
    # Get gradients for special tokens
    if tok_emb.grad is not None:
        special_tok_grad = tok_emb.grad[_SPECIAL_START:_SPECIAL_END]
        print(f"\n  token_embedding_table gradients for special tokens:")
        print(f"    grad_norm (special): {special_tok_grad.norm().item():.8f}")
        print(f"    grad_mean (special): {special_tok_grad.mean().item():.8f}")
        print(f"    grad_std (special):  {special_tok_grad.std().item():.8f}")
        
        # Compare to regular tokens
        regular_tok_grad = tok_emb.grad[:_SPECIAL_START]
        print(f"\n    grad_norm (regular): {regular_tok_grad.norm().item():.8f}")
        print(f"    grad_mean (regular): {regular_tok_grad.mean().item():.8f}")
    else:
        print("  WARNING: token_embedding_table has no gradient!")
    
    if lm_head.grad is not None:
        special_lm_grad = lm_head.grad[_SPECIAL_START:_SPECIAL_END]
        print(f"\n  lm_head gradients for special tokens:")
        print(f"    grad_norm (special): {special_lm_grad.norm().item():.8f}")
        print(f"    grad_mean (special): {special_lm_grad.mean().item():.8f}")
        print(f"    grad_std (special):  {special_lm_grad.std().item():.8f}")
        
        regular_lm_grad = lm_head.grad[:_SPECIAL_START]
        print(f"\n    grad_norm (regular): {regular_lm_grad.norm().item():.8f}")
        print(f"    grad_mean (regular): {regular_lm_grad.mean().item():.8f}")
    else:
        print("  WARNING: lm_head has no gradient!")
    
    # Optimizer step
    print("\n8. Optimizer step...")
    optimizer.step()
    
    # Compare weights after update
    print("\n" + "=" * 60)
    print("WEIGHT CHANGES AFTER OPTIMIZER STEP")
    print("=" * 60)
    
    changes = []
    for name, param in model.named_parameters():
        if name in initial_weights:
            diff = (param - initial_weights[name]).norm().item()
            original = initial_weights[name].norm().item()
            changes.append({
                'name': name,
                'change': diff,
                'original': original,
                'ratio': diff / (original + 1e-8)
            })
    
    # Sort by change magnitude
    changes.sort(key=lambda x: x['change'], reverse=True)
    
    print("\nTop 10 by weight change:")
    print("-" * 80)
    for i, info in enumerate(changes[:10]):
        print(f"  {i+1}. {info['name']}")
        print(f"      change={info['change']:.8f}, original={info['original']:.4f}, ratio={info['ratio']:.8f}")
    
    # Check embedding changes specifically
    print("\n" + "-" * 60)
    print("EMBEDDING WEIGHT CHANGES")
    print("-" * 60)
    
    emb_changes = [c for c in changes if 'embedding' in c['name'] or 'lm_head' in c['name']]
    for info in emb_changes:
        print(f"  {info['name']}")
        print(f"    change={info['change']:.8f}, ratio={info['ratio']:.8f}")
    
    # Special token weight changes
    print("\n" + "-" * 60)
    print("SPECIAL TOKEN EMBEDDING CHANGES")
    print("-" * 60)
    
    tok_emb_new = model.token_embedding_table.weight[_SPECIAL_START:_SPECIAL_END].detach()
    tok_emb_old = initial_weights['token_embedding_table.weight'][_SPECIAL_START:_SPECIAL_END]
    tok_change = (tok_emb_new - tok_emb_old).norm().item()
    print(f"  token_embedding_table (special): change={tok_change:.8f}")
    
    lm_head_new = model.lm_head.weight[_SPECIAL_START:_SPECIAL_END].detach()
    lm_head_old = initial_weights['lm_head.weight'][_SPECIAL_START:_SPECIAL_END]
    lm_change = (lm_head_new - lm_head_old).norm().item()
    print(f"  lm_head (special): change={lm_change:.8f}")
    
    # Compare to regular tokens
    tok_emb_reg_new = model.token_embedding_table.weight[:_SPECIAL_START].detach()
    tok_emb_reg_old = initial_weights['token_embedding_table.weight'][:_SPECIAL_START]
    tok_reg_change = (tok_emb_reg_new - tok_emb_reg_old).norm().item()
    print(f"\n  token_embedding_table (regular): change={tok_reg_change:.8f}")
    
    # Show per-token changes
    print("\n" + "-" * 60)
    print("PER-SPECIAL-TOKEN CHANGES")
    print("-" * 60)
    
    special_token_names = list(SPECIAL_TOKEN_STRINGS.keys())
    
    for i, token_id in enumerate(range(_SPECIAL_START, _SPECIAL_END)):
        emb_change = (tok_emb_new[i] - tok_emb_old[i]).norm().item()
        lm_change = (lm_head_new[i] - lm_head_old[i]).norm().item()
        name = special_token_names[i] if i < len(special_token_names) else f"token_{token_id}"
        print(f"  {token_id} ({name}): emb_change={emb_change:.6f}, lm_change={lm_change:.6f}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    # Check if embeddings are in no_decay group (they should be)
    print("\n  Checking optimizer parameter groups...")
    for i, group in enumerate(optimizer.param_groups):
        wd = group['weight_decay']
        num_params = sum(p.numel() for p in group['params'])
        print(f"    Group {i}: weight_decay={wd}, num_params={num_params:,}")
    
    print("\n  Done!")


if __name__ == "__main__":
    main()
