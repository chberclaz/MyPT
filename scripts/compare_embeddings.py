#!/usr/bin/env python3
"""
Compare special token embeddings between two models.

Shows if SFT training actually updated the special token representations.

Usage:
    python scripts/compare_embeddings.py --before domain_v5 --after phase3a_chat_test_250iters
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT


def main():
    parser = argparse.ArgumentParser(description="Compare embeddings between models")
    parser.add_argument("--before", type=str, required=True, help="Model before SFT")
    parser.add_argument("--after", type=str, required=True, help="Model after SFT")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EMBEDDING COMPARISON DIAGNOSTIC")
    print("=" * 70)
    
    # Load models
    print(f"\nLoading '{args.before}'...")
    model_before, _, _, _ = GPT.load(f"checkpoints/{args.before}")
    
    print(f"\nLoading '{args.after}'...")
    model_after, _, _, _ = GPT.load(f"checkpoints/{args.after}")
    
    # Get embedding tables
    emb_before = model_before.token_embedding_table.weight.detach().cpu().float()
    emb_after = model_after.token_embedding_table.weight.detach().cpu().float()
    
    # Get lm_head weights
    lm_before = model_before.lm_head.weight.detach().cpu().float()
    lm_after = model_after.lm_head.weight.detach().cpu().float()
    
    print(f"\nEmbedding shape: {emb_before.shape}")
    print(f"LM head shape: {lm_before.shape}")
    
    # Compare special tokens (50257-50271)
    special_start = 50257
    special_end = 50272
    
    # Token names
    token_names = {
        50257: "<myPT_system>",
        50258: "</myPT_system>",
        50259: "<myPT_user>",
        50260: "</myPT_user>",
        50261: "<myPT_user_context>",
        50262: "</myPT_user_context>",
        50263: "<myPT_assistant>",
        50264: "</myPT_assistant>",
        50265: "<myPT_toolcall>",
        50266: "</myPT_toolcall>",
        50267: "<myPT_toolresult>",
        50268: "</myPT_toolresult>",
        50269: "<myPT_thinking>",
        50270: "</myPT_thinking>",
        50271: "<myPT_eot>",
    }
    
    print("\n" + "=" * 70)
    print("INPUT EMBEDDINGS (token_embedding_table)")
    print("=" * 70)
    print(f"{'Token ID':<10} {'Name':<25} {'L2 Dist':<12} {'Cosine':<12} {'Changed?':<10}")
    print("-" * 70)
    
    for tid in range(special_start, special_end):
        if tid >= emb_before.shape[0]:
            break
            
        vec_b = emb_before[tid]
        vec_a = emb_after[tid]
        
        l2_dist = torch.norm(vec_a - vec_b).item()
        cosine_sim = torch.nn.functional.cosine_similarity(vec_b.unsqueeze(0), vec_a.unsqueeze(0)).item()
        
        name = token_names.get(tid, "?")
        changed = "YES" if l2_dist > 0.1 else "NO"
        
        print(f"{tid:<10} {name:<25} {l2_dist:<12.4f} {cosine_sim:<12.4f} {changed:<10}")
    
    # Compare a few regular tokens for reference
    print("\n" + "-" * 70)
    print("Reference: Regular tokens (should also change if model trained)")
    print("-" * 70)
    for tid in [100, 1000, 10000, 50000]:
        vec_b = emb_before[tid]
        vec_a = emb_after[tid]
        l2_dist = torch.norm(vec_a - vec_b).item()
        cosine_sim = torch.nn.functional.cosine_similarity(vec_b.unsqueeze(0), vec_a.unsqueeze(0)).item()
        print(f"{tid:<10} {'(regular)':<25} {l2_dist:<12.4f} {cosine_sim:<12.4f}")
    
    print("\n" + "=" * 70)
    print("OUTPUT PROJECTION (lm_head)")
    print("=" * 70)
    print(f"{'Token ID':<10} {'Name':<25} {'L2 Dist':<12} {'Cosine':<12} {'Changed?':<10}")
    print("-" * 70)
    
    for tid in range(special_start, special_end):
        if tid >= lm_before.shape[0]:
            break
            
        vec_b = lm_before[tid]
        vec_a = lm_after[tid]
        
        l2_dist = torch.norm(vec_a - vec_b).item()
        cosine_sim = torch.nn.functional.cosine_similarity(vec_b.unsqueeze(0), vec_a.unsqueeze(0)).item()
        
        name = token_names.get(tid, "?")
        changed = "YES" if l2_dist > 0.1 else "NO"
        
        print(f"{tid:<10} {name:<25} {l2_dist:<12.4f} {cosine_sim:<12.4f} {changed:<10}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Total change in special token embeddings
    special_emb_b = emb_before[special_start:special_end]
    special_emb_a = emb_after[special_start:special_end]
    special_lm_b = lm_before[special_start:special_end]
    special_lm_a = lm_after[special_start:special_end]
    
    total_emb_change = torch.norm(special_emb_a - special_emb_b).item()
    total_lm_change = torch.norm(special_lm_a - special_lm_b).item()
    
    # Average change per token
    avg_emb_change = total_emb_change / (special_end - special_start)
    avg_lm_change = total_lm_change / (special_end - special_start)
    
    print(f"Total embedding change (special tokens): {total_emb_change:.4f}")
    print(f"Average per token: {avg_emb_change:.4f}")
    print(f"Total lm_head change (special tokens): {total_lm_change:.4f}")
    print(f"Average per token: {avg_lm_change:.4f}")
    
    if avg_emb_change < 0.01:
        print("\n⚠️  WARNING: Special token embeddings barely changed!")
        print("   This could mean gradients aren't flowing to these tokens.")
    elif avg_emb_change < 0.1:
        print("\n⚠️  WARNING: Small embedding changes - may need more training.")
    else:
        print("\n✓ Embeddings show significant changes.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
