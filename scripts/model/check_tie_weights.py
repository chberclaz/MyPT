#!/usr/bin/env python3
"""
Check tie_weights correctness for a checkpoint.

Verifies that model.lm_head.weight and model.token_embedding_table.weight
are the SAME tensor (data_ptr equality) when tie_weights=True.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import load_model


def check_tie_weights(model_name: str, verbose: bool = False) -> bool:
    """Check if tie_weights is correctly configured and enforced.
    
    Returns True if check passes, False if it fails.
    """
    print(f"Loading model: {model_name}")
    model, tokenizer, config = load_model(model_name)
    
    # Check config
    tie_weights_config = getattr(config, 'tie_weights', None)
    print(f"\nConfig tie_weights: {tie_weights_config}")
    
    # Get weight tensors
    embedding_weight = model.token_embedding_table.weight
    lm_head_weight = model.lm_head.weight
    
    # Check pointer equality (same underlying data)
    embedding_ptr = embedding_weight.data_ptr()
    lm_head_ptr = lm_head_weight.data_ptr()
    same_tensor = (embedding_ptr == lm_head_ptr)
    
    print(f"\nWeight Analysis:")
    print(f"  token_embedding_table.weight shape: {embedding_weight.shape}")
    print(f"  lm_head.weight shape:               {lm_head_weight.shape}")
    print(f"  token_embedding_table.weight ptr:   {embedding_ptr}")
    print(f"  lm_head.weight ptr:                 {lm_head_ptr}")
    print(f"  Same tensor (ptr equality):         {same_tensor}")
    
    if verbose:
        # Also check if values are equal (even if not same pointer)
        import torch
        values_equal = torch.equal(embedding_weight, lm_head_weight)
        print(f"  Values equal (torch.equal):         {values_equal}")
        
        # Check a few specific values
        print(f"\n  Sample values [0,0]: emb={embedding_weight[0,0].item():.6f}, lm={lm_head_weight[0,0].item():.6f}")
        print(f"  Sample values [100,100]: emb={embedding_weight[100,100].item():.6f}, lm={lm_head_weight[100,100].item():.6f}")
    
    # Verdict
    print("\n" + "="*60)
    if tie_weights_config is True:
        if same_tensor:
            print("✅ PASS: tie_weights=True and weights ARE the same tensor")
            return True
        else:
            print("❌ FAIL: tie_weights=True but weights are DIFFERENT tensors!")
            print("   This means tie_weights is not actually working.")
            return False
    elif tie_weights_config is False:
        if not same_tensor:
            print("✅ PASS: tie_weights=False and weights are different tensors")
            return True
        else:
            print("⚠️  WARNING: tie_weights=False but weights ARE the same tensor")
            print("   This is unusual but not necessarily wrong.")
            return True
    else:
        print(f"⚠️  WARNING: tie_weights not set in config (value: {tie_weights_config})")
        if same_tensor:
            print("   Weights ARE the same tensor (tie_weights effectively True)")
        else:
            print("   Weights are different tensors (tie_weights effectively False)")
        return True  # Not a failure, just informational


def main():
    parser = argparse.ArgumentParser(description="Check tie_weights correctness")
    parser.add_argument("--model", type=str, required=True, help="Model name to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show additional details")
    args = parser.parse_args()
    
    success = check_tie_weights(args.model, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
