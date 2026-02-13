#!/usr/bin/env python3
"""
Convert a checkpoint to use weight tying.

Weight tying shares weights between token_embedding_table and lm_head.
This helps special tokens learn faster during SFT.

Strategy options:
1. "embedding" - Use token_embedding_table weights for both (default)
2. "lm_head" - Use lm_head weights for both  
3. "average" - Average the two weight matrices

Usage:
    python scripts/enable_weight_tying.py --model domain_v5 --output domain_v5_tied --strategy embedding
"""

import argparse
import torch
import os
import sys
import shutil
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description="Enable weight tying on a checkpoint")
    parser.add_argument("--model", type=str, required=True, help="Source model name")
    parser.add_argument("--output", type=str, required=True, help="Output model name")
    parser.add_argument("--strategy", type=str, default="average",
                       choices=["embedding", "lm_head", "average"],
                       help="Which weights to use: embedding, lm_head, or average (default: average)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENABLE WEIGHT TYING")
    print("=" * 60)
    
    source_dir = f"checkpoints/{args.model}"
    output_dir = f"checkpoints/{args.output}"
    
    if not os.path.exists(source_dir):
        print(f"Error: Source model not found: {source_dir}")
        return
    
    # Load model
    print(f"\n1. Loading model '{args.model}'...")
    model, tokenizer_state, step, _ = GPT.load(source_dir)
    
    # Check current state
    emb_weight = model.token_embedding_table.weight
    lm_weight = model.lm_head.weight
    
    print(f"\n2. Current weight shapes:")
    print(f"   token_embedding_table: {emb_weight.shape}")
    print(f"   lm_head: {lm_weight.shape}")
    
    # Check if already tied
    if emb_weight.data_ptr() == lm_weight.data_ptr():
        print("\n⚠️  Weights are already tied! Nothing to do.")
        return
    
    # Compare weights
    diff = (emb_weight - lm_weight).norm().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        emb_weight.flatten().unsqueeze(0),
        lm_weight.flatten().unsqueeze(0)
    ).item()
    
    print(f"\n3. Weight comparison:")
    print(f"   L2 distance: {diff:.4f}")
    print(f"   Cosine similarity: {cos_sim:.6f}")
    
    # Create new weights based on strategy
    print(f"\n4. Applying strategy: '{args.strategy}'...")
    
    if args.strategy == "embedding":
        new_weight = emb_weight.clone()
        print("   Using token_embedding_table weights")
    elif args.strategy == "lm_head":
        new_weight = lm_weight.clone()
        print("   Using lm_head weights")
    else:  # average
        new_weight = (emb_weight + lm_weight) / 2
        print("   Using averaged weights")
    
    # Update model weights
    model.token_embedding_table.weight.data = new_weight
    model.lm_head.weight = model.token_embedding_table.weight  # Tie!
    
    # Verify tying worked
    assert model.token_embedding_table.weight.data_ptr() == model.lm_head.weight.data_ptr(), \
        "Weight tying failed!"
    
    print("   ✓ Weights are now tied")
    
    # Update config
    model.config.tie_weights = True
    
    # Copy source directory structure
    print(f"\n5. Saving to '{output_dir}'...")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model weights
    model_path = os.path.join(output_dir, "model.pt")
    state_dict = model.state_dict()
    
    # Get dtype info
    checkpoint_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    
    checkpoint_data = {
        "state_dict": state_dict,
        "checkpoint_dtype": checkpoint_dtype,
        "model_dtype": checkpoint_dtype,
    }
    torch.save(checkpoint_data, model_path)
    print(f"   Saved model.pt ({checkpoint_dtype})")
    
    # Save config with tie_weights=True
    config_path = os.path.join(output_dir, "config.json")
    config_dict = model.config.to_dict()
    config_dict['tie_weights'] = True  # Ensure it's set
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"   Saved config.json (tie_weights=True)")
    
    # Copy tokenizer
    src_tokenizer = os.path.join(source_dir, "tokenizer.json")
    if os.path.exists(src_tokenizer):
        shutil.copy(src_tokenizer, os.path.join(output_dir, "tokenizer.json"))
        print(f"   Copied tokenizer.json")
    
    # Create training state
    training_state = {
        "step": 0,
        "source_model": args.model,
        "tie_weights_strategy": args.strategy,
        "note": "Checkpoint converted to use weight tying"
    }
    with open(os.path.join(output_dir, "training_state.json"), 'w') as f:
        json.dump(training_state, f, indent=2)
    print(f"   Saved training_state.json")
    
    print(f"\n✅ Done! Weight-tied model saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"   python train.py --init_from_model {args.output} ...")


if __name__ == "__main__":
    main()
