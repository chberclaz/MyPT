#!/usr/bin/env python3
"""
Initialize special token embeddings from meaningful tokens.

Special tokens (50257-50271) start with random embeddings because they were
never used during pre-training or domain adaptation. This script initializes
them from semantically related existing tokens.

Usage:
    python scripts/init_special_embeddings.py --model domain_v5 --output domain_v5_sft_ready
"""

import argparse
import os
import sys
import torch
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT


# Map special tokens to meaningful source tokens
# We use tokens that appear in similar contexts to bootstrap the embeddings
INIT_SOURCES = {
    # Format: special_token_id -> [list of source token texts to average]
    50257: ["system", ":", "<"],           # <myPT_system>
    50258: ["system", ">", "/"],           # </myPT_system>  
    50259: ["user", ":", "<"],             # <myPT_user>
    50260: ["user", ">", "/"],             # </myPT_user>
    50261: ["context", ":", "<"],          # <myPT_user_context>
    50262: ["context", ">", "/"],          # </myPT_user_context>
    50263: ["assistant", ":", "<"],        # <myPT_assistant>
    50264: ["assistant", ">", "/"],        # </myPT_assistant>
    50265: ["tool", "call", "<"],          # <myPT_toolcall>
    50266: ["tool", "call", ">"],          # </myPT_toolcall>
    50267: ["result", ":", "<"],           # <myPT_toolresult>
    50268: ["result", ">", "/"],           # </myPT_toolresult>
    50269: ["think", ":", "<"],            # <myPT_thinking>
    50270: ["think", ">", "/"],            # </myPT_thinking>
    50271: ["end", ".", "\n"],             # <myPT_eot>
}

TOKEN_NAMES = {
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


def main():
    parser = argparse.ArgumentParser(description="Initialize special token embeddings")
    parser.add_argument("--model", type=str, required=True, help="Source model name")
    parser.add_argument("--output", type=str, required=True, help="Output model name")
    parser.add_argument("--scale", type=float, default=1.0, 
                        help="Scale factor for initialized embeddings (default: 1.0)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SPECIAL TOKEN EMBEDDING INITIALIZATION")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model '{args.model}'...")
    model, tokenizer_state, step, opt_state = GPT.load(f"checkpoints/{args.model}")
    tokenizer = model.tokenizer
    
    # Get embedding table
    emb = model.token_embedding_table.weight.data
    lm_head = model.lm_head.weight.data
    
    print(f"Embedding shape: {emb.shape}")
    print(f"LM head shape: {lm_head.shape}")
    
    # Store original norms for comparison
    orig_norms = {}
    for tid in range(50257, 50272):
        if tid < emb.shape[0]:
            orig_norms[tid] = torch.norm(emb[tid]).item()
    
    print("\n" + "=" * 70)
    print("Initializing special token embeddings...")
    print("=" * 70)
    
    for tid, source_texts in INIT_SOURCES.items():
        if tid >= emb.shape[0]:
            continue
            
        # Encode source tokens and get their embeddings
        source_embeds = []
        source_ids = []
        for text in source_texts:
            try:
                encoded = tokenizer.encode(text)
                for enc_id in encoded:
                    if enc_id < 50257:  # Only use regular tokens
                        source_embeds.append(emb[enc_id].clone())
                        source_ids.append(enc_id)
            except:
                pass
        
        if source_embeds:
            # Average the source embeddings
            avg_emb = torch.stack(source_embeds).mean(dim=0) * args.scale
            
            # Update embedding table
            old_norm = torch.norm(emb[tid]).item()
            emb[tid] = avg_emb
            new_norm = torch.norm(emb[tid]).item()
            
            # Also update lm_head (output projection) for consistency
            lm_head[tid] = avg_emb
            
            name = TOKEN_NAMES.get(tid, "?")
            print(f"{tid} {name:<25} norm: {old_norm:.3f} -> {new_norm:.3f} "
                  f"(from {len(source_embeds)} tokens: {source_ids[:5]}...)")
        else:
            print(f"{tid} {TOKEN_NAMES.get(tid, '?'):<25} SKIPPED (no valid source tokens)")
    
    # Save the modified model
    output_dir = f"checkpoints/{args.output}"
    print(f"\nSaving to {output_dir}...")
    
    # Copy the original checkpoint directory
    source_dir = f"checkpoints/{args.model}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    
    # Save updated model weights
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    print(f"\n✅ Saved model with initialized special token embeddings to: {output_dir}")
    
    # Verify the changes
    print("\n" + "=" * 70)
    print("Verification - comparing original vs initialized embeddings:")
    print("=" * 70)
    
    for tid in range(50257, 50272):
        if tid < emb.shape[0]:
            old_norm = orig_norms.get(tid, 0)
            new_norm = torch.norm(emb[tid]).item()
            name = TOKEN_NAMES.get(tid, "?")
            changed = "YES" if abs(new_norm - old_norm) > 0.1 else "NO"
            print(f"{tid} {name:<25} {old_norm:.3f} -> {new_norm:.3f} Changed: {changed}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Run SFT training from the initialized model:")
    print(f"     python train.py --init_from_model {args.output} ...")
    print(f"  2. The special tokens now have meaningful embeddings as starting point")


if __name__ == "__main__":
    main()
