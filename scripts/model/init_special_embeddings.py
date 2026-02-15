#!/usr/bin/env python3
"""
Initialize special token embeddings from meaningful tokens.

Special tokens start with random embeddings because they were never used
during pre-training. This script initializes them from semantically related
existing tokens.

Token IDs are resolved dynamically from core/special_tokens.py -- never
hardcoded.

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
from core.special_tokens import (
    SPECIAL_TOKEN_STRINGS, get_special_token_ids, BASE_VOCAB_SIZE,
)

# Dynamic token ID lookup
_IDS = get_special_token_ids()
_N_SPECIAL = len(SPECIAL_TOKEN_STRINGS)

# TOKEN_NAMES: id -> string (built dynamically from the canonical source)
TOKEN_NAMES = {tid: SPECIAL_TOKEN_STRINGS[name]
               for name, tid in _IDS.items()}

# Map special tokens to meaningful source tokens (for embedding initialization).
# We use tokens that appear in similar contexts to bootstrap the embeddings.
#
# IMPORTANT: Only initialize tokens that appear in Phase 3a training data!
# Unused tokens should keep their random embeddings so the model doesn't generate them.

# Phase 3a tokens (conversational, no tools):
INIT_SOURCES_PHASE3A = {
    _IDS["myPT_system_open"]:    ["system", ":", "<"],
    _IDS["myPT_system_close"]:   ["system", ">", "/"],
    _IDS["myPT_user_open"]:      ["user", ":", "<"],
    _IDS["myPT_user_close"]:     ["user", ">", "/"],
    _IDS["myPT_assistant_open"]: ["assistant", ":", "<"],
    _IDS["myPT_assistant_close"]:["assistant", ">", "/"],
    _IDS["myPT_eot"]:            ["end", ".", "\n"],
}

# Phase 3b tokens (agentic with tools) - add these later:
INIT_SOURCES_PHASE3B = {
    _IDS["myPT_user_context_open"]:      ["context", ":", "<"],
    _IDS["myPT_user_context_close"]:     ["context", ">", "/"],
    _IDS["myPT_assistant_context_open"]: ["context", "assistant", "<"],
    _IDS["myPT_assistant_context_close"]:["context", "assistant", ">"],
    _IDS["myPT_toolcall_open"]:          ["tool", "call", "<"],
    _IDS["myPT_toolcall_close"]:         ["tool", "call", ">"],
    _IDS["myPT_toolresult_open"]:        ["result", ":", "<"],
    _IDS["myPT_toolresult_close"]:       ["result", ">", "/"],
    _IDS["myPT_think_open"]:             ["think", ":", "<"],
    _IDS["myPT_think_close"]:            ["think", ">", "/"],
    _IDS["myPT_cite_open"]:              ["cite", "source", "<"],
    _IDS["myPT_cite_close"]:             ["cite", "source", ">"],
}

# Default: only Phase 3a tokens
INIT_SOURCES = INIT_SOURCES_PHASE3A


def main():
    parser = argparse.ArgumentParser(description="Initialize special token embeddings")
    parser.add_argument("--model", type=str, required=True, help="Source model name")
    parser.add_argument("--output", type=str, required=True, help="Output model name")
    parser.add_argument("--scale", type=float, default=1.0, 
                        help="Scale factor for initialized embeddings (default: 1.0)")
    parser.add_argument("--phase", type=str, default="3a", choices=["3a", "3b", "all"],
                        help="Which phase's tokens to initialize: 3a (conversational), 3b (agentic), all")
    args = parser.parse_args()
    
    # Select which tokens to initialize based on phase
    global INIT_SOURCES
    if args.phase == "3a":
        INIT_SOURCES = INIT_SOURCES_PHASE3A
        print(f"Initializing Phase 3a tokens only (conversational)")
    elif args.phase == "3b":
        INIT_SOURCES = {**INIT_SOURCES_PHASE3A, **INIT_SOURCES_PHASE3B}
        print(f"Initializing Phase 3a + 3b tokens (conversational + agentic)")
    else:  # all
        INIT_SOURCES = {**INIT_SOURCES_PHASE3A, **INIT_SOURCES_PHASE3B}
        print(f"Initializing all special tokens")
    
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
    for tid in range(BASE_VOCAB_SIZE, BASE_VOCAB_SIZE + _N_SPECIAL):
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
                    if enc_id < BASE_VOCAB_SIZE:  # Only use regular tokens
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
    
    # Copy the original checkpoint directory (preserves config.json, tokenizer.json, etc.)
    source_dir = f"checkpoints/{args.model}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    
    # Load the original checkpoint to get the format
    original_checkpoint = torch.load(os.path.join(source_dir, "model.pt"), map_location="cpu", weights_only=True)
    
    # Update the state_dict in the checkpoint
    # Keep the same format (dict with state_dict, checkpoint_dtype, model_dtype)
    if isinstance(original_checkpoint, dict) and "state_dict" in original_checkpoint:
        # New format checkpoint
        original_checkpoint["state_dict"] = model.state_dict()
        # Move to CPU to avoid VRAM issues
        for k, v in original_checkpoint["state_dict"].items():
            if torch.is_tensor(v):
                original_checkpoint["state_dict"][k] = v.cpu()
        torch.save(original_checkpoint, os.path.join(output_dir, "model.pt"))
        print(f"  Saved in new checkpoint format (with state_dict wrapper)")
    else:
        # Old format - just save the state dict directly
        sd = model.state_dict()
        for k, v in sd.items():
            if torch.is_tensor(v):
                sd[k] = v.cpu()
        torch.save(sd, os.path.join(output_dir, "model.pt"))
        print(f"  Saved in old checkpoint format (raw state_dict)")
    
    print(f"\n✅ Saved model with initialized special token embeddings to: {output_dir}")
    
    # Verify the changes
    print("\n" + "=" * 70)
    print("Verification - comparing original vs initialized embeddings:")
    print("=" * 70)
    
    for tid in range(BASE_VOCAB_SIZE, BASE_VOCAB_SIZE + _N_SPECIAL):
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
