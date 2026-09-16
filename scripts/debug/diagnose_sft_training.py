#!/usr/bin/env python3
"""
Diagnose SFT training pipeline.

Verifies:
1. Data loader returns (X, Y, mask) tuple
2. Mask has correct values (mix of 0s and 1s)
3. Mask is aligned correctly with Y (targets)
4. Model.forward() uses the mask correctly
5. Loss computation differs with vs without mask

Usage:
    python scripts/diagnose_sft_training.py --dataset data/sft_phase3a --model domain_v5
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT, GPTConfig
from core.episode_data_loader import GPTEpisodeDataLoader
from core.special_tokens import SPECIAL_TOKEN_STRINGS, BASE_VOCAB_SIZE


def main():
    parser = argparse.ArgumentParser(description="Diagnose SFT training pipeline")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset directory")
    parser.add_argument("--model", type=str, default=None, help="Base model to load (optional)")
    parser.add_argument("--show_episode", type=int, default=0, help="Episode index to display in detail")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SFT TRAINING PIPELINE DIAGNOSTIC")
    print("=" * 70)
    
    # Step 1: Load model or create dummy config
    print("\n[1] Loading model/config...")
    if args.model:
        model, _, _, _ = GPT.load(f"checkpoints/{args.model}")
        config = model.config
        print(f"    Loaded model: {args.model}")
        print(f"    use_loss_mask in config: {config.use_loss_mask}")
    else:
        config = GPTConfig(
            batch_size=4,
            block_size=1024,
            vocab_size=50304,
            use_loss_mask=True
        )
        model = None
        print("    Using dummy config (no model loaded)")
    
    # Force use_loss_mask for this test
    config.use_loss_mask = True
    print(f"    Forcing use_loss_mask=True for diagnostic")
    
    # Step 2: Create data loader
    print("\n[2] Creating episode data loader...")
    
    # Need a tokenizer
    from core.tokenizer import Tokenizer
    tokenizer = Tokenizer(config, kind="gpt2")
    
    loader = GPTEpisodeDataLoader(
        config=config,
        tokenizer=tokenizer,
        dataset_dir=args.dataset,
        use_loss_mask=True
    )
    
    # Step 3: Get a batch and inspect
    print("\n[3] Getting a batch...")
    batch = loader.get_batch('train')
    
    print(f"    Batch type: {type(batch)}")
    print(f"    Batch length: {len(batch)}")
    
    if len(batch) == 3:
        X, Y, mask = batch
        print(f"    ✓ Got 3-tuple (X, Y, mask)")
    elif len(batch) == 2:
        X, Y = batch
        mask = None
        print(f"    ✗ Got 2-tuple (X, Y) - MASK NOT RETURNED!")
        print(f"    This means the data loader isn't providing the mask!")
        return
    else:
        print(f"    ✗ Unexpected batch format!")
        return
    
    print(f"\n    X shape: {X.shape}")
    print(f"    Y shape: {Y.shape}")
    print(f"    mask shape: {mask.shape}")
    print(f"    mask dtype: {mask.dtype}")
    
    # Step 4: Analyze mask
    print("\n[4] Analyzing mask...")
    mask_np = mask.cpu().numpy()
    
    total = mask_np.size
    ones = (mask_np == 1).sum()
    zeros = (mask_np == 0).sum()
    other = total - ones - zeros
    
    print(f"    Total positions: {total}")
    print(f"    mask=1 (train): {ones} ({100*ones/total:.1f}%)")
    print(f"    mask=0 (ignore): {zeros} ({100*zeros/total:.1f}%)")
    if other > 0:
        print(f"    other values: {other} (UNEXPECTED!)")
    
    if ones == 0:
        print(f"    ✗ ERROR: No mask=1 positions! Model won't learn anything!")
        return
    elif ones == total:
        print(f"    ✗ WARNING: All mask=1! This is like no masking at all.")
    else:
        print(f"    ✓ Mask has mix of 0s and 1s")
    
    # Step 5: Check mask alignment with special tokens
    print("\n[5] Checking mask alignment with special tokens...")
    
    # Get special token IDs
    assistant_open_id = tokenizer.special_tokens.get('myPT_assistant_open')
    assistant_close_id = tokenizer.special_tokens.get('myPT_assistant_close')
    user_close_id = tokenizer.special_tokens.get('myPT_user_close')
    
    print(f"    <myPT_assistant> ID: {assistant_open_id}")
    print(f"    </myPT_assistant> ID: {assistant_close_id}")
    print(f"    </myPT_user> ID: {user_close_id}")
    
    # Check first sample in batch
    y_sample = Y[0].cpu().numpy()
    mask_sample = mask[0].cpu().numpy()
    
    # Find positions of special tokens in Y (targets)
    for i, (tok, m) in enumerate(zip(y_sample[:100], mask_sample[:100])):
        if tok == assistant_open_id:
            print(f"    Position {i}: Y=<myPT_assistant>, mask={m} {'✓' if m == 1 else '✗ SHOULD BE 1!'}")
        elif tok == assistant_close_id:
            print(f"    Position {i}: Y=</myPT_assistant>, mask={m} {'✓' if m == 1 else '✗ SHOULD BE 1!'}")
        elif tok == user_close_id:
            print(f"    Position {i}: Y=</myPT_user>, mask={m} {'✓' if m == 0 else '✗ SHOULD BE 0!'}")
    
    # Step 6: Test model forward with and without mask
    if model is not None:
        print("\n[6] Testing model forward pass...")
        
        model.eval()
        with torch.no_grad():
            # With mask
            _, loss_with_mask, _ = model(X, Y, loss_mask=mask)
            
            # Without mask
            _, loss_without_mask, _ = model(X, Y, loss_mask=None)
            
            # With all-ones mask (equivalent to no mask)
            all_ones = torch.ones_like(mask)
            _, loss_all_ones, _ = model(X, Y, loss_mask=all_ones)
        
        print(f"    Loss WITH mask: {loss_with_mask.item():.4f}")
        print(f"    Loss WITHOUT mask: {loss_without_mask.item():.4f}")
        print(f"    Loss with all-ones mask: {loss_all_ones.item():.4f}")
        
        if abs(loss_with_mask.item() - loss_without_mask.item()) < 0.01:
            print(f"    ✗ WARNING: Losses are nearly identical! Mask might not be working!")
        else:
            print(f"    ✓ Losses differ, mask is having an effect")
        
        if abs(loss_without_mask.item() - loss_all_ones.item()) < 0.01:
            print(f"    ✓ All-ones mask ≈ no mask (expected)")
        else:
            print(f"    ? All-ones mask differs from no mask (unexpected)")
    
    # Step 7: Decode an actual episode and show structure
    print("\n[7] Decoding training episode to check structure...")
    
    # Load raw episode data
    import numpy as np
    train_dir = os.path.join(args.dataset, "train")
    tokens_file = os.path.join(train_dir, "tokens.bin")
    mask_file = os.path.join(train_dir, "mask.bin")
    episodes_file = os.path.join(train_dir, "episodes.idx")
    
    if os.path.exists(tokens_file) and os.path.exists(episodes_file):
        tokens_raw = np.memmap(tokens_file, dtype=np.uint32, mode='r')
        episodes_raw = np.memmap(episodes_file, dtype=np.uint64, mode='r').reshape(-1, 2)
        
        mask_raw = None
        if os.path.exists(mask_file):
            mask_raw = np.memmap(mask_file, dtype=np.uint8, mode='r')
        
        ep_idx = min(args.show_episode, len(episodes_raw) - 1)
        start, length = episodes_raw[ep_idx]
        start, length = int(start), int(length)
        
        ep_tokens = tokens_raw[start:start+length]
        ep_mask = mask_raw[start:start+length] if mask_raw is not None else None
        
        print(f"\n    Episode {ep_idx}: {length} tokens (offset {start})")
        
        # Decode and show
        text = tokenizer.decode(list(ep_tokens))
        print(f"\n    === DECODED TEXT ===")
        print("    " + text.replace("\n", "\n    ")[:2000])  # First 2000 chars
        
        # Show special token positions
        print(f"\n    === SPECIAL TOKEN POSITIONS ===")
        for i, tok in enumerate(ep_tokens[:200]):  # First 200 tokens
            tok = int(tok)
            if tok >= BASE_VOCAB_SIZE:  # Special token range
                tok_str = tokenizer.decode([tok])
                m = ep_mask[i] if ep_mask is not None else "N/A"
                print(f"    Position {i}: token={tok} '{tok_str}' mask={m}")
        
        # Show mask transitions
        if ep_mask is not None:
            print(f"\n    === MASK TRANSITIONS ===")
            prev_m = -1
            for i, (tok, m) in enumerate(zip(ep_tokens, ep_mask)):
                if m != prev_m:
                    tok_str = tokenizer.decode([int(tok)])
                    direction = "START>>>" if m == 1 else "<<<END"
                    print(f"    Position {i}: {direction} mask={m}, token={tok} '{repr(tok_str)}'")
                prev_m = m
    else:
        print(f"    Cannot find episode data files in {train_dir}")
    
    # Step 8: Test special token round-trip
    print("\n[8] Testing special token encoding round-trip...")
    test_text = "<myPT_system>Test system</myPT_system>\n<myPT_user>Test user</myPT_user>\n<myPT_assistant>Test response</myPT_assistant>\n<myPT_eot>\n"
    
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"    Original: {repr(test_text[:100])}")
    print(f"    Encoded:  {encoded[:20]}...")
    print(f"    Decoded:  {repr(decoded[:100])}")
    
    if test_text == decoded:
        print(f"    ✓ Perfect round-trip!")
    else:
        print(f"    ✗ MISMATCH! Text changed during encode/decode!")
        print(f"    This could cause training/inference discrepancy!")
    
    # Check special tokens are single tokens
    print(f"\n    Checking special tokens are single IDs:")
    for tag_name, tag_str in [
        ("system_open", "<myPT_system>"),
        ("system_close", "</myPT_system>"),
        ("user_open", "<myPT_user>"),
        ("user_close", "</myPT_user>"),
        ("assistant_open", "<myPT_assistant>"),
        ("assistant_close", "</myPT_assistant>"),
        ("eot", "<myPT_eot>"),
    ]:
        enc = tokenizer.encode(tag_str)
        if len(enc) == 1:
            print(f"    ✓ {tag_str} -> single token {enc[0]}")
        else:
            print(f"    ✗ {tag_str} -> MULTIPLE tokens {enc}! BUG!")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
