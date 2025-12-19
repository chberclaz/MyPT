#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare chat SFT dataset with loss masking.

Reads JSONL with conversations, serializes to special token format,
and creates shards with parallel loss mask files.

Input JSONL format:
    {"system": "...", "context": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output:
    output_dir/
        train/shard_00000.bin
        train/shard_00000_mask.bin
        val/shard_00000.bin
        val/shard_00000_mask.bin
        tokenizer_state.json
        dataset_metadata.json

Usage:
    python scripts/prepare_chat_sft.py --input data/chat.jsonl --output_dir data/chat_sft
    
    # With custom settings
    python scripts/prepare_chat_sft.py --input data/chat.jsonl --output_dir data/chat_sft \
        --val_split 0.1 --shard_size 50000000 --tokenization gpt2
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.special_tokens import SPECIAL_TOKEN_STRINGS
from core.tokenizer import Tokenizer
from core.model import GPTConfig


# Get tags from special_tokens.py
SYSTEM_OPEN = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
SYSTEM_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_system_close"]
CONTEXT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_context_open"]
CONTEXT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_context_close"]
USER_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
USER_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_close"]
ASSISTANT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
ASSISTANT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]
EOT = SPECIAL_TOKEN_STRINGS["myPT_eot"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare chat SFT dataset with loss masking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with conversations")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shards")
    
    parser.add_argument("--tokenization", type=str, default="gpt2",
                        choices=["gpt2", "char"],
                        help="Tokenization method (default: gpt2)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--shard_size", type=int, default=50_000_000,
                        help="Target shard size in bytes (default: 50MB)")
    parser.add_argument("--vocab_size", type=int, default=50304,
                        help="Vocab size for tokenizer config (default: 50304)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def serialize_conversation(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Serialize a conversation to text + char-level mask.
    
    Returns:
        (text, char_mask) where char_mask has '1' for assistant chars, '0' otherwise
    """
    text_parts = []
    mask_parts = []
    
    # System message (masked out)
    if item.get("system"):
        s = f"{SYSTEM_OPEN}{item['system']}{SYSTEM_CLOSE}\n"
        text_parts.append(s)
        mask_parts.append("0" * len(s))
    
    # Context (masked out) - used for RAG
    if item.get("context"):
        c = f"{CONTEXT_OPEN}{item['context']}{CONTEXT_CLOSE}\n"
        text_parts.append(c)
        mask_parts.append("0" * len(c))
    
    # Messages
    for msg in item.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            u = f"{USER_OPEN}{content}{USER_CLOSE}\n"
            text_parts.append(u)
            mask_parts.append("0" * len(u))  # Don't train on user messages
        
        elif role == "assistant":
            # Include tags in training - model needs to learn the format
            a = f"{ASSISTANT_OPEN}{content}{ASSISTANT_CLOSE}\n"
            text_parts.append(a)
            mask_parts.append("1" * len(a))  # Train on assistant responses!
    
    # End of turn marker
    text_parts.append(EOT + "\n")
    mask_parts.append("0" * (len(EOT) + 1))
    
    return "".join(text_parts), "".join(mask_parts)


def char_mask_to_token_mask(
    text: str, 
    char_mask: str, 
    tokenizer: Tokenizer
) -> Tuple[List[int], List[int]]:
    """
    Convert character-level mask to token-level mask.
    
    Strategy: For each token, if ANY of its characters are masked as '1',
    the entire token is masked as 1 (trainable).
    
    Returns:
        (token_ids, token_mask) where token_mask[i] is 1 if token i should be trained on
    """
    assert len(text) == len(char_mask), f"Length mismatch: {len(text)} vs {len(char_mask)}"
    
    # Encode text to get token IDs
    token_ids = tokenizer.encode(text)
    
    # Decode each token to get its character span
    # This is a bit tricky - we need to map tokens back to character positions
    
    token_mask = []
    char_pos = 0
    
    for token_id in token_ids:
        # Get the text for this token
        token_text = tokenizer.decode([token_id])
        token_len = len(token_text)
        
        # Check if any character in this token's span is masked as '1'
        if char_pos + token_len <= len(char_mask):
            span_mask = char_mask[char_pos:char_pos + token_len]
            # If any char is '1', mark token as trainable
            token_mask.append(1 if '1' in span_mask else 0)
        else:
            # Edge case: token extends beyond mask (shouldn't happen)
            token_mask.append(0)
        
        char_pos += token_len
    
    return token_ids, token_mask


def save_shard(
    tokens: np.ndarray, 
    masks: np.ndarray, 
    shard_path: str
) -> None:
    """Save token and mask shards."""
    # Save tokens
    tokens.astype(np.uint16).tofile(shard_path)
    
    # Save masks with _mask suffix
    mask_path = shard_path.replace('.bin', '_mask.bin')
    masks.astype(np.uint8).tofile(mask_path)


def main():
    args = parse_args()
    
    from core.banner import print_banner
    print_banner("MyPT Chat SFT", "Conversation Dataset Preparer")
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directories
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Initialize tokenizer
    config = GPTConfig(vocab_size=args.vocab_size)
    tokenizer = Tokenizer(config, args.tokenization)
    
    print(f"\nConfiguration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output_dir}")
    print(f"  Tokenization: {args.tokenization}")
    print(f"  Val split: {args.val_split}")
    print()
    
    # Load and process conversations
    print("Loading conversations...")
    conversations = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                conversations.append(item)
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num} (invalid JSON): {e}")
    
    print(f"  Loaded {len(conversations)} conversations")
    
    if not conversations:
        print("Error: No valid conversations found")
        sys.exit(1)
    
    # Shuffle and split
    np.random.seed(args.seed)
    indices = np.random.permutation(len(conversations))
    
    val_size = int(len(conversations) * args.val_split)
    val_indices = set(indices[:val_size])
    
    print(f"  Train: {len(conversations) - val_size}, Val: {val_size}")
    
    # Process conversations
    print("\nProcessing and tokenizing...")
    
    train_tokens = []
    train_masks = []
    val_tokens = []
    val_masks = []
    
    total_train_tokens = 0
    total_val_tokens = 0
    
    for i, conv in enumerate(conversations):
        # Serialize to text + char mask
        text, char_mask = serialize_conversation(conv)
        
        # Convert to token-level mask
        tokens, mask = char_mask_to_token_mask(text, char_mask, tokenizer)
        
        if i in val_indices:
            val_tokens.extend(tokens)
            val_masks.extend(mask)
            total_val_tokens += len(tokens)
        else:
            train_tokens.extend(tokens)
            train_masks.extend(mask)
            total_train_tokens += len(tokens)
        
        if args.verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(conversations)} conversations")
    
    print(f"  Total train tokens: {total_train_tokens:,}")
    print(f"  Total val tokens: {total_val_tokens:,}")
    
    # Calculate mask statistics
    train_mask_ratio = sum(train_masks) / len(train_masks) if train_masks else 0
    val_mask_ratio = sum(val_masks) / len(val_masks) if val_masks else 0
    print(f"  Train mask ratio (assistant tokens): {train_mask_ratio:.1%}")
    print(f"  Val mask ratio (assistant tokens): {val_mask_ratio:.1%}")
    
    # Save shards
    print("\nSaving shards...")
    
    def save_split(tokens, masks, output_dir, split_name):
        if not tokens:
            return 0
        
        tokens_arr = np.array(tokens, dtype=np.uint16)
        masks_arr = np.array(masks, dtype=np.uint8)
        
        # Calculate number of shards
        bytes_per_token = 2  # uint16
        tokens_per_shard = args.shard_size // bytes_per_token
        num_shards = (len(tokens_arr) + tokens_per_shard - 1) // tokens_per_shard
        
        for shard_idx in range(num_shards):
            start = shard_idx * tokens_per_shard
            end = min((shard_idx + 1) * tokens_per_shard, len(tokens_arr))
            
            shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
            save_shard(tokens_arr[start:end], masks_arr[start:end], shard_path)
            
            print(f"  Saved {split_name}/shard_{shard_idx:05d}.bin ({end - start:,} tokens)")
        
        return num_shards
    
    train_shards = save_split(train_tokens, train_masks, train_dir, "train")
    val_shards = save_split(val_tokens, val_masks, val_dir, "val")
    
    # Save tokenizer state
    tokenizer_path = os.path.join(args.output_dir, "tokenizer_state.json")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.get_state(), f, indent=2)
    print(f"  Saved tokenizer state: {tokenizer_path}")
    
    # Save metadata
    metadata = {
        "num_conversations": len(conversations),
        "num_train_tokens": total_train_tokens,
        "num_val_tokens": total_val_tokens,
        "num_train_shards": train_shards,
        "num_val_shards": val_shards,
        "train_mask_ratio": train_mask_ratio,
        "val_mask_ratio": val_mask_ratio,
        "tokenization": args.tokenization,
        "vocab_size": args.vocab_size,
        "val_split": args.val_split,
        "shard_size": args.shard_size,
        "special_tokens_used": list(SPECIAL_TOKEN_STRINGS.keys()),
    }
    
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ SFT Dataset prepared successfully!")
    print(f"   Conversations: {len(conversations)}")
    print(f"   Train tokens: {total_train_tokens:,} ({train_shards} shards)")
    print(f"   Val tokens: {total_val_tokens:,} ({val_shards} shards)")
    print(f"   Mask ratio: {train_mask_ratio:.1%} (assistant tokens)")
    print(f"   Output: {args.output_dir}")
    print("=" * 60)
    
    print(f"\nTo train with loss masking:")
    print(f"  python train.py --model_name my_sft_model \\")
    print(f"      --dataset_dir {args.output_dir} \\")
    print(f"      --config_file configs/sft1/micro.json")


if __name__ == "__main__":
    main()

