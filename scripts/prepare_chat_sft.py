#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare chat SFT dataset with episode-indexed format and loss masking.

Reads JSONL with conversations, serializes to special token format,
and creates episode-indexed binary files with loss masks.

Input JSONL format:
    {"system": "...", "context": "...", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output (episode-indexed format):
    output_dir/
        train/tokens.bin        # uint32 token IDs (all episodes concatenated)
        train/mask.bin          # uint8 loss mask (aligned to tokens)
        train/episodes.idx      # uint64 pairs: (start, length) per episode
        val/tokens.bin
        val/mask.bin
        val/episodes.idx
        tokenizer_state.json
        dataset_metadata.json

For large datasets (>100M tokens), outputs multi-shard format:
    output_dir/
        train/shard_00000/tokens.bin
        train/shard_00000/mask.bin
        train/shard_00000/episodes.idx
        train/shard_00001/...
        ...

Usage:
    python scripts/prepare_chat_sft.py --input data/chat.jsonl --output_dir data/chat_sft
    
    # With custom settings
    python scripts/prepare_chat_sft.py --input data/chat.jsonl --output_dir data/chat_sft \\
        --val_split 0.1 --tokens_per_shard 50000000 --tokenization gpt2
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
from core.system_prompts import CONVERSATION_SYSTEM_PROMPT

# Audit logging for compliance
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# Get tags from special_tokens.py
SYSTEM_OPEN = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
SYSTEM_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_system_close"]
# NOTE: CONTEXT tags NOT used in Phase 3a - reserved for Phase 3b agentic SFT
USER_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
USER_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_close"]
ASSISTANT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
ASSISTANT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]
EOT = SPECIAL_TOKEN_STRINGS["myPT_eot"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare chat SFT dataset with episode-indexed format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with conversations")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for episode-indexed dataset")
    
    parser.add_argument("--tokenization", type=str, default="gpt2",
                        choices=["gpt2", "char"],
                        help="Tokenization method (default: gpt2)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--tokens_per_shard", type=int, default=50_000_000,
                        help="Max tokens per shard (default: 50M, for multi-shard support)")
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
    
    # System message (masked out) - always use centralized prompt for consistency
    # NO newlines between tags - tags alone define structure
    s = f"{SYSTEM_OPEN}{CONVERSATION_SYSTEM_PROMPT}{SYSTEM_CLOSE}"
    text_parts.append(s)
    mask_parts.append("0" * len(s))
    
    # NOTE: We do NOT include <myPT_user_context> here!
    # The JSONL "context" field is just metadata (episode_id, language), NOT RAG context.
    # <myPT_user_context> is reserved for Phase 3b agentic SFT where actual RAG data goes.
    
    # Messages
    for msg in item.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            u = f"{USER_OPEN}{content}{USER_CLOSE}"
            text_parts.append(u)
            mask_parts.append("0" * len(u))  # Don't train on user messages
        
        elif role == "assistant":
            # Opening tag: mask=0 (given in prompt, don't predict it)
            # Content + closing tag: mask=1 (model learns what to say and when to stop)
            text_parts.append(ASSISTANT_OPEN)
            mask_parts.append("0" * len(ASSISTANT_OPEN))  # Don't train on opening tag!
            
            content_and_close = f"{content}{ASSISTANT_CLOSE}"
            text_parts.append(content_and_close)
            mask_parts.append("1" * len(content_and_close))  # Train on content + closing tag
    
    # End of turn marker
    text_parts.append(EOT)
    mask_parts.append("0" * len(EOT))
    
    return "".join(text_parts), "".join(mask_parts)


def char_mask_to_token_mask(
    text: str, 
    char_mask: str, 
    tokenizer: Tokenizer
) -> Tuple[List[int], List[int]]:
    """
    Convert character-level mask to token-level mask.
    
    Strategy: Use token ID detection for robust masking.
    - Find <myPT_assistant> token (ID 50263) - mask=0
    - Everything after it until </myPT_assistant> (ID 50264) - mask=1
    - </myPT_assistant> itself - mask=1 (model learns to stop)
    - <myPT_eot> (ID 50271) - mask=0
    
    This avoids character/byte alignment issues with BPE tokenization.
    
    Returns:
        (token_ids, token_mask) where token_mask[i] is 1 if token i should be trained on
    """
    # Special token IDs
    ASSISTANT_OPEN_ID = 50263   # <myPT_assistant>
    ASSISTANT_CLOSE_ID = 50264  # </myPT_assistant>
    EOT_ID = 50271              # <myPT_eot>
    
    # Encode full text to get token IDs
    token_ids = tokenizer.encode(text)
    
    # Build mask based on token structure
    token_mask = []
    in_assistant_response = False
    
    for i, token_id in enumerate(token_ids):
        if token_id == ASSISTANT_OPEN_ID:
            # Opening tag - don't train on it
            token_mask.append(0)
            in_assistant_response = True
        elif token_id == ASSISTANT_CLOSE_ID:
            # Closing tag - DO train on it (model learns when to stop)
            token_mask.append(1)
            in_assistant_response = False
        elif token_id == EOT_ID:
            # End of turn - DO train on it (future-proofing for multi-turn)
            token_mask.append(1)
        elif in_assistant_response:
            # Inside assistant response - train on content
            token_mask.append(1)
        else:
            # System/user content - don't train
            token_mask.append(0)
    
    return token_ids, token_mask


class EpisodeShardWriter:
    """
    Writes episode-indexed shards with tokens, masks, and episode index.
    
    Handles multi-shard output when data exceeds tokens_per_shard.
    """
    
    def __init__(self, output_dir: str, split: str, tokens_per_shard: int):
        self.output_dir = output_dir
        self.split = split
        self.tokens_per_shard = tokens_per_shard
        
        # Current shard data
        self.shard_idx = 0
        self.tokens: List[int] = []
        self.masks: List[int] = []
        self.episodes: List[Tuple[int, int]] = []  # (start, length) pairs
        self.current_offset = 0  # Token offset in current shard
        
        # Statistics
        self.total_tokens = 0
        self.total_episodes = 0
        self.shards_written = 0
    
    def add_episode(self, token_ids: List[int], mask: List[int]):
        """Add an episode to the current shard."""
        episode_len = len(token_ids)
        
        # Check if we need to start a new shard
        if self.tokens and (len(self.tokens) + episode_len > self.tokens_per_shard):
            self._flush_shard()
        
        # Record episode boundary
        start = len(self.tokens)
        self.episodes.append((start, episode_len))
        
        # Append data
        self.tokens.extend(token_ids)
        self.masks.extend(mask)
        
        self.total_tokens += episode_len
        self.total_episodes += 1
    
    def _get_shard_dir(self) -> str:
        """Get directory for current shard."""
        split_dir = os.path.join(self.output_dir, self.split)
        
        # Use multi-shard format if we've written any shards already,
        # or if this is going to be a multi-shard dataset
        if self.shards_written > 0 or len(self.tokens) >= self.tokens_per_shard:
            return os.path.join(split_dir, f"shard_{self.shard_idx:05d}")
        else:
            # Single shard: write directly to split dir
            return split_dir
    
    def _flush_shard(self):
        """Write current shard to disk."""
        if not self.tokens:
            return
        
        shard_dir = self._get_shard_dir()
        os.makedirs(shard_dir, exist_ok=True)
        
        # Write tokens.bin (uint32)
        tokens_arr = np.array(self.tokens, dtype=np.uint32)
        tokens_path = os.path.join(shard_dir, "tokens.bin")
        tokens_arr.tofile(tokens_path)
        
        # Write mask.bin (uint8)
        masks_arr = np.array(self.masks, dtype=np.uint8)
        mask_path = os.path.join(shard_dir, "mask.bin")
        masks_arr.tofile(mask_path)
        
        # Write episodes.idx (uint64 pairs: start, length)
        # Format: each episode is 16 bytes (2 × uint64)
        episodes_arr = np.array(self.episodes, dtype=np.uint64)
        episodes_path = os.path.join(shard_dir, "episodes.idx")
        episodes_arr.tofile(episodes_path)
        
        print(f"  Written {self.split}/shard_{self.shard_idx:05d}: "
              f"{len(self.tokens):,} tokens, {len(self.episodes)} episodes")
        
        # Reset for next shard
        self.tokens = []
        self.masks = []
        self.episodes = []
        self.shard_idx += 1
        self.shards_written += 1
    
    def finalize(self) -> int:
        """Flush remaining data and return number of shards written."""
        self._flush_shard()
        return self.shards_written


def main():
    args = parse_args()
    
    from core.banner import print_banner
    print_banner("MyPT Chat SFT", "Episode-Indexed Dataset Preparer")
    
    # Audit: Dataset preparation started
    if AUDIT_AVAILABLE:
        audit.training(
            "dataset_prepare_start",
            dataset_type="chat_sft",
            input_file=args.input,
            output_dir=args.output_dir,
            tokenization=args.tokenization,
            details=f"Chat SFT dataset preparation started: {args.input}"
        )
    
    # Validate input
    if not os.path.exists(args.input):
        error_msg = f"Input file not found: {args.input}"
        print(f"Error: {error_msg}")
        if AUDIT_AVAILABLE:
            audit.training(
                "dataset_prepare_error",
                level=audit.AuditLevel.ERROR,
                dataset_type="chat_sft",
                error="file_not_found",
                details=error_msg
            )
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
    print(f"  Tokens per shard: {args.tokens_per_shard:,}")
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
        error_msg = "No valid conversations found in input file"
        print(f"Error: {error_msg}")
        if AUDIT_AVAILABLE:
            audit.training(
                "dataset_prepare_error",
                level=audit.AuditLevel.ERROR,
                dataset_type="chat_sft",
                input_file=args.input,
                error="no_conversations",
                details=error_msg
            )
        sys.exit(1)
    
    # Shuffle and split
    np.random.seed(args.seed)
    indices = np.random.permutation(len(conversations))
    
    val_size = int(len(conversations) * args.val_split)
    val_indices = set(indices[:val_size])
    
    print(f"  Train: {len(conversations) - val_size}, Val: {val_size}")
    
    # Initialize shard writers
    train_writer = EpisodeShardWriter(args.output_dir, "train", args.tokens_per_shard)
    val_writer = EpisodeShardWriter(args.output_dir, "val", args.tokens_per_shard)
    
    # Process conversations
    print("\nProcessing and tokenizing...")
    
    train_mask_sum = 0
    train_mask_count = 0
    val_mask_sum = 0
    val_mask_count = 0
    
    for i, conv in enumerate(conversations):
        # Serialize to text + char mask
        text, char_mask = serialize_conversation(conv)
        
        # Determine split for this episode
        split = "val" if i in val_indices else "train"
        episode_idx = val_writer.total_episodes if split == "val" else train_writer.total_episodes
        
        # Audit: Log episode text before tokenization (for traceability)
        if AUDIT_AVAILABLE:
            # Truncate text for log (first 500 chars) to avoid huge log entries
            text_preview = text[:500] + "..." if len(text) > 500 else text
            # Escape newlines for single-line log format
            text_preview_escaped = text_preview.replace("\n", "\\n")
            audit.training(
                "episode_text",
                dataset_type="chat_sft",
                split=split,
                episode_idx=episode_idx,
                conversation_idx=i,
                text_length=len(text),
                num_assistant_chars=char_mask.count("1"),
                details=text_preview_escaped
            )
        
        # Convert to token-level mask
        tokens, mask = char_mask_to_token_mask(text, char_mask, tokenizer)
        
        if i in val_indices:
            val_writer.add_episode(tokens, mask)
            val_mask_sum += sum(mask)
            val_mask_count += len(mask)
        else:
            train_writer.add_episode(tokens, mask)
            train_mask_sum += sum(mask)
            train_mask_count += len(mask)
        
        if args.verbose and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(conversations)} conversations")
    
    # Finalize shards
    print("\nWriting shards...")
    train_shards = train_writer.finalize()
    val_shards = val_writer.finalize()
    
    # Calculate statistics
    train_mask_ratio = train_mask_sum / train_mask_count if train_mask_count > 0 else 0
    val_mask_ratio = val_mask_sum / val_mask_count if val_mask_count > 0 else 0
    
    print(f"\n  Total train tokens: {train_writer.total_tokens:,}")
    print(f"  Total val tokens: {val_writer.total_tokens:,}")
    print(f"  Train mask ratio (assistant tokens): {train_mask_ratio:.1%}")
    print(f"  Val mask ratio (assistant tokens): {val_mask_ratio:.1%}")
    
    # Save tokenizer state
    tokenizer_path = os.path.join(args.output_dir, "tokenizer_state.json")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.get_state(), f, indent=2)
    print(f"  Saved tokenizer state: {tokenizer_path}")
    
    # Save metadata
    metadata = {
        "schema": "episode_indexed_sft_v1",
        "has_loss_mask": True,
        "num_conversations": len(conversations),
        "num_train_episodes": train_writer.total_episodes,
        "num_val_episodes": val_writer.total_episodes,
        "num_train_tokens": train_writer.total_tokens,
        "num_val_tokens": val_writer.total_tokens,
        "num_train_shards": train_shards,
        "num_val_shards": val_shards,
        "train_mask_ratio": train_mask_ratio,
        "val_mask_ratio": val_mask_ratio,
        "tokenization": args.tokenization,
        "vocab_size": args.vocab_size,
        "tokens_per_shard": args.tokens_per_shard,
        "val_split": args.val_split,
        "special_tokens_used": list(SPECIAL_TOKEN_STRINGS.keys()),
    }
    
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Episode-indexed SFT dataset prepared successfully!")
    print(f"   Conversations: {len(conversations)}")
    print(f"   Train: {train_writer.total_tokens:,} tokens, {train_writer.total_episodes} episodes ({train_shards} shard(s))")
    print(f"   Val: {val_writer.total_tokens:,} tokens, {val_writer.total_episodes} episodes ({val_shards} shard(s))")
    print(f"   Mask ratio: {train_mask_ratio:.1%} (assistant tokens)")
    print(f"   Output: {args.output_dir}")
    print("=" * 60)
    
    # Audit: Dataset preparation completed
    if AUDIT_AVAILABLE:
        audit.training(
            "dataset_prepare_complete",
            dataset_type="chat_sft",
            input_file=args.input,
            output_dir=args.output_dir,
            num_conversations=len(conversations),
            num_train_episodes=train_writer.total_episodes,
            num_val_episodes=val_writer.total_episodes,
            num_train_tokens=train_writer.total_tokens,
            num_val_tokens=val_writer.total_tokens,
            train_shards=train_shards,
            val_shards=val_shards,
            mask_ratio=round(train_mask_ratio, 4),
            details=f"Chat SFT dataset prepared: {len(conversations)} conversations, "
                    f"{train_writer.total_tokens:,} train tokens, {val_writer.total_tokens:,} val tokens"
        )
    
    print(f"\nTo train with this dataset:")
    print(f"  python train.py --model_name my_sft_model \\")
    print(f"      --dataset_dir {args.output_dir} \\")
    print(f"      --config_file configs/sft1/micro.json")


if __name__ == "__main__":
    main()
