#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare toolcall SFT dataset with episode-indexed format and loss masking.

Reads JSONL with tool-augmented conversations, serializes to special token format,
and creates episode-indexed binary files with loss masks.

Input JSONL format:
    {
        "system": "You are MyPT workspace assistant...",
        "messages": [
            {"role": "user", "content": "Find docs about X"},
            {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": "X"}},
            {"role": "toolresult", "name": "workspace.search", "content": {"documents": [...]}},
            {"role": "assistant", "content": "Here is what I found..."}
        ]
    }

Mask rules:
    - assistant, assistant_toolcall → mask = 1 (train on model outputs)
    - user, system, toolresult → mask = 0 (don't train)

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
        ...

Usage:
    python scripts/prepare_tool_sft.py --input data/tool_conversations.jsonl --output_dir data/tool_sft
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from core.tokenizer import Tokenizer
from core.model import GPTConfig

# Audit logging for compliance
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# Get tags from special_tokens.py
SYSTEM_OPEN = SPECIAL_TOKEN_STRINGS["myPT_system_open"]
SYSTEM_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_system_close"]
USER_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_open"]
USER_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_close"]
ASSISTANT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_open"]
ASSISTANT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]
TOOLCALL_OPEN = SPECIAL_TOKEN_STRINGS["myPT_toolcall_open"]
TOOLCALL_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_toolcall_close"]
TOOLRESULT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"]
TOOLRESULT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"]
USER_CONTEXT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_user_context_open"]
USER_CONTEXT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_user_context_close"]
ASSISTANT_CONTEXT_OPEN = SPECIAL_TOKEN_STRINGS["myPT_assistant_context_open"]
ASSISTANT_CONTEXT_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_assistant_context_close"]
THINK_OPEN = SPECIAL_TOKEN_STRINGS["myPT_think_open"]
THINK_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_think_close"]
CITE_OPEN = SPECIAL_TOKEN_STRINGS["myPT_cite_open"]
CITE_CLOSE = SPECIAL_TOKEN_STRINGS["myPT_cite_close"]
EOT = SPECIAL_TOKEN_STRINGS["myPT_eot"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare toolcall SFT dataset with episode-indexed format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with tool conversations")
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
    
    # Packing options
    parser.add_argument("--enable_packing", action="store_true",
                        help="Enable sequence packing (fill block_size with multiple episodes)")
    parser.add_argument("--pack_block_size", type=int, default=1024,
                        help="Target packed sequence length (default: 1024)")
    
    # Weighted loss masking (WeFT-style)
    parser.add_argument("--weighted_mask", action="store_true",
                        help="Use weighted loss masks: structural tokens (stop, think, cite, toolcall) "
                             "get higher weights (1.5-2.0x) to emphasize generation steering decisions.")
    
    return parser.parse_args()


def serialize_toolcall(name: str, arguments: dict) -> str:
    """Serialize a toolcall to the expected format."""
    data = {"name": name, **arguments}
    json_str = json.dumps(data, ensure_ascii=False)
    return f"{TOOLCALL_OPEN}{json_str}{TOOLCALL_CLOSE}"


def serialize_toolresult(content: Any) -> str:
    """Serialize a tool result to the expected format."""
    if isinstance(content, str):
        # Already a string (might be JSON string)
        try:
            json.loads(content)  # Validate it's JSON
            return f"{TOOLRESULT_OPEN}{content}{TOOLRESULT_CLOSE}"
        except:
            json_str = json.dumps(content, ensure_ascii=False)
            return f"{TOOLRESULT_OPEN}{json_str}{TOOLRESULT_CLOSE}"
    else:
        json_str = json.dumps(content, ensure_ascii=False)
        return f"{TOOLRESULT_OPEN}{json_str}{TOOLRESULT_CLOSE}"


def serialize_conversation(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    Serialize a tool conversation to text + char-level mask.
    
    Handles all myPT tag types including think, cite, user_context,
    and assistant_context. See docs/sft/TAG_NESTING_REFERENCE.md for
    canonical nesting rules.
    
    JSONL fields that trigger new tags:
        msg["context"]  → <myPT_user_context> inside user block
        msg["think"]    → <myPT_think> at start of assistant block
        msg["cite"]     → <myPT_cite> at end of assistant content
        role "assistant_context" → standalone <myPT_assistant_context>
    
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
    
    # Messages
    for msg in item.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            # Build user block, optionally with user_context inside
            user_context = msg.get("context", "")
            if user_context:
                ctx = f"{USER_CONTEXT_OPEN}{user_context}{USER_CONTEXT_CLOSE}"
                u = f"{USER_OPEN}{ctx}{content}{USER_CLOSE}\n"
            else:
                u = f"{USER_OPEN}{content}{USER_CLOSE}\n"
            text_parts.append(u)
            mask_parts.append("0" * len(u))
        
        elif role == "assistant_context":
            # Standalone block between user and assistant (system-injected)
            ac = f"{ASSISTANT_CONTEXT_OPEN}{content}{ASSISTANT_CONTEXT_CLOSE}\n"
            text_parts.append(ac)
            mask_parts.append("0" * len(ac))
        
        elif role == "assistant":
            # Build assistant content with optional think prefix and cite suffix
            think_text = msg.get("think", "")
            cite_text = msg.get("cite", "")
            
            inner = ""
            if think_text:
                inner += f"{THINK_OPEN}{think_text}{THINK_CLOSE}"
            inner += content
            if cite_text:
                inner += f"{CITE_OPEN}{cite_text}{CITE_CLOSE}"
            
            a = f"{ASSISTANT_OPEN}{inner}{ASSISTANT_CLOSE}\n"
            text_parts.append(a)
            mask_parts.append("1" * len(a))
        
        elif role == "assistant_toolcall":
            # Assistant message with toolcall, optionally with think prefix
            name = msg.get("name", "")
            arguments = msg.get("arguments", {})
            toolcall_str = serialize_toolcall(name, arguments)
            think_text = msg.get("think", "")
            
            inner = ""
            if think_text:
                inner += f"{THINK_OPEN}{think_text}{THINK_CLOSE}"
            inner += toolcall_str
            
            a = f"{ASSISTANT_OPEN}{inner}{ASSISTANT_CLOSE}\n"
            text_parts.append(a)
            mask_parts.append("1" * len(a))
        
        elif role == "toolresult":
            # Tool result (don't train)
            result_content = msg.get("content", {})
            r = serialize_toolresult(result_content) + "\n"
            text_parts.append(r)
            mask_parts.append("0" * len(r))
    
    # End of turn marker (trained - model learns to stop)
    text_parts.append(EOT + "\n")
    mask_parts.append("1" * len(EOT) + "0")
    
    return "".join(text_parts), "".join(mask_parts)


def char_mask_to_token_mask(
    text: str, 
    char_mask: str, 
    tokenizer: Tokenizer,
    weighted: bool = False
) -> Tuple[List[int], List[float]]:
    """
    Convert character-level mask to token-level mask.
    
    When weighted=False: binary mask (0.0 / 1.0).
    When weighted=True: WeFT-style weighted mask with higher weights for
    structural control tokens that steer generation decisions.
    
    Uses dynamic token ID lookup -- never hardcodes IDs.
    """
    assert len(text) == len(char_mask), f"Length mismatch: {len(text)} vs {len(char_mask)}"
    
    # Dynamic token ID lookup
    _IDS = get_special_token_ids()
    
    ASSISTANT_OPEN_ID  = _IDS["myPT_assistant_open"]
    ASSISTANT_CLOSE_ID = _IDS["myPT_assistant_close"]
    EOT_ID             = _IDS["myPT_eot"]
    
    # Additional IDs for weighted masking
    TOOLCALL_OPEN_ID   = _IDS["myPT_toolcall_open"]
    TOOLCALL_CLOSE_ID  = _IDS["myPT_toolcall_close"]
    THINK_OPEN_ID      = _IDS["myPT_think_open"]
    THINK_CLOSE_ID     = _IDS["myPT_think_close"]
    CITE_OPEN_ID       = _IDS["myPT_cite_open"]
    CITE_CLOSE_ID      = _IDS["myPT_cite_close"]
    
    # Weighted mask values
    W_STOP   = 2.0   # Critical stop signals (assistant_close, eot)
    W_STEER  = 1.5   # Steering tokens (think open/close, cite open/close)
    W_ACTION = 2.0   # Action triggers (toolcall open/close)
    W_NORMAL = 1.0   # Normal assistant content
    W_OFF    = 0.0   # Masked (system, user, toolresult)
    
    # High-weight token sets (only used when weighted=True)
    STOP_IDS   = {ASSISTANT_CLOSE_ID, EOT_ID}
    STEER_IDS  = {THINK_OPEN_ID, THINK_CLOSE_ID, CITE_OPEN_ID, CITE_CLOSE_ID}
    ACTION_IDS = {TOOLCALL_OPEN_ID, TOOLCALL_CLOSE_ID}
    
    # Encode full text to get token IDs
    token_ids = tokenizer.encode(text)
    
    # Build mask based on token structure
    token_mask: List[float] = []
    in_assistant_response = False
    
    for token_id in token_ids:
        if token_id == ASSISTANT_OPEN_ID:
            token_mask.append(W_OFF)
            in_assistant_response = True
        elif token_id == ASSISTANT_CLOSE_ID:
            token_mask.append(W_STOP if weighted else W_NORMAL)
            in_assistant_response = False
        elif token_id == EOT_ID:
            token_mask.append(W_STOP if weighted else W_NORMAL)
        elif in_assistant_response:
            if weighted and token_id in STEER_IDS:
                token_mask.append(W_STEER)
            elif weighted and token_id in ACTION_IDS:
                token_mask.append(W_ACTION)
            else:
                token_mask.append(W_NORMAL)
        else:
            token_mask.append(W_OFF)
    
    return token_ids, token_mask


class EpisodeShardWriter:
    """
    Writes episode-indexed shards with tokens, masks, and episode index.
    
    Handles multi-shard output when data exceeds tokens_per_shard.
    """
    
    def __init__(self, output_dir: str, split: str, tokens_per_shard: int,
                 weighted_mask: bool = False):
        self.output_dir = output_dir
        self.split = split
        self.tokens_per_shard = tokens_per_shard
        self.weighted_mask = weighted_mask
        
        # Current shard data
        self.shard_idx = 0
        self.tokens: List[int] = []
        self.masks: List[float] = []
        self.episodes: List[Tuple[int, int]] = []  # (start, length) pairs
        
        # Statistics
        self.total_tokens = 0
        self.total_episodes = 0
        self.shards_written = 0
    
    def add_episode(self, token_ids: List[int], mask: List[float],
                    segment_ids: Optional[List[int]] = None):
        """Add an episode to the current shard.
        
        Args:
            token_ids: Token IDs for this episode/packed sequence
            mask: Loss mask values aligned with token_ids
            segment_ids: Optional per-token segment IDs for packed sequences.
                segment 0 = padding, 1+ = episode within pack.
        """
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
        
        # Segment IDs (for packed sequences with segment-isolated attention)
        if segment_ids is not None:
            if not hasattr(self, 'segment_ids'):
                self.segment_ids = []
            self.segment_ids.extend(segment_ids)
        
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
        
        # Write mask.bin (uint8) -- always written for backward compatibility
        masks_binary = np.array([1 if m > 0 else 0 for m in self.masks], dtype=np.uint8)
        mask_path = os.path.join(shard_dir, "mask.bin")
        masks_binary.tofile(mask_path)
        
        # Write mask_weighted.bin (float32) when using WeFT-style weighted masks
        if self.weighted_mask:
            masks_weighted = np.array(self.masks, dtype=np.float32)
            mask_weighted_path = os.path.join(shard_dir, "mask_weighted.bin")
            masks_weighted.tofile(mask_weighted_path)
        
        # Write segment_ids.bin (uint8) if segment IDs were provided
        has_segments = hasattr(self, 'segment_ids') and len(self.segment_ids) > 0
        if has_segments:
            seg_arr = np.array(self.segment_ids, dtype=np.uint8)
            seg_path = os.path.join(shard_dir, "segment_ids.bin")
            seg_arr.tofile(seg_path)
        
        # Write episodes.idx (uint64 pairs: start, length)
        episodes_arr = np.array(self.episodes, dtype=np.uint64)
        episodes_path = os.path.join(shard_dir, "episodes.idx")
        episodes_arr.tofile(episodes_path)
        
        seg_info = " (with segment_ids)" if has_segments else ""
        print(f"  Written {self.split}/shard_{self.shard_idx:05d}: "
              f"{len(self.tokens):,} tokens, {len(self.episodes)} episodes{seg_info}")
        
        # Reset for next shard
        self.tokens = []
        self.masks = []
        self.episodes = []
        if hasattr(self, 'segment_ids'):
            self.segment_ids = []
        self.shard_idx += 1
        self.shards_written += 1
    
    def finalize(self) -> int:
        """Flush remaining data and return number of shards written."""
        self._flush_shard()
        return self.shards_written


def main():
    args = parse_args()
    
    from core.banner import print_banner
    print_banner("MyPT Tool SFT", "Episode-Indexed Tool-Calling Dataset Preparer")
    
    # Audit: Dataset preparation started
    if AUDIT_AVAILABLE:
        audit.training(
            "dataset_prepare_start",
            dataset_type="tool_sft",
            input_file=args.input,
            output_dir=args.output_dir,
            tokenization=args.tokenization,
            details=f"Tool SFT dataset preparation started: {args.input}"
        )
    
    # Validate input
    if not os.path.exists(args.input):
        error_msg = f"Input file not found: {args.input}"
        print(f"Error: {error_msg}")
        if AUDIT_AVAILABLE:
            audit.training(
                "dataset_prepare_error",
                level=audit.AuditLevel.ERROR,
                dataset_type="tool_sft",
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
    
    # Load conversations
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
                dataset_type="tool_sft",
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
    
    # Count tool usage
    tool_counts = {}
    for conv in conversations:
        for msg in conv.get("messages", []):
            if msg.get("role") == "assistant_toolcall":
                name = msg.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1
    
    print(f"\nTool usage in dataset:")
    for name, count in sorted(tool_counts.items()):
        print(f"  {name}: {count}")
    
    # Initialize shard writers
    use_weighted = getattr(args, 'weighted_mask', False)
    train_writer = EpisodeShardWriter(args.output_dir, "train", args.tokens_per_shard,
                                       weighted_mask=use_weighted)
    val_writer = EpisodeShardWriter(args.output_dir, "val", args.tokens_per_shard,
                                     weighted_mask=use_weighted)
    
    # Pad token for packing
    pad_token_id = get_special_token_ids()["myPT_eot"]
    
    # Process conversations -- tokenize all episodes first
    print("\nProcessing and tokenizing...")
    
    train_episodes = []
    val_episodes = []
    
    train_mask_sum = 0
    train_mask_count = 0
    val_mask_sum = 0
    val_mask_count = 0
    
    for i, conv in enumerate(conversations):
        # Pre-training replay episodes: raw text, full loss (mask=1 everywhere)
        if conv.get("_replay", False):
            raw_text = conv.get("text", "")
            tokens = tokenizer.encode(raw_text)
            mask = [1.0] * len(tokens)
            meta = {"_conv_idx": i, "_replay": True}
        else:
            text, char_mask = serialize_conversation(conv)
            
            # Audit: Log episode text before tokenization (for traceability)
            split_name = "val" if i in val_indices else "train"
            if AUDIT_AVAILABLE:
                text_preview = text[:500] + "..." if len(text) > 500 else text
                text_preview_escaped = text_preview.replace("\n", "\\n")
                audit.training(
                    "episode_text",
                    dataset_type="tool_sft",
                    split=split_name,
                    conversation_idx=i,
                    text_length=len(text),
                    num_assistant_chars=char_mask.count("1"),
                    details=text_preview_escaped
                )
            
            tokens, mask = char_mask_to_token_mask(
                text, char_mask, tokenizer,
                weighted=use_weighted
            )
            meta = {"_conv_idx": i}
        
        if i in val_indices:
            val_episodes.append((tokens, mask, meta))
            val_mask_sum += sum(mask)
            val_mask_count += len(mask)
        else:
            train_episodes.append((tokens, mask, meta))
            train_mask_sum += sum(mask)
            train_mask_count += len(mask)
        
        if args.verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(conversations)}")
    
    print(f"  Tokenized {len(train_episodes)} train + {len(val_episodes)} val episodes")
    
    # ==========================================================================
    # PACKING PATH
    # ==========================================================================
    if args.enable_packing:
        from scripts.sft.pack_utils import greedy_bin_pack, compute_packing_stats
        
        print(f"\n{'='*60}")
        print(f"  PACKING MODE ENABLED")
        print(f"  Block size: {args.pack_block_size}")
        print(f"{'='*60}")
        
        import random
        random.shuffle(train_episodes)
        train_packed = greedy_bin_pack(train_episodes, args.pack_block_size, pad_token_id)
        val_packed = greedy_bin_pack(val_episodes, args.pack_block_size, pad_token_id)
        
        packing_stats_train = compute_packing_stats(
            [(t, m) for t, m, _ in train_episodes],
            train_packed,
            args.pack_block_size
        )
        packing_stats_val = compute_packing_stats(
            [(t, m) for t, m, _ in val_episodes],
            val_packed,
            args.pack_block_size
        )
        
        # Write packed sequences
        print(f"\nWriting packed sequences...")
        for pack in train_packed:
            tokens, mask, segment_ids, meta = pack
            train_writer.add_episode(tokens, mask, segment_ids=segment_ids)
        
        for pack in val_packed:
            tokens, mask, segment_ids, meta = pack
            val_writer.add_episode(tokens, mask, segment_ids=segment_ids)
        
        # Finalize
        print("\nFinalizing shards...")
        train_shards = train_writer.finalize()
        val_shards = val_writer.finalize()
        
        # Print packing summary
        print(f"\n{'='*60}")
        print(f"  PACKING RESULTS")
        print(f"{'='*60}")
        print(f"\n  TRAIN:")
        print(f"    Original: {packing_stats_train['original_episodes']} episodes, "
              f"{packing_stats_train['original_tokens']:,} tokens")
        print(f"    Avg episode length: {packing_stats_train['avg_episode_len']:.1f} tokens")
        print(f"    Packed:   {packing_stats_train['packed_sequences']} sequences x "
              f"{args.pack_block_size} = {packing_stats_train['packed_tokens']:,} tokens")
        print(f"    Content utilization: {packing_stats_train['nonpad_ratio']:.1%}")
        print(f"    Supervised tokens/step: "
              f"{packing_stats_train['unpacked_effective_mask']:.1%} -> "
              f"{packing_stats_train['packed_effective_mask']:.1%}")
        print(f"    Training efficiency: {packing_stats_train['training_efficiency_gain']:.1f}x")
        print(f"\n  VAL:")
        print(f"    Original: {packing_stats_val['original_episodes']} episodes, "
              f"{packing_stats_val['original_tokens']:,} tokens")
        print(f"    Packed:   {packing_stats_val['packed_sequences']} sequences x "
              f"{args.pack_block_size} = {packing_stats_val['packed_tokens']:,} tokens")
        
        train_mask_ratio = packing_stats_train['packed_mask_ratio']
        val_mask_ratio = packing_stats_val['packed_mask_ratio']
    
    # ==========================================================================
    # NON-PACKING PATH (original behavior)
    # ==========================================================================
    else:
        for tokens, mask, meta in train_episodes:
            train_writer.add_episode(tokens, mask)
        
        for tokens, mask, meta in val_episodes:
            val_writer.add_episode(tokens, mask)
        
        # Finalize shards
        print("\nWriting shards...")
        train_shards = train_writer.finalize()
        val_shards = val_writer.finalize()
        
        # Calculate statistics
        train_mask_ratio = train_mask_sum / train_mask_count if train_mask_count > 0 else 0
        val_mask_ratio = val_mask_sum / val_mask_count if val_mask_count > 0 else 0
    
    print(f"\n  Total train tokens: {train_writer.total_tokens:,}")
    print(f"  Total val tokens: {val_writer.total_tokens:,}")
    print(f"  Train mask ratio: {train_mask_ratio:.1%}")
    print(f"  Val mask ratio: {val_mask_ratio:.1%}")
    
    # Save tokenizer state
    tokenizer_path = os.path.join(args.output_dir, "tokenizer_state.json")
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer.get_state(), f, indent=2)
    print(f"  Saved tokenizer state")
    
    # Save metadata
    metadata = {
        "schema": "episode_indexed_sft_v1",
        "has_loss_mask": True,
        "dataset_type": "toolcall_sft",
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
        "tool_counts": tool_counts,
        "weighted_mask": use_weighted,
        "special_tokens_used": [
            "myPT_system", "myPT_user", "myPT_assistant",
            "myPT_toolcall", "myPT_toolresult", "myPT_eot",
            "myPT_think", "myPT_cite"
        ],
    }
    
    if args.enable_packing:
        metadata["packing_enabled"] = True
        metadata["pack_block_size"] = args.pack_block_size
    
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Episode-indexed Toolcall SFT dataset prepared!")
    print(f"   Conversations: {len(conversations)}")
    print(f"   Train: {train_writer.total_tokens:,} tokens, {train_writer.total_episodes} episodes ({train_shards} shard(s))")
    print(f"   Val: {val_writer.total_tokens:,} tokens, {val_writer.total_episodes} episodes ({val_shards} shard(s))")
    print(f"   Mask ratio: {train_mask_ratio:.1%}")
    print(f"   Output: {args.output_dir}")
    print("=" * 60)
    
    # Audit: Dataset preparation completed
    if AUDIT_AVAILABLE:
        audit.training(
            "dataset_prepare_complete",
            dataset_type="tool_sft",
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
            num_tools=len(tool_counts),
            details=f"Tool SFT dataset prepared: {len(conversations)} conversations, "
                    f"{train_writer.total_tokens:,} train tokens, {len(tool_counts)} unique tools"
        )
    
    print(f"\nTo train:")
    print(f"  python train.py --model_name my_agent \\")
    print(f"      --dataset_dir {args.output_dir} \\")
    print(f"      --config_file configs/sft/phase5_simple_toolcall.json \\")
    print(f"      --init_from_model base_model")


if __name__ == "__main__":
    main()
