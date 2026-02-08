#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare chat SFT dataset with episode-indexed format and loss masking.

Reads JSONL with conversations, serializes to special token format,
and creates episode-indexed binary files with loss masks.

PACKING MODE (--enable_packing):
    For short episodes (like operator learning), packing combines multiple
    episodes into single fixed-length sequences to maximize supervised token
    density. This can improve mask ratio from ~13% to ~60-70%, dramatically
    improving training efficiency.

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
    # Basic (unpacked)
    python scripts/prepare_chat_sft.py --input data/chat.jsonl --output_dir data/chat_sft
    
    # With separate val file
    python scripts/prepare_chat_sft.py --input train.jsonl --output_dir data/sft \\
        --val_file val.jsonl
    
    # PACKED (for short episodes like operator learning)
    python scripts/prepare_chat_sft.py --input data/operator_train.jsonl --output_dir data/sft_packed \\
        --val_file data/operator_val.jsonl \\
        --enable_packing --pack_block_size 1024 --pack_by_field "_meta.operator"
"""

import argparse
import os
import sys
import json
import re
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set, Optional

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


# =============================================================================
# OPERATOR DATASET VALIDATION FUNCTIONS
# =============================================================================

def extract_payload_from_conv(conv: Dict[str, Any]) -> Optional[str]:
    """Extract payload from a conversation.
    
    First tries _meta.payload (reliable for operator dataset).
    Falls back to heuristics on user message text.
    """
    # Try _meta.payload first (operator dataset stores this)
    meta = conv.get("_meta", {})
    if "payload" in meta:
        return meta["payload"]
    
    # Fallback: extract from user message using heuristics
    messages = conv.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return extract_payload_from_text(msg.get("content", ""))
    
    return None


def extract_payload_from_text(user_msg: str) -> Optional[str]:
    """Extract payload from user message text using heuristics."""
    # EXTRACT operator: look for quoted text
    quote_match = re.search(r'"([^"]+)"', user_msg)
    if quote_match:
        return quote_match.group(1)
    
    # COPY/WRAP operators: text after colon
    colon_match = re.search(r':\s*(.+)$', user_msg)
    if colon_match:
        return colon_match.group(1).strip()
    
    return None


def get_template_signature(user_msg: str, payload: Optional[str]) -> str:
    """Replace payload with {PAYLOAD} to get template signature."""
    if payload and payload in user_msg:
        # Replace payload, handling both quoted and unquoted
        sig = user_msg.replace(f'"{payload}"', '"{PAYLOAD}"')
        sig = sig.replace(payload, "{PAYLOAD}")
        # Collapse whitespace
        return ' '.join(sig.split())
    return ' '.join(user_msg.split())


def get_user_message(conv: Dict[str, Any]) -> str:
    """Extract user message from conversation."""
    for msg in conv.get("messages", []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def file_sha256(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def check_train_val_separation(
    train_convs: List[Dict],
    val_convs: List[Dict],
    verbose: bool = False
) -> Tuple[int, int, List[str], List[str]]:
    """Check for payload and template overlap between train and val.
    
    Returns:
        (payload_overlap_count, template_overlap_count, payload_examples, template_examples)
    """
    # Extract payloads
    train_payloads: Set[str] = set()
    val_payloads: Set[str] = set()
    
    for conv in train_convs:
        payload = extract_payload_from_conv(conv)
        if payload:
            train_payloads.add(payload)
    
    for conv in val_convs:
        payload = extract_payload_from_conv(conv)
        if payload:
            val_payloads.add(payload)
    
    payload_overlap = train_payloads & val_payloads
    payload_examples = list(payload_overlap)[:5]
    
    # Extract template signatures
    train_templates: Set[str] = set()
    val_templates: Set[str] = set()
    
    for conv in train_convs:
        user_msg = get_user_message(conv)
        payload = extract_payload_from_conv(conv)
        sig = get_template_signature(user_msg, payload)
        train_templates.add(sig)
    
    for conv in val_convs:
        user_msg = get_user_message(conv)
        payload = extract_payload_from_conv(conv)
        sig = get_template_signature(user_msg, payload)
        val_templates.add(sig)
    
    template_overlap = train_templates & val_templates
    template_examples = list(template_overlap)[:5]
    
    if verbose:
        print(f"\n  Train/Val Separation Check:")
        print(f"    Train payloads: {len(train_payloads)}")
        print(f"    Val payloads: {len(val_payloads)}")
        print(f"    Payload overlap: {len(payload_overlap)}")
        print(f"    Train templates: {len(train_templates)}")
        print(f"    Val templates: {len(val_templates)}")
        print(f"    Template overlap: {len(template_overlap)}")
    
    return len(payload_overlap), len(template_overlap), payload_examples, template_examples


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
                        help="Validation split ratio (default: 0.1). Ignored if --val_file is provided.")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Separate validation JSONL file (overrides --val_split)")
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
    parser.add_argument("--pack_by_field", type=str, default=None,
                        help="Group episodes by this metadata field before packing (e.g., '_meta.operator')")
    parser.add_argument("--pack_shuffle", action="store_true", default=True,
                        help="Shuffle episodes before packing (default: True)")
    parser.add_argument("--no_pack_shuffle", action="store_false", dest="pack_shuffle",
                        help="Don't shuffle episodes before packing")
    
    # System prompt options
    parser.add_argument("--no_system_prompt", action="store_true",
                        help="Skip system prompt (for operators/mechanical tasks to maximize mask ratio)")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Override system prompt (default: use CONVERSATION_SYSTEM_PROMPT)")
    
    return parser.parse_args()


def serialize_conversation(
    item: Dict[str, Any],
    skip_system: bool = False,
    system_prompt_override: Optional[str] = None
) -> Tuple[str, str]:
    """
    Serialize a conversation to text + char-level mask.
    
    Args:
        item: Conversation dict with 'messages' list
        skip_system: If True, omit system prompt entirely (for operators, increases mask ratio)
        system_prompt_override: Custom system prompt to use instead of default
    
    Returns:
        (text, char_mask) where char_mask has '1' for assistant chars, '0' otherwise
    """
    text_parts = []
    mask_parts = []
    
    # System message (masked out) - skip for mechanical tasks to maximize mask ratio
    # NO newlines between tags - tags alone define structure
    if not skip_system:
        prompt = system_prompt_override if system_prompt_override else CONVERSATION_SYSTEM_PROMPT
        s = f"{SYSTEM_OPEN}{prompt}{SYSTEM_CLOSE}"
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


# =============================================================================
# PACKING FUNCTIONS
# =============================================================================

def get_nested_field(obj: Dict, field_path: str) -> Optional[str]:
    """
    Get a nested field from a dictionary using dot notation.
    
    Example: get_nested_field({"_meta": {"operator": "COPY"}}, "_meta.operator") -> "COPY"
    """
    parts = field_path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return str(current) if current is not None else None


def group_by_field(items: List[Dict], field_path: str) -> Dict[str, List[Dict]]:
    """
    Group items by a nested field value.
    
    Returns dict mapping field_value -> list of items.
    Items with missing field go to "_ungrouped" key.
    """
    groups: Dict[str, List[Dict]] = {}
    for item in items:
        key = get_nested_field(item, field_path)
        if key is None:
            key = "_ungrouped"
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups


def greedy_bin_pack(
    episodes: List[Tuple[List[int], List[int], Dict]],
    block_size: int,
    pad_token_id: int,
) -> List[Tuple[List[int], List[int], List[int], Dict]]:
    """
    Pack multiple episodes into fixed-size blocks using greedy bin-packing.
    
    Episodes that don't fit entirely are NOT split - they go in the next block.
    Each packed sequence is exactly block_size tokens (padded if needed).
    
    Also produces per-token segment_ids for segment-isolated attention:
    - segment_id 0 = padding (not attended to)
    - segment_id 1, 2, 3, ... = episodes within the packed sequence
    This enables the model to build an attention mask that prevents
    cross-episode attention bleeding in packed sequences.
    
    Args:
        episodes: List of (tokens, mask, metadata) tuples
        block_size: Target sequence length (e.g., 1024)
        pad_token_id: Token ID for padding
    
    Returns:
        List of packed (tokens, mask, segment_ids, metadata) tuples
    
    The metadata for packed sequences includes:
        - _packed: True
        - _num_episodes: count of episodes in this pack
        - _original_categories: list of category values (if present)
    """
    if not episodes:
        return []
    
    packed_sequences = []
    
    # Current pack being filled
    current_tokens: List[int] = []
    current_mask: List[int] = []
    current_segment_ids: List[int] = []
    current_episode_count = 0
    current_categories: List[str] = []
    
    for tokens, mask, meta in episodes:
        ep_len = len(tokens)
        
        # Episode too long for block_size? Truncate (keep end to preserve closers)
        if ep_len > block_size:
            # Keep last block_size tokens
            tokens = tokens[-block_size:]
            mask = mask[-block_size:]
            ep_len = block_size
        
        # Would adding this episode exceed block_size?
        if len(current_tokens) + ep_len > block_size:
            # Finalize current pack (if non-empty)
            if current_tokens:
                # Pad to block_size
                pad_len = block_size - len(current_tokens)
                if pad_len > 0:
                    current_tokens.extend([pad_token_id] * pad_len)
                    current_mask.extend([0] * pad_len)  # Don't train on padding
                    current_segment_ids.extend([0] * pad_len)  # segment 0 = padding
                
                packed_meta = {
                    "_packed": True,
                    "_num_episodes": current_episode_count,
                    "_original_categories": current_categories,
                }
                packed_sequences.append((current_tokens, current_mask, current_segment_ids, packed_meta))
            
            # Start new pack
            current_tokens = []
            current_mask = []
            current_segment_ids = []
            current_episode_count = 0
            current_categories = []
        
        # Add episode to current pack
        current_episode_count += 1
        current_tokens.extend(tokens)
        current_mask.extend(mask)
        current_segment_ids.extend([current_episode_count] * ep_len)  # 1-indexed segment ID
        
        # Track category if present
        cat = meta.get("_category")
        if cat:
            current_categories.append(cat)
    
    # Finalize last pack (if non-empty)
    if current_tokens:
        pad_len = block_size - len(current_tokens)
        if pad_len > 0:
            current_tokens.extend([pad_token_id] * pad_len)
            current_mask.extend([0] * pad_len)
            current_segment_ids.extend([0] * pad_len)
        
        packed_meta = {
            "_packed": True,
            "_num_episodes": current_episode_count,
            "_original_categories": current_categories,
        }
        packed_sequences.append((current_tokens, current_mask, current_segment_ids, packed_meta))
    
    return packed_sequences


def compute_packing_stats(
    original_episodes: List[Tuple[List[int], List[int]]],
    packed_sequences: list,
    block_size: int,
) -> Dict:
    """
    Compute statistics comparing unpacked vs packed datasets.
    
    Key insight: Without packing, each episode would be padded to block_size,
    wasting most of the compute on padding. With packing, we fill the block
    with multiple episodes, dramatically increasing supervised tokens per step.
    """
    # Original stats
    orig_tokens = sum(len(tokens) for tokens, _ in original_episodes)
    orig_mask_tokens = sum(sum(mask) for _, mask in original_episodes)
    orig_mask_ratio = orig_mask_tokens / orig_tokens if orig_tokens > 0 else 0
    avg_episode_len = orig_tokens / len(original_episodes) if original_episodes else 0
    
    # Packed stats (tuples are (tokens, mask, segment_ids, meta))
    packed_tokens = sum(len(p[0]) for p in packed_sequences)
    packed_mask_tokens = sum(sum(p[1]) for p in packed_sequences)
    packed_mask_ratio = packed_mask_tokens / packed_tokens if packed_tokens > 0 else 0
    
    # Non-pad ratio (how much of packed sequences is actual content)
    total_content_tokens = orig_tokens
    total_packed_slots = len(packed_sequences) * block_size
    nonpad_ratio = total_content_tokens / total_packed_slots if total_packed_slots > 0 else 0
    
    # Episodes per pack (metadata is last element)
    eps_per_pack = [p[-1].get("_num_episodes", 1) for p in packed_sequences]
    avg_eps_per_pack = sum(eps_per_pack) / len(eps_per_pack) if eps_per_pack else 0
    
    # TRAINING EFFICIENCY: Compare supervised tokens per training step
    # WITHOUT packing: each episode padded to block_size
    #   effective_mask_per_step = orig_mask_ratio * (avg_episode_len / block_size)
    # WITH packing: packed sequences at block_size
    #   effective_mask_per_step = packed_mask_ratio
    
    unpacked_effective_mask = orig_mask_ratio * (avg_episode_len / block_size) if block_size > 0 else 0
    packed_effective_mask = packed_mask_ratio
    
    # This is the real improvement: how many more supervised tokens per step
    training_efficiency_gain = packed_effective_mask / unpacked_effective_mask if unpacked_effective_mask > 0 else 1.0
    
    return {
        "original_episodes": len(original_episodes),
        "original_tokens": orig_tokens,
        "original_mask_tokens": orig_mask_tokens,
        "original_mask_ratio": orig_mask_ratio,
        "avg_episode_len": avg_episode_len,
        "packed_sequences": len(packed_sequences),
        "packed_tokens": packed_tokens,
        "packed_mask_tokens": packed_mask_tokens,
        "packed_mask_ratio": packed_mask_ratio,
        "nonpad_ratio": nonpad_ratio,
        "avg_episodes_per_pack": avg_eps_per_pack,
        "min_episodes_per_pack": min(eps_per_pack) if eps_per_pack else 0,
        "max_episodes_per_pack": max(eps_per_pack) if eps_per_pack else 0,
        # Old metric (content ratio change, always ~1.0x)
        "content_mask_change": packed_mask_ratio / orig_mask_ratio if orig_mask_ratio > 0 else 1.0,
        # NEW: Real training efficiency gain (supervised tokens per step)
        "unpacked_effective_mask": unpacked_effective_mask,
        "packed_effective_mask": packed_effective_mask,
        "training_efficiency_gain": training_efficiency_gain,
    }


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
    
    def add_episode(self, token_ids: List[int], mask: List[int],
                    segment_ids: Optional[List[int]] = None):
        """Add an episode to the current shard.
        
        Args:
            token_ids: Token IDs for this episode/packed sequence
            mask: Loss mask values (0/1) aligned with token_ids
            segment_ids: Optional per-token segment IDs for packed sequences.
                segment 0 = padding, 1+ = episode within pack.
                If provided, written to segment_ids.bin for segment-isolated attention.
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
        
        # Write mask.bin (uint8)
        masks_arr = np.array(self.masks, dtype=np.uint8)
        mask_path = os.path.join(shard_dir, "mask.bin")
        masks_arr.tofile(mask_path)
        
        # Write segment_ids.bin (uint8) if segment IDs were provided
        # This enables segment-isolated attention for packed sequences
        has_segments = hasattr(self, 'segment_ids') and len(self.segment_ids) > 0
        if has_segments:
            seg_arr = np.array(self.segment_ids, dtype=np.uint8)
            seg_path = os.path.join(shard_dir, "segment_ids.bin")
            seg_arr.tofile(seg_path)
        
        # Write episodes.idx (uint64 pairs: start, length)
        # Format: each episode is 16 bytes (2 × uint64)
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
    
    # Validate --val_file if provided
    if args.val_file:
        if not os.path.exists(args.val_file):
            print(f"Error: Validation file not found: {args.val_file}")
            sys.exit(1)
        
        # Mutual exclusivity: --val_file and --val_split > 0
        if args.val_split > 0:
            print("=" * 60)
            print("  VAL SOURCE: Using explicit val file (no random split)")
            print(f"  Note: --val_split={args.val_split} ignored because --val_file is provided")
            print("=" * 60)
    
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
    if args.val_file:
        print(f"  Val file: {args.val_file} (separate file)")
    else:
        print(f"  Val split: {args.val_split}")
    print(f"  Tokens per shard: {args.tokens_per_shard:,}")
    if args.no_system_prompt:
        print(f"  System prompt: SKIPPED (--no_system_prompt, maximizes mask ratio)")
    elif args.system_prompt:
        print(f"  System prompt: Custom override ({len(args.system_prompt)} chars)")
    else:
        print(f"  System prompt: Default (CONVERSATION_SYSTEM_PROMPT)")
    print()
    
    # Load conversations
    def load_jsonl(filepath: str) -> list:
        """Load conversations from JSONL file."""
        convs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    convs.append(item)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping line {line_num} (invalid JSON): {e}")
        return convs
    
    print("Loading conversations...")
    conversations = load_jsonl(args.input)
    print(f"  Loaded {len(conversations)} conversations from {args.input}")
    
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
    
    # Handle train/val split
    payload_overlap_count = 0
    template_overlap_count = 0
    
    if args.val_file:
        # Separate validation file provided
        val_conversations = load_jsonl(args.val_file)
        print(f"  Loaded {len(val_conversations)} validation conversations from {args.val_file}")
        train_conversations = conversations
        val_indices = set()  # Not used when val_file is provided
        use_separate_val = True
        
        # STRICT SEPARATION CHECK for explicit val files
        print("\n" + "=" * 60)
        print("  TRAIN/VAL SEPARATION CHECK (--val_file mode)")
        print("=" * 60)
        
        payload_overlap_count, template_overlap_count, payload_examples, template_examples = \
            check_train_val_separation(train_conversations, val_conversations, verbose=True)
        
        # Hard fail on payload overlap
        if payload_overlap_count > 0:
            print(f"\n  ❌ ERROR: Train/Val PAYLOAD overlap detected!")
            print(f"     {payload_overlap_count} overlapping payloads found.")
            print(f"     Examples: {payload_examples}")
            print(f"\n     Operator benchmark INVALID - payloads must be unique.")
            print(f"     Regenerate dataset with different seeds for train/val.")
            sys.exit(1)
        else:
            print(f"\n  ✅ Payload overlap: 0 (PASS)")
        
        # Hard fail on template overlap
        if template_overlap_count > 0:
            print(f"\n  ❌ ERROR: Train/Val TEMPLATE overlap detected!")
            print(f"     {template_overlap_count} overlapping templates found.")
            print(f"     Examples: {template_examples}")
            print(f"\n     Operator benchmark requires template generalization.")
            print(f"     Val templates must differ from train templates.")
            sys.exit(1)
        else:
            print(f"  ✅ Template overlap: 0 (PASS)")
        
        print("=" * 60 + "\n")
        
    else:
        # Random split
        np.random.seed(args.seed)
        indices = np.random.permutation(len(conversations))
        val_size = int(len(conversations) * args.val_split)
        val_indices = set(indices[:val_size])
        train_conversations = None  # Will use conversations with val_indices
        val_conversations = None
        use_separate_val = False
    
    if use_separate_val:
        print(f"  Train: {len(train_conversations)}, Val: {len(val_conversations)}")
    else:
        print(f"  Train: {len(conversations) - len(val_indices)}, Val: {len(val_indices)}")
    
    # Get pad token ID for packing
    pad_token_id = tokenizer.special_tokens.get('myPT_eot', 50271)
    
    # Initialize shard writers
    train_writer = EpisodeShardWriter(args.output_dir, "train", args.tokens_per_shard)
    val_writer = EpisodeShardWriter(args.output_dir, "val", args.tokens_per_shard)
    
    # Stats tracking
    train_mask_sum = 0
    train_mask_count = 0
    val_mask_sum = 0
    val_mask_count = 0
    packing_stats_train = None
    packing_stats_val = None
    
    def tokenize_conversation(conv: Dict, conv_idx: int) -> Tuple[List[int], List[int], Dict]:
        """Tokenize a single conversation, return (tokens, mask, metadata)."""
        text, char_mask = serialize_conversation(
            conv,
            skip_system=args.no_system_prompt,
            system_prompt_override=args.system_prompt
        )
        tokens, mask = char_mask_to_token_mask(text, char_mask, tokenizer)
        
        # Build metadata for packing
        meta = {"_conv_idx": conv_idx}
        if args.pack_by_field:
            category = get_nested_field(conv, args.pack_by_field)
            if category:
                meta["_category"] = category
        
        return tokens, mask, meta
    
    def tokenize_all(convs: List[Dict], split_name: str) -> List[Tuple[List[int], List[int], Dict]]:
        """Tokenize all conversations, return list of (tokens, mask, metadata)."""
        episodes = []
        for i, conv in enumerate(convs):
            tokens, mask, meta = tokenize_conversation(conv, i)
            episodes.append((tokens, mask, meta))
            
            if args.verbose and (i + 1) % 1000 == 0:
                print(f"    Tokenized {i + 1}/{len(convs)} {split_name} conversations...")
        
        return episodes
    
    # ==========================================================================
    # PACKING PATH
    # ==========================================================================
    if args.enable_packing:
        print(f"\n{'='*60}")
        print(f"  PACKING MODE ENABLED")
        print(f"  Block size: {args.pack_block_size}")
        print(f"  Group by: {args.pack_by_field or '(none)'}")
        print(f"{'='*60}")
        
        # Prepare train/val conversation lists
        if use_separate_val:
            train_convs = train_conversations
            val_convs = val_conversations
        else:
            train_convs = [c for i, c in enumerate(conversations) if i not in val_indices]
            val_convs = [c for i, c in enumerate(conversations) if i in val_indices]
        
        # --- TOKENIZE ALL ---
        print(f"\nTokenizing {len(train_convs)} train conversations...")
        train_episodes = tokenize_all(train_convs, "train")
        
        print(f"Tokenizing {len(val_convs)} val conversations...")
        val_episodes = tokenize_all(val_convs, "val")
        
        # --- PACK TRAIN ---
        print(f"\nPacking train episodes...")
        if args.pack_by_field:
            # Group by category
            train_groups = group_by_field(
                [{"_tokens": t, "_mask": m, **meta} for t, m, meta in train_episodes],
                "_category"
            )
            print(f"  Found {len(train_groups)} groups: {list(train_groups.keys())}")
            
            # Pack within each group
            train_packed = []
            for group_name, items in train_groups.items():
                group_episodes = [(item["_tokens"], item["_mask"], item) for item in items]
                if args.pack_shuffle:
                    import random
                    random.seed(args.seed)
                    random.shuffle(group_episodes)
                
                packed = greedy_bin_pack(group_episodes, args.pack_block_size, pad_token_id)
                print(f"    {group_name}: {len(group_episodes)} episodes → {len(packed)} packed sequences")
                train_packed.extend(packed)
        else:
            # Pack all together
            if args.pack_shuffle:
                import random
                random.seed(args.seed)
                random.shuffle(train_episodes)
            train_packed = greedy_bin_pack(train_episodes, args.pack_block_size, pad_token_id)
        
        # --- PACK VAL ---
        print(f"\nPacking val episodes...")
        if args.pack_by_field:
            val_groups = group_by_field(
                [{"_tokens": t, "_mask": m, **meta} for t, m, meta in val_episodes],
                "_category"
            )
            val_packed = []
            for group_name, items in val_groups.items():
                group_episodes = [(item["_tokens"], item["_mask"], item) for item in items]
                packed = greedy_bin_pack(group_episodes, args.pack_block_size, pad_token_id)
                print(f"    {group_name}: {len(group_episodes)} episodes → {len(packed)} packed sequences")
                val_packed.extend(packed)
        else:
            val_packed = greedy_bin_pack(val_episodes, args.pack_block_size, pad_token_id)
        
        # --- COMPUTE STATS ---
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
        
        # --- WRITE PACKED SEQUENCES ---
        print(f"\nWriting packed sequences...")
        for pack in train_packed:
            tokens, mask, segment_ids = pack[0], pack[1], pack[2]
            train_writer.add_episode(tokens, mask, segment_ids=segment_ids)
            train_mask_sum += sum(mask)
            train_mask_count += len(mask)
        
        for pack in val_packed:
            tokens, mask, segment_ids = pack[0], pack[1], pack[2]
            val_writer.add_episode(tokens, mask, segment_ids=segment_ids)
            val_mask_sum += sum(mask)
            val_mask_count += len(mask)
        
        # --- PACKING SUMMARY ---
        print(f"\n{'='*60}")
        print(f"  PACKING RESULTS")
        print(f"{'='*60}")
        print(f"\n  TRAIN:")
        print(f"    Original: {packing_stats_train['original_episodes']} episodes, {packing_stats_train['original_tokens']:,} tokens")
        print(f"    Avg episode length: {packing_stats_train['avg_episode_len']:.1f} tokens")
        print(f"    Packed:   {packing_stats_train['packed_sequences']} sequences × {args.pack_block_size} = {packing_stats_train['packed_tokens']:,} tokens")
        print(f"    Content utilization: {packing_stats_train['nonpad_ratio']:.1%} (was {packing_stats_train['avg_episode_len']/args.pack_block_size:.1%} without packing)")
        print(f"    Supervised tokens/step: {packing_stats_train['unpacked_effective_mask']:.1%} → {packing_stats_train['packed_effective_mask']:.1%}")
        print(f"    ⚡ TRAINING EFFICIENCY: {packing_stats_train['training_efficiency_gain']:.1f}x more supervised signal per step")
        print(f"    Avg episodes/pack: {packing_stats_train['avg_episodes_per_pack']:.1f} (range: {packing_stats_train['min_episodes_per_pack']}-{packing_stats_train['max_episodes_per_pack']})")
        
        print(f"\n  VAL:")
        print(f"    Original: {packing_stats_val['original_episodes']} episodes, {packing_stats_val['original_tokens']:,} tokens")
        print(f"    Packed:   {packing_stats_val['packed_sequences']} sequences × {args.pack_block_size} = {packing_stats_val['packed_tokens']:,} tokens")
        print(f"    ⚡ TRAINING EFFICIENCY: {packing_stats_val['training_efficiency_gain']:.1f}x")
        print(f"{'='*60}")
    
    # ==========================================================================
    # NON-PACKING PATH (original behavior)
    # ==========================================================================
    else:
        print("\nProcessing and tokenizing...")
        
        def process_conversations(convs, split_name, writer, check_val_indices=False):
            """Process a list of conversations into a split."""
            nonlocal train_mask_sum, train_mask_count, val_mask_sum, val_mask_count
            
            for i, conv in enumerate(convs):
                # For random split mode, check if this index belongs to val
                if check_val_indices:
                    actual_split = "val" if i in val_indices else "train"
                    if actual_split != split_name:
                        continue
                
                # Serialize to text + char mask
                text, char_mask = serialize_conversation(
                    conv,
                    skip_system=args.no_system_prompt,
                    system_prompt_override=args.system_prompt
                )
                
                episode_idx = writer.total_episodes
                
                # Audit: Log episode text before tokenization (for traceability)
                if AUDIT_AVAILABLE:
                    text_preview = text[:500] + "..." if len(text) > 500 else text
                    text_preview_escaped = text_preview.replace("\n", "\\n")
                    audit.training(
                        "episode_text",
                        dataset_type="chat_sft",
                        split=split_name,
                        episode_idx=episode_idx,
                        conversation_idx=i,
                        text_length=len(text),
                        num_assistant_chars=char_mask.count("1"),
                        details=text_preview_escaped
                    )
                
                # Convert to token-level mask
                tokens, mask = char_mask_to_token_mask(text, char_mask, tokenizer)
                
                writer.add_episode(tokens, mask)
                if split_name == "train":
                    train_mask_sum += sum(mask)
                    train_mask_count += len(mask)
                else:
                    val_mask_sum += sum(mask)
                    val_mask_count += len(mask)
        
        if use_separate_val:
            print("  Processing train conversations...")
            process_conversations(train_conversations, "train", train_writer)
            print("  Processing val conversations...")
            process_conversations(val_conversations, "val", val_writer)
        else:
            process_conversations(conversations, "train", train_writer, check_val_indices=True)
            process_conversations(conversations, "val", val_writer, check_val_indices=True)
    
    if args.verbose:
        print(f"  Processed {train_writer.total_episodes + val_writer.total_episodes} total sequences")
    
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
        "num_conversations": len(train_conversations) + len(val_conversations) if use_separate_val else len(conversations),
        "num_train_conversations": len(train_conversations) if use_separate_val else len(conversations) - len(val_indices),
        "num_val_conversations": len(val_conversations) if use_separate_val else len(val_indices),
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
        "special_tokens_used": list(SPECIAL_TOKEN_STRINGS.keys()),
        # Provenance
        "prepare_mode": "explicit_val_file" if use_separate_val else "val_split",
        "source_train_file": os.path.abspath(args.input),
        "source_val_file": os.path.abspath(args.val_file) if args.val_file else None,
        "val_split": 0.0 if use_separate_val else args.val_split,
        # System prompt settings
        "skip_system_prompt": args.no_system_prompt,
        "system_prompt_override": args.system_prompt,
    }
    
    # Add packing metadata
    if args.enable_packing:
        metadata["packing_enabled"] = True
        metadata["pack_block_size"] = args.pack_block_size
        metadata["pack_by_field"] = args.pack_by_field
        metadata["num_train_mask_tokens"] = packing_stats_train["packed_mask_tokens"]
        metadata["num_val_mask_tokens"] = packing_stats_val["packed_mask_tokens"]
        metadata["original_train_episodes"] = packing_stats_train["original_episodes"]
        metadata["original_val_episodes"] = packing_stats_val["original_episodes"]
        metadata["original_train_mask_ratio"] = packing_stats_train["original_mask_ratio"]
        metadata["avg_episode_len"] = packing_stats_train["avg_episode_len"]
        metadata["avg_episodes_per_pack"] = packing_stats_train["avg_episodes_per_pack"]
        metadata["nonpad_ratio"] = packing_stats_train["nonpad_ratio"]
        # Training efficiency: how many more supervised tokens per step vs unpacked
        metadata["training_efficiency_gain"] = packing_stats_train["training_efficiency_gain"]
        metadata["unpacked_effective_mask"] = packing_stats_train["unpacked_effective_mask"]
        metadata["packed_effective_mask"] = packing_stats_train["packed_effective_mask"]
    else:
        metadata["packing_enabled"] = False
        metadata["num_train_mask_tokens"] = train_mask_sum
        metadata["num_val_mask_tokens"] = val_mask_sum
    
    # Add file hashes for provenance (when using explicit val file)
    if use_separate_val:
        metadata["source_train_sha256"] = file_sha256(args.input)
        metadata["source_val_sha256"] = file_sha256(args.val_file)
        metadata["payload_overlap_count"] = payload_overlap_count
        metadata["template_overlap_count"] = template_overlap_count
    
    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Episode-indexed SFT dataset prepared successfully!")
    print(f"   Conversations: {len(conversations)}")
    if args.enable_packing:
        print(f"   Train: {train_writer.total_tokens:,} tokens, {train_writer.total_episodes} packed sequences ({train_shards} shard(s))")
        print(f"          ({packing_stats_train['original_episodes']} original episodes packed)")
        print(f"   Val: {val_writer.total_tokens:,} tokens, {val_writer.total_episodes} packed sequences ({val_shards} shard(s))")
        print(f"          ({packing_stats_val['original_episodes']} original episodes packed)")
        print(f"   Content mask ratio: {train_mask_ratio:.1%}")
        print(f"   ⚡ Training efficiency: {packing_stats_train['training_efficiency_gain']:.1f}x more supervised tokens per step")
    else:
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
