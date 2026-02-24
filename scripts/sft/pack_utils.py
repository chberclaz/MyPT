#!/usr/bin/env python3
"""
Shared packing utilities for SFT dataset preparation.

Greedy bin-packing combines multiple short episodes into fixed-length
sequences to maximize supervised token density per training step.
Segment IDs enable segment-isolated attention to prevent cross-episode
attention bleeding.

Used by: prepare_chat_sft.py, prepare_tool_sft.py
"""

from typing import Dict, List, Optional, Tuple


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
    episodes: List[Tuple[List[int], List[float], Dict]],
    block_size: int,
    pad_token_id: int,
) -> List[Tuple[List[int], List[float], List[int], Dict]]:
    """
    Pack multiple episodes into fixed-size blocks using greedy bin-packing.

    Episodes that don't fit entirely are NOT split - they go in the next block.
    Each packed sequence is exactly block_size tokens (padded if needed).

    Also produces per-token segment_ids for segment-isolated attention:
    - segment_id 0 = padding (not attended to)
    - segment_id 1, 2, 3, ... = episodes within the packed sequence

    Args:
        episodes: List of (tokens, mask, metadata) tuples
        block_size: Target sequence length (e.g., 1024)
        pad_token_id: Token ID for padding

    Returns:
        List of packed (tokens, mask, segment_ids, metadata) tuples
    """
    if not episodes:
        return []

    packed_sequences = []

    current_tokens: List[int] = []
    current_mask: List[float] = []
    current_segment_ids: List[int] = []
    current_episode_count = 0
    current_categories: List[str] = []

    for tokens, mask, meta in episodes:
        ep_len = len(tokens)

        # Episode too long for block_size? Truncate (keep end to preserve closers)
        if ep_len > block_size:
            tokens = tokens[-block_size:]
            mask = mask[-block_size:]
            ep_len = block_size

        # Would adding this episode exceed block_size or uint8 segment_id capacity?
        # segment_id=0 is reserved for padding; real episodes use 1..255.
        if len(current_tokens) + ep_len > block_size or current_episode_count >= 255:
            # Finalize current pack (if non-empty)
            if current_tokens:
                pad_len = block_size - len(current_tokens)
                if pad_len > 0:
                    current_tokens.extend([pad_token_id] * pad_len)
                    current_mask.extend([0.0] * pad_len)
                    current_segment_ids.extend([0] * pad_len)

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
        current_segment_ids.extend([current_episode_count] * ep_len)

        cat = meta.get("_category")
        if cat:
            current_categories.append(cat)

    # Finalize last pack
    if current_tokens:
        pad_len = block_size - len(current_tokens)
        if pad_len > 0:
            current_tokens.extend([pad_token_id] * pad_len)
            current_mask.extend([0.0] * pad_len)
            current_segment_ids.extend([0] * pad_len)

        packed_meta = {
            "_packed": True,
            "_num_episodes": current_episode_count,
            "_original_categories": current_categories,
        }
        packed_sequences.append((current_tokens, current_mask, current_segment_ids, packed_meta))

    return packed_sequences


def compute_packing_stats(
    original_episodes: List[Tuple[List[int], List[float]]],
    packed_sequences: list,
    block_size: int,
) -> Dict:
    """
    Compute statistics comparing unpacked vs packed datasets.

    Key insight: Without packing, each episode would be padded to block_size,
    wasting most compute on padding. With packing, we fill the block with
    multiple episodes, dramatically increasing supervised tokens per step.
    """
    orig_tokens = sum(len(tokens) for tokens, _ in original_episodes)
    orig_mask_tokens = sum(sum(mask) for _, mask in original_episodes)
    orig_mask_ratio = orig_mask_tokens / orig_tokens if orig_tokens > 0 else 0
    avg_episode_len = orig_tokens / len(original_episodes) if original_episodes else 0

    packed_tokens = sum(len(p[0]) for p in packed_sequences)
    packed_mask_tokens = sum(sum(p[1]) for p in packed_sequences)
    packed_mask_ratio = packed_mask_tokens / packed_tokens if packed_tokens > 0 else 0

    total_content_tokens = orig_tokens
    total_packed_slots = len(packed_sequences) * block_size
    nonpad_ratio = total_content_tokens / total_packed_slots if total_packed_slots > 0 else 0

    eps_per_pack = [p[-1].get("_num_episodes", 1) for p in packed_sequences]
    avg_eps_per_pack = sum(eps_per_pack) / len(eps_per_pack) if eps_per_pack else 0

    unpacked_effective_mask = orig_mask_ratio * (avg_episode_len / block_size) if block_size > 0 else 0
    packed_effective_mask = packed_mask_ratio

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
        "content_mask_change": packed_mask_ratio / orig_mask_ratio if orig_mask_ratio > 0 else 1.0,
        "unpacked_effective_mask": unpacked_effective_mask,
        "packed_effective_mask": packed_effective_mask,
        "training_efficiency_gain": training_efficiency_gain,
    }
