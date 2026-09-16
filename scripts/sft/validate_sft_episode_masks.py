#!/usr/bin/env python3
"""
SFT Episode Mask Validator

Validates loss-mask alignment across all rows in a prepared SFT dataset.

Packed 4096 blocks and toolcall chains have many <myPT_assistant> spans per row.
"Exactly 1 open/close/eot" is the unpacked format-lock check — it is not a
packed-data failure. This script validates every assistant span instead.

Real failures:
1. Zero <myPT_assistant> (BPE fusion / empty assistant)
2. Open/close count mismatch or overlapping spans
3. Mask 0 on assistant content, mask 1 on <myPT_assistant>, mask 0 on closer
4. All-zero mask on a row or segment
5. Pad region (segment_id 0) with mask 1

Usage:
    python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/sft_phase6_3_ground
    python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/sft_format_lock_prepared --verbose
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.special_tokens import get_special_token_ids
_IDS = get_special_token_ids()
ASSISTANT_OPEN_ID = _IDS["myPT_assistant_open"]
ASSISTANT_CLOSE_ID = _IDS["myPT_assistant_close"]
EOT_ID = _IDS["myPT_eot"]


def load_metadata(dataset_dir: str) -> dict:
    path = Path(dataset_dir) / "dataset_metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_dataset(dataset_dir: str, split: str = "train"):
    """Load tokens, mask, episode index, and optional segment_ids."""
    split_dir = Path(dataset_dir) / split

    tokens_path = split_dir / "tokens.bin"
    mask_path = split_dir / "mask.bin"
    episodes_path = split_dir / "episodes.idx"
    seg_path = split_dir / "segment_ids.bin"

    if not tokens_path.exists():
        raise FileNotFoundError(f"Tokens file not found: {tokens_path}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes index not found: {episodes_path}")

    tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
    episodes = np.memmap(episodes_path, dtype=np.uint64, mode="r").reshape(-1, 2)

    mask = None
    if mask_path.exists():
        mask = np.memmap(mask_path, dtype=np.uint8, mode="r")

    segments = None
    if seg_path.exists():
        segments = np.memmap(seg_path, dtype=np.uint8, mode="r")

    return tokens, mask, episodes, segments


def validate_assistant_spans(ep_tokens: np.ndarray, ep_mask: np.ndarray, prefix: str = "") -> dict:
    """Validate every <myPT_assistant> … </myPT_assistant> span. Multi-span is OK."""
    failures = {}
    opens = np.where(ep_tokens == ASSISTANT_OPEN_ID)[0]
    closes = np.where(ep_tokens == ASSISTANT_CLOSE_ID)[0]

    if len(opens) == 0:
        failures[f"{prefix}assistant_open_count"] = "Expected >=1, got 0"
        return failures
    if len(opens) != len(closes):
        failures[f"{prefix}assistant_span_mismatch"] = (
            f"opens={len(opens)} closes={len(closes)}"
        )
        return failures

    for i, (open_idx, close_idx) in enumerate(zip(opens, closes)):
        if open_idx >= close_idx:
            failures[f"{prefix}span_{i}_order"] = f"open={open_idx} close={close_idx}"
            continue
        if i + 1 < len(opens) and close_idx >= opens[i + 1]:
            failures[f"{prefix}span_{i}_overlap"] = (
                f"close={close_idx} next_open={opens[i + 1]}"
            )

        if ep_mask is None:
            continue

        if ep_mask[open_idx] != 0:
            failures[f"{prefix}assistant_open_mask"] = (
                f"<myPT_assistant> should have mask=0, got {ep_mask[open_idx]}"
            )
        content_start = open_idx + 1
        if content_start < close_idx:
            content_mask = ep_mask[content_start:close_idx]
            if not np.all(content_mask == 1):
                zero_positions = np.where(content_mask == 0)[0]
                failures[f"{prefix}assistant_content_mask"] = (
                    f"Found mask=0 at content positions: {zero_positions[:5]}..."
                )
        if ep_mask[close_idx] != 1:
            failures[f"{prefix}assistant_close_mask"] = (
                f"</myPT_assistant> should have mask=1, got {ep_mask[close_idx]}"
            )

    if ep_mask is not None and np.sum(ep_mask == 1) == 0:
        failures[f"{prefix}mask_ratio_zero"] = "no trained tokens"

    return failures


def validate_episode(
    ep_idx: int,
    tokens: np.ndarray,
    mask: np.ndarray,
    start: int,
    length: int,
    segments: np.ndarray = None,
    verbose: bool = False,
) -> dict:
    """Validate one packed or unpacked row."""
    failures = {}

    ep_tokens = tokens[start:start + length]
    ep_mask = mask[start:start + length] if mask is not None else None
    ep_seg = segments[start:start + length] if segments is not None else None

    if ep_mask is None:
        failures["no_mask"] = "Mask file not found"
        return failures

    if ep_seg is not None:
        pad = ep_seg == 0
        if np.any(pad) and np.any(ep_mask[pad] != 0):
            failures["pad_mask"] = "Padding (segment_id 0) must have mask=0"
        max_seg = int(ep_seg.max()) if length else 0
        if max_seg == 0:
            failures["no_segments"] = "Packed row has no non-pad segments"
        for sid in range(1, max_seg + 1):
            sel = ep_seg == sid
            if not np.any(sel):
                continue
            failures.update(
                validate_assistant_spans(ep_tokens[sel], ep_mask[sel], prefix=f"seg{sid}_")
            )
            # Conversation EOT is trained; pad EOTs live in segment 0 and are ignored.
            if np.sum(ep_tokens[sel] == EOT_ID) < 1:
                failures[f"seg{sid}_eot_count"] = "Expected >=1 content EOT, got 0"
    else:
        failures.update(validate_assistant_spans(ep_tokens, ep_mask))

    if verbose and not failures:
        trained = int(np.sum(ep_mask == 1))
        n_open = int(np.sum(ep_tokens == ASSISTANT_OPEN_ID))
        print(f"  Episode {ep_idx}: {length} tokens, {trained} trained, {n_open} assistant spans, OK")

    return failures


def validate_alignment_offbyone(ep_idx: int, tokens: np.ndarray, mask: np.ndarray,
                                 start: int, length: int, block_size: int = 1024) -> dict:
    """
    Validate that mask alignment is correct for x/y/mask_y formation.

    In training: x=seq[:-1], y=seq[1:], mask_y=mask[1:]
    So if mask[p]==1 (first trained position), then mask_y[p-1] should be 1.
    """
    failures = {}

    ep_tokens = tokens[start:start + length]
    ep_mask = mask[start:start + length] if mask is not None else None

    if ep_mask is None:
        return failures

    L = min(length, block_size + 1)
    m = ep_mask[:L]
    mask_y = m[1:]

    first_trained = np.where(m == 1)[0]
    if len(first_trained) > 0:
        p = first_trained[0]
        if p > 0 and p - 1 < len(mask_y):
            if mask_y[p - 1] != 1:
                failures["alignment_offbyone"] = f"mask[{p}]==1 but mask_y[{p-1}]=={mask_y[p-1]}"

    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate SFT episode masks")
    parser.add_argument("--dataset_dir", required=True, help="Path to prepared dataset")
    parser.add_argument("--split", default="train", help="Split to validate (train/val)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to check")
    args = parser.parse_args()

    meta = load_metadata(args.dataset_dir)
    packing = bool(meta.get("packing_enabled"))
    pack_block = meta.get("pack_block_size")
    n_conv = meta.get("num_conversations")

    print(f"Loading dataset from {args.dataset_dir}/{args.split}...")
    tokens, mask, episodes, segments = load_dataset(args.dataset_dir, args.split)

    print(f"Found {len(episodes)} rows, {len(tokens):,} tokens")
    if packing:
        print(
            f"Packed dataset: block_size={pack_block}, "
            f"source conversations={n_conv}, segment_ids={'yes' if segments is not None else 'no'}"
        )
        print("Multi-span <myPT_assistant> per row is expected (packing + toolcall chains).")
    if mask is not None:
        print(f"Mask file present: {len(mask):,} entries, ratio={float(mask.mean()):.1%}")
    else:
        print("WARNING: No mask file found!")

    failure_counts = defaultdict(int)
    trained_token_counts = []
    open_counts = []
    failed_episodes = []

    n_episodes = len(episodes)
    if args.max_episodes:
        n_episodes = min(n_episodes, args.max_episodes)

    print(f"\nValidating {n_episodes} rows (span-level, packing-aware)...")

    for i in range(n_episodes):
        start, length = int(episodes[i, 0]), int(episodes[i, 1])

        failures = validate_episode(
            i, tokens, mask, start, length, segments=segments, verbose=args.verbose
        )
        failures.update(validate_alignment_offbyone(i, tokens, mask, start, length))

        if failures:
            for key in failures:
                failure_counts[key] += 1
            failed_episodes.append((i, failures))

        if mask is not None:
            ep_mask = mask[start:start + length]
            trained_token_counts.append(np.sum(ep_mask == 1))
        ep_tokens = tokens[start:start + length]
        open_counts.append(int(np.sum(ep_tokens == ASSISTANT_OPEN_ID)))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_episodes} rows...")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nRows validated: {n_episodes}")
    print(f"Rows with failures: {len(failed_episodes)}")
    if open_counts:
        arr_o = np.array(open_counts)
        print(
            f"Assistant spans per row: min={arr_o.min()} median={np.median(arr_o):.0f} max={arr_o.max()}"
        )

    if failure_counts:
        print("\nFailure counts by type:")
        for key, count in sorted(failure_counts.items()):
            print(f"  {key}: {count}")

    if trained_token_counts:
        arr = np.array(trained_token_counts)
        print(f"\nTrained tokens per row:")
        print(f"  Min: {arr.min()}")
        print(f"  Median: {np.median(arr):.1f}")
        print(f"  Max: {arr.max()}")
        print(f"  Total: {arr.sum():,}")
        if arr.min() == 0:
            print("  WARNING: at least one all-zero mask row")

    if failed_episodes and args.verbose:
        print("\nFirst 5 failed rows:")
        for ep_idx, failures in failed_episodes[:5]:
            start, length = int(episodes[ep_idx, 0]), int(episodes[ep_idx, 1])
            print(f"\n  Row {ep_idx} (start={start}, length={length}):")
            for key, msg in failures.items():
                print(f"    {key}: {msg}")

    if failed_episodes:
        print(f"\nVALIDATION FAILED: {len(failed_episodes)} rows have real mask/span issues")
        return 1

    print("\nVALIDATION PASSED: All assistant spans OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
