#!/usr/bin/env python
"""
Tokenize Context Extension Dataset into Episode-Indexed Binary Format
======================================================================
Combines two data sources into a single episode-indexed dataset:

1. QA episodes (~40%): From build_context_extension_dataset.py output.
   Plain-text passage + question + answer, <|endoftext|> delimited.

2. General text (~60%): Sampled from existing pre-training shards.
   Re-chunked at document boundaries (EOS token 50256) into ~4096-token
   episodes for positional encoding adaptation on diverse text.

Output uses the episode-indexed format expected by GPTEpisodeDataLoader:
    output_dir/
        train/
            tokens.bin      # uint32 (packed blocks concatenated)
            mask.bin         # uint8 (1=real, 0=padding)
            episodes.idx    # uint64 (start, length) per packed block
        val/
            ...
        dataset_metadata.json
        tokenizer_state.json

Why episode-indexed (not token-stream)?
    Token-stream samples random windows that can cut episodes mid-sentence.
    A partial QA episode (question without passage) teaches hallucination.
    Episode-indexed packing guarantees complete episodes only.

Prerequisites:
    pip install tiktoken numpy

Usage:
    # QA only (no general text)
    python scripts/data_prep/prepare_context_extension.py

    # QA + general text from pre-training shards (recommended)
    python scripts/data_prep/prepare_context_extension.py \\
        --general_shards_dir data/unified_6B \\
        --general_target_tokens 480000000

    # Custom paths
    python scripts/data_prep/prepare_context_extension.py \\
        --input_file data/context_extension_raw/context_ext.txt \\
        --general_shards_dir data/unified_6B \\
        --general_target_tokens 480000000 \\
        --out_dir data/context_extension
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import tiktoken
except ImportError:
    print("ERROR: 'tiktoken' not installed. Run: pip install tiktoken")
    sys.exit(1)


EOS_TOKEN_ID = 50256
EOS_DELIMITER = "<|endoftext|>"
PAD_TOKEN_ID = EOS_TOKEN_ID

SHORT_THRESHOLD = 500
LONG_THRESHOLD = 1200


# ---------------------------------------------------------------------------
# QA episode tokenization
# ---------------------------------------------------------------------------

def tokenize_qa_episodes(input_path: Path, enc, max_tokens=None):
    """Split raw QA text on <|endoftext|>, tokenize each episode."""
    raw_text = input_path.read_text(encoding="utf-8")
    documents = raw_text.split(EOS_DELIMITER)
    del raw_text

    episodes = []
    n_empty = 0
    n_too_long = 0

    for doc in documents:
        doc = doc.strip()
        if not doc:
            n_empty += 1
            continue

        tokens = enc.encode_ordinary(doc)
        if not tokens:
            n_empty += 1
            continue

        tokens.append(EOS_TOKEN_ID)

        if max_tokens is not None and len(tokens) > max_tokens:
            n_too_long += 1
            continue

        episodes.append(np.array(tokens, dtype=np.uint32))

    return episodes, n_empty, n_too_long


# ---------------------------------------------------------------------------
# General text sampling from pre-training shards
# ---------------------------------------------------------------------------

def sample_general_episodes(shards_dir: str, target_tokens: int, block_size: int, seed: int):
    """
    Sample general text episodes from pre-training shards.

    Reads uint32 binary shards. If shards contain EOS (50256) boundaries,
    splits on them and concatenates short documents into ~block_size
    mega-episodes. If shards are continuous streams (no EOS), chunks them
    directly into block_size episodes with an EOS appended.

    Both paths ensure episodes fill the full block_size for RoPE training.
    """
    train_dir = os.path.join(shards_dir, "train")
    if not os.path.exists(train_dir):
        train_dir = shards_dir

    shard_files = sorted(glob.glob(os.path.join(train_dir, "*.bin")))
    shard_files = [f for f in shard_files if not f.endswith("_mask.bin")]

    if not shard_files:
        print(f"  ERROR: No .bin shards found in {train_dir}")
        return []

    print(f"  Found {len(shard_files)} shards in {train_dir}")

    rng = np.random.RandomState(seed)
    rng.shuffle(shard_files)

    # Probe first shard for EOS tokens to determine strategy
    probe = np.memmap(shard_files[0], dtype=np.uint32, mode='r')
    has_eos = int((probe == EOS_TOKEN_ID).sum()) > 0
    del probe

    if has_eos:
        print(f"  Mode: EOS-delimited (document boundaries found)")
        return _sample_eos_delimited(shard_files, target_tokens, block_size, rng)
    else:
        print(f"  Mode: continuous stream (no EOS found, chunking at {block_size})")
        return _sample_continuous(shard_files, target_tokens, block_size, rng)


def _sample_continuous(shard_files, target_tokens, block_size, rng):
    """Chunk continuous token streams into fixed block_size episodes."""
    chunk_content = block_size - 1  # reserve 1 token for trailing EOS
    episodes = []
    total_tokens = 0

    for shard_idx, shard_path in enumerate(shard_files):
        if total_tokens >= target_tokens:
            break

        data = np.memmap(shard_path, dtype=np.uint32, mode='r')
        n_chunks = len(data) // chunk_content

        for c in range(n_chunks):
            if total_tokens >= target_tokens:
                break

            chunk = np.empty(block_size, dtype=np.uint32)
            chunk[:chunk_content] = data[c * chunk_content:(c + 1) * chunk_content]
            chunk[-1] = EOS_TOKEN_ID
            episodes.append(chunk)
            total_tokens += block_size

        del data

        if (shard_idx + 1) % 50 == 0:
            print(f"    Shard {shard_idx+1}/{len(shard_files)}: "
                  f"{len(episodes):,} episodes, {total_tokens:,} tokens so far")

    print(f"  Chunked into {len(episodes):,} full-length episodes "
          f"({total_tokens:,} tokens, all {block_size} tokens each)")
    return episodes


def _sample_eos_delimited(shard_files, target_tokens, block_size, rng):
    """Split on EOS boundaries, concatenate short docs into mega-episodes."""
    min_mega_len = int(block_size * 0.75)
    episodes = []
    total_tokens = 0
    n_long = 0
    n_mega = 0
    mega_buffer = np.empty(0, dtype=np.uint32)

    for shard_idx, shard_path in enumerate(shard_files):
        if total_tokens >= target_tokens:
            break

        data = np.memmap(shard_path, dtype=np.uint32, mode='r')
        eos_positions = np.where(data == EOS_TOKEN_ID)[0]

        if len(eos_positions) == 0:
            del data
            continue

        prev_end = 0
        for eos_pos in eos_positions:
            if total_tokens >= target_tokens:
                break

            doc = data[prev_end:eos_pos + 1]
            prev_end = eos_pos + 1
            doc_len = len(doc)

            if doc_len < 65:
                continue

            if doc_len >= block_size:
                if len(mega_buffer) >= min_mega_len:
                    episodes.append(np.array(mega_buffer, dtype=np.uint32))
                    total_tokens += len(mega_buffer)
                    n_mega += 1
                mega_buffer = np.empty(0, dtype=np.uint32)

                episodes.append(np.array(doc[:block_size], dtype=np.uint32))
                total_tokens += block_size
                n_long += 1
            else:
                if len(mega_buffer) + doc_len <= block_size:
                    mega_buffer = np.concatenate([mega_buffer, doc])
                else:
                    if len(mega_buffer) >= min_mega_len:
                        episodes.append(np.array(mega_buffer, dtype=np.uint32))
                        total_tokens += len(mega_buffer)
                        n_mega += 1
                    mega_buffer = np.array(doc, dtype=np.uint32)

        del data

        if (shard_idx + 1) % 50 == 0:
            print(f"    Shard {shard_idx+1}/{len(shard_files)}: "
                  f"{len(episodes):,} episodes, {total_tokens:,} tokens so far")

    if len(mega_buffer) >= min_mega_len:
        episodes.append(np.array(mega_buffer, dtype=np.uint32))
        total_tokens += len(mega_buffer)
        n_mega += 1

    print(f"  Concatenated into {len(episodes):,} episodes "
          f"({n_long:,} full-length docs, {n_mega:,} mega-episodes)")
    return episodes


# ---------------------------------------------------------------------------
# Diversity interleaving and packing
# ---------------------------------------------------------------------------

def classify_episode(tokens):
    """Classify episode by length for diversity bucketing."""
    length = len(tokens)
    if length <= SHORT_THRESHOLD:
        return "short"
    elif length >= LONG_THRESHOLD:
        return "long"
    else:
        return "medium"


def interleave_episodes(episodes, label=""):
    """
    Round-robin interleave episodes from short/medium/long buckets
    so packed blocks contain a mix of episode types.
    """
    buckets = {"short": [], "medium": [], "long": []}
    for ep in episodes:
        bucket = classify_episode(ep)
        buckets[bucket].append(ep)

    print(f"  {label}Episode buckets: short={len(buckets['short'])}, "
          f"medium={len(buckets['medium'])}, long={len(buckets['long'])}")

    interleaved = []
    bucket_iters = {k: iter(v) for k, v in buckets.items()}
    active_keys = [k for k in ["short", "long", "medium"] if buckets[k]]

    while active_keys:
        exhausted = []
        for key in active_keys:
            try:
                interleaved.append(next(bucket_iters[key]))
            except StopIteration:
                exhausted.append(key)
        for key in exhausted:
            active_keys.remove(key)

    return interleaved


def greedy_pack(episodes, block_size, pad_token_id):
    """
    Pack episodes into fixed-size blocks. No episode splitting.
    Produces segment_ids for segment-isolated attention (prevents
    cross-episode attention bleeding within packed blocks).

    Returns list of (tokens, mask, segment_ids) tuples.
    segment_ids: 0=padding, 1=first episode, 2=second episode, etc.
    """
    blocks = []
    current_tokens = []
    current_mask = []
    current_segments = []
    current_seg_id = 1

    for ep_tokens in episodes:
        ep_len = len(ep_tokens)

        if ep_len > block_size:
            ep_tokens = ep_tokens[:block_size]
            ep_len = block_size

        remaining = block_size - len(current_tokens)

        if ep_len <= remaining:
            current_tokens.extend(ep_tokens)
            current_mask.extend([1] * ep_len)
            current_segments.extend([current_seg_id] * ep_len)
            current_seg_id += 1
            if current_seg_id > 255:
                current_seg_id = 1
        else:
            pad_len = block_size - len(current_tokens)
            current_tokens.extend([pad_token_id] * pad_len)
            current_mask.extend([0] * pad_len)
            current_segments.extend([0] * pad_len)

            assert len(current_tokens) == block_size
            blocks.append((current_tokens, current_mask, current_segments))

            current_tokens = list(ep_tokens)
            current_mask = [1] * ep_len
            current_segments = [1] * ep_len
            current_seg_id = 2

    if current_tokens:
        pad_len = block_size - len(current_tokens)
        current_tokens.extend([pad_token_id] * pad_len)
        current_mask.extend([0] * pad_len)
        current_segments.extend([0] * pad_len)
        blocks.append((current_tokens, current_mask, current_segments))

    return blocks


def _finalize_bin(bin_state, block_size, pad_token_id):
    """Finalize one open bin into fixed-size token/mask/segment arrays."""
    used = bin_state["used"]
    pad_len = block_size - used

    tokens = list(bin_state["tokens"])
    mask = list(bin_state["mask"])
    segments = list(bin_state["segments"])

    if pad_len > 0:
        tokens.extend([pad_token_id] * pad_len)
        mask.extend([0] * pad_len)
        segments.extend([0] * pad_len)

    return (
        np.asarray(tokens, dtype=np.uint32),
        np.asarray(mask, dtype=np.uint8),
        np.asarray(segments, dtype=np.uint8),
        used,
        bin_state["next_seg"] - 1,
    )


def pack_and_write_split(episodes, split_dir: Path, block_size: int, pad_token_id: int, max_open_bins: int = 256):
    """
    Pack episodes with online best-fit and stream directly to disk.

    This improves density versus strict append-only greedy packing by placing each
    episode into the currently best-fitting open block (least remaining space after
    insertion). It also avoids constructing giant in-memory packed block lists.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    if not episodes:
        print(f"  WARNING: No blocks for {split_dir.name}, skipping")
        return 0, 0, 0.0

    tokens_path = split_dir / "tokens.bin"
    mask_path = split_dir / "mask.bin"
    segments_path = split_dir / "segment_ids.bin"
    idx_entries = []
    real_tokens = 0
    total_segments = 0
    n_blocks = 0
    open_bins = []

    def finalize_open_bin(bin_idx, f_tokens, f_mask, f_segments):
        nonlocal real_tokens, total_segments, n_blocks
        t_arr, m_arr, s_arr, used, seg_count = _finalize_bin(
            open_bins[bin_idx], block_size, pad_token_id
        )
        t_arr.tofile(f_tokens)
        m_arr.tofile(f_mask)
        s_arr.tofile(f_segments)
        idx_entries.append((n_blocks * block_size, block_size))
        real_tokens += int(used)
        total_segments += int(seg_count)
        n_blocks += 1
        open_bins.pop(bin_idx)

    with open(tokens_path, "wb") as f_tokens, open(mask_path, "wb") as f_mask, open(segments_path, "wb") as f_segments:
        for i, ep in enumerate(episodes):
            ep_tokens = ep
            if len(ep_tokens) > block_size:
                ep_tokens = ep_tokens[:block_size]
            ep_len = len(ep_tokens)
            if ep_len == 0:
                continue

            # Best-fit: choose open bin with smallest remaining slack after placement.
            best_idx = -1
            best_slack = block_size + 1
            for j, b in enumerate(open_bins):
                if b["next_seg"] > 255:
                    continue
                remaining = block_size - b["used"]
                if ep_len <= remaining:
                    slack = remaining - ep_len
                    if slack < best_slack:
                        best_slack = slack
                        best_idx = j
                        if slack == 0:
                            break

            if best_idx < 0:
                open_bins.append({
                    "tokens": [],
                    "mask": [],
                    "segments": [],
                    "used": 0,
                    "next_seg": 1,
                })
                best_idx = len(open_bins) - 1

            b = open_bins[best_idx]
            seg_id = b["next_seg"]
            b["next_seg"] += 1

            if isinstance(ep_tokens, np.ndarray):
                b["tokens"].extend(ep_tokens.tolist())
            else:
                b["tokens"].extend(ep_tokens)
            b["mask"].extend([1] * ep_len)
            b["segments"].extend([seg_id] * ep_len)
            b["used"] += ep_len

            # Close bins that are full or reached segment ID limit.
            if b["used"] >= block_size or b["next_seg"] > 255:
                finalize_open_bin(best_idx, f_tokens, f_mask, f_segments)

            # Bound memory: if too many open bins, close the fullest one.
            if len(open_bins) > max_open_bins:
                fullest_idx = max(range(len(open_bins)), key=lambda k: open_bins[k]["used"])
                finalize_open_bin(fullest_idx, f_tokens, f_mask, f_segments)

            if (i + 1) % 100000 == 0:
                print(
                    f"    {split_dir.name}: processed {i+1:,}/{len(episodes):,} episodes, "
                    f"wrote {n_blocks:,} blocks, open_bins={len(open_bins)}"
                )

        # Flush remaining open bins (fullest first for slightly better locality).
        open_bins.sort(key=lambda x: x["used"], reverse=True)
        while open_bins:
            finalize_open_bin(0, f_tokens, f_mask, f_segments)

    episodes_idx = np.asarray(idx_entries, dtype=np.uint64)
    episodes_idx.tofile(str(split_dir / "episodes.idx"))
    avg_segments = (total_segments / n_blocks) if n_blocks > 0 else 0.0
    return n_blocks, real_tokens, avg_segments


# ---------------------------------------------------------------------------
# Length distribution diagnostics
# ---------------------------------------------------------------------------

def print_length_histogram(episodes, label, block_size):
    """Print length distribution histogram for a list of token-list episodes."""
    if not episodes:
        return
    bins = [
        (0, 500, "0-500"),
        (500, 1000, "500-1K"),
        (1000, 2000, "1K-2K"),
        (2000, 3000, "2K-3K"),
        (3000, block_size + 1, f"3K-{block_size}"),
    ]
    lengths = [len(ep) for ep in episodes]
    total_tokens = sum(lengths)
    long_tokens = sum(l for l in lengths if l >= 3000)

    print(f"  {label} length distribution ({len(episodes):,} episodes):")
    for lo, hi, name in bins:
        count = sum(1 for l in lengths if lo <= l < hi)
        tok = sum(l for l in lengths if lo <= l < hi)
        if count > 0:
            print(f"    {name:>10}: {count:>7,} episodes ({count/len(episodes):>5.1%}), "
                  f"{tok:>12,} tokens ({tok/total_tokens:>5.1%})")
    print(f"    Tokens in episodes >=3K: {long_tokens:,} / {total_tokens:,} "
          f"({long_tokens/total_tokens:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize context extension dataset (QA + general text) into episode-indexed format",
    )
    parser.add_argument("--input_file", type=str,
                        default="data/context_extension_raw/context_ext.txt",
                        help="Input QA text file with <|endoftext|> delimiters")
    parser.add_argument("--general_shards_dir", type=str, default=None,
                        help="Pre-training shard directory for general text (e.g. data/unified_6B)")
    parser.add_argument("--general_target_tokens", type=int, default=480_000_000,
                        help="Target tokens for general text portion (default: 480M for 60%% of ~800M)")
    parser.add_argument("--out_dir", type=str,
                        default="data/context_extension",
                        help="Output directory")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Block size for packing (default: 4096)")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Fraction of episodes for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Run build_context_extension_dataset.py first.")
        sys.exit(1)

    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("Context Extension Tokenizer (Episode-Indexed)")
    print("=" * 70)
    print(f"QA input:   {input_path}")
    print(f"General:    {args.general_shards_dir or 'NONE (QA only)'}")
    if args.general_shards_dir:
        print(f"  Target:   {args.general_target_tokens:,} tokens")
    print(f"Output:     {out_dir}")
    print(f"Block size: {args.block_size}")

    enc = tiktoken.get_encoding("gpt2")

    # Step 1: Tokenize QA episodes
    print("\n[1/6] Tokenizing QA episodes...")
    qa_episodes, n_empty, n_too_long = tokenize_qa_episodes(
        input_path,
        enc,
        max_tokens=args.block_size,
    )
    qa_tokens = sum(len(ep) for ep in qa_episodes)
    qa_lengths = [len(ep) for ep in qa_episodes]
    print(f"  QA episodes: {len(qa_episodes):,} (skipped {n_empty} empty, {n_too_long} >{args.block_size} tokens)")
    print(f"  QA tokens:   {qa_tokens:,}")
    print(f"  Lengths:     min={min(qa_lengths)}, median={sorted(qa_lengths)[len(qa_lengths)//2]}, "
          f"max={max(qa_lengths)}, mean={qa_tokens/len(qa_episodes):.0f}")
    print_length_histogram(qa_episodes, "QA", args.block_size)

    # Step 2: Sample general text episodes
    general_episodes = []
    general_tokens = 0
    if args.general_shards_dir:
        print(f"\n[2/6] Sampling general text from pre-training shards...")
        general_episodes = sample_general_episodes(
            args.general_shards_dir,
            target_tokens=args.general_target_tokens,
            block_size=args.block_size,
            seed=args.seed,
        )
        general_tokens = sum(len(ep) for ep in general_episodes)
        gen_lengths = [len(ep) for ep in general_episodes]
        print(f"  General episodes: {len(general_episodes):,}")
        print(f"  General tokens:   {general_tokens:,}")
        if gen_lengths:
            print(f"  Lengths: min={min(gen_lengths)}, median={sorted(gen_lengths)[len(gen_lengths)//2]}, "
                  f"max={max(gen_lengths)}, mean={general_tokens/len(general_episodes):.0f}")
        print_length_histogram(general_episodes, "General", args.block_size)
    else:
        print("\n[2/6] No general shards specified, QA only mode.")

    total_tokens = qa_tokens + general_tokens
    total_episodes_raw = len(qa_episodes) + len(general_episodes)
    qa_pct = qa_tokens / total_tokens * 100 if total_tokens > 0 else 100
    gen_pct = general_tokens / total_tokens * 100 if total_tokens > 0 else 0
    print(f"\n  Combined: {total_tokens:,} tokens ({total_episodes_raw:,} episodes)")
    print(f"  Mix: QA {qa_pct:.0f}% / General {gen_pct:.0f}%")

    # Step 3: Combine all episodes and train/val split at episode level
    print(f"\n[3/6] Train/val split...")
    all_raw_episodes = qa_episodes + general_episodes
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_raw_episodes))

    n_val = max(1, int(len(all_raw_episodes) * args.val_fraction))
    val_indices = set(indices[:n_val])
    train_eps = [all_raw_episodes[i] for i in range(len(all_raw_episodes)) if i not in val_indices]
    val_eps = [all_raw_episodes[i] for i in range(len(all_raw_episodes)) if i in val_indices]

    print(f"  Train: {len(train_eps):,} episodes ({sum(len(e) for e in train_eps):,} tokens)")
    print(f"  Val:   {len(val_eps):,} episodes ({sum(len(e) for e in val_eps):,} tokens)")

    # Step 4: Diversity interleave (mix short QA, long QA, general in each block)
    print(f"\n[4/6] Diversity interleaving...")
    train_ordered = interleave_episodes(train_eps, label="Train: ")
    val_ordered = interleave_episodes(val_eps, label="Val:   ")

    # Step 5: Pack + write train split with best-fit
    print(f"\n[5/6] Packing + writing train split ({args.block_size}-token blocks, best-fit)...")
    n_train_blocks, n_train_real, avg_train_segs = pack_and_write_split(
        train_ordered,
        out_dir / "train",
        block_size=args.block_size,
        pad_token_id=PAD_TOKEN_ID,
    )
    train_total = n_train_blocks * args.block_size
    print(f"  Train avg episodes/block: {avg_train_segs:.1f} (segment-isolated attention active)")
    print(f"  Train: {n_train_blocks:,} blocks, "
          f"{n_train_real:,}/{train_total:,} real tokens ({n_train_real/train_total:.1%} density)")

    # Step 6: Pack + write val split with same strategy
    print(f"\n[6/6] Packing + writing val split...")
    n_val_blocks, n_val_real, avg_val_segs = pack_and_write_split(
        val_ordered,
        out_dir / "val",
        block_size=args.block_size,
        pad_token_id=PAD_TOKEN_ID,
    )
    val_total = n_val_blocks * args.block_size
    if n_val_blocks:
        print(f"  Val avg episodes/block:   {avg_val_segs:.1f}")
        print(f"  Val:   {n_val_blocks:,} blocks, "
              f"{n_val_real:,}/{val_total:,} real tokens ({n_val_real/val_total:.1%} density)")

    # Metadata
    metadata = {
        "schema": "episode_indexed_v1",
        "description": "Context extension dataset (Phase 1b). QA episodes + general text, packed.",
        "num_train_episodes": n_train_blocks,
        "num_val_episodes": n_val_blocks,
        "num_train_tokens": n_train_blocks * args.block_size,
        "num_val_tokens": n_val_blocks * args.block_size,
        "num_train_real_tokens": n_train_real,
        "num_val_real_tokens": n_val_real,
        "block_size": args.block_size,
        "qa_episodes": len(qa_episodes),
        "qa_tokens": qa_tokens,
        "general_episodes": len(general_episodes),
        "general_tokens": general_tokens,
        "qa_fraction": round(qa_pct / 100, 3),
        "general_fraction": round(gen_pct / 100, 3),
        "packing": True,
        "packing_strategy": "online_best_fit_with_diversity_interleaving",
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": EOS_TOKEN_ID,
        "loss_mask": "all_real_tokens (mask=1 for content, mask=0 for padding)",
        "qa_source": str(input_path),
        "general_source": args.general_shards_dir,
    }
    with open(out_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    tok_state = {
        "token_kind": "gpt2",
        "base_vocab_size": 50257,
        "model_vocab_size": 50304,
    }
    with open(out_dir / "tokenizer_state.json", "w") as f:
        json.dump(tok_state, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("Tokenization Complete!")
    print("=" * 70)
    print(f"Total real tokens: {n_train_real + n_val_real:,}")
    print(f"  QA:      {qa_tokens:,} ({qa_pct:.0f}%)")
    print(f"  General: {general_tokens:,} ({gen_pct:.0f}%)")
    print(f"Train: {n_train_blocks:,} blocks ({n_train_real:,} real tokens, "
          f"{n_train_real/(n_train_blocks * args.block_size):.1%} density)")
    print(f"Val:   {n_val_blocks:,} blocks ({n_val_real:,} real tokens)")
    print(f"Format: episode-indexed (tokens.bin + mask.bin + episodes.idx)")
    print(f"Output: {out_dir}")
    print(f"\nTo train:")
    print(f"  python train.py \\")
    print(f"    --model_name phase1b_context_ext \\")
    print(f"    --config_file configs/phase1b_context_extension.json \\")
    print(f"    --dataset_dir {out_dir} \\")
    print(f"    --init_from_model GOLD_unified_v1")


if __name__ == "__main__":
    main()
