#!/usr/bin/env python
"""
Fetch and Prepare Phase 1.5 Induction Strengthening Dataset

This script automates the complete pipeline for creating the Phase 1.5 dataset
designed to strengthen induction heads in the base model. The dataset is composed
of repetition-rich data: code, dialogue, and structured text.

Pipeline:
1. Read source definitions from data/sources/phase1_5_induction.json
2. Download sources from HuggingFace / OPUS / web
3. Clean and filter using tools/clean_code_corpus.py and tools/clean_dialogue_corpus.py
4. Tokenize and shard with prepare_weighted_dataset.py

Usage:
    # Full pipeline
    python scripts/fetch_and_prepare_phase1_5.py \\
        --sources_file data/sources/phase1_5_induction.json \\
        --out_dir data/phase1_5_induction_raw \\
        --total_tokens 3500000000

    # Skip download (use existing raw data)
    python scripts/fetch_and_prepare_phase1_5.py \\
        --skip_download \\
        --out_dir data/phase1_5_induction_raw \\
        --total_tokens 3500000000

    # Download only (inspect before cleaning)
    python scripts/fetch_and_prepare_phase1_5.py \\
        --download_only \\
        --out_dir data/phase1_5_induction_raw

    # Select specific sources
    python scripts/fetch_and_prepare_phase1_5.py \\
        --sources codeparrot opensub_en \\
        --out_dir data/phase1_5_induction_raw \\
        --total_tokens 500000000
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.banner import print_banner

# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_header(text: str, char: str = "=", width: int = 70):
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_subheader(text: str, char: str = "-", width: int = 60):
    print()
    print(f"  {char * 3} {text} {char * 3}")


def print_status(status: str, message: str, indent: int = 2):
    indicators = {
        "OK": "[OK]", "SKIP": "[SKIP]", "WARN": "[WARN]",
        "ERROR": "[ERROR]", "INFO": "[INFO]", "DONE": "[DONE]",
        "DL": "[DOWN]", "CLEAN": "[CLN]",
    }
    ind = indicators.get(status, f"[{status}]")
    prefix = " " * indent
    print(f"{prefix}{ind} {message}")


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ---------------------------------------------------------------------------
# Source config loading
# ---------------------------------------------------------------------------

DEFAULT_SOURCES_FILE = "data/sources/phase1_5_induction.json"


def load_sources(json_path: str) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """Load source definitions, weights, and metadata from JSON config."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sources = data.get("sources", {})
    weights = data.get("default_weights", {})
    replay = data.get("replay", {})
    metadata = {
        "name": data.get("name", "Phase 1.5 Dataset"),
        "description": data.get("description", ""),
        "version": data.get("version", "1.0"),
        "replay": replay,
    }
    return sources, weights, metadata


# ---------------------------------------------------------------------------
# HuggingFace download helpers
# ---------------------------------------------------------------------------

def download_hf_dataset(source_key: str, source_cfg: dict, raw_dir: str,
                        max_tokens: int = 0) -> Tuple[bool, str, float]:
    """
    Download a HuggingFace dataset and save as JSONL.
    
    Uses the `datasets` library to stream data, avoiding full download into RAM.
    Falls back to a subset if the dataset is gated or very large.
    """
    start = time.time()
    dataset_id = source_cfg["dataset_id"]
    subset = source_cfg.get("subset", None)
    split = source_cfg.get("split", "train")
    text_field = source_cfg.get("text_field", "content")
    est_tokens = source_cfg.get("est_tokens", 500_000_000)
    max_tok = source_cfg.get("max_tokens", max_tokens) or est_tokens

    dest_dir = os.path.join(raw_dir, source_key)
    os.makedirs(dest_dir, exist_ok=True)
    out_jsonl = os.path.join(dest_dir, f"{source_key}_raw.jsonl")

    if os.path.exists(out_jsonl):
        size = os.path.getsize(out_jsonl)
        print_status("SKIP", f"{source_key}: JSONL already exists ({format_size(size)})")
        return True, dest_dir, time.time() - start

    print_status("DL", f"Streaming {dataset_id}" + (f" [{subset}]" if subset else "") + f" split={split}")

    try:
        from datasets import load_dataset
    except ImportError:
        print_status("ERROR", "The 'datasets' library is required: pip install datasets")
        return False, dest_dir, time.time() - start

    try:
        from itertools import chain as itertools_chain

        data_dir = source_cfg.get("data_dir", None)

        # Handle multi-split (e.g. "todayilearned+programming+science")
        splits = [s.strip() for s in split.split("+")]

        datasets_to_chain = []
        for sp in splits:
            kwargs = {"streaming": True, "split": sp}
            if data_dir:
                kwargs["data_dir"] = data_dir
            try:
                if subset:
                    ds_part = load_dataset(dataset_id, subset, **kwargs)
                else:
                    ds_part = load_dataset(dataset_id, **kwargs)
                datasets_to_chain.append(ds_part)
                if len(splits) > 1:
                    print_status("OK", f"Loaded split '{sp}'")
            except Exception as e_split:
                print_status("WARN", f"Failed to load split '{sp}': {e_split}")
                continue

        if not datasets_to_chain:
            print_status("ERROR", f"No splits loaded for {dataset_id}")
            return False, dest_dir, time.time() - start

        if data_dir:
            print_status("INFO", f"Using data_dir='{data_dir}'")
        if len(splits) > 1:
            print_status("INFO", f"Chaining {len(datasets_to_chain)} splits")

        ds = itertools_chain(*datasets_to_chain)
    except Exception as e:
        print_status("ERROR", f"Failed to load {dataset_id}: {e}")
        return False, dest_dir, time.time() - start

    # Optional post-load filtering (e.g. filter by language column)
    filter_field = source_cfg.get("filter_field", None)
    filter_value = source_cfg.get("filter_value", None)
    if filter_field and filter_value:
        print_status("INFO", f"Filtering: {filter_field} == '{filter_value}'")

    # Estimate ~4 chars per token for budget tracking
    char_budget = max_tok * 4
    chars_written = 0
    docs_written = 0
    skipped_filter = 0

    total_scanned = 0

    try:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for record in ds:
                total_scanned += 1

                # Apply filter if configured
                if filter_field and filter_value:
                    if record.get(filter_field, "") != filter_value:
                        skipped_filter += 1
                        # Show scanning progress so user knows it's not stuck
                        if skipped_filter % 50000 == 0:
                            print(f"\r    {source_key}: scanning... {total_scanned:,} records, "
                                  f"{docs_written:,} matched '{filter_value}'   ", end="", flush=True)
                        continue

                text = record.get(text_field, "")
                if not text or len(text) < 50:
                    continue
                # Write as JSONL
                f.write(json.dumps({"content": text, "source": source_key}) + "\n")
                chars_written += len(text)
                docs_written += 1

                if docs_written % 10000 == 0:
                    est_tok = chars_written // 4
                    if filter_field:
                        hit_rate = docs_written / max(total_scanned, 1) * 100
                        print(f"\r    {source_key}: {docs_written:,} docs ({hit_rate:.0f}% hit rate), "
                              f"~{est_tok:,} tokens, scanned {total_scanned:,}   ", end="", flush=True)
                    else:
                        print(f"\r    {source_key}: {docs_written:,} docs, ~{est_tok:,} tokens estimated   ", end="", flush=True)

                if chars_written >= char_budget:
                    break
        print()  # newline
    except KeyboardInterrupt:
        print()
        print_status("WARN", f"Interrupted. Partial data saved ({docs_written:,} docs)")
    except Exception as e:
        print_status("ERROR", f"Stream error: {e}")
        if docs_written == 0 and os.path.exists(out_jsonl):
            os.remove(out_jsonl)
        return False, dest_dir, time.time() - start

    elapsed = time.time() - start
    size = os.path.getsize(out_jsonl) if os.path.exists(out_jsonl) else 0
    print_status("OK", f"{source_key}: {docs_written:,} docs, {format_size(size)} in {format_duration(elapsed)}")
    return True, dest_dir, elapsed


def download_opus_file(source_key: str, source_cfg: dict, raw_dir: str) -> Tuple[bool, str, float]:
    """Download an OPUS monolingual file (typically .txt.gz)."""
    start = time.time()
    url = source_cfg["url"]
    dest_dir = os.path.join(raw_dir, source_key)
    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(url)
    dest_path = os.path.join(dest_dir, filename)
    txt_path = dest_path.replace(".gz", "") if dest_path.endswith(".gz") else dest_path

    if os.path.exists(txt_path):
        size = os.path.getsize(txt_path)
        print_status("SKIP", f"{source_key}: Already downloaded ({format_size(size)})")
        return True, dest_dir, time.time() - start

    if os.path.exists(dest_path):
        print_status("SKIP", f"{source_key}: Archive exists, will extract")
    else:
        print_status("DL", f"Downloading {url[:80]}...")
        try:
            import ssl
            import urllib.request
            import urllib.error
            # Try normal download first, fall back to unverified SSL if cert fails
            try:
                urllib.request.urlretrieve(url, dest_path)
            except urllib.error.URLError as ssl_err:
                if "CERTIFICATE_VERIFY_FAILED" in str(ssl_err):
                    print_status("WARN", "SSL cert issue, retrying with unverified context...")
                    ctx = ssl.create_default_context()
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, context=ctx) as resp, open(dest_path, "wb") as out_f:
                        shutil.copyfileobj(resp, out_f)
                else:
                    raise
        except Exception as e:
            print_status("ERROR", f"Download failed: {e}")
            return False, dest_dir, time.time() - start
        print_status("OK", f"Downloaded {format_size(os.path.getsize(dest_path))}")

    # Extract .gz
    if dest_path.endswith(".gz") and not os.path.exists(txt_path):
        print_status("INFO", f"Extracting {filename}...")
        try:
            with gzip.open(dest_path, "rb") as gz_in, open(txt_path, "wb") as f_out:
                shutil.copyfileobj(gz_in, f_out)
            print_status("OK", f"Extracted to {format_size(os.path.getsize(txt_path))}")
        except Exception as e:
            print_status("ERROR", f"Extract failed: {e}")
            return False, dest_dir, time.time() - start

    elapsed = time.time() - start
    return True, dest_dir, elapsed


# ---------------------------------------------------------------------------
# Cleaning orchestration
# ---------------------------------------------------------------------------

def clean_code_source(source_key: str, source_cfg: dict, raw_dir: str,
                      clean_dir: str) -> Tuple[bool, str, float]:
    """Clean a code source using tools/clean_code_corpus.py."""
    start = time.time()
    src_dir = os.path.join(raw_dir, source_key)
    out_dir = os.path.join(clean_dir, source_key)

    if os.path.exists(out_dir):
        shards = list(Path(out_dir).glob("shard_*.txt"))
        total_size = sum(s.stat().st_size for s in shards) if shards else 0
        if shards and total_size > 1000:  # Non-empty shards (>1KB total)
            print_status("SKIP", f"{source_key}: Already cleaned ({len(shards)} shards, {total_size // 1024}KB)")
            return True, out_dir, 0
        elif shards:
            # Stale empty shards from a previous failed run — remove and redo
            print_status("WARN", f"{source_key}: Found {len(shards)} empty shards, re-cleaning...")
            shutil.rmtree(out_dir)

    jsonl_path = os.path.join(src_dir, f"{source_key}_raw.jsonl")
    language = source_cfg.get("language", "python")

    script = os.path.join(os.path.dirname(__file__), "..", "tools", "clean_code_corpus.py")

    cmd = [sys.executable, script]

    if os.path.exists(jsonl_path):
        cmd.extend(["--input_jsonl", jsonl_path])
    elif os.path.isdir(src_dir):
        cmd.extend(["--input_dir", src_dir])
    else:
        print_status("ERROR", f"No raw data found for {source_key} in {src_dir}")
        return False, out_dir, time.time() - start

    cmd.extend([
        "--output_dir", out_dir,
        "--language", language,
        "--source_name", source_key,
        "--shard_mb", "25",
    ])

    print_status("CLEAN", f"Cleaning {source_key} ({language})...")

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        return result.returncode == 0, out_dir, elapsed
    except subprocess.CalledProcessError as e:
        print_status("ERROR", f"clean_code_corpus failed for {source_key}: rc={e.returncode}")
        return False, out_dir, time.time() - start


def clean_dialogue_source(source_key: str, source_cfg: dict, raw_dir: str,
                          clean_dir: str) -> Tuple[bool, str, float]:
    """Clean a dialogue/structured text source using tools/clean_dialogue_corpus.py."""
    start = time.time()
    src_dir = os.path.join(raw_dir, source_key)
    out_dir = os.path.join(clean_dir, source_key)

    if os.path.exists(out_dir):
        shards = list(Path(out_dir).glob("shard_*.txt"))
        total_size = sum(s.stat().st_size for s in shards) if shards else 0
        if shards and total_size > 1000:  # Non-empty shards (>1KB total)
            print_status("SKIP", f"{source_key}: Already cleaned ({len(shards)} shards, {total_size // 1024}KB)")
            return True, out_dir, 0
        elif shards:
            print_status("WARN", f"{source_key}: Found {len(shards)} empty shards, re-cleaning...")
            shutil.rmtree(out_dir)

    script = os.path.join(os.path.dirname(__file__), "..", "tools", "clean_dialogue_corpus.py")

    # Determine format and input file
    # Note: download_hf_dataset always writes {"content": ...} regardless of
    # the original field name, so we use "content" for all HF-downloaded JSONL
    src_type = source_cfg.get("type", "huggingface")
    text_field = "content" if src_type == "huggingface" else source_cfg.get("text_field", "content")

    # Find input file
    jsonl_path = os.path.join(src_dir, f"{source_key}_raw.jsonl")
    txt_files = list(Path(src_dir).glob("*.txt"))

    # Map source types to dialogue cleaner formats
    format_map = {
        "opensub_en": "opensub",
        "reddit_threaded": "reddit_threaded",
        "stackexchange_qa": "stackexchange",
        "reddit_comments": "reddit",
        "stackoverflow": "stackoverflow",
    }
    fmt = format_map.get(source_key, "generic")

    cmd = [sys.executable, script]

    if fmt == "opensub" and txt_files:
        # OpenSubtitles uses plain text input
        cmd.extend(["--input_file", str(txt_files[0])])
    elif os.path.exists(jsonl_path):
        cmd.extend(["--input_jsonl", jsonl_path])
    else:
        print_status("ERROR", f"No raw data for {source_key} in {src_dir}")
        return False, out_dir, time.time() - start

    cmd.extend([
        "--output_dir", out_dir,
        "--format", fmt,
        "--text_field", text_field,
        "--source_name", source_key,
        "--shard_mb", "25",
    ])

    print_status("CLEAN", f"Cleaning {source_key} (format={fmt})...")

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        return result.returncode == 0, out_dir, elapsed
    except subprocess.CalledProcessError as e:
        print_status("ERROR", f"clean_dialogue_corpus failed for {source_key}: rc={e.returncode}")
        return False, out_dir, time.time() - start


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def run_tokenization(clean_dir: str, out_dir: str, weights: Dict[str, float],
                     total_tokens: int, tokens_per_shard: int,
                     val_fraction: float) -> Tuple[bool, float]:
    """Run prepare_weighted_dataset.py on cleaned shards."""
    start = time.time()

    print_header("Tokenizing and Sharding")
    print(f"    Clean data:     {clean_dir}")
    print(f"    Output:         {out_dir}")
    print(f"    Total tokens:   {total_tokens:,}")
    print(f"    Tokens/shard:   {tokens_per_shard:,}")
    print(f"    Val fraction:   {val_fraction}")

    script = os.path.join(os.path.dirname(__file__), "prepare_weighted_dataset.py")
    if not os.path.exists(script):
        print_status("ERROR", f"Cannot find {script}")
        return False, time.time() - start

    # Build --source and --weight arguments from available cleaned data
    source_args = []
    weight_args = []

    for source_key, w in weights.items():
        source_dir = os.path.join(clean_dir, source_key)
        if not os.path.isdir(source_dir):
            print_status("WARN", f"No cleaned data for {source_key}, skipping")
            continue

        shard_files = sorted(Path(source_dir).glob("shard_*.txt"))
        if not shard_files:
            print_status("WARN", f"No shards for {source_key}, skipping")
            continue

        # Use glob pattern instead of listing every file (avoids Windows cmd length limit)
        glob_pattern = os.path.join(source_dir, "shard_*.txt")
        source_args.append(f"--source={source_key}:{glob_pattern}")
        weight_args.append(f"--weight={source_key}:{w}")
        print(f"    {source_key}: {len(shard_files)} shards, weight={w:.2f}")

    if not source_args:
        print_status("ERROR", "No source data available for tokenization")
        return False, time.time() - start

    cmd = [
        sys.executable, script,
        *source_args,
        *weight_args,
        f"--total_tokens={total_tokens}",
        f"--out_dir={out_dir}",
        "--tokenization=gpt2",
        f"--tokens_per_shard={tokens_per_shard}",
        f"--val_fraction={val_fraction}",
        "--repeat",  # Repeat smaller sources to meet weight targets
    ]

    print()
    print_subheader("Launching prepare_weighted_dataset.py")

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start
        return result.returncode == 0, elapsed
    except subprocess.CalledProcessError as e:
        print_status("ERROR", f"Tokenization failed: rc={e.returncode}")
        return False, time.time() - start


# ---------------------------------------------------------------------------
# Source classification (which cleaner to use)
# ---------------------------------------------------------------------------

CODE_SOURCES = {
    "codeparrot", "starcoderdata_python", "starcoderdata_javascript",
    "starcoderdata_java",
}

DIALOGUE_SOURCES = {
    "opensub_en", "reddit_threaded", "stackexchange_qa",
    "reddit_comments", "stackoverflow",  # Legacy keys (kept for backward compat)
    "github_readmes",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    overall_start = time.time()

    parser = argparse.ArgumentParser(
        description="Fetch and prepare Phase 1.5 induction strengthening dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--sources_file", type=str, default=DEFAULT_SOURCES_FILE,
                        help=f"Source definitions JSON (default: {DEFAULT_SOURCES_FILE})")
    parser.add_argument("--raw_dir", type=str, default="data/phase1_5_raw",
                        help="Directory for raw downloads (default: data/phase1_5_raw)")
    parser.add_argument("--clean_dir", type=str, default="data/phase1_5_clean",
                        help="Directory for cleaned text (default: data/phase1_5_clean)")
    parser.add_argument("--out_dir", type=str, default="data/phase1_5_induction_raw",
                        help="Output directory for tokenized shards")
    parser.add_argument("--total_tokens", type=int, default=3_500_000_000,
                        help="Total tokens to prepare (default: 3,500,000,000)")
    parser.add_argument("--tokens_per_shard", type=int, default=10_000_000,
                        help="Tokens per shard (default: 10,000,000)")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Validation fraction (default: 0.05)")

    # Control flags
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step")
    parser.add_argument("--skip_clean", action="store_true",
                        help="Skip cleaning step")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download, don't clean or tokenize")
    parser.add_argument("--clean_only", action="store_true",
                        help="Only clean (skip download and tokenization)")

    # Source selection
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Select specific sources (default: all)")

    # Weight overrides
    parser.add_argument("--weight", type=str, action="append", default=[],
                        help="Override weight: --weight source_key:0.3")

    args = parser.parse_args()

    # Banner
    print_banner("MyPT", "Phase 1.5 Induction Data Pipeline")

    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Sources file: {args.sources_file}")

    # Load sources
    try:
        sources, weights, metadata = load_sources(args.sources_file)
    except FileNotFoundError:
        print_status("ERROR", f"Sources file not found: {args.sources_file}")
        sys.exit(1)
    except Exception as e:
        print_status("ERROR", f"Failed to load sources: {e}")
        sys.exit(1)

    print(f"  Dataset: {metadata['name']}")
    print(f"  {metadata['description']}")

    # Filter sources if requested
    if args.sources:
        sources = {k: v for k, v in sources.items() if k in args.sources}
        weights = {k: v for k, v in weights.items() if k in sources}
        if not sources:
            print_status("ERROR", f"None of {args.sources} found in config")
            sys.exit(1)

    # Apply weight overrides
    for w_arg in args.weight:
        if ":" in w_arg:
            k, v = w_arg.split(":", 1)
            try:
                weights[k] = float(v)
            except ValueError:
                print_status("WARN", f"Invalid weight: {w_arg}")

    # Normalize weights to active sources
    active_weights = {k: weights.get(k, 0.1) for k in sources}
    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}

    # Print config summary
    print_header("Configuration")
    print(f"    Raw downloads:   {args.raw_dir}")
    print(f"    Cleaned data:    {args.clean_dir}")
    print(f"    Tokenized out:   {args.out_dir}")
    print(f"    Total tokens:    {args.total_tokens:,}")
    print(f"    Sources:         {len(sources)}")
    print()

    for key, src in sources.items():
        w = active_weights.get(key, 0)
        est = src.get("est_tokens", 0)
        category = "CODE" if key in CODE_SOURCES else "DIALOGUE/STRUCT"
        print(f"    {key:<28} {w*100:5.1f}%  ~{est/1e9:.1f}B tok  [{category}]")

    print()

    # -----------------------------------------------------------------------
    # Step 1: Download
    # -----------------------------------------------------------------------
    download_time = 0.0
    if not args.skip_download and not args.clean_only:
        print_header("STEP 1: Downloading Sources", char="*")

        dl_start = time.time()
        for i, (key, cfg) in enumerate(sources.items(), 1):
            print_subheader(f"[{i}/{len(sources)}] {cfg.get('name', key)}")
            src_type = cfg.get("type", "huggingface")

            if src_type == "huggingface":
                ok, _, elapsed = download_hf_dataset(key, cfg, args.raw_dir)
            elif src_type == "opus":
                ok, _, elapsed = download_opus_file(key, cfg, args.raw_dir)
            else:
                print_status("WARN", f"Unknown source type '{src_type}' for {key}")
                continue

            if not ok:
                print_status("WARN", f"Failed to download {key}, continuing...")

        download_time = time.time() - dl_start
        print(f"\n  Download complete: {format_duration(download_time)}")
    else:
        print_header("STEP 1: Download")
        print_status("SKIP", "Download skipped")

    if args.download_only:
        print(f"\n  Raw data in: {args.raw_dir}")
        print(f"  Total time: {format_duration(time.time() - overall_start)}")
        return

    # -----------------------------------------------------------------------
    # Step 2: Clean
    # -----------------------------------------------------------------------
    clean_time = 0.0
    if not args.skip_clean:
        print_header("STEP 2: Cleaning Sources", char="*")

        cl_start = time.time()
        for i, (key, cfg) in enumerate(sources.items(), 1):
            print_subheader(f"[{i}/{len(sources)}] Cleaning {key}")

            if key in CODE_SOURCES:
                ok, _, elapsed = clean_code_source(key, cfg, args.raw_dir, args.clean_dir)
            else:
                ok, _, elapsed = clean_dialogue_source(key, cfg, args.raw_dir, args.clean_dir)

            if not ok:
                print_status("WARN", f"Cleaning failed for {key}, continuing...")

        clean_time = time.time() - cl_start
        print(f"\n  Cleaning complete: {format_duration(clean_time)}")
    else:
        print_header("STEP 2: Cleaning")
        print_status("SKIP", "Cleaning skipped")

    if args.clean_only:
        print(f"\n  Cleaned data in: {args.clean_dir}")
        print(f"  Total time: {format_duration(time.time() - overall_start)}")
        return

    # -----------------------------------------------------------------------
    # Step 3: Tokenize
    # -----------------------------------------------------------------------
    print_header("STEP 3: Tokenizing and Sharding", char="*")

    ok, tokenize_time = run_tokenization(
        clean_dir=args.clean_dir,
        out_dir=args.out_dir,
        weights=active_weights,
        total_tokens=args.total_tokens,
        tokens_per_shard=args.tokens_per_shard,
        val_fraction=args.val_fraction,
    )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_time = time.time() - overall_start

    if ok:
        print()
        print("=" * 70)
        print("  Phase 1.5 Data Pipeline: SUCCESS")
        print("=" * 70)
        print()
        print(f"    Output directory:  {args.out_dir}")
        print(f"    Target tokens:     {args.total_tokens:,}")
        print()
        print("    Timing:")
        print(f"      Download:   {format_duration(download_time)}")
        print(f"      Clean:      {format_duration(clean_time)}")
        print(f"      Tokenize:   {format_duration(tokenize_time)}")
        print(f"      TOTAL:      {format_duration(total_time)}")
        print()
        print("    Next steps:")
        print()
        print("    # Mix with replay data:")
        print(f"    python scripts/mix_tokenized_datasets.py \\")
        print(f"        --domain_dir {args.out_dir} \\")
        print(f"        --replay_dir data/multilingual_1.5B_wiki90 \\")
        print(f"        --replay2_dir data/domain_161M_corpus_tokenized \\")
        print(f"        --output_dir data/phase1_5_mixed \\")
        print(f"        --replay_ratio 0.10 \\")
        print(f"        --replay2_ratio 0.03")
        print()
        print("    # Train:")
        print(f"    python train.py \\")
        print(f"        --model_name domain_v6_induction \\")
        print(f"        --dataset_dir data/phase1_5_mixed \\")
        print(f"        --config_file configs/base/750M_phase1_5_induction.json \\")
        print(f"        --init_from_model domain_v5")
        print()
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("  Phase 1.5 Data Pipeline: FAILED")
        print("=" * 70)
        print(f"    Total time: {format_duration(total_time)}")
        print("    Check errors above. Partial data may be available.")
        print("    Re-run with --skip_download to resume from cleaning step.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
