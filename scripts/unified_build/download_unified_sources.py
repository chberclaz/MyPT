#!/usr/bin/env python
"""
Download New Sources for From-Scratch Unified Training

Downloads and cleans FineWeb-Edu and peS2o datasets from HuggingFace.
These are the only two new sources needed -- all others already exist
in data/phase1_5_clean/ (after metadata stripping).

Sources:
1. FineWeb-Edu (HuggingFaceFW/fineweb-edu) - Educational web content
   Target: ~1.5B tokens (~10GB raw parquet)
   
2. peS2o (allenai/peS2o) - Scientific papers
   Target: ~0.3B tokens (~3GB raw)

Usage:
    # Download both sources
    python scripts/unified_build/download_unified_sources.py \
        --output_dir data/unified_clean \
        --fineweb_tokens 1500000000 \
        --pes2o_tokens 300000000

    # Download only FineWeb-Edu
    python scripts/unified_build/download_unified_sources.py \
        --output_dir data/unified_clean \
        --sources fineweb_edu

    # Download only peS2o
    python scripts/unified_build/download_unified_sources.py \
        --output_dir data/unified_clean \
        --sources pes2o
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate tokens per character (GPT-2 BPE average)
CHARS_PER_TOKEN = 4.0

DOC_DELIMITER = "\n\n"

SHARD_MB = 25.0


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_fineweb_text(text: str) -> Optional[str]:
    """
    Clean a FineWeb-Edu document.
    
    FineWeb-Edu is already heavily filtered (quality classifier scored),
    but we still want to:
    - Remove cookie notices, navigation boilerplate
    - Remove excessive whitespace
    - Filter very short or long documents
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Remove common web boilerplate patterns
    boilerplate_patterns = [
        re.compile(r"^Accept\s+(?:all\s+)?cookies?.*$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^We use cookies.*$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^Cookie\s+(?:policy|settings|preferences).*$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^Skip to (?:main )?content.*$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^(?:Share|Tweet|Pin|Email)\s*(?:on\s+\w+)?\s*$", re.MULTILINE | re.IGNORECASE),
        re.compile(r"^(?:Previous|Next)\s+(?:post|article).*$", re.MULTILINE | re.IGNORECASE),
    ]
    
    for pattern in boilerplate_patterns:
        text = pattern.sub("", text)
    
    # Collapse excessive whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = re.sub(r"[ \t]{4,}", "   ", text)
    text = text.strip()
    
    # Length filter
    word_count = len(text.split())
    if word_count < 30 or word_count > 10000:
        return None
    
    return text


def clean_pes2o_text(text: str) -> Optional[str]:
    """
    Clean a peS2o scientific paper.
    
    peS2o is already cleaned by Allen AI, but we want to:
    - Remove very short entries (abstracts only)
    - Remove reference sections that are just lists of citations
    - Keep structured content (sections, equations, citations)
    """
    if not text or not text.strip():
        return None
    
    text = text.strip()
    
    # Length filter - want substantial papers, not just abstracts
    word_count = len(text.split())
    if word_count < 100 or word_count > 30000:
        return None
    
    # Remove excessive reference lists at the end
    # (pattern: many lines starting with numbers or [N])
    lines = text.split("\n")
    ref_start = None
    for i in range(len(lines) - 1, max(len(lines) // 2, 0), -1):
        line = lines[i].strip()
        if re.match(r"^\[?\d+\]?\s*[A-Z]", line) or re.match(r"^\d+\.\s+[A-Z]", line):
            ref_start = i
        else:
            break
    
    if ref_start is not None and (len(lines) - ref_start) > 20:
        # Large reference section - truncate to keep just first 5 refs as examples
        lines = lines[:ref_start + 5]
        text = "\n".join(lines)
    
    # Collapse whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    
    # Re-check length after cleaning
    if len(text.split()) < 80:
        return None
    
    return text


# ---------------------------------------------------------------------------
# Shard Writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes cleaned documents to sharded text files (no metadata headers)."""
    
    def __init__(self, output_dir: str, shard_mb: float = 25.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = int(shard_mb * 1024 * 1024)
        
        self.shard_idx = 0
        self.current_bytes = 0
        self.current_file = None
        self.total_docs = 0
        self.total_bytes = 0
        self._open_new_shard()
    
    def _open_new_shard(self):
        if self.current_file:
            self.current_file.close()
        path = self.output_dir / f"shard_{self.shard_idx:04d}.txt"
        self.current_file = open(path, "w", encoding="utf-8")
        self.current_bytes = 0
    
    def write_document(self, content: str):
        """Write a clean document (NO metadata headers)."""
        doc = content + DOC_DELIMITER
        doc_bytes = len(doc.encode("utf-8"))
        
        if self.current_bytes + doc_bytes > self.shard_bytes and self.current_bytes > 0:
            self.shard_idx += 1
            self._open_new_shard()
        
        self.current_file.write(doc)
        self.current_bytes += doc_bytes
        self.total_bytes += doc_bytes
        self.total_docs += 1
    
    def close(self):
        if self.current_file:
            self.current_file.close()
            self.current_file = None
    
    def stats(self) -> dict:
        return {
            "total_docs": self.total_docs,
            "total_shards": self.shard_idx + 1,
            "total_bytes": self.total_bytes,
            "total_mb": self.total_bytes / (1024 * 1024),
        }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class ExactDeduplicator:
    """Deduplicate by SHA-256 hash."""
    
    def __init__(self):
        self.seen_hashes = set()
        self.duplicates_found = 0
    
    def is_duplicate(self, content: str) -> bool:
        h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        if h in self.seen_hashes:
            self.duplicates_found += 1
            return True
        self.seen_hashes.add(h)
        return False


# ---------------------------------------------------------------------------
# Download and Process
# ---------------------------------------------------------------------------

def download_fineweb_edu(output_dir: Path, target_tokens: int,
                         report_every: int = 10000) -> dict:
    """Download and clean FineWeb-Edu dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)
    
    print(f"\n  Loading FineWeb-Edu from HuggingFace (streaming)...")
    print(f"  Target: {target_tokens:,} tokens (~{target_tokens * CHARS_PER_TOKEN / 1e9:.1f} GB text)")
    
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # Use the 10B token sample (much faster than full)
        split="train",
        streaming=True,
    )
    
    writer = ShardWriter(str(output_dir / "fineweb_edu"), shard_mb=SHARD_MB)
    deduper = ExactDeduplicator()
    
    stats = {"seen": 0, "kept": 0, "filtered": 0, "duplicates": 0, "est_tokens": 0}
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    total_chars = 0
    
    start_time = time.time()
    
    for record in ds:
        stats["seen"] += 1
        
        text = record.get("text", "")
        cleaned = clean_fineweb_text(text)
        
        if not cleaned:
            stats["filtered"] += 1
            continue
        
        if deduper.is_duplicate(cleaned):
            stats["duplicates"] += 1
            continue
        
        writer.write_document(cleaned)
        total_chars += len(cleaned)
        stats["kept"] += 1
        
        if stats["kept"] % report_every == 0:
            elapsed = time.time() - start_time
            est_tokens = int(total_chars / CHARS_PER_TOKEN)
            pct = est_tokens / target_tokens * 100
            print(f"    {stats['kept']:,} docs, ~{est_tokens:,} tokens ({pct:.1f}%), "
                  f"{elapsed:.0f}s elapsed")
        
        if total_chars >= target_chars:
            print(f"    Reached target ({target_tokens:,} tokens)")
            break
    
    writer.close()
    
    stats["est_tokens"] = int(total_chars / CHARS_PER_TOKEN)
    stats["total_mb"] = writer.stats()["total_mb"]
    stats["total_shards"] = writer.stats()["total_shards"]
    
    elapsed = time.time() - start_time
    print(f"\n  FineWeb-Edu complete:")
    print(f"    Docs: {stats['seen']:,} seen, {stats['kept']:,} kept")
    print(f"    Tokens: ~{stats['est_tokens']:,}")
    print(f"    Size: {stats['total_mb']:.1f} MB ({stats['total_shards']} shards)")
    print(f"    Time: {elapsed:.0f}s")
    
    return stats


def download_pes2o(output_dir: Path, target_tokens: int,
                   report_every: int = 5000) -> dict:
    """Download and clean peS2o scientific papers."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)
    
    print(f"\n  Loading peS2o from HuggingFace (streaming)...")
    print(f"  Target: {target_tokens:,} tokens (~{target_tokens * CHARS_PER_TOKEN / 1e9:.1f} GB text)")
    
    # peS2o uses a custom loading script that newer datasets versions reject.
    # Bypass by loading the raw json.gz files directly with the "json" builder.
    ds = load_dataset(
        "json",
        data_files="hf://datasets/allenai/peS2o/data/v2/train-*.json.gz",
        split="train",
        streaming=True,
    )
    
    writer = ShardWriter(str(output_dir / "pes2o"), shard_mb=SHARD_MB)
    deduper = ExactDeduplicator()
    
    stats = {"seen": 0, "kept": 0, "filtered": 0, "duplicates": 0, "est_tokens": 0}
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    total_chars = 0
    
    start_time = time.time()
    
    for record in ds:
        stats["seen"] += 1
        
        text = record.get("text", "")
        cleaned = clean_pes2o_text(text)
        
        if not cleaned:
            stats["filtered"] += 1
            continue
        
        if deduper.is_duplicate(cleaned):
            stats["duplicates"] += 1
            continue
        
        writer.write_document(cleaned)
        total_chars += len(cleaned)
        stats["kept"] += 1
        
        if stats["kept"] % report_every == 0:
            elapsed = time.time() - start_time
            est_tokens = int(total_chars / CHARS_PER_TOKEN)
            pct = est_tokens / target_tokens * 100
            print(f"    {stats['kept']:,} docs, ~{est_tokens:,} tokens ({pct:.1f}%), "
                  f"{elapsed:.0f}s elapsed")
        
        if total_chars >= target_chars:
            print(f"    Reached target ({target_tokens:,} tokens)")
            break
    
    writer.close()
    
    stats["est_tokens"] = int(total_chars / CHARS_PER_TOKEN)
    stats["total_mb"] = writer.stats()["total_mb"]
    stats["total_shards"] = writer.stats()["total_shards"]
    
    elapsed = time.time() - start_time
    print(f"\n  peS2o complete:")
    print(f"    Docs: {stats['seen']:,} seen, {stats['kept']:,} kept")
    print(f"    Tokens: ~{stats['est_tokens']:,}")
    print(f"    Size: {stats['total_mb']:.1f} MB ({stats['total_shards']} shards)")
    print(f"    Time: {elapsed:.0f}s")
    
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download new sources for unified from-scratch training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--output_dir", type=str, default="data/unified_clean",
                        help="Output base directory")
    parser.add_argument("--sources", nargs="*", default=["fineweb_edu", "pes2o"],
                        choices=["fineweb_edu", "pes2o"],
                        help="Which sources to download (default: both)")
    parser.add_argument("--fineweb_tokens", type=int, default=1_500_000_000,
                        help="Target tokens for FineWeb-Edu (default: 1.5B)")
    parser.add_argument("--pes2o_tokens", type=int, default=300_000_000,
                        help="Target tokens for peS2o (default: 300M)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("  Unified Source Downloader")
    print("  From-Scratch Training Dataset Preparation")
    print("=" * 70)
    print(f"\n  Output: {output_dir}")
    print(f"  Sources: {', '.join(args.sources)}")
    
    results = {}
    
    if "fineweb_edu" in args.sources:
        results["fineweb_edu"] = download_fineweb_edu(
            output_dir, args.fineweb_tokens
        )
    
    if "pes2o" in args.sources:
        results["pes2o"] = download_pes2o(
            output_dir, args.pes2o_tokens
        )
    
    # Write download metadata
    meta = {
        "downloaded_at": datetime.now().isoformat(),
        "sources": results,
    }
    meta_path = output_dir / "download_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  Metadata: {meta_path}")
    print("\n  All downloads complete!\n")


if __name__ == "__main__":
    main()
