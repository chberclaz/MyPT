#!/usr/bin/env python
"""
Clean Dialogue Corpus for Phase 1.5 Induction Training

Processes raw dialogue data (OpenSubtitles, Reddit comments) into clean training
text. Dialogue is valuable for induction heads because:
- Names and references are repeated across turns
- Quote-reply patterns force the model to track and copy references
- Conversational echoing (partial repetition) is natural

Supported formats:
1. OpenSubtitles (OPUS monolingual .txt / .txt.gz) - line-based dialogue
2. Reddit comments (HuggingFace JSONL) - threaded discussions with quotes
3. Generic JSONL with text field

Pipeline:
1. Parse raw text into dialogue turns
2. Filter: remove very short turns (<3 words), very long turns (>500 words)
3. Optional language detection (remove non-English)
4. Group into conversation windows
5. Add document boundary markers
6. Write to sharded text files

Usage:
    # Process OpenSubtitles monolingual dump
    python tools/clean_dialogue_corpus.py \\
        --input_file data/raw/opensub_en.txt \\
        --output_dir data/phase1_5_induction_raw/opensub_en \\
        --format opensub

    # Process Reddit JSONL
    python tools/clean_dialogue_corpus.py \\
        --input_jsonl data/raw/reddit_comments.jsonl \\
        --output_dir data/phase1_5_induction_raw/reddit \\
        --format reddit

    # Process generic JSONL with text field
    python tools/clean_dialogue_corpus.py \\
        --input_jsonl data/raw/stackoverflow.jsonl \\
        --output_dir data/phase1_5_induction_raw/stackoverflow \\
        --format generic \\
        --text_field text
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOC_START = "==== DOC START ===="
DOC_END = "==== DOC END ===="

# OpenSubtitles metadata patterns (timing lines, etc.)
SUBTITLE_META = re.compile(
    r"^\d+\s*$|"                         # Subtitle number
    r"^\d{2}:\d{2}:\d{2}|"              # Timestamp
    r"^<[^>]+>|"                          # HTML tags
    r"^\{\\an?\d\}|"                      # ASS/SSA formatting
    r"^♪|"                                # Music indicators  
    r"^\[.*\]\s*$",                       # Stage directions [laughs]
    re.MULTILINE
)

# URL pattern for removal
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Reddit quote pattern (lines starting with >)
REDDIT_QUOTE = re.compile(r"^>\s*", re.MULTILINE)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_opensub_lines(text: str, window_size: int = 20) -> Iterator[str]:
    """
    Parse OpenSubtitles monolingual text into conversation windows.
    
    The OPUS mono dump is one subtitle line per text line.
    We group consecutive lines into windows of `window_size` lines
    to create natural dialogue chunks.
    """
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        
        # Skip metadata lines
        if not line or SUBTITLE_META.match(line):
            continue
        
        # Clean up formatting
        line = re.sub(r"<[^>]+>", "", line)  # Remove HTML
        line = re.sub(r"\{\\[^}]+\}", "", line)  # Remove ASS tags
        line = line.strip("- ").strip()
        
        if not line or len(line.split()) < 2:
            continue
        
        lines.append(line)
        
        if len(lines) >= window_size:
            yield "\n".join(lines)
            # Overlap by half for continuity
            lines = lines[window_size // 2:]
    
    # Remaining lines
    if len(lines) >= 5:  # Minimum window
        yield "\n".join(lines)


def parse_reddit_record(record: dict, text_field: str = "content") -> Optional[str]:
    """
    Parse a Reddit comment/post record into training text.
    
    Preserves quote structure (>) as it shows explicit repetition patterns.
    """
    content = record.get(text_field, "")
    if not content:
        return None
    
    # Clean up
    content = URL_PATTERN.sub("[URL]", content)
    content = content.strip()
    
    # Preserve Reddit quotes (they show repetition!) but clean formatting
    lines = content.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines)
    
    # Check length
    word_count = len(text.split())
    if word_count < 10 or word_count > 2000:
        return None
    
    return text


def parse_stackoverflow_record(record: dict, text_field: str = "text") -> Optional[str]:
    """
    Parse a StackOverflow Q&A record into training text.
    
    SO is valuable because:
    - Questions contain code snippets
    - Answers quote the question and include modified versions of the same code
    - Error messages are repeated and referenced
    """
    content = record.get(text_field, "")
    if not content:
        return None
    
    # Basic HTML tag removal (SO uses some HTML)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"&[a-z]+;", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    
    word_count = len(content.split())
    if word_count < 15 or word_count > 3000:
        return None
    
    return content


def parse_reddit_threaded_record(record: dict, text_field: str = "text") -> Optional[str]:
    """
    Parse a Reddit threaded conversation record from REDDIT_threaded dataset.
    
    Each record contains a full multi-turn conversation thread with parent-child
    reply chains in the format:
        username1: message text
        username2: reply text
        username3: nested reply text
    
    This is the exact structure we need for training induction heads:
    - Usernames are repeated across turns (model must track who said what)
    - Replies reference and quote parent comments
    - Conversational echoing (partial repetition) is natural
    """
    content = record.get(text_field, "")
    if not content:
        return None
    
    # Clean URLs but preserve the conversation structure
    content = URL_PATTERN.sub("[URL]", content)
    content = content.strip()
    
    # Require minimum thread depth (at least 3 messages for meaningful threading)
    num_messages = record.get("num_messages", 0)
    if num_messages > 0 and num_messages < 3:
        return None
    
    # Check total length - need enough content for meaningful context
    word_count = len(content.split())
    if word_count < 30 or word_count > 8000:
        return None
    
    return content


def parse_stackexchange_qa_record(record: dict, text_field: str = "text") -> Optional[str]:
    """
    Parse a StackExchange Q&A record from common-pile/stackexchange.
    
    Each record contains a FULL Q&A thread: question text + all answer(s) +
    comments interleaved. This is extremely valuable because:
    - Answers directly reference and quote the question
    - Code snippets are repeated with modifications across Q/A
    - Error messages and technical terms are echoed across Q/A/comments
    - Multiple answers to the same question create natural repetition
    """
    content = record.get(text_field, "")
    if not content:
        return None
    
    content = content.strip()
    
    # Require substantial Q&A (not just a short question with no answers)
    word_count = len(content.split())
    if word_count < 50 or word_count > 15000:
        return None
    
    return content


# ---------------------------------------------------------------------------
# Shard Writer (same pattern as clean_code_corpus.py)
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes cleaned documents to sharded text files."""
    
    def __init__(self, output_dir: str, shard_mb: float = 25.0, source_name: str = "dialogue"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = int(shard_mb * 1024 * 1024)
        self.source_name = source_name
        
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
    
    def write_document(self, content: str, doc_id: str = ""):
        header = f"SOURCE: {self.source_name}"
        if doc_id:
            header += f" | ID: {doc_id}"
        doc = f"{DOC_START}\n{header}\n====\n{content}\n{DOC_END}\n\n"
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
    """Deduplicate by SHA-256 hash of content."""
    
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
# Processing Pipelines
# ---------------------------------------------------------------------------

def process_opensub(input_file: str, writer: ShardWriter, deduper: ExactDeduplicator,
                    window_size: int = 20, max_docs: int = 0,
                    report_every: int = 10000) -> dict:
    """Process OpenSubtitles monolingual dump."""
    stats = defaultdict(int)
    
    # Open file (handle .gz)
    if input_file.endswith(".gz"):
        f = gzip.open(input_file, "rt", encoding="utf-8", errors="replace")
    else:
        f = open(input_file, "r", encoding="utf-8", errors="replace")
    
    try:
        # Read in chunks to handle very large files
        chunk_size = 1024 * 1024  # 1 MB
        buffer = ""
        chunk_num = 0
        
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            
            buffer += data
            
            # Process complete windows from buffer
            # Split at double newlines to find natural boundaries
            while "\n" * 3 in buffer or len(buffer) > chunk_size * 2:
                # Find a good split point
                split_at = buffer.rfind("\n", 0, chunk_size)
                if split_at == -1:
                    split_at = len(buffer)
                
                chunk_text = buffer[:split_at]
                buffer = buffer[split_at:]
                
                for window_text in parse_opensub_lines(chunk_text, window_size):
                    stats["windows_seen"] += 1
                    
                    if max_docs > 0 and writer.total_docs >= max_docs:
                        return dict(stats)
                    
                    if deduper.is_duplicate(window_text):
                        stats["duplicates"] += 1
                        continue
                    
                    writer.write_document(window_text, f"window_{stats['windows_seen']}")
                    
                    if writer.total_docs % report_every == 0:
                        print(f"    Processed {writer.total_docs:,} windows, "
                              f"deduped {deduper.duplicates_found:,}")
            
            chunk_num += 1
        
        # Process remaining buffer
        if buffer.strip():
            for window_text in parse_opensub_lines(buffer, window_size):
                stats["windows_seen"] += 1
                if max_docs > 0 and writer.total_docs >= max_docs:
                    break
                if not deduper.is_duplicate(window_text):
                    writer.write_document(window_text, f"window_{stats['windows_seen']}")
    finally:
        f.close()
    
    return dict(stats)


def process_jsonl(input_jsonl: str, format_type: str, writer: ShardWriter,
                  deduper: ExactDeduplicator, text_field: str = "content",
                  max_docs: int = 0, report_every: int = 10000) -> dict:
    """Process JSONL files (Reddit, StackOverflow, generic)."""
    stats = defaultdict(int)
    
    parse_fn = {
        "reddit": lambda r: parse_reddit_record(r, text_field),
        "reddit_threaded": lambda r: parse_reddit_threaded_record(r, text_field),
        "stackoverflow": lambda r: parse_stackoverflow_record(r, text_field),
        "stackexchange": lambda r: parse_stackexchange_qa_record(r, text_field),
        "generic": lambda r: r.get(text_field, "").strip() or None,
    }.get(format_type, lambda r: r.get(text_field, "").strip() or None)
    
    # Handle .gz files
    if input_jsonl.endswith(".gz"):
        f = gzip.open(input_jsonl, "rt", encoding="utf-8", errors="replace")
    else:
        f = open(input_jsonl, "r", encoding="utf-8", errors="replace")
    
    try:
        for i, line in enumerate(f):
            stats["lines_seen"] += 1
            
            if max_docs > 0 and writer.total_docs >= max_docs:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                continue
            
            text = parse_fn(record)
            if not text:
                stats["filtered_empty_or_short"] += 1
                continue
            
            # Min word count
            if len(text.split()) < 5:
                stats["filtered_too_short"] += 1
                continue
            
            # Max length
            if len(text) > 50_000:
                stats["filtered_too_long"] += 1
                continue
            
            if deduper.is_duplicate(text):
                stats["duplicates"] += 1
                continue
            
            writer.write_document(text, f"doc_{i:08d}")
            
            if writer.total_docs % report_every == 0 and writer.total_docs > 0:
                print(f"    Processed {stats['lines_seen']:,} lines, "
                      f"kept {writer.total_docs:,}, "
                      f"deduped {deduper.duplicates_found:,}")
    finally:
        f.close()
    
    return dict(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clean dialogue corpus for Phase 1.5 induction training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=str,
                             help="Input text file (OpenSubtitles .txt or .txt.gz)")
    input_group.add_argument("--input_jsonl", type=str,
                             help="Input JSONL file (Reddit, SO, generic)")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cleaned shards")
    parser.add_argument("--format", type=str, required=True,
                        choices=["opensub", "reddit", "reddit_threaded", "stackoverflow", "stackexchange", "generic"],
                        help="Input format type")
    parser.add_argument("--text_field", type=str, default="content",
                        help="JSON field containing text (for JSONL input)")
    parser.add_argument("--source_name", type=str, default=None,
                        help="Source name for document markers (default: format)")
    parser.add_argument("--shard_mb", type=float, default=25.0,
                        help="Target shard size in MB (default: 25)")
    parser.add_argument("--window_size", type=int, default=20,
                        help="Dialogue lines per window (OpenSubtitles, default: 20)")
    parser.add_argument("--max_docs", type=int, default=0,
                        help="Max documents to produce (0 = unlimited)")
    parser.add_argument("--report_every", type=int, default=10000,
                        help="Print progress every N documents")
    
    args = parser.parse_args()
    
    source_name = args.source_name or args.format
    
    print("=" * 60)
    print("  Dialogue Corpus Cleaner")
    print("  Phase 1.5 Induction Training Data")
    print("=" * 60)
    print(f"\n  Format:      {args.format}")
    print(f"  Source name: {source_name}")
    print(f"  Shard size:  {args.shard_mb} MB")
    
    if args.input_file:
        print(f"  Input:       {args.input_file}")
    else:
        print(f"  Input:       {args.input_jsonl}")
    print(f"  Output:      {args.output_dir}")
    
    deduper = ExactDeduplicator()
    writer = ShardWriter(args.output_dir, shard_mb=args.shard_mb, source_name=source_name)
    
    print(f"\n  Processing...")
    
    if args.format == "opensub":
        if not args.input_file:
            print("  ERROR: OpenSubtitles format requires --input_file")
            sys.exit(1)
        stats = process_opensub(
            args.input_file, writer, deduper,
            window_size=args.window_size,
            max_docs=args.max_docs,
            report_every=args.report_every,
        )
    else:
        if not args.input_jsonl:
            print("  ERROR: JSONL formats require --input_jsonl")
            sys.exit(1)
        stats = process_jsonl(
            args.input_jsonl, args.format, writer, deduper,
            text_field=args.text_field,
            max_docs=args.max_docs,
            report_every=args.report_every,
        )
    
    writer.close()
    
    # Summary
    w_stats = writer.stats()
    print(f"\n" + "=" * 60)
    print(f"  SUMMARY")
    print(f"=" * 60)
    print(f"  Documents kept:  {w_stats['total_docs']:,}")
    print(f"  Duplicates:      {deduper.duplicates_found:,}")
    for key, value in sorted(stats.items()):
        if key not in ("duplicates",):
            print(f"  {key}: {value:,}")
    print(f"  Output shards:   {w_stats['total_shards']}")
    print(f"  Output size:     {w_stats['total_mb']:.1f} MB")
    print(f"  Output dir:      {args.output_dir}")
    print()
    
    # Write processing metadata
    meta = {
        "source_name": source_name,
        "format": args.format,
        "input": args.input_file or args.input_jsonl,
        "total_kept": w_stats["total_docs"],
        "duplicates": deduper.duplicates_found,
        "shards": w_stats["total_shards"],
        "total_mb": round(w_stats["total_mb"], 1),
        "processing_stats": stats,
    }
    meta_path = Path(args.output_dir) / "cleaning_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
