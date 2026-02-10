#!/usr/bin/env python
"""
Strip Metadata from Cleaned Text Shards

Removes document boundary markers and metadata headers that would leak
into training data. Also strips StarCoder metadata tags and cleans
Reddit/dialogue artifacts.

Issues fixed:
1. Document markers: ==== DOC START ====, ==== DOC END ====
2. Source headers: SOURCE: xxx | PATH/ID: doc_xxx
3. Separator lines: ==== (between header and content)
4. StarCoder tags: <reponame>..., <filename>..., <gh_stars>...
5. Reddit artifacts: [deleted]: prefix usernames, HTML entities like &#x200B;
6. Empty lines left behind after stripping

Usage:
    # Strip a single source
    python tools/strip_metadata.py \
        --input_dir data/phase1_5_clean/codeparrot \
        --output_dir data/unified_clean/codeparrot

    # Strip all sources at once
    python tools/strip_metadata.py --all \
        --clean_base data/phase1_5_clean \
        --output_base data/unified_clean

    # Dry-run (report only, no writing)
    python tools/strip_metadata.py --all \
        --clean_base data/phase1_5_clean \
        --output_base data/unified_clean \
        --dry_run
"""

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Patterns to strip
# ---------------------------------------------------------------------------

# Document boundary markers
DOC_START_PATTERN = re.compile(r"^==== DOC START ====\s*$", re.MULTILINE)
DOC_END_PATTERN = re.compile(r"^==== DOC END ====\s*$", re.MULTILINE)

# Source/metadata header: SOURCE: xxx | PATH: xxx or SOURCE: xxx | ID: xxx
SOURCE_HEADER_PATTERN = re.compile(
    r"^SOURCE:\s*\S+(?:\s*\|\s*(?:PATH|ID):\s*\S+)?\s*$", re.MULTILINE
)

# Separator line (==== alone on a line, immediately after header)
SEPARATOR_PATTERN = re.compile(r"^====\s*$", re.MULTILINE)

# StarCoder metadata tags (appear at start of documents)
REPONAME_PATTERN = re.compile(r"<reponame>[^\n]*\n?")
FILENAME_PATTERN = re.compile(r"<filename>[^\n]*\n?")
GH_STARS_PATTERN = re.compile(r"<gh_stars>[^\n]*\n?")

# Reddit artifacts
DELETED_USER_PATTERN = re.compile(r"^\[deleted\]:\s*", re.MULTILINE)
REMOVED_CONTENT_PATTERN = re.compile(r"^\[removed\]\s*$", re.MULTILINE)

# HTML entities that shouldn't be in clean text
HTML_ENTITY_PATTERN = re.compile(r"&(?:#x?[0-9a-fA-F]+|[a-zA-Z]+);")

# Multiple blank lines (collapse to max 2)
MULTI_BLANK_PATTERN = re.compile(r"\n{4,}")

# Document delimiter (for splitting documents in a shard)
DOC_DELIMITER = "\n\n"  # Documents separated by double newline in output


def strip_document_markers(text: str) -> str:
    """Remove document boundary markers and metadata headers."""
    text = DOC_START_PATTERN.sub("", text)
    text = DOC_END_PATTERN.sub("", text)
    text = SOURCE_HEADER_PATTERN.sub("", text)
    text = SEPARATOR_PATTERN.sub("", text)
    return text


def strip_starcoder_tags(text: str) -> str:
    """Remove StarCoder metadata tags from code."""
    text = REPONAME_PATTERN.sub("", text)
    text = FILENAME_PATTERN.sub("", text)
    text = GH_STARS_PATTERN.sub("", text)
    return text


def clean_reddit_artifacts(text: str) -> str:
    """Clean Reddit-specific artifacts."""
    # Replace [deleted]: with empty string (removes the username marker)
    text = DELETED_USER_PATTERN.sub("", text)
    # Remove [removed] content markers
    text = REMOVED_CONTENT_PATTERN.sub("", text)
    # Decode HTML entities
    text = html.unescape(text)
    # Remove zero-width spaces and similar
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    return text


def clean_document(content: str, source_type: str) -> str:
    """
    Clean a single document's content.
    
    Args:
        content: Raw document text (between DOC START and DOC END)
        source_type: One of 'code', 'dialogue', 'reddit', 'stackexchange', 'readme', 'opensub'
    """
    # Strip StarCoder tags for code and readme sources
    if source_type in ("code", "readme"):
        content = strip_starcoder_tags(content)
    
    # Clean Reddit artifacts
    if source_type == "reddit":
        content = clean_reddit_artifacts(content)
    
    # Decode any remaining HTML entities for all sources
    if source_type in ("stackexchange", "readme"):
        content = html.unescape(content)
    
    # Strip leading/trailing whitespace per line, remove fully empty documents
    lines = content.split("\n")
    # Remove leading empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    # Remove trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()
    
    return "\n".join(lines)


def process_shard(input_path: Path, output_path: Path, source_type: str,
                  dry_run: bool = False) -> Dict[str, int]:
    """
    Process a single shard file: strip metadata, clean content, write output.
    
    Returns stats dict.
    """
    stats = {
        "docs_in": 0,
        "docs_out": 0,
        "docs_empty": 0,
        "bytes_in": 0,
        "bytes_out": 0,
        "markers_stripped": 0,
        "tags_stripped": 0,
    }
    
    text = input_path.read_text(encoding="utf-8", errors="replace")
    stats["bytes_in"] = len(text.encode("utf-8"))
    
    # Split into documents using DOC START/END markers
    # Pattern: ==== DOC START ====\nSOURCE: ...\n====\n<content>\n==== DOC END ====
    doc_pattern = re.compile(
        r"==== DOC START ====\n"
        r"(?:SOURCE:[^\n]*\n)?"
        r"(?:====\n)?"
        r"(.*?)"
        r"(?:==== DOC END ====)",
        re.DOTALL
    )
    
    documents = doc_pattern.findall(text)
    stats["docs_in"] = len(documents)
    
    if not documents:
        # If no document markers found, treat entire file as one document
        # (this handles already-clean files)
        cleaned = strip_document_markers(text)
        cleaned = clean_document(cleaned, source_type)
        if cleaned.strip():
            documents = [cleaned]
            stats["docs_in"] = 1
        else:
            stats["docs_empty"] = 1
            return stats
    
    # Clean each document
    cleaned_docs = []
    for doc_content in documents:
        cleaned = clean_document(doc_content, source_type)
        if cleaned.strip() and len(cleaned.strip()) > 20:
            cleaned_docs.append(cleaned)
            stats["docs_out"] += 1
        else:
            stats["docs_empty"] += 1
    
    if not cleaned_docs:
        return stats
    
    # Join documents with double newline separator
    output_text = DOC_DELIMITER.join(cleaned_docs) + "\n"
    
    # Collapse excessive blank lines
    output_text = MULTI_BLANK_PATTERN.sub("\n\n\n", output_text)
    
    stats["bytes_out"] = len(output_text.encode("utf-8"))
    stats["markers_stripped"] = stats["docs_in"]  # Each doc had markers
    
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    
    return stats


def get_source_type(source_name: str) -> str:
    """Map source directory name to cleaning type."""
    mapping = {
        "codeparrot": "code",
        "starcoderdata_python": "code",
        "starcoderdata_javascript": "code",
        "starcoderdata_java": "code",
        "github_readmes": "readme",
        "reddit_threaded": "reddit",
        "stackexchange_qa": "stackexchange",
        "opensub_en": "opensub",
        "fineweb_edu": "generic",
        "pes2o": "generic",
    }
    return mapping.get(source_name, "generic")


def process_source(input_dir: Path, output_dir: Path, source_name: str,
                   dry_run: bool = False) -> Dict[str, int]:
    """Process all shards in a source directory."""
    source_type = get_source_type(source_name)
    
    # Find all text shards
    shards = sorted(input_dir.glob("shard_*.txt"))
    if not shards:
        print(f"    [WARN] No shards found in {input_dir}")
        return {"shards": 0}
    
    total_stats = {
        "shards": len(shards),
        "docs_in": 0,
        "docs_out": 0,
        "docs_empty": 0,
        "bytes_in": 0,
        "bytes_out": 0,
    }
    
    for shard in shards:
        output_path = output_dir / shard.name
        stats = process_shard(shard, output_path, source_type, dry_run)
        for key in ("docs_in", "docs_out", "docs_empty", "bytes_in", "bytes_out"):
            total_stats[key] += stats.get(key, 0)
    
    # Copy cleaning_metadata.json if it exists
    meta_src = input_dir / "cleaning_metadata.json"
    if meta_src.exists() and not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_dst = output_dir / "cleaning_metadata.json"
        import shutil
        shutil.copy2(meta_src, meta_dst)
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Strip metadata from cleaned text shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Single source mode
    parser.add_argument("--input_dir", type=str,
                        help="Input directory with text shards")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for cleaned shards")
    parser.add_argument("--source_name", type=str, default=None,
                        help="Source name (for cleaning type detection)")
    
    # Batch mode
    parser.add_argument("--all", action="store_true",
                        help="Process all sources in clean_base directory")
    parser.add_argument("--clean_base", type=str, default="data/phase1_5_clean",
                        help="Base directory containing source subdirectories")
    parser.add_argument("--output_base", type=str, default="data/unified_clean",
                        help="Base output directory")
    
    # Options
    parser.add_argument("--dry_run", action="store_true",
                        help="Report only, don't write files")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Specific sources to process (with --all)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Metadata Stripper")
    print("  Remove document markers and metadata from cleaned shards")
    print("=" * 70)
    
    if args.dry_run:
        print("\n  *** DRY RUN - no files will be written ***")
    
    if args.all:
        clean_base = Path(args.clean_base)
        output_base = Path(args.output_base)
        
        if not clean_base.exists():
            print(f"\n  ERROR: Clean base directory not found: {clean_base}")
            sys.exit(1)
        
        # Find all source directories
        source_dirs = sorted([d for d in clean_base.iterdir() if d.is_dir()])
        
        if args.sources:
            source_dirs = [d for d in source_dirs if d.name in args.sources]
        
        print(f"\n  Clean base:  {clean_base}")
        print(f"  Output base: {output_base}")
        print(f"  Sources:     {len(source_dirs)}")
        
        all_results = {}
        
        for source_dir in source_dirs:
            source_name = source_dir.name
            output_dir = output_base / source_name
            source_type = get_source_type(source_name)
            
            print(f"\n  Processing: {source_name} (type: {source_type})")
            print(f"    Input:  {source_dir}")
            print(f"    Output: {output_dir}")
            
            stats = process_source(source_dir, output_dir, source_name, args.dry_run)
            all_results[source_name] = stats
            
            if stats.get("shards", 0) > 0:
                reduction = 0
                if stats["bytes_in"] > 0:
                    reduction = (1 - stats["bytes_out"] / stats["bytes_in"]) * 100
                print(f"    Shards: {stats['shards']}")
                print(f"    Docs:   {stats['docs_in']} -> {stats['docs_out']} "
                      f"({stats['docs_empty']} empty removed)")
                print(f"    Size:   {stats['bytes_in'] / (1024*1024):.1f} MB -> "
                      f"{stats['bytes_out'] / (1024*1024):.1f} MB "
                      f"({reduction:.1f}% reduction)")
        
        # Summary
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        total_in = sum(r.get("bytes_in", 0) for r in all_results.values())
        total_out = sum(r.get("bytes_out", 0) for r in all_results.values())
        total_docs_in = sum(r.get("docs_in", 0) for r in all_results.values())
        total_docs_out = sum(r.get("docs_out", 0) for r in all_results.values())
        total_empty = sum(r.get("docs_empty", 0) for r in all_results.values())
        
        print(f"\n  Sources processed: {len(all_results)}")
        print(f"  Total docs:        {total_docs_in:,} -> {total_docs_out:,}")
        print(f"  Empty removed:     {total_empty:,}")
        print(f"  Total size:        {total_in / (1024**3):.2f} GB -> "
              f"{total_out / (1024**3):.2f} GB")
        if total_in > 0:
            print(f"  Reduction:         {(1 - total_out / total_in) * 100:.1f}%")
        
        # Write summary metadata
        if not args.dry_run:
            output_base.mkdir(parents=True, exist_ok=True)
            summary = {
                "stripped_from": str(clean_base),
                "output_to": str(output_base),
                "sources": {name: stats for name, stats in all_results.items()},
                "total_docs_in": total_docs_in,
                "total_docs_out": total_docs_out,
                "total_bytes_in": total_in,
                "total_bytes_out": total_out,
            }
            summary_path = output_base / "strip_metadata_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n  Summary written to: {summary_path}")
    
    elif args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        source_name = args.source_name or input_dir.name
        source_type = get_source_type(source_name)
        
        print(f"\n  Source:      {source_name} (type: {source_type})")
        print(f"  Input:       {input_dir}")
        print(f"  Output:      {output_dir}")
        
        stats = process_source(input_dir, output_dir, source_name, args.dry_run)
        
        if stats.get("shards", 0) > 0:
            reduction = 0
            if stats["bytes_in"] > 0:
                reduction = (1 - stats["bytes_out"] / stats["bytes_in"]) * 100
            print(f"\n  Shards: {stats['shards']}")
            print(f"  Docs:   {stats['docs_in']} -> {stats['docs_out']} "
                  f"({stats['docs_empty']} empty removed)")
            print(f"  Size:   {stats['bytes_in'] / (1024*1024):.1f} MB -> "
                  f"{stats['bytes_out'] / (1024*1024):.1f} MB "
                  f"({reduction:.1f}% reduction)")
    else:
        parser.print_help()
        print("\n  ERROR: Provide --input_dir + --output_dir, or use --all")
        sys.exit(1)
    
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
