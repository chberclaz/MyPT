#!/usr/bin/env python
"""
Convert OpenSubtitles Parallel Corpus to TSV

The OPUS OpenSubtitles Moses format contains parallel files:
- OpenSubtitles.de-en.de (German sentences, one per line)
- OpenSubtitles.de-en.en (English sentences, one per line)

This script:
1. Extracts the zip file (if not already extracted)
2. Merges the parallel files into a single TSV (tab-separated)
3. Output format: german_sentence<TAB>english_sentence

Usage:
    python scripts/convert_opensubtitles.py --input_dir data/raw/opensubtitles_de_en
    
    # Or with custom output
    python scripts/convert_opensubtitles.py \
        --input_dir data/raw/opensubtitles_de_en \
        --output_file data/raw/opensubtitles_de_en/opensubtitles_de_en.tsv
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_if_needed(input_dir: str) -> bool:
    """Extract the zip file if .de/.en files don't exist."""
    de_file = os.path.join(input_dir, "OpenSubtitles.de-en.de")
    en_file = os.path.join(input_dir, "OpenSubtitles.de-en.en")
    
    if os.path.exists(de_file) and os.path.exists(en_file):
        print(f"[OK] Parallel files already extracted")
        return True
    
    # Find the zip file
    zip_candidates = [
        os.path.join(input_dir, "de-en.txt.zip"),
        os.path.join(input_dir, "OpenSubtitles2018.de-en.txt.zip"),
    ]
    
    zip_path = None
    for candidate in zip_candidates:
        if os.path.exists(candidate):
            zip_path = candidate
            break
    
    if not zip_path:
        # Try any .zip file
        for f in os.listdir(input_dir):
            if f.endswith('.zip'):
                zip_path = os.path.join(input_dir, f)
                break
    
    if not zip_path:
        print(f"[ERROR] No zip file found in {input_dir}")
        return False
    
    print(f"[INFO] Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(input_dir)
        print(f"[OK] Extracted to {input_dir}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract: {e}")
        return False


def convert_to_tsv(input_dir: str, output_file: str, max_lines: int = None) -> int:
    """
    Merge parallel .de and .en files into a single TSV.
    
    Returns number of lines written.
    """
    de_file = os.path.join(input_dir, "OpenSubtitles.de-en.de")
    en_file = os.path.join(input_dir, "OpenSubtitles.de-en.en")
    
    if not os.path.exists(de_file):
        print(f"[ERROR] German file not found: {de_file}")
        return 0
    if not os.path.exists(en_file):
        print(f"[ERROR] English file not found: {en_file}")
        return 0
    
    # Get file sizes
    de_size = os.path.getsize(de_file) / (1024**3)
    en_size = os.path.getsize(en_file) / (1024**3)
    print(f"[INFO] German file: {de_size:.2f} GB")
    print(f"[INFO] English file: {en_size:.2f} GB")
    
    print(f"[INFO] Converting to TSV: {output_file}")
    
    lines_written = 0
    skipped_empty = 0
    skipped_mismatch = 0
    
    try:
        with open(de_file, 'r', encoding='utf-8', errors='ignore') as f_de, \
             open(en_file, 'r', encoding='utf-8', errors='ignore') as f_en, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for i, (de_line, en_line) in enumerate(zip(f_de, f_en)):
                if max_lines and i >= max_lines:
                    print(f"[INFO] Reached max_lines limit: {max_lines}")
                    break
                
                de_text = de_line.strip()
                en_text = en_line.strip()
                
                # Skip empty lines
                if not de_text or not en_text:
                    skipped_empty += 1
                    continue
                
                # Skip very short lines (likely noise)
                if len(de_text) < 3 or len(en_text) < 3:
                    skipped_empty += 1
                    continue
                
                # Write as TSV
                f_out.write(f"{de_text}\t{en_text}\n")
                lines_written += 1
                
                # Progress
                if lines_written % 1_000_000 == 0:
                    print(f"[INFO] Written {lines_written:,} lines...")
        
        print(f"[DONE] Wrote {lines_written:,} parallel sentence pairs")
        print(f"[INFO] Skipped {skipped_empty:,} empty/short lines")
        
        # Report output size
        out_size = os.path.getsize(output_file) / (1024**3)
        print(f"[INFO] Output file size: {out_size:.2f} GB")
        
        return lines_written
        
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenSubtitles parallel corpus to TSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/opensubtitles_de_en",
        help="Directory containing OpenSubtitles zip or extracted files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output TSV file path (default: input_dir/opensubtitles_de_en.tsv)"
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum lines to process (for testing)"
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip extraction step"
    )
    
    args = parser.parse_args()
    
    # Default output file
    if args.output_file is None:
        args.output_file = os.path.join(args.input_dir, "opensubtitles_de_en.tsv")
    
    print("=" * 60)
    print("  OpenSubtitles Parallel Corpus Converter")
    print("=" * 60)
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output file: {args.output_file}")
    print()
    
    # Step 1: Extract if needed
    if not args.skip_extract:
        if not extract_if_needed(args.input_dir):
            sys.exit(1)
    
    # Step 2: Convert to TSV
    lines = convert_to_tsv(args.input_dir, args.output_file, args.max_lines)
    
    if lines > 0:
        print()
        print("=" * 60)
        print("  SUCCESS!")
        print("=" * 60)
        print(f"  Output: {args.output_file}")
        print(f"  Lines: {lines:,}")
        print()
        print("  To use with prepare_weighted_dataset.py:")
        print(f"    --source opensub:{args.output_file}")
        print("    --split_tsv")
        print()
    else:
        print("[ERROR] No lines written. Check the input files.")
        sys.exit(1)


if __name__ == "__main__":
    main()

