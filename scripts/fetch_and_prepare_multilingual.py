#!/usr/bin/env python
"""
Fetch and Prepare Multilingual Dataset

This script automates the complete pipeline for creating a weighted multilingual
dataset for MyPT training:

1. Reads data source configurations from a JSON file (data/sources/*.json)

2. Downloads all specified data sources

3. Extracts and processes the downloaded files

4. Calls prepare_weighted_dataset.py to create sharded, weighted training data

Usage:
    python scripts/fetch_and_prepare_multilingual.py --sources_file data/sources/multilingual_de_en.json

Options:
    --sources_file      JSON file with data source definitions (required or use default)
    --data_dir          Directory for raw downloads (default: data/raw)
    --out_dir           Output directory for prepared dataset (default: data/multilingual_weighted)
    --total_tokens      Total tokens to prepare (default: 450000000)
    --tokenization      Tokenizer: gpt2 or char (default: gpt2)
    --tokens_per_shard  Tokens per shard file (default: 10000000)
    --val_fraction      Fraction for validation (default: 0.05)
    --skip_download     Skip download step (use existing files)
    --skip_extract      Skip extraction step
    --download_only     Only download, don't prepare dataset
    --list_sources      List available source files in data/sources/

    Weight overrides (optional, defaults from JSON file):
    --weight            Override weight for a source: --weight wiki_en:0.4 --weight wiki_de:0.3

Example:
    # Use default sources file
    python scripts/fetch_and_prepare_multilingual.py \\
        --data_dir data/raw \\
        --out_dir data/multilingual_450M \\
        --total_tokens 450000000

    # Use custom sources file
    python scripts/fetch_and_prepare_multilingual.py \\
        --sources_file data/sources/my_custom_sources.json \\
        --out_dir data/custom_dataset

    # List available source files
    python scripts/fetch_and_prepare_multilingual.py --list_sources
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
import gzip
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
import time
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def print_header(text: str, char: str = "=", width: int = 70):
    """Print a header with decorative characters."""
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_subheader(text: str, char: str = "-", width: int = 60):
    """Print a subheader."""
    print()
    print(f"  {char * 3} {text} {char * 3}")


def print_status(status: str, message: str, indent: int = 2):
    """Print a status message with indicator."""
    indicators = {
        "OK": "[OK]",
        "SKIP": "[SKIP]",
        "WARN": "[WARN]",
        "ERROR": "[ERROR]",
        "INFO": "[INFO]",
        "DONE": "[DONE]",
        "WAIT": "[....]",
        "DL": "[DOWN]",
        "EXTRACT": "[EXTR]",
    }
    ind = indicators.get(status, f"[{status}]")
    prefix = " " * indent
    print(f"{prefix}{ind} {message}")


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Data source configuration loading
# ---------------------------------------------------------------------------

DEFAULT_SOURCES_FILE = "data/sources/multilingual_de_en.json"
SOURCES_DIR = "data/sources"


def list_available_sources() -> List[str]:
    """List all available source JSON files in data/sources/."""
    if not os.path.exists(SOURCES_DIR):
        return []
    return sorted([f for f in os.listdir(SOURCES_DIR) if f.endswith('.json')])


def load_sources_from_json(json_path: str) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """
    Load data source configurations from a JSON file.
    
    Args:
        json_path: Path to the JSON configuration file
        
    Returns:
        Tuple of (sources_dict, default_weights, metadata)
        
    The JSON file should have this structure:
    {
        "name": "Dataset Name",
        "description": "Dataset description",
        "version": "1.0",
        "default_weights": {
            "source_key": 0.30,
            ...
        },
        "sources": {
            "source_key": {
                "name": "Human Readable Name",
                "subdir": "directory_name",
                "urls": [
                    ["filename", "url"],
                    ...
                ],
                "final_pattern": "*.txt",
                "type": "zip|gzip|split_zip",
                "est_size_mb": 1000,
                "description": "Source description"
            },
            ...
        }
    }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Sources file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate required fields
    if 'sources' not in data:
        raise ValueError(f"Sources file must contain 'sources' key: {json_path}")
    
    sources = data['sources']
    
    # Convert URL lists from [filename, url] to tuples for compatibility
    for source_key, source_config in sources.items():
        if 'urls' in source_config:
            source_config['urls'] = [tuple(u) for u in source_config['urls']]
    
    # Get default weights
    default_weights = data.get('default_weights', {})
    
    # Get metadata
    metadata = {
        'name': data.get('name', 'Unknown Dataset'),
        'description': data.get('description', ''),
        'version': data.get('version', '1.0'),
    }
    
    return sources, default_weights, metadata


def print_sources_info(sources: Dict[str, Any], weights: Dict[str, float], metadata: Dict[str, Any]):
    """Print information about loaded sources."""
    print(f"\n    Dataset: {metadata.get('name', 'Unknown')}")
    if metadata.get('description'):
        print(f"    Description: {metadata['description']}")
    print(f"    Version: {metadata.get('version', '1.0')}")
    print(f"    Sources loaded: {len(sources)}")
    print()
    
    total_est_size = sum(s.get("est_size_mb", 0) for s in sources.values())
    
    print(f"    {'Source':<15} {'Weight':<10} {'Est. Size':<12} {'Description'}")
    print(f"    {'-'*15} {'-'*10} {'-'*12} {'-'*35}")
    for key, src in sources.items():
        est = src.get("est_size_mb", 0)
        desc = src.get("description", "")[:35]
        weight = weights.get(key, 0)
        print(f"    {key:<15} {weight*100:>6.1f}%    {est:>8,} MB   {desc}")
    print(f"    {'-'*15} {'-'*10} {'-'*12}")
    print(f"    {'TOTAL':<15} {'100.0%':<10} {total_est_size:>8,} MB")


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def download_with_progress(url: str, dest_path: str, desc: str = "", file_num: int = 0, total_files: int = 0) -> Tuple[bool, float]:
    """Download a file with progress indication. Returns (success, elapsed_seconds)."""
    start_time = time.time()
    last_print_time = start_time
    
    def report_hook(block_num, block_size, total_size):
        nonlocal last_print_time
        downloaded = block_num * block_size
        elapsed = time.time() - start_time
        
        # Calculate speed
        if elapsed > 0:
            speed_mbps = (downloaded / (1024 * 1024)) / elapsed
        else:
            speed_mbps = 0
        
        # Progress string
        file_progress = f"[{file_num}/{total_files}]" if total_files > 0 else ""
        
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            
            # Estimate remaining time
            if percent > 0:
                eta_seconds = (elapsed / percent) * (100 - percent)
                eta_str = format_duration(eta_seconds)
            else:
                eta_str = "calculating..."
            
            bar_width = 20
            filled = int(bar_width * percent / 100)
            bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
            
            print(f"\r    {file_progress} [{bar}] {percent:3d}% | {mb_down:.1f}/{mb_total:.1f} MB | {speed_mbps:.1f} MB/s | ETA: {eta_str}    ", end="", flush=True)
        else:
            mb_down = downloaded / (1024 * 1024)
            print(f"\r    {file_progress} {mb_down:.1f} MB downloaded | {speed_mbps:.1f} MB/s    ", end="", flush=True)
    
    try:
        urlretrieve(url, dest_path, reporthook=report_hook)
        elapsed = time.time() - start_time
        print()  # newline after progress
        return True, elapsed
    except (URLError, HTTPError) as e:
        print(f"\n    [ERROR] Failed to download: {e}")
        return False, time.time() - start_time
    except Exception as e:
        print(f"\n    [ERROR] Unexpected error: {e}")
        return False, time.time() - start_time


def download_source(source_key: str, source_config: dict, data_dir: str, 
                   source_num: int = 0, total_sources: int = 0) -> Tuple[bool, str, float]:
    """Download all files for a data source. Returns (success, source_dir, elapsed_seconds)."""
    start_time = time.time()
    
    source_progress = f"[{source_num}/{total_sources}]" if total_sources > 0 else ""
    est_size = source_config.get("est_size_mb", 0)
    description = source_config.get("description", "")
    
    print_header(f"{source_progress} Downloading: {source_config['name']}")
    print(f"    Description: {description}")
    print(f"    Estimated size: ~{est_size:,} MB")
    print(f"    Files to download: {len(source_config['urls'])}")
    print(f"    Target directory: {os.path.join(data_dir, source_config['subdir'])}")
    print()
    
    source_dir = os.path.join(data_dir, source_config["subdir"])
    os.makedirs(source_dir, exist_ok=True)
    
    total_files = len(source_config["urls"])
    downloaded_count = 0
    skipped_count = 0
    total_bytes = 0
    
    for i, (filename, url) in enumerate(source_config["urls"], 1):
        dest_path = os.path.join(source_dir, filename)
        
        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            total_bytes += size
            skipped_count += 1
            print_status("SKIP", f"[{i}/{total_files}] {filename} already exists ({format_size(size)})")
            continue
        
        print(f"    [{i}/{total_files}] Downloading: {filename}")
        print(f"         URL: {url[:60]}...")
        
        success, elapsed = download_with_progress(url, dest_path, filename, i, total_files)
        if not success:
            print_status("ERROR", f"Failed to download {filename}")
            return False, source_dir, time.time() - start_time
        
        if os.path.exists(dest_path):
            size = os.path.getsize(dest_path)
            total_bytes += size
            downloaded_count += 1
            print_status("OK", f"Downloaded {format_size(size)} in {format_duration(elapsed)}")
        
        # Small delay between downloads to be polite
        time.sleep(0.5)
    
    elapsed = time.time() - start_time
    print()
    print(f"    {'-'*50}")
    print(f"    Source complete: {source_config['name']}")
    print(f"    Downloaded: {downloaded_count} files | Skipped: {skipped_count} files")
    print(f"    Total size: {format_size(total_bytes)}")
    print(f"    Time elapsed: {format_duration(elapsed)}")
    print_status("DONE", f"All files for {source_config['name']} ready")
    
    return True, source_dir, elapsed


# ---------------------------------------------------------------------------
# Extraction utilities
# ---------------------------------------------------------------------------

def combine_split_parts(source_dir: str, parts: List[str], combined_name: str) -> Tuple[bool, float]:
    """Combine split archive parts into single file. Returns (success, elapsed_seconds)."""
    start_time = time.time()
    combined_path = os.path.join(source_dir, combined_name)
    
    if os.path.exists(combined_path):
        size = os.path.getsize(combined_path)
        print_status("SKIP", f"{combined_name} already exists ({format_size(size)})", indent=6)
        return True, 0
    
    print(f"      Combining {len(parts)} parts into {combined_name}...")
    
    part_files = [os.path.join(source_dir, p) for p in parts]
    missing = [os.path.basename(p) for p in part_files if not os.path.exists(p)]
    if missing:
        print_status("ERROR", f"Missing parts: {missing}", indent=6)
        return False, time.time() - start_time
    
    # Calculate total size
    total_size = sum(os.path.getsize(p) for p in part_files)
    print(f"      Total parts size: {format_size(total_size)}")
    
    try:
        written = 0
        with open(combined_path, 'wb') as outfile:
            for i, part_path in enumerate(sorted(part_files), 1):
                part_size = os.path.getsize(part_path)
                print(f"        [{i}/{len(parts)}] Adding: {os.path.basename(part_path)} ({format_size(part_size)})")
                with open(part_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                written += part_size
                progress = (written / total_size) * 100
                print(f"              Progress: {progress:.0f}%")
        
        elapsed = time.time() - start_time
        print_status("OK", f"Combined to {combined_name} ({format_size(written)}) in {format_duration(elapsed)}", indent=6)
        return True, elapsed
    except Exception as e:
        print_status("ERROR", f"Failed to combine: {e}", indent=6)
        return False, time.time() - start_time


def extract_zip(zip_path: str, dest_dir: str) -> Tuple[bool, float]:
    """Extract a zip archive. Returns (success, elapsed_seconds)."""
    start_time = time.time()
    zip_name = os.path.basename(zip_path)
    
    print(f"      Extracting ZIP: {zip_name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            total_files = len(file_list)
            total_size = sum(info.file_size for info in zf.infolist())
            print(f"        Archive contains: {total_files} files, {format_size(total_size)} uncompressed")
            
            for i, member in enumerate(file_list, 1):
                zf.extract(member, dest_dir)
                if i % 100 == 0 or i == total_files:
                    progress = (i / total_files) * 100
                    print(f"\r        Extracting: {i}/{total_files} files ({progress:.0f}%)    ", end="", flush=True)
            print()  # newline
        
        elapsed = time.time() - start_time
        print_status("OK", f"Extracted {total_files} files in {format_duration(elapsed)}", indent=6)
        return True, elapsed
    except Exception as e:
        print_status("ERROR", f"Failed to extract: {e}", indent=6)
        return False, time.time() - start_time


def extract_gzip(gz_path: str) -> Tuple[bool, float]:
    """Extract a gzip file (removes .gz extension). Returns (success, elapsed_seconds)."""
    start_time = time.time()
    
    if not gz_path.endswith('.gz'):
        print_status("SKIP", f"{os.path.basename(gz_path)} is not a .gz file", indent=6)
        return True, 0
    
    out_path = gz_path[:-3]  # remove .gz
    out_name = os.path.basename(out_path)
    
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print_status("SKIP", f"{out_name} already exists ({format_size(size)})", indent=6)
        return True, 0
    
    gz_size = os.path.getsize(gz_path)
    print(f"      Extracting GZIP: {os.path.basename(gz_path)} ({format_size(gz_size)})...")
    
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                # Copy in chunks with progress
                chunk_size = 1024 * 1024  # 1 MB chunks
                written = 0
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    written += len(chunk)
                    print(f"\r        Decompressed: {format_size(written)}    ", end="", flush=True)
                print()  # newline
        
        elapsed = time.time() - start_time
        final_size = os.path.getsize(out_path)
        ratio = final_size / gz_size if gz_size > 0 else 1
        print_status("OK", f"Extracted to {out_name} ({format_size(final_size)}, {ratio:.1f}x expansion) in {format_duration(elapsed)}", indent=6)
        return True, elapsed
    except Exception as e:
        print_status("ERROR", f"Failed to extract: {e}", indent=6)
        return False, time.time() - start_time


def extract_source(source_key: str, source_config: dict, data_dir: str,
                  source_num: int = 0, total_sources: int = 0) -> Tuple[bool, float]:
    """Extract downloaded files for a source. Returns (success, elapsed_seconds)."""
    start_time = time.time()
    
    source_progress = f"[{source_num}/{total_sources}]" if total_sources > 0 else ""
    print_subheader(f"{source_progress} Extracting: {source_config['name']}")
    
    source_dir = os.path.join(data_dir, source_config["subdir"])
    source_type = source_config["type"]
    
    print(f"    Type: {source_type}")
    print(f"    Directory: {source_dir}")
    print()
    
    if source_type == "split_zip":
        # Combine split parts, then extract zip
        parts = [u[0] for u in source_config["urls"]]
        combined_name = source_config["combined_name"]
        
        print(f"    Step 1/2: Combining {len(parts)} split parts...")
        success, _ = combine_split_parts(source_dir, parts, combined_name)
        if not success:
            return False, time.time() - start_time
        
        print(f"    Step 2/2: Extracting combined archive...")
        combined_path = os.path.join(source_dir, combined_name)
        success, _ = extract_zip(combined_path, source_dir)
        if not success:
            return False, time.time() - start_time
            
    elif source_type == "zip":
        # Just extract zip
        for i, (filename, _) in enumerate(source_config["urls"], 1):
            zip_path = os.path.join(source_dir, filename)
            if zip_path.endswith('.zip'):
                print(f"    Step {i}/{len(source_config['urls'])}: Extracting {filename}...")
                success, _ = extract_zip(zip_path, source_dir)
                if not success:
                    return False, time.time() - start_time
                    
    elif source_type == "gzip":
        # Extract gzip files
        for i, (filename, _) in enumerate(source_config["urls"], 1):
            gz_path = os.path.join(source_dir, filename)
            if gz_path.endswith('.gz'):
                print(f"    Step {i}/{len(source_config['urls'])}: Extracting {filename}...")
                success, _ = extract_gzip(gz_path)
                if not success:
                    return False, time.time() - start_time
    
    elapsed = time.time() - start_time
    print_status("DONE", f"Extraction complete for {source_config['name']} ({format_duration(elapsed)})", indent=4)
    return True, elapsed


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def find_data_files(source_dir: str, pattern: str) -> List[str]:
    """Find all data files matching pattern in source directory."""
    import glob as glob_module
    search_pattern = os.path.join(source_dir, pattern)
    files = glob_module.glob(search_pattern)
    
    # Also check subdirectories
    search_pattern_deep = os.path.join(source_dir, "**", pattern)
    files.extend(glob_module.glob(search_pattern_deep, recursive=True))
    
    # Deduplicate and sort
    files = sorted(set(files))
    return files


def run_prepare_weighted_dataset(
    data_dir: str,
    out_dir: str,
    weights: dict,
    total_tokens: int,
    tokenization: str,
    tokens_per_shard: int,
    val_fraction: float,
    sources: Dict[str, Any],
    split_tsv: bool = True,
) -> Tuple[bool, float]:
    """Run prepare_weighted_dataset.py with the downloaded data. Returns (success, elapsed_seconds)."""
    start_time = time.time()
    
    print_header("Preparing Weighted Dataset")
    print(f"    Output directory: {out_dir}")
    print(f"    Target tokens: {total_tokens:,}")
    print(f"    Tokenization: {tokenization}")
    print(f"    Tokens per shard: {tokens_per_shard:,}")
    print(f"    Validation fraction: {val_fraction}")
    print()
    
    # Build source arguments
    print_subheader("Scanning for data files")
    source_args = []
    weight_args = []
    total_files_found = 0
    
    for source_key, source_config in sources.items():
        source_dir = os.path.join(data_dir, source_config["subdir"])
        pattern = source_config["final_pattern"]
        files = find_data_files(source_dir, pattern)
        
        if not files:
            print_status("WARN", f"No files found for {source_config['name']} ({pattern}) in {source_dir}")
            continue
        
        # Calculate total size of found files
        total_size = sum(os.path.getsize(f) for f in files)
        total_files_found += len(files)
        
        print(f"    {source_key}: {len(files)} files found ({format_size(total_size)})")
        for f in files[:2]:  # Show first 2
            fsize = os.path.getsize(f)
            print(f"      - {os.path.basename(f)} ({format_size(fsize)})")
        if len(files) > 2:
            print(f"      ... and {len(files) - 2} more files")
        
        # Create source argument with comma-separated paths
        files_str = ",".join(files)
        source_args.append(f"--source={source_key}:{files_str}")
        
        # Add weight
        if source_key in weights:
            weight_args.append(f"--weight={source_key}:{weights[source_key]}")
    
    if not source_args:
        print_status("ERROR", "No data sources found!")
        return False, time.time() - start_time
    
    print()
    print(f"    Total files found: {total_files_found}")
    print(f"    Sources with data: {len(source_args)}")
    
    # Build command
    script_path = os.path.join(os.path.dirname(__file__), "prepare_weighted_dataset.py")
    
    if not os.path.exists(script_path):
        print_status("ERROR", f"Could not find prepare_weighted_dataset.py at {script_path}")
        return False, time.time() - start_time
    
    cmd = [
        sys.executable,
        script_path,
    ]
    cmd.extend(source_args)
    cmd.extend(weight_args)
    cmd.extend([
        f"--total_tokens={total_tokens}",
        f"--out_dir={out_dir}",
        f"--tokenization={tokenization}",
        f"--tokens_per_shard={tokens_per_shard}",
        f"--val_fraction={val_fraction}",
    ])
    if split_tsv:
        cmd.append("--split_tsv")
    
    print()
    print_subheader("Launching prepare_weighted_dataset.py")
    print(f"    Script: {script_path}")
    print(f"    Started at: {get_timestamp()}")
    print()
    print("    " + "-" * 50)
    print("    NOTE: This process has TWO passes:")
    print("      Pass 1: Count tokens in all sources (may take a while)")
    print("      Pass 2: Sample and write shards")
    print("    " + "-" * 50)
    print()
    
    try:
        # Run with output visible in real-time
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_status("DONE", f"Dataset preparation completed in {format_duration(elapsed)}")
            return True, elapsed
        else:
            print_status("ERROR", f"prepare_weighted_dataset.py failed with return code {result.returncode}")
            return False, elapsed
            
    except subprocess.CalledProcessError as e:
        print_status("ERROR", f"prepare_weighted_dataset.py failed with return code {e.returncode}")
        return False, time.time() - start_time
    except FileNotFoundError:
        print_status("ERROR", f"Could not find Python or prepare_weighted_dataset.py")
        return False, time.time() - start_time
    except KeyboardInterrupt:
        print()
        print_status("WARN", "Process interrupted by user")
        return False, time.time() - start_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    overall_start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Fetch and prepare multilingual dataset for MyPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Sources file argument
    parser.add_argument("--sources_file", type=str, default=DEFAULT_SOURCES_FILE,
                        help=f"JSON file with data source definitions (default: {DEFAULT_SOURCES_FILE})")
    parser.add_argument("--list_sources", action="store_true",
                        help="List available source files in data/sources/ and exit")
    
    # Directory arguments
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory for raw downloads (default: data/raw)")
    parser.add_argument("--out_dir", type=str, default="data/multilingual_weighted",
                        help="Output directory for prepared dataset (default: data/multilingual_weighted)")
    
    # Dataset arguments
    parser.add_argument("--total_tokens", type=int, default=450_000_000,
                        help="Total tokens to prepare (default: 450000000)")
    parser.add_argument("--tokenization", type=str, default="gpt2",
                        choices=["gpt2", "char"],
                        help="Tokenizer type (default: gpt2)")
    parser.add_argument("--tokens_per_shard", type=int, default=10_000_000,
                        help="Tokens per shard file (default: 10000000)")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Fraction for validation (default: 0.05)")
    
    # Weight overrides (optional)
    parser.add_argument("--weight", type=str, action="append", default=[],
                        help="Override weight for a source: --weight source_key:0.4 (can be repeated)")
    
    # Control flags
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download step (use existing files)")
    parser.add_argument("--skip_extract", action="store_true",
                        help="Skip extraction step")
    parser.add_argument("--download_only", action="store_true",
                        help="Only download, don't prepare dataset")
    
    # Source selection (filter which sources to use from the JSON)
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Which sources to include (default: all from JSON file)")
    
    args = parser.parse_args()
    
    # Handle --list_sources
    if args.list_sources:
        print("\nAvailable source files in data/sources/:")
        print("-" * 50)
        available = list_available_sources()
        if not available:
            print("  No source files found.")
            print(f"  Create JSON files in {SOURCES_DIR}/")
        else:
            for f in available:
                json_path = os.path.join(SOURCES_DIR, f)
                try:
                    with open(json_path, 'r') as jf:
                        data = json.load(jf)
                    name = data.get('name', 'Unknown')
                    num_sources = len(data.get('sources', {}))
                    print(f"  {f:<35} - {name} ({num_sources} sources)")
                except Exception as e:
                    print(f"  {f:<35} - Error reading: {e}")
        print()
        return
    
    # Load sources from JSON file
    print(f"\n  Loading sources from: {args.sources_file}")
    try:
        SOURCES, default_weights, metadata = load_sources_from_json(args.sources_file)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print(f"\n  Available source files:")
        for f in list_available_sources():
            print(f"    - {os.path.join(SOURCES_DIR, f)}")
        print(f"\n  Use --sources_file to specify a different file")
        print(f"  Or --list_sources to see all available files")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR loading sources file: {e}")
        sys.exit(1)
    
    # Print banner with robot head
    from core.banner import print_banner
    print_banner("MyPT Dataset Fetcher", "Multilingual Data Preparation Pipeline")
    print()
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Sources file: {args.sources_file}")
    print()
    
    # Print configuration summary
    print_header("Configuration Summary")
    print(f"    Sources file:        {args.sources_file}")
    print(f"    Data directory:      {args.data_dir}")
    print(f"    Output directory:    {args.out_dir}")
    print(f"    Total tokens:        {args.total_tokens:,}")
    print(f"    Tokenization:        {args.tokenization}")
    print(f"    Tokens per shard:    {args.tokens_per_shard:,}")
    print(f"    Validation fraction: {args.val_fraction}")
    print()
    
    # Filter sources if --sources was provided
    if args.sources:
        active_sources = {k: v for k, v in SOURCES.items() if k in args.sources}
        if not active_sources:
            print(f"    ERROR: None of the specified sources {args.sources} found in {args.sources_file}")
            print(f"    Available sources: {list(SOURCES.keys())}")
            sys.exit(1)
    else:
        active_sources = SOURCES
    
    total_est_size = sum(s.get("est_size_mb", 0) for s in active_sources.values())
    
    print(f"    Selected sources ({len(active_sources)}):")
    print(f"    {'Source':<15} {'Est. Size':<12} {'Description'}")
    print(f"    {'-'*15} {'-'*12} {'-'*35}")
    for key, src in active_sources.items():
        est = src.get("est_size_mb", 0)
        desc = src.get("description", "")[:35]
        print(f"    {key:<15} {est:>8,} MB   {desc}")
    print(f"    {'-'*15} {'-'*12}")
    print(f"    {'TOTAL':<15} {total_est_size:>8,} MB")
    print()
    
    # Build weights dict from JSON defaults
    weights = {k: default_weights.get(k, 0) for k in active_sources.keys()}
    
    # Apply weight overrides from CLI
    for weight_arg in args.weight:
        if ':' not in weight_arg:
            print(f"    WARNING: Invalid weight format '{weight_arg}'. Use 'source_key:value'")
            continue
        key, value = weight_arg.split(':', 1)
        try:
            weights[key] = float(value)
            print(f"    Weight override: {key} = {float(value)}")
        except ValueError:
            print(f"    WARNING: Invalid weight value '{value}' for '{key}'")
    
    # Filter weights to active sources only
    weights = {k: v for k, v in weights.items() if k in active_sources}
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    print("    Normalized weights:")
    for name, w in weights.items():
        bar_width = int(w * 30)
        bar = "█" * bar_width + "░" * (30 - bar_width)
        print(f"      {name:<12} [{bar}] {w*100:5.1f}%")
    print()
    
    # Show pipeline overview
    print_header("Pipeline Overview")
    steps = []
    if not args.skip_download:
        steps.append(("1", "Download", f"Fetch {len(active_sources)} sources (~{total_est_size:,} MB)"))
    else:
        steps.append(("1", "Download", "SKIPPED (--skip_download)"))
    
    if not args.skip_extract:
        steps.append(("2", "Extract", "Decompress and extract archives"))
    else:
        steps.append(("2", "Extract", "SKIPPED (--skip_extract)"))
    
    if not args.download_only:
        steps.append(("3", "Prepare", f"Create {args.total_tokens:,} token dataset"))
    else:
        steps.append(("3", "Prepare", "SKIPPED (--download_only)"))
    
    for num, name, desc in steps:
        print(f"    Step {num}: {name:<10} - {desc}")
    print()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Track timing
    download_time = 0
    extract_time = 0
    prepare_time = 0
    
    # Step 1: Download
    if not args.skip_download:
        print_header("STEP 1/3: Downloading Data Sources", char="*")
        print(f"    Estimated total download: ~{total_est_size:,} MB")
        print(f"    This may take a while depending on your connection speed...")
        
        download_start = time.time()
        total_sources = len(active_sources)
        
        for i, (source_key, source_config) in enumerate(active_sources.items(), 1):
            success, _, elapsed = download_source(source_key, source_config, args.data_dir, i, total_sources)
            if not success:
                print_status("WARN", f"Failed to download {source_config['name']}, continuing with other sources...")
                # Continue with other sources
        
        download_time = time.time() - download_start
        print()
        print(f"    Step 1 complete. Total download time: {format_duration(download_time)}")
    else:
        print_header("STEP 1/3: Download", char="*")
        print_status("SKIP", "Download step skipped (--skip_download)")
    
    # Step 2: Extract
    if not args.skip_extract:
        print_header("STEP 2/3: Extracting Archives", char="*")
        
        extract_start = time.time()
        total_sources = len(active_sources)
        
        for i, (source_key, source_config) in enumerate(active_sources.items(), 1):
            success, elapsed = extract_source(source_key, source_config, args.data_dir, i, total_sources)
            if not success:
                print_status("WARN", f"Failed to extract {source_config['name']}, continuing with other sources...")
                # Continue with other sources
        
        extract_time = time.time() - extract_start
        print()
        print(f"    Step 2 complete. Total extraction time: {format_duration(extract_time)}")
    else:
        print_header("STEP 2/3: Extraction", char="*")
        print_status("SKIP", "Extraction step skipped (--skip_extract)")
    
    if args.download_only:
        print_header("Download Complete", char="*")
        print_status("INFO", "Skipping dataset preparation (--download_only)")
        print()
        print(f"    Downloaded data is in: {args.data_dir}")
        print(f"    To prepare dataset later, run:")
        print(f"      python {sys.argv[0]} --skip_download --out_dir {args.out_dir}")
        
        overall_time = time.time() - overall_start_time
        print()
        print(f"    Total time: {format_duration(overall_time)}")
        return
    
    # Step 3: Prepare weighted dataset
    print_header("STEP 3/3: Preparing Weighted Dataset", char="*")
    
    success, prepare_time = run_prepare_weighted_dataset(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        weights=weights,
        total_tokens=args.total_tokens,
        tokenization=args.tokenization,
        tokens_per_shard=args.tokens_per_shard,
        val_fraction=args.val_fraction,
        sources=active_sources,
        split_tsv=True,  # Always split TSV for parallel corpora
    )
    
    # Final summary
    overall_time = time.time() - overall_start_time
    
    if success:
        print()
        print("=" * 70)
        print("  ╔═══════════════════════════════════════════════════════════════╗")
        print("  ║                    SUCCESS! Dataset Ready.                    ║")
        print("  ╚═══════════════════════════════════════════════════════════════╝")
        print("=" * 70)
        print()
        print_header("Summary")
        print(f"    Output directory: {args.out_dir}")
        print()
        print("    Timing breakdown:")
        print(f"      Download:    {format_duration(download_time)}")
        print(f"      Extraction:  {format_duration(extract_time)}")
        print(f"      Preparation: {format_duration(prepare_time)}")
        print(f"      {'─' * 30}")
        print(f"      TOTAL:       {format_duration(overall_time)}")
        print()
        print("    Next steps - Train your model:")
        print()
        print(f"      python train.py \\")
        print(f"          --dataset_dir {args.out_dir} \\")
        print(f"          --config_file configs/150M_1024.json \\")
        print(f"          --model_name multilingual_150M \\")
        print(f"          --max_iters 50000")
        print()
        print("=" * 70)
        print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    else:
        print()
        print("=" * 70)
        print("  ╔═══════════════════════════════════════════════════════════════╗")
        print("  ║                    ERROR: Preparation Failed                  ║")
        print("  ╚═══════════════════════════════════════════════════════════════╝")
        print("=" * 70)
        print()
        print(f"    Total time elapsed: {format_duration(overall_time)}")
        print()
        print("    Troubleshooting:")
        print("      1. Check the error messages above")
        print("      2. Ensure all downloads completed successfully")
        print("      3. Check disk space availability")
        print("      4. Try running with --skip_download if data is already downloaded")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()

