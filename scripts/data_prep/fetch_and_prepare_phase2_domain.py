#!/usr/bin/env python
"""
Fetch and Prepare Phase 2 Domain Dataset

This script automates the complete pipeline for creating a domain-specific
dataset for MyPT Phase 2 training:

1. Clones source repositories (security docs, protocols, man pages, etc.)
2. Processes files through format-specific converters
3. Cleans, normalizes, and deduplicates content
4. Calls prepare_weighted_dataset.py to create tokenized training data

Usage:
    # Full pipeline
    python scripts/fetch_and_prepare_phase2_domain.py --out_dir data/phase2_domain

    # Skip fetching (use existing clones)
    python scripts/fetch_and_prepare_phase2_domain.py --skip_fetch --out_dir data/phase2_domain

    # Build corpus only (no tokenization)
    python scripts/fetch_and_prepare_phase2_domain.py --corpus_only --out_dir data/phase2_corpus

Options:
    --sources_dir       Directory for cloned repos (default: sources)
    --work_dir          Working directory (default: work)
    --out_dir           Output directory for final dataset (default: data/phase2_domain)
    --config_file       Optional JSON config file (default: data/sources/phase2_domain.json)
    --total_tokens      Target token count (default: 100000000)
    --skip_fetch        Skip repository cloning step
    --corpus_only       Build corpus only, don't tokenize
    --sources           Filter which sources to process
    --seed              Random seed for reproducibility (default: 42)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import banner
from core.banner import print_banner

# Audit logging for compliance
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase2_pipeline_{timestamp}.log"
    
    logger = logging.getLogger("phase2_pipeline")
    logger.setLevel(log_level)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('  [%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

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
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ---------------------------------------------------------------------------
# Repository cloning (Python-based, Windows compatible)
# ---------------------------------------------------------------------------

DEFAULT_REPOS = {
    "rfcxml": "https://github.com/ietf-tools/rfcxml.git",
    "owasp-top10": "https://github.com/OWASP/Top10.git",
    "owasp-cheatsheets": "https://github.com/OWASP/CheatSheetSeries.git",
    "owasp-wstg": "https://github.com/OWASP/wstg.git",
    "mitre-cti": "https://github.com/mitre/cti.git",
    "man-pages": "https://github.com/mkerrisk/man-pages.git",
    "bash-docs": "https://github.com/bminor/bash.git",
    "python-docs": "https://github.com/python/cpython.git",
    "python-peps": "https://github.com/python/peps.git",
    "nodejs-docs": "https://github.com/nodejs/node.git",
    "mdn-content": "https://github.com/mdn/content.git",
    "openjdk": "https://github.com/openjdk/jdk.git",
}


def clone_repository(
    name: str, 
    url: str, 
    target_dir: Path,
    logger: logging.Logger,
    depth: int = 1
) -> Tuple[bool, float, str]:
    """Clone a git repository (cross-platform)."""
    start_time = time.time()
    target_path = target_dir / name
    
    if target_path.exists():
        elapsed = time.time() - start_time
        logger.info(f"'{name}' already exists, skipping")
        return True, elapsed, "already_exists"
    
    logger.info(f"Cloning {name}...")
    logger.debug(f"  URL: {url}")
    
    try:
        cmd = ["git", "clone", "--depth", str(depth), "--quiet", url, str(target_path)]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            total_size = sum(
                f.stat().st_size for f in target_path.rglob('*') if f.is_file()
            )
            logger.info(f"Cloned {name} ({format_size(total_size)}) in {format_duration(elapsed)}")
            return True, elapsed, "cloned"
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            logger.error(f"Failed to clone {name}: {error_msg}")
            return False, elapsed, f"error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"Clone timeout for {name}")
        return False, elapsed, "timeout"
    except FileNotFoundError:
        elapsed = time.time() - start_time
        logger.error("Git not found - please install git and add to PATH")
        return False, elapsed, "git_not_found"
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Clone error for {name}: {e}")
        return False, elapsed, f"exception: {e}"


def fetch_sources(
    sources_dir: Path, 
    repos: Dict[str, str],
    logger: logging.Logger,
    filter_sources: Optional[List[str]] = None
) -> Tuple[bool, float, Dict[str, Tuple[bool, str]]]:
    """
    Clone all source repositories.
    
    Returns:
        (any_success, total_time, results_dict)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching Source Repositories")
    logger.info("=" * 60)
    
    sources_dir.mkdir(parents=True, exist_ok=True)
    
    if filter_sources:
        repos = {k: v for k, v in repos.items() if k in filter_sources}
    
    total = len(repos)
    results = {}
    total_time = 0
    success_count = 0
    
    logger.info(f"Cloning {total} repositories to {sources_dir}")
    
    for i, (name, url) in enumerate(repos.items(), 1):
        logger.info(f"[{i}/{total}] {name}")
        success, elapsed, msg = clone_repository(name, url, sources_dir, logger)
        results[name] = (success, msg)
        total_time += elapsed
        if success:
            success_count += 1
    
    logger.info(f"Fetch complete: {success_count}/{total} successful")
    logger.info(f"Total fetch time: {format_duration(total_time)}")
    
    return success_count > 0, total_time, results


# ---------------------------------------------------------------------------
# Corpus building (delegates to build_phase2_corpus.py)
# ---------------------------------------------------------------------------

def build_corpus(
    sources_dir: Path,
    work_dir: Path,
    out_dir: Path,
    logger: logging.Logger,
    config_file: Optional[str] = None,
    filter_sources: Optional[List[str]] = None,
    min_chars: int = 400,
    dedupe: str = "exact,simhash",
    seed: int = 42,
) -> Tuple[bool, float]:
    """Build the domain corpus."""
    logger.info("=" * 60)
    logger.info("STEP 2: Building Domain Corpus")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    build_script = Path(__file__).parent.parent / "tools" / "build_phase2_corpus.py"
    
    if not build_script.exists():
        logger.error(f"Build script not found: {build_script}")
        return False, 0
    
    corpus_dir = out_dir / "corpus"
    
    cmd = [
        sys.executable,
        str(build_script),
        f"--sources_dir={sources_dir}",
        f"--work_dir={work_dir}",
        f"--out_dir={corpus_dir}",
        f"--min_chars={min_chars}",
        f"--dedupe={dedupe}",
        f"--seed={seed}",
    ]
    
    if config_file and Path(config_file).exists():
        cmd.append(f"--config_file={config_file}")
    
    if filter_sources:
        cmd.extend(["--sources"] + filter_sources)
    
    logger.info(f"Running: {build_script.name}")
    logger.info(f"Output: {corpus_dir}")
    
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"Corpus built in {format_duration(elapsed)}")
            return True, elapsed
        else:
            logger.error(f"Build failed with code {result.returncode}")
            return False, elapsed
    except Exception as e:
        logger.error(f"Build error: {e}")
        return False, time.time() - start_time


# ---------------------------------------------------------------------------
# Tokenization (delegates to prepare_weighted_dataset.py)
# ---------------------------------------------------------------------------

def tokenize_corpus(
    corpus_dir: Path,
    out_dir: Path,
    total_tokens: int,
    logger: logging.Logger,
    tokenization: str = "gpt2",
    tokens_per_shard: int = 10_000_000,
    val_fraction: float = 0.05,
) -> Tuple[bool, float]:
    """Tokenize corpus."""
    logger.info("=" * 60)
    logger.info("STEP 3: Tokenizing Corpus")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    shard_dir = corpus_dir / "corpus_shards"
    if not shard_dir.exists():
        logger.error(f"Shard directory not found: {shard_dir}")
        return False, 0
    
    shards = sorted(shard_dir.glob("shard_*.txt"))
    if not shards:
        logger.error("No shard files found!")
        return False, 0
    
    total_size = sum(s.stat().st_size for s in shards)
    logger.info(f"Found {len(shards)} corpus shards ({format_size(total_size)})")
    
    prepare_script = Path(__file__).parent / "prepare_weighted_dataset.py"
    if not prepare_script.exists():
        logger.error(f"Tokenizer script not found: {prepare_script}")
        return False, 0
    
    shard_paths = ",".join(str(s) for s in shards)
    
    cmd = [
        sys.executable,
        str(prepare_script),
        f"--source=domain:{shard_paths}",
        "--weight=domain:1.0",
        f"--total_tokens={total_tokens}",
        f"--out_dir={out_dir}",
        f"--tokenization={tokenization}",
        f"--tokens_per_shard={tokens_per_shard}",
        f"--val_fraction={val_fraction}",
        "--no_normalize",
        "--no_filter",
    ]
    
    logger.info(f"Running: {prepare_script.name}")
    logger.info(f"Target: {total_tokens:,} tokens")
    logger.info(f"Output: {out_dir}")
    
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"Tokenization complete in {format_duration(elapsed)}")
            return True, elapsed
        else:
            logger.error(f"Tokenization failed with code {result.returncode}")
            return False, elapsed
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        return False, time.time() - start_time


# ---------------------------------------------------------------------------
# Pipeline metadata and audit
# ---------------------------------------------------------------------------

def write_pipeline_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    timing: Dict[str, float],
    fetch_results: Optional[Dict[str, Tuple[bool, str]]] = None,
) -> None:
    """Write pipeline execution metadata for reproducibility."""
    metadata = {
        'pipeline': 'phase2_domain',
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'parameters': {
            'sources_dir': str(args.sources_dir),
            'work_dir': str(args.work_dir),
            'out_dir': str(args.out_dir),
            'total_tokens': args.total_tokens,
            'tokenization': args.tokenization,
            'min_chars': args.min_chars,
            'dedupe': args.dedupe,
            'val_fraction': args.val_fraction,
            'skip_fetch': args.skip_fetch,
            'corpus_only': args.corpus_only,
            'sources_filter': args.sources,
        },
        'timing': timing,
        'fetch_results': fetch_results,
    }
    
    metadata_path = out_dir / 'pipeline_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch and prepare Phase 2 domain dataset for MyPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Directory arguments
    parser.add_argument("--sources_dir", type=str, default="sources",
                        help="Directory for cloned repos (default: sources)")
    parser.add_argument("--work_dir", type=str, default="work",
                        help="Working directory (default: work)")
    parser.add_argument("--out_dir", type=str, default="data/phase2_domain",
                        help="Output directory for final dataset (default: data/phase2_domain)")
    
    # Config
    parser.add_argument("--config_file", type=str, default="data/sources/phase2_domain.json",
                        help="JSON config file (default: data/sources/phase2_domain.json)")
    
    # Dataset parameters
    parser.add_argument("--total_tokens", type=int, default=100_000_000,
                        help="Target token count (default: 100000000)")
    parser.add_argument("--tokenization", type=str, default="gpt2",
                        choices=["gpt2", "char"],
                        help="Tokenization type (default: gpt2)")
    parser.add_argument("--tokens_per_shard", type=int, default=10_000_000,
                        help="Tokens per shard (default: 10000000)")
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Validation fraction (default: 0.05)")
    
    # Processing options
    parser.add_argument("--min_chars", type=int, default=400,
                        help="Minimum characters per document (default: 400)")
    parser.add_argument("--dedupe", type=str, default="exact,simhash",
                        help="Deduplication methods (default: exact,simhash)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    # Control flags
    parser.add_argument("--skip_fetch", action="store_true",
                        help="Skip repository cloning step")
    parser.add_argument("--skip_build", action="store_true",
                        help="Skip corpus building step")
    parser.add_argument("--corpus_only", action="store_true",
                        help="Build corpus only, don't tokenize")
    
    # Source filtering
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Filter which sources to process (default: all)")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner("MyPT Phase 2", "Domain Dataset Pipeline")
    
    overall_start = time.time()
    
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Setup logging
    work_dir = Path(args.work_dir)
    logger = setup_logging(work_dir / 'logs')
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info(f"Random seed: {args.seed}")
    
    # Audit: Pipeline started
    if AUDIT_AVAILABLE:
        audit.training(
            "phase2_pipeline_start",
            pipeline_type="domain_corpus",
            sources_dir=args.sources_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            total_tokens=args.total_tokens,
            details="Phase 2 domain dataset pipeline started"
        )
    
    # Show configuration
    print("=" * 60)
    print("  Configuration")
    print("=" * 60)
    print(f"  Sources directory:  {args.sources_dir}")
    print(f"  Work directory:     {args.work_dir}")
    print(f"  Output directory:   {args.out_dir}")
    print(f"  Config file:        {args.config_file}")
    print(f"  Total tokens:       {args.total_tokens:,}")
    print(f"  Tokenization:       {args.tokenization}")
    print(f"  Min chars:          {args.min_chars}")
    print(f"  Deduplication:      {args.dedupe}")
    print(f"  Random seed:        {args.seed}")
    print()
    
    logger.info("Configuration:")
    logger.info(f"  sources_dir: {args.sources_dir}")
    logger.info(f"  out_dir: {args.out_dir}")
    logger.info(f"  total_tokens: {args.total_tokens}")
    logger.info(f"  seed: {args.seed}")
    
    sources_dir = Path(args.sources_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Track timing
    timing = {'fetch': 0, 'build': 0, 'tokenize': 0}
    fetch_results = None
    
    # Step 1: Fetch sources
    if not args.skip_fetch:
        any_success, timing['fetch'], fetch_results = fetch_sources(
            sources_dir=sources_dir,
            repos=DEFAULT_REPOS,
            logger=logger,
            filter_sources=args.sources,
        )
        
        if AUDIT_AVAILABLE:
            success_count = sum(1 for s, _ in fetch_results.values() if s)
            audit.training(
                "phase2_fetch_complete",
                success_count=success_count,
                total_count=len(fetch_results),
            )
        
        if not any_success:
            logger.warning("No repositories were fetched successfully")
    else:
        print("\n" + "=" * 60)
        print("  STEP 1: Fetch Sources - SKIPPED")
        print("=" * 60)
        logger.info("Fetch step skipped (--skip_fetch)")
    
    # Step 2: Build corpus
    corpus_dir = out_dir / "corpus"
    
    if not args.skip_build:
        success, timing['build'] = build_corpus(
            sources_dir=sources_dir,
            work_dir=work_dir,
            out_dir=out_dir,
            logger=logger,
            config_file=args.config_file if Path(args.config_file).exists() else None,
            filter_sources=args.sources,
            min_chars=args.min_chars,
            dedupe=args.dedupe,
            seed=args.seed,
        )
        
        if not success:
            logger.error("Corpus building failed!")
            if AUDIT_AVAILABLE:
                audit.training(
                    "phase2_pipeline_error",
                    level=audit.AuditLevel.ERROR,
                    error="build_failed",
                    details="Corpus building failed"
                )
            sys.exit(1)
        
        if AUDIT_AVAILABLE:
            audit.training("phase2_build_complete", build_time=timing['build'])
    else:
        print("\n" + "=" * 60)
        print("  STEP 2: Build Corpus - SKIPPED")
        print("=" * 60)
        logger.info("Build step skipped (--skip_build)")
    
    # Step 3: Tokenize
    if not args.corpus_only:
        success, timing['tokenize'] = tokenize_corpus(
            corpus_dir=corpus_dir,
            out_dir=out_dir,
            total_tokens=args.total_tokens,
            logger=logger,
            tokenization=args.tokenization,
            tokens_per_shard=args.tokens_per_shard,
            val_fraction=args.val_fraction,
        )
        
        if not success:
            logger.error("Tokenization failed!")
            if AUDIT_AVAILABLE:
                audit.training(
                    "phase2_pipeline_error",
                    level=audit.AuditLevel.ERROR,
                    error="tokenize_failed",
                    details="Tokenization failed"
                )
            sys.exit(1)
        
        if AUDIT_AVAILABLE:
            audit.training("phase2_tokenize_complete", tokenize_time=timing['tokenize'])
    else:
        print("\n" + "=" * 60)
        print("  STEP 3: Tokenization - SKIPPED")
        print("=" * 60)
        logger.info("Tokenization skipped (--corpus_only)")
    
    # Write pipeline metadata
    timing['total'] = time.time() - overall_start
    write_pipeline_metadata(out_dir, args, timing, fetch_results)
    
    # Final audit
    if AUDIT_AVAILABLE:
        audit.training(
            "phase2_pipeline_complete",
            pipeline_type="domain_corpus",
            out_dir=str(out_dir),
            total_time=timing['total'],
            details="Phase 2 domain dataset pipeline completed successfully"
        )
    
    # Summary
    print()
    print("=" * 60)
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║              SUCCESS! Dataset Ready.                  ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print("=" * 60)
    print()
    print("  Timing breakdown:")
    print(f"    Fetch:        {format_duration(timing['fetch'])}")
    print(f"    Build:        {format_duration(timing['build'])}")
    print(f"    Tokenize:     {format_duration(timing['tokenize'])}")
    print(f"    {'─' * 25}")
    print(f"    TOTAL:        {format_duration(timing['total'])}")
    print()
    print(f"  Output directory: {args.out_dir}")
    print(f"  Pipeline metadata: {out_dir / 'pipeline_metadata.json'}")
    print()
    
    if not args.corpus_only:
        print("  Next steps - Train your model:")
        print()
        print(f"    python train.py \\")
        print(f"        --dataset_dir {args.out_dir} \\")
        print(f"        --config_file configs/750M_2048.json \\")
        print(f"        --model_name phase2_domain \\")
        print(f"        --max_iters 50000")
    else:
        print("  Corpus built. To tokenize:")
        print()
        print(f"    python scripts/prepare_weighted_dataset.py \\")
        print(f"        --source domain:{corpus_dir}/corpus_shards/*.txt \\")
        print(f"        --total_tokens {args.total_tokens} \\")
        print(f"        --out_dir {args.out_dir}")
    
    print()
    print("=" * 60)
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    print("  Note: You are responsible for verifying licenses of included sources.")
    print()
    
    logger.info(f"Pipeline completed at {datetime.now().isoformat()}")
    logger.info(f"Total time: {format_duration(timing['total'])}")


if __name__ == "__main__":
    main()
