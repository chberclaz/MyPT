#!/usr/bin/env python
"""
Build Unified 6B Token From-Scratch Dataset (local tokenization)
=================================================================
Runs the full data preparation pipeline locally on Windows/Linux/Mac:
  1. Downloads FineWeb-Edu and peS2o (new sources)
  2. Holds back code shards for eval set (never seen during training)
  3. Tokenizes all 7 text sources individually
  4. Mixes all 9 sources into unified ~6B dataset

After completion, upload data/unified_6B/ (and data/code_eval_tokenized/)
to RunPod for training.

v3 mix (dual retrieval sources):
  - 27% FineWeb-Edu, 15% Wiki, 15% Python code, 8% JS/Java code
  - 13% StackExchange, 7% NQ/TriviaQA, 3% Reddit, 7% Domain, 3% peS2o, 2% README
  - OpenSubtitles REMOVED (quality audit)

Prerequisites:
    pip install datasets tiktoken numpy

Usage:
    python scripts/unified_build/build_unified_dataset.py                    # run all steps
    python scripts/unified_build/build_unified_dataset.py --step download    # download only
    python scripts/unified_build/build_unified_dataset.py --step holdback    # code eval holdback only
    python scripts/unified_build/build_unified_dataset.py --step tokenize    # tokenize only
    python scripts/unified_build/build_unified_dataset.py --step mix         # mix only
    python scripts/unified_build/build_unified_dataset.py --step tokenize --only code_python  # single source
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON = sys.executable  # use the same Python that's running this script

# Where stripped text sources live
CLEAN_DIR = PROJECT_ROOT / "data" / "unified_clean"

# Where individually tokenized sources go
TOKENIZED_DIR = PROJECT_ROOT / "data" / "unified_tokenized"

# Code eval holdback
CODE_EVAL_CLEAN = PROJECT_ROOT / "data" / "code_eval_clean"
CODE_EVAL_TOKENIZED = PROJECT_ROOT / "data" / "code_eval_tokenized"

# Final mixed output
MIX_CONFIG = PROJECT_ROOT / "data" / "sources" / "unified_from_scratch.json"
MIX_OUTPUT = PROJECT_ROOT / "data" / "unified_6B"

# Tokenizer script
TOKENIZE_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_weighted_dataset.py"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "unified_build" / "download_unified_sources.py"
DOWNLOAD_NQ_SCRIPT = PROJECT_ROOT / "scripts" / "unified_build" / "download_nq_triviaqa.py"
MIX_SCRIPT = PROJECT_ROOT / "scripts" / "unified_build" / "mix_multi_source.py"

# Source definitions: name -> tokenization config
# Each entry has: sources (list of name:glob pairs), weights, target_tokens, description
SOURCES = {
    "code_python": {
        "sources": [
            ("codeparrot", str(CLEAN_DIR / "codeparrot" / "shard_*.txt")),
            ("starcoder_py", str(CLEAN_DIR / "starcoderdata_python" / "shard_*.txt")),
        ],
        "weights": {"codeparrot": 0.65, "starcoder_py": 0.35},
        "target_tokens": 900_000_000,
        "description": "Code Python (15%) -- codeparrot + starcoder python",
    },
    "code_js_java": {
        "sources": [
            ("js", str(CLEAN_DIR / "starcoderdata_javascript" / "shard_*.txt")),
            ("java", str(CLEAN_DIR / "starcoderdata_java" / "shard_*.txt")),
        ],
        "weights": {"js": 0.5, "java": 0.5},
        "target_tokens": 480_000_000,
        "description": "Code JS+Java (8%)",
    },
    "stackexchange_qa": {
        "sources": [
            ("stackexchange", str(CLEAN_DIR / "stackexchange_qa" / "shard_*.txt")),
        ],
        "weights": None,
        "target_tokens": 780_000_000,
        "description": "StackExchange Q&A (13%) -- strongest retrieval source",
    },
    "nq_triviaqa": {
        "sources": [
            ("nq_triviaqa", str(CLEAN_DIR / "nq_triviaqa" / "shard_*.txt")),
        ],
        "weights": None,
        "target_tokens": 420_000_000,
        "description": "NQ + TriviaQA (7%) -- extractive retrieval, passage-grounded QA",
    },
    "reddit_threaded": {
        "sources": [
            ("reddit", str(CLEAN_DIR / "reddit_threaded" / "shard_*.txt")),
        ],
        "weights": None,
        "target_tokens": 180_000_000,
        "description": "Reddit threaded (3%) -- general language",
    },
    "github_readme": {
        "sources": [
            ("readme", str(CLEAN_DIR / "github_readmes" / "shard_*.txt")),
        ],
        "weights": None,
        "target_tokens": 120_000_000,
        "description": "GitHub README (2%) -- reduced after audit",
    },
    "fineweb_edu": {
        "sources": [
            ("fineweb", str(CLEAN_DIR / "fineweb_edu" / "shard_*.txt")),
        ],
        "weights": None,
        "target_tokens": 1_620_000_000,
        "description": "FineWeb-Edu (27%) -- educational web content",
    },
    "pes2o": {
        "sources": [
            ("pes2o", str(CLEAN_DIR / "pes2o" / "shard_*.txt")),
        ],
        "weights": None,
        "target_tokens": 180_000_000,
        "description": "peS2o scientific (3%)",
    },
}


def banner(text: str):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run(cmd: list, desc: str = ""):
    """Run a subprocess, streaming output in real-time."""
    if desc:
        print(f"\n  >> {desc}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [ERROR] Command failed with exit code {result.returncode}")
        print(f"  Elapsed: {elapsed:.0f}s")
        sys.exit(1)

    print(f"\n  Completed in {elapsed:.0f}s")
    return result


# ---------------------------------------------------------------------------
# Step 1: Download new sources
# ---------------------------------------------------------------------------

def step_download():
    banner("STEP 1: Download New Sources (FineWeb-Edu + peS2o + NQ/TriviaQA)")

    # --- FineWeb-Edu + peS2o ---
    fineweb_dir = CLEAN_DIR / "fineweb_edu"
    pes2o_dir = CLEAN_DIR / "pes2o"

    existing = []
    if fineweb_dir.exists() and list(fineweb_dir.glob("shard_*.txt")):
        n = len(list(fineweb_dir.glob("shard_*.txt")))
        print(f"  FineWeb-Edu already exists ({n} shards) -- skipping")
        existing.append("fineweb_edu")

    if pes2o_dir.exists() and list(pes2o_dir.glob("shard_*.txt")):
        n = len(list(pes2o_dir.glob("shard_*.txt")))
        print(f"  peS2o already exists ({n} shards) -- skipping")
        existing.append("pes2o")

    to_download = [s for s in ["fineweb_edu", "pes2o"] if s not in existing]

    if to_download:
        cmd = [
            PYTHON, str(DOWNLOAD_SCRIPT),
            "--output_dir", str(CLEAN_DIR),
            "--fineweb_tokens", "1620000000",
            "--pes2o_tokens", "300000000",
            "--sources",
        ] + to_download
        run(cmd, f"Downloading: {', '.join(to_download)}")
    else:
        print("  FineWeb-Edu + peS2o already downloaded.")

    # --- NQ + TriviaQA ---
    nq_dir = CLEAN_DIR / "nq_triviaqa"
    if nq_dir.exists() and list(nq_dir.glob("shard_*.txt")):
        n = len(list(nq_dir.glob("shard_*.txt")))
        print(f"  NQ/TriviaQA already exists ({n} shards) -- skipping")
    else:
        cmd = [
            PYTHON, str(DOWNLOAD_NQ_SCRIPT),
            "--output_dir", str(CLEAN_DIR),
            "--target_tokens", "420000000",
        ]
        run(cmd, "Downloading: NQ + TriviaQA")


# ---------------------------------------------------------------------------
# Step 2: Code eval holdback
# ---------------------------------------------------------------------------

def step_holdback():
    banner("STEP 2: Hold Back Code Shards for Eval Set")

    if CODE_EVAL_CLEAN.exists() and list(CODE_EVAL_CLEAN.glob("*.txt")):
        n = len(list(CODE_EVAL_CLEAN.glob("*.txt")))
        print(f"  Code eval clean dir already has {n} shards -- skipping holdback")
        print(f"  (Delete {CODE_EVAL_CLEAN} to re-run)")
        _tokenize_eval_holdback()
        return

    CODE_EVAL_CLEAN.mkdir(parents=True, exist_ok=True)

    # Hold back last 3 codeparrot shards
    cp_shards = sorted(glob.glob(str(CLEAN_DIR / "codeparrot" / "shard_*.txt")))
    cp_holdback = cp_shards[-3:] if len(cp_shards) >= 3 else cp_shards[-1:]
    for shard in cp_holdback:
        dest = CODE_EVAL_CLEAN / Path(shard).name
        print(f"    Moving: {shard} -> {dest}")
        shutil.move(shard, dest)

    # Hold back last 2 starcoder_python shards
    sp_shards = sorted(glob.glob(str(CLEAN_DIR / "starcoderdata_python" / "shard_*.txt")))
    sp_holdback = sp_shards[-2:] if len(sp_shards) >= 2 else sp_shards[-1:]
    for shard in sp_holdback:
        # Rename to avoid collision with codeparrot shard names
        dest = CODE_EVAL_CLEAN / f"starcoder_py_{Path(shard).name}"
        print(f"    Moving: {shard} -> {dest}")
        shutil.move(shard, dest)

    print(f"\n  Held back {len(cp_holdback)} codeparrot + {len(sp_holdback)} starcoder_python shards")
    print(f"  These will NOT be seen during training.")

    _tokenize_eval_holdback()


def _tokenize_eval_holdback():
    """Tokenize the held-back code eval shards."""
    # Check if already tokenized
    val_dir = CODE_EVAL_TOKENIZED / "val"
    if val_dir.exists() and list(val_dir.glob("*.bin")):
        n = len(list(val_dir.glob("*.bin")))
        print(f"  Code eval already tokenized ({n} val shards) -- skipping")
        return

    print()
    print("  Tokenizing code eval holdback set...")

    cmd = [
        PYTHON, str(TOKENIZE_SCRIPT),
        "--source", f"eval_code:{CODE_EVAL_CLEAN / 'shard_*.txt'}",
        "--source", f"eval_code_sc:{CODE_EVAL_CLEAN / 'starcoder_py_shard_*.txt'}",
        "--weight", "eval_code:0.6",
        "--weight", "eval_code_sc:0.4",
        "--total_tokens", "50000000",
        "--out_dir", str(CODE_EVAL_TOKENIZED),
        "--tokenization", "gpt2",
        "--tokens_per_shard", "10000000",
        "--val_fraction", "0.05",
        "--no_normalize", "--no_filter",
    ]
    run(cmd, "Tokenizing code eval holdback (~50M tokens)")

    # Move all train shards to val (eval-only loader reads from val/)
    train_dir = CODE_EVAL_TOKENIZED / "train"
    val_dir = CODE_EVAL_TOKENIZED / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    if train_dir.exists():
        for bin_file in sorted(train_dir.glob("*.bin")):
            dest = val_dir / bin_file.name
            if not dest.exists():
                shutil.move(str(bin_file), str(dest))
        # Clean up empty train dir
        remaining = list(train_dir.iterdir())
        if not any(f.suffix == ".bin" for f in remaining):
            shutil.rmtree(train_dir, ignore_errors=True)

    n_val = len(list(val_dir.glob("*.bin")))
    print(f"  Code eval set ready: {val_dir} ({n_val} shards)")


# ---------------------------------------------------------------------------
# Step 3: Tokenize all text sources
# ---------------------------------------------------------------------------

def step_tokenize(only: str = None):
    banner("STEP 3: Tokenize All Text Sources")

    sources_to_run = SOURCES
    if only:
        if only not in SOURCES:
            print(f"  [ERROR] Unknown source '{only}'. Available: {', '.join(SOURCES.keys())}")
            sys.exit(1)
        sources_to_run = {only: SOURCES[only]}
        print(f"  Running single source: {only}")

    total_start = time.time()

    for name, cfg in sources_to_run.items():
        out_dir = TOKENIZED_DIR / name

        # Check if already tokenized
        if out_dir.exists():
            train_dir = out_dir / "train"
            if train_dir.exists() and list(train_dir.glob("*.bin")):
                n = len(list(train_dir.glob("*.bin")))
                print(f"\n  {name}: already tokenized ({n} train shards) -- skipping")
                print(f"  (Delete {out_dir} to re-tokenize)")
                continue

        print(f"\n  --- {cfg['description']} ---")
        print(f"  Target: {cfg['target_tokens']:,} tokens")

        # Check source files exist
        total_files = 0
        for src_name, src_glob in cfg["sources"]:
            found = glob.glob(src_glob)
            total_files += len(found)
            if not found:
                print(f"  [WARN] No files found for {src_name}: {src_glob}")

        if total_files == 0:
            print(f"  [ERROR] No source files found for {name} -- skipping")
            continue

        # Build command
        cmd = [PYTHON, str(TOKENIZE_SCRIPT)]

        for src_name, src_glob in cfg["sources"]:
            cmd.extend(["--source", f"{src_name}:{src_glob}"])

        if cfg["weights"]:
            for wname, wval in cfg["weights"].items():
                cmd.extend(["--weight", f"{wname}:{wval}"])

        cmd.extend([
            "--total_tokens", str(cfg["target_tokens"]),
            "--out_dir", str(out_dir),
            "--tokenization", "gpt2",
            "--tokens_per_shard", "10000000",
            "--val_fraction", "0.05",
            "--no_normalize", "--no_filter",
        ])

        run(cmd, f"Tokenizing {name}")

    total_elapsed = time.time() - total_start
    print(f"\n  All tokenization complete in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    # Summary
    print("\n  Tokenized source summary:")
    for name in SOURCES:
        out_dir = TOKENIZED_DIR / name
        train_dir = out_dir / "train"
        val_dir = out_dir / "val"
        n_train = len(list(train_dir.glob("*.bin"))) if train_dir.exists() else 0
        n_val = len(list(val_dir.glob("*.bin"))) if val_dir.exists() else 0
        total_m = (n_train + n_val) * 10
        status = "OK" if n_train > 0 else "MISSING"
        print(f"    [{status}] {name}: {n_train} train + {n_val} val shards (~{total_m}M tokens)")


# ---------------------------------------------------------------------------
# Step 4: Mix all sources into unified dataset
# ---------------------------------------------------------------------------

def step_mix():
    banner("STEP 4: Mix All 9 Sources into Unified 6B Dataset")

    if not MIX_CONFIG.exists():
        print(f"  [ERROR] Mix config not found: {MIX_CONFIG}")
        sys.exit(1)

    # Verify all sources exist
    with open(MIX_CONFIG) as f:
        config = json.load(f)

    print("  Checking source availability:")
    all_ok = True
    for name, src in config["sources"].items():
        src_dir = PROJECT_ROOT / src["directory"]
        train_dir = src_dir / "train"
        val_dir = src_dir / "val"

        n_train = len(list(train_dir.glob("*.bin"))) if train_dir.exists() else 0
        n_val = len(list(val_dir.glob("*.bin"))) if val_dir.exists() else 0

        if n_train == 0 and n_val == 0:
            print(f"    [MISSING] {name}: {src_dir}")
            all_ok = False
        else:
            print(f"    [OK]      {name}: {n_train} train + {n_val} val shards")

    if not all_ok:
        print("\n  [ERROR] Some sources are missing. Run tokenize step first.")
        sys.exit(1)

    cmd = [
        PYTHON, str(MIX_SCRIPT),
        "--config", str(MIX_CONFIG),
        "--output_dir", str(MIX_OUTPUT),
        "--seed", "42",
    ]

    run(cmd, "Mixing all sources")

    # Final summary
    train_dir = MIX_OUTPUT / "train"
    val_dir = MIX_OUTPUT / "val"
    n_train = len(list(train_dir.glob("*.bin"))) if train_dir.exists() else 0
    n_val = len(list(val_dir.glob("*.bin"))) if val_dir.exists() else 0
    total_b = (n_train + n_val) * 10 / 1000

    print(f"\n  Dataset ready at: {MIX_OUTPUT}")
    print(f"  Train: {n_train} shards")
    print(f"  Val:   {n_val} shards")
    print(f"  Total: ~{total_b:.1f}B tokens")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = {
    "download": step_download,
    "holdback": step_holdback,
    "tokenize": step_tokenize,
    "mix": step_mix,
}

ALL_STEPS_ORDER = ["download", "holdback", "tokenize", "mix"]


def main():
    parser = argparse.ArgumentParser(
        description="Build unified 6B token from-scratch dataset (local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        default=None,
        help="Run a single step (default: run all steps in order)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="For tokenize step: run only this source (e.g. code_python, fineweb_edu)",
    )

    args = parser.parse_args()

    banner("Building Unified 6B Token From-Scratch Dataset")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python:       {PYTHON}")
    print(f"  Clean data:   {CLEAN_DIR}")
    print(f"  Output:       {MIX_OUTPUT}")

    if args.step:
        # Single step
        if args.step == "tokenize" and args.only:
            step_tokenize(only=args.only)
        else:
            STEPS[args.step]()
    else:
        # All steps
        for step_name in ALL_STEPS_ORDER:
            if step_name == "tokenize" and args.only:
                step_tokenize(only=args.only)
            else:
                STEPS[step_name]()

    banner("Pipeline Complete!")
    print(f"  Unified dataset: {MIX_OUTPUT}")
    print(f"  Code eval set:   {CODE_EVAL_TOKENIZED}")
    print()
    print("  Upload to RunPod:")
    print(f"    - {MIX_OUTPUT}/")
    print(f"    - {CODE_EVAL_TOKENIZED}/")
    print(f"    - data/multilingual_1.5B_wiki90/  (if not already on RunPod)")
    print(f"    - data/domain_161M_corpus_tokenized/  (if not already on RunPod)")
    print()


if __name__ == "__main__":
    main()
