#!/usr/bin/env python3
"""
Phase 1 Format Lock: Complete Preparation Pipeline

Generates, mixes, and tokenizes all data needed for SFT Phase 1 (Format Lock).
Run this once after pre-training completes to prepare the Phase 1 dataset.

Pipeline:
    1. Generate format lock dataset (EN + DE, combinatorial templates)
    2. Generate echo dataset (diverse instruction following)
    3. Mix datasets (70% format lock + 30% echo, shuffled)
    4. Tokenize mixed dataset with loss masking + PACKING (via prepare_chat_sft.py)
       Packing is ON by default: Phase 1 episodes are ~30 tokens each, so packing
       fits many episodes per 4096-token block (major training efficiency gain).
    5. Print stats and validate output

Usage:
    python scripts/sft/prepare_phase1_format_lock.py
    python scripts/sft/prepare_phase1_format_lock.py --output_dir data/sft_phase1_v2
    python scripts/sft/prepare_phase1_format_lock.py --no_packing   # disable packing
    python scripts/sft/prepare_phase1_format_lock.py --dry_run

After running:
    python train.py \\
        --model_name phase1_format_lock \\
        --config_file configs/sft/phase1_format_lock.json \\
        --dataset_dir data/sft_phase1_format_lock \\
        --init_from_model checkpoints/GOLD_unified_v1
"""

import argparse
import subprocess
import sys
import json
import shutil
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _abs_path(p: Path) -> Path:
    """Resolve path relative to project root if not absolute."""
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _resolve_existing_path(candidates):
    """Return first existing path from candidate list, else None."""
    for p in candidates:
        if p.exists():
            return p
    return None


def run_cmd(cmd: list, description: str, dry_run: bool = False):
    """Run a command with description and error handling."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  CMD: {cmd_str}")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    project_root_str = str(PROJECT_ROOT)
    if existing_pythonpath:
        if project_root_str not in existing_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = project_root_str + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = project_root_str

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: Command failed with exit code {result.returncode}")
        return False
    return True


def count_jsonl(filepath: Path) -> tuple:
    """Count episodes and language distribution in a JSONL file."""
    total = 0
    en_count = 0
    de_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
                total += 1
                lang = ep.get("language", "en")
                if lang == "de":
                    de_count += 1
                else:
                    en_count += 1
            except json.JSONDecodeError:
                pass
    return total, en_count, de_count


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Format Lock: Complete preparation pipeline"
    )
    parser.add_argument("--output_dir", type=str,
                        default="data/sft_phase1_format_lock",
                        help="Final tokenized output directory")
    parser.add_argument("--intermediate_dir", type=str,
                        default="data/sft_phase1_intermediate",
                        help="Directory for intermediate JSONL files")
    parser.add_argument("--format_lock_mode", type=str, default="full",
                        choices=["full", "minimal"],
                        help="Format lock dataset mode (default: full)")
    parser.add_argument("--format_lock_math", type=str, default="include",
                        choices=["include", "exclude", "only", "minimal"],
                        help="Format lock math mode (default: include)")
    parser.add_argument("--echo_gibberish", type=str, default="exclude",
                        choices=["include", "exclude", "only"],
                        help="Echo gibberish mode (default: exclude)")
    parser.add_argument("--format_lock_ratio", type=float, default=0.7,
                        help="Fraction of format lock in mix (default: 0.7)")
    parser.add_argument("--echo_ratio", type=float, default=0.3,
                        help="Fraction of echo in mix (default: 0.3)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split (default: 0.05)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip_tokenize", action="store_true",
                        help="Stop after mixing (don't tokenize)")
    parser.add_argument("--no_packing", action="store_true",
                        help="Disable episode packing (packing is ON by default for Phase 1)")
    parser.add_argument("--pack_block_size", type=int, default=4096,
                        help="Block size for episode packing (default: 4096)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    intermediate = _abs_path(Path(args.intermediate_dir))
    output = _abs_path(Path(args.output_dir))
    py = sys.executable

    print("\n" + "="*70)
    print("  PHASE 1 FORMAT LOCK: PREPARATION PIPELINE")
    print("="*70)
    print(f"  Output dir:       {output}")
    print(f"  Intermediate dir: {intermediate}")
    print(f"  Format lock:      mode={args.format_lock_mode}, math={args.format_lock_math}")
    print(f"  Echo:             gibberish={args.echo_gibberish}")
    print(f"  Mix ratio:        {args.format_lock_ratio:.0%} format_lock + {args.echo_ratio:.0%} echo")
    print(f"  Val split:        {args.val_split:.0%}")

    # Step 1: Generate format lock dataset
    format_lock_dir = intermediate / "format_lock"
    format_lock_file = format_lock_dir / "mypt_format_lock_v1.jsonl"

    ok = run_cmd([
        py, "scripts/sft/generate_format_lock_dataset.py",
        "--output_dir", str(format_lock_dir),
        "--mode", args.format_lock_mode,
        "--math", args.format_lock_math,
    ], "Step 1/4: Generate format lock dataset", args.dry_run)
    if not ok:
        return 1

    # Step 2: Generate echo dataset
    echo_dir = intermediate / "echo"
    echo_file = echo_dir / "mypt_echo_diverse.jsonl"

    ok = run_cmd([
        py, "scripts/sft/generate_echo_dataset.py",
        "--output_dir", str(echo_dir),
        "--gibberish", args.echo_gibberish,
        "--seed", str(args.seed),
    ], "Step 2/4: Generate echo dataset", args.dry_run)
    if not ok:
        return 1

    # Step 3: Mix datasets
    mixed_file = intermediate / "phase1_mixed.jsonl"

    ok = run_cmd([
        py, "scripts/sft/mix_sft_jsonl.py",
        "--inputs",
        f"{format_lock_file}:{args.format_lock_ratio}",
        f"{echo_file}:{args.echo_ratio}",
        "--output", str(mixed_file),
        "--shuffle",
        "--seed", str(args.seed),
    ], "Step 3/4: Mix format_lock + echo datasets", args.dry_run)
    if not ok:
        return 1

    # Resolve mixed output path robustly (some scripts may resolve relative paths differently)
    mixed_file_resolved = mixed_file
    if not args.dry_run:
        alt_mixed = PROJECT_ROOT / "scripts" / Path(args.intermediate_dir) / "phase1_mixed.jsonl"
        found = _resolve_existing_path([mixed_file, alt_mixed])
        if found is not None:
            mixed_file_resolved = found
            if found != mixed_file:
                print(f"  Note: using mixed file from alternate path: {found}")
        else:
            print(f"  ERROR: mixed output not found at expected locations:")
            print(f"    - {mixed_file}")
            print(f"    - {alt_mixed}")
            return 1

    # Print mix stats
    if not args.dry_run and mixed_file_resolved.exists():
        total, en, de = count_jsonl(mixed_file_resolved)
        de_pct = (de / total * 100) if total > 0 else 0
        print(f"\n  Mixed dataset: {total:,} episodes")
        print(f"    English: {en:,}  German: {de:,} ({de_pct:.1f}%)")

    if args.skip_tokenize:
        print(f"\n  Skipping tokenization (--skip_tokenize)")
        print(f"  Mixed JSONL ready at: {mixed_file_resolved}")
        return 0

    # Step 4: Tokenize with loss masking + packing
    tokenize_cmd = [
        py, "scripts/sft/prepare_chat_sft.py",
        "--input", str(mixed_file_resolved),
        "--output_dir", str(output),
        "--val_split", str(args.val_split),
    ]
    if not args.no_packing:
        tokenize_cmd.extend(["--enable_packing", "--pack_block_size", str(args.pack_block_size)])
    step_label = "Step 4/4: Tokenize with loss masking"
    if not args.no_packing:
        step_label += f" + PACKING (block_size={args.pack_block_size})"
    ok = run_cmd(tokenize_cmd, step_label, args.dry_run)
    if not ok:
        return 1

    # Final summary
    print("\n" + "="*70)
    print("  PHASE 1 FORMAT LOCK: PREPARATION COMPLETE")
    print("="*70)
    if not args.dry_run:
        print(f"  Tokenized dataset: {output}")
        if (output / "dataset_metadata.json").exists():
            with open(output / "dataset_metadata.json") as f:
                meta = json.load(f)
            def _fmt_num(val):
                return f"{val:,}" if isinstance(val, int) else str(val)

            total_tokens = meta.get('total_tokens', meta.get('num_train_tokens', '?'))
            train_eps = meta.get('train_episodes', meta.get('num_train_episodes', '?'))
            val_eps = meta.get('val_episodes', meta.get('num_val_episodes', '?'))

            print(f"  Total tokens:      {_fmt_num(total_tokens)}")
            print(f"  Train episodes:    {_fmt_num(train_eps)}")
            print(f"  Val episodes:      {_fmt_num(val_eps)}")

        print(f"\n  Next step:")
        print(f"    python train.py \\")
        print(f"        --model_name phase1_format_lock \\")
        print(f"        --config_file configs/sft/phase1_format_lock.json \\")
        print(f"        --dataset_dir {output} \\")
        print(f"        --init_from_model checkpoints/GOLD_unified_v1")
    else:
        print("  [DRY RUN] No files were created")

    return 0


if __name__ == "__main__":
    sys.exit(main())
