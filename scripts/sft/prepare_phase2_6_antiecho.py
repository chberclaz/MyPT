#!/usr/bin/env python3
"""
Build Phase 2.6 anti-echo micro-phase dataset.

Target train composition:
- 60% anti-echo
- 20% echo
- 20% replay from current phase2 operators

Outputs:
- <output_dir>/phase2_6_mixed_train.jsonl
- <output_dir>/phase2_6_val.jsonl
- <output_dir>/phase2_6_meta.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import GPTConfig, Tokenizer
from scripts.sft.generate_echo_dataset import generate_echo_pairs, create_episode as create_echo_episode


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pairs_to_episodes(pairs, start_idx: int, phase_name: str, kind: str) -> List[Dict]:
    out: List[Dict] = []
    for i, (q, a, cat) in enumerate(pairs):
        ep = create_echo_episode(q, a, start_idx + i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = phase_name
        ep["_meta"]["operator"] = kind
        ep["_meta"]["category"] = cat
        out.append(ep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Phase 2.6 anti-echo micro-phase dataset")
    ap.add_argument("--output_dir", type=str, default="data/sft_phase2_6_intermediate")
    ap.add_argument("--replay_file", type=str, default="data/sft_phase2_intermediate/operators/operator_train.jsonl")
    ap.add_argument("--target_train_size", type=int, default=60000)
    ap.add_argument("--val_size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2601)
    ap.add_argument("--anti_ratio", type=float, default=0.60)
    ap.add_argument("--echo_ratio", type=float, default=0.20)
    ap.add_argument("--replay_ratio", type=float, default=0.20)
    ap.add_argument("--echo_gibberish_mode", type=str, default="include", choices=["include", "exclude", "only"])
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    args = ap.parse_args()

    # Validate ratios
    total_ratio = args.anti_ratio + args.echo_ratio + args.replay_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Build tokenizer for BPE-safe gibberish in echo generator.
    tok_cfg = GPTConfig(vocab_size=50304)
    tok = Tokenizer(tok_cfg, "gpt2")

    # Oversample generated pairs, then select exact anti/echo budgets.
    needed_anti = int(args.target_train_size * args.anti_ratio)
    needed_echo = int(args.target_train_size * args.echo_ratio)
    needed_replay = args.target_train_size - needed_anti - needed_echo

    oversample_n = max(args.target_train_size * 3, 120000)
    pairs = generate_echo_pairs(
        max_examples=oversample_n,
        seed=args.seed + 11,
        gibberish_mode=args.echo_gibberish_mode,
        bpe_safe=True,
        max_target_tokens=4,
        anti_echo_ratio=0.60,
        contrast_ratio=0.35,
        tokenizer=tok,
    )

    anti_pairs = [p for p in pairs if "anti_echo" in p[2]]
    echo_pairs = [p for p in pairs if "anti_echo" not in p[2]]
    if len(anti_pairs) < needed_anti:
        raise RuntimeError(f"Not enough anti-echo pairs ({len(anti_pairs)}) for target {needed_anti}")
    if len(echo_pairs) < (needed_echo + args.val_size):
        raise RuntimeError(f"Not enough echo pairs ({len(echo_pairs)}) for train+val")

    anti_train_pairs = rng.sample(anti_pairs, needed_anti)
    anti_train_keys = {(q, a, c) for (q, a, c) in anti_train_pairs}
    anti_val_pool = [p for p in anti_pairs if (p[0], p[1], p[2]) not in anti_train_keys]

    echo_train_pairs = rng.sample(echo_pairs, needed_echo)
    echo_train_keys = {(q, a, c) for (q, a, c) in echo_train_pairs}
    echo_val_pool = [p for p in echo_pairs if (p[0], p[1], p[2]) not in echo_train_keys]

    # Replay sampling from operator train set
    replay_path = (PROJECT_ROOT / args.replay_file).resolve()
    replay_rows = _read_jsonl(replay_path)
    if len(replay_rows) < needed_replay:
        raise RuntimeError(f"Replay file too small: {len(replay_rows)} < {needed_replay}")
    replay_sample = rng.sample(replay_rows, needed_replay)
    for r in replay_sample:
        r.setdefault("_meta", {})
        r["_meta"]["phase"] = "phase2_replay"

    anti_eps = _pairs_to_episodes(anti_train_pairs, 0, "phase2_6_anti_echo", "ANTI_ECHO")
    echo_eps = _pairs_to_episodes(echo_train_pairs, len(anti_eps), "phase2_6_echo", "ECHO")

    train_rows: List[Dict] = []
    train_rows.extend(anti_eps)
    train_rows.extend(echo_eps)
    train_rows.extend(replay_sample)
    if args.shuffle:
        rng.shuffle(train_rows)

    # Val composition: half anti + half echo (no replay in val)
    val_half = args.val_size // 2
    anti_val_pairs = rng.sample(anti_val_pool, min(val_half, len(anti_val_pool)))
    echo_val_pairs = rng.sample(echo_val_pool, min(args.val_size - len(anti_val_pairs), len(echo_val_pool)))
    val_rows = _pairs_to_episodes(anti_val_pairs, 900000, "phase2_6_anti_echo_val", "ANTI_ECHO")
    val_rows.extend(_pairs_to_episodes(echo_val_pairs, 910000, "phase2_6_echo_val", "ECHO"))
    if args.shuffle:
        rng.shuffle(val_rows)

    train_file = out_dir / "phase2_6_mixed_train.jsonl"
    val_file = out_dir / "phase2_6_val.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(val_file, val_rows)

    meta = {
        "seed": args.seed,
        "target_train_size": args.target_train_size,
        "ratios": {
            "anti": args.anti_ratio,
            "echo": args.echo_ratio,
            "replay": args.replay_ratio,
        },
        "counts": {
            "anti_train": len(anti_eps),
            "echo_train": len(echo_eps),
            "replay_train": len(replay_sample),
            "train_total": len(train_rows),
            "val_total": len(val_rows),
        },
        "sources": {
            "replay_file": str(replay_path),
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
    }
    meta_file = out_dir / "phase2_6_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=== Phase 2.6 dataset ready ===")
    print(f"Train: {train_file} ({len(train_rows):,})")
    print(f"Val:   {val_file} ({len(val_rows):,})")
    print(f"Meta:  {meta_file}")
    print("\nNext step:")
    print(
        "python scripts/sft/prepare_chat_sft.py "
        "--input data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl "
        "--output_dir data/sft_phase2_6_antiecho "
        "--val_file data/sft_phase2_6_intermediate/phase2_6_val.jsonl "
        "--no_system_prompt --enable_packing --pack_block_size 4096 --pack_by_field \"_meta.operator\""
    )


if __name__ == "__main__":
    main()
