#!/usr/bin/env python3
"""
Build Phase 2.5 dataset: WRAP-focused + anti-echo + operator replay.

Goal:
- Strengthen WRAP operator robustness with high template/delimiter diversity.
- Add anti-echo constraints ("do not blindly copy" behavior).
- Preserve prior operator skill via 20% replay from current operator train set.

Outputs:
- <output_dir>/wrap_focus_train.jsonl
- <output_dir>/wrap_focus_val.jsonl
- <output_dir>/echo_antiecho_train.jsonl
- <output_dir>/phase2_5_mixed.jsonl
- <output_dir>/phase2_5_meta.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import GPTConfig, Tokenizer
from scripts.sft.generate_operator_dataset import generate_unique_payloads, create_episode
from scripts.sft.generate_echo_dataset import generate_echo_pairs, create_episode as create_echo_episode


# ASCII-only wrapper styles for broad delimiter coverage.
WRAP_STYLES: List[Tuple[str, str, str]] = [
    ("square", "[", "]"),
    ("paren", "(", ")"),
    ("curly", "{", "}"),
    ("angle", "<", ">"),
    ("double_square", "[[", "]]"),
    ("double_paren", "((", "))"),
    ("double_curly", "{{", "}}"),
    ("single_quote", "'", "'"),
    ("double_quote", '"', '"'),
]

WRAP_TEMPLATES_TRAIN = [
    "Wrap this in {L}{R}: {X}",
    "Surround with {STYLE} brackets: {X}",
    "Return {X} wrapped by {L} and {R}.",
    "Enclose exactly: {X} using {L}{R}",
    "Put {L} and {R} around: {X}",
    "Output only the wrapped form of: {X} with {STYLE}",
]

WRAP_TEMPLATES_VAL = [
    "Bracket this with {L}{R}: {X}",
    "Wrap using {STYLE}: {X}",
    "Add delimiters {L}...{R} around: {X}",
]


def _format_wrap_question(template: str, payload: str, style_name: str, left: str, right: str) -> str:
    return (
        template.replace("{X}", payload)
        .replace("{STYLE}", style_name)
        .replace("{L}", left)
        .replace("{R}", right)
    )


def _write_jsonl(path: Path, episodes: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict]:
    episodes: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def generate_wrap_focus_dataset(
    n_train_payloads: int,
    n_val_payloads: int,
    reps_per_style: int,
    seed_train: int,
    seed_val: int,
    tokenizer: Tokenizer,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Generate WRAP-specialized train/val episodes with diverse delimiters and templates.
    """
    # Keep payload diversity aligned with operator phase assumptions.
    train_payloads = generate_unique_payloads(
        n_train_payloads,
        seed_train,
        tokenizer=tokenizer,
        max_tokens=12,
        max_words=4,
        word_distribution=(0.35, 0.30, 0.20, 0.15),
    )
    val_payloads = generate_unique_payloads(
        n_val_payloads,
        seed_val,
        exclude=set(train_payloads),
        tokenizer=tokenizer,
        max_tokens=12,
        max_words=4,
        word_distribution=(0.35, 0.30, 0.20, 0.15),
    )

    train_eps: List[Dict] = []
    val_eps: List[Dict] = []

    rng_train = random.Random(seed_train)
    rng_val = random.Random(seed_val)

    # Train: all payloads x all styles x reps_per_style templates
    for payload in train_payloads:
        for style_name, left, right in WRAP_STYLES:
            n_tpl = min(reps_per_style, len(WRAP_TEMPLATES_TRAIN))
            templates = rng_train.sample(WRAP_TEMPLATES_TRAIN, n_tpl)
            for tpl in templates:
                q = _format_wrap_question(tpl, payload, style_name, left, right)
                a = f"{left}{payload}{right}"
                ep = create_episode(q, a, language="en")
                ep["_meta"] = {
                    "operator": "WRAP",
                    "payload": payload,
                    "expected": a,
                    "wrap_style": style_name,
                    "left_delim": left,
                    "right_delim": right,
                    "phase": "phase2_5_wrap_focus",
                }
                train_eps.append(ep)

    # Val: all payloads x all styles x 1 val template
    for payload in val_payloads:
        for style_name, left, right in WRAP_STYLES:
            tpl = rng_val.choice(WRAP_TEMPLATES_VAL)
            q = _format_wrap_question(tpl, payload, style_name, left, right)
            a = f"{left}{payload}{right}"
            ep = create_episode(q, a, language="en")
            ep["_meta"] = {
                "operator": "WRAP",
                "payload": payload,
                "expected": a,
                "wrap_style": style_name,
                "left_delim": left,
                "right_delim": right,
                "split": "val",
                "phase": "phase2_5_wrap_focus",
            }
            val_eps.append(ep)

    rng_train.shuffle(train_eps)
    rng_val.shuffle(val_eps)
    meta = {
        "n_train_payloads": n_train_payloads,
        "n_val_payloads": n_val_payloads,
        "n_wrap_styles": len(WRAP_STYLES),
        "reps_per_style": reps_per_style,
        "train_episodes": len(train_eps),
        "val_episodes": len(val_eps),
    }
    return train_eps, val_eps, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Phase 2.5 wrap+anti-echo mixed JSONL")
    parser.add_argument("--output_dir", type=str, default="data/sft_phase2_5_intermediate")
    parser.add_argument("--replay_file", type=str, default="data/sft_phase2_intermediate/operators/operator_train.jsonl")
    parser.add_argument("--replay_ratio", type=float, default=0.20, help="Replay ratio from current operator train set")
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--wrap_train_payloads", type=int, default=9000)
    parser.add_argument("--wrap_val_payloads", type=int, default=800)
    parser.add_argument("--wrap_reps_per_style", type=int, default=2)
    parser.add_argument("--echo_max_examples", type=int, default=70000)
    parser.add_argument("--echo_anti_ratio", type=float, default=0.40)
    parser.add_argument("--echo_contrast_ratio", type=float, default=0.35)
    parser.add_argument("--echo_gibberish_mode", type=str, default="include", choices=["include", "exclude", "only"])
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    args = parser.parse_args()

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2.5 dataset build ===")
    print(f"Output dir:   {out_dir}")
    print(f"Replay file:  {args.replay_file} @ {args.replay_ratio:.0%}")
    print(f"Seed:         {args.seed}")

    # Tokenizer for BPE-safe payload/gibberish generation
    tok_cfg = GPTConfig(vocab_size=50304)
    tok = Tokenizer(tok_cfg, "gpt2")

    # 1) WRAP-focused dataset
    wrap_train, wrap_val, wrap_meta = generate_wrap_focus_dataset(
        n_train_payloads=args.wrap_train_payloads,
        n_val_payloads=args.wrap_val_payloads,
        reps_per_style=args.wrap_reps_per_style,
        seed_train=args.seed,
        seed_val=args.seed + 111,
        tokenizer=tok,
    )
    wrap_train_file = out_dir / "wrap_focus_train.jsonl"
    wrap_val_file = out_dir / "wrap_focus_val.jsonl"
    _write_jsonl(wrap_train_file, wrap_train)
    _write_jsonl(wrap_val_file, wrap_val)
    print(f"WRAP focus:   {len(wrap_train):,} train, {len(wrap_val):,} val")

    # 2) Echo + anti-echo dataset
    echo_pairs = generate_echo_pairs(
        max_examples=args.echo_max_examples,
        seed=args.seed + 222,
        gibberish_mode=args.echo_gibberish_mode,
        bpe_safe=True,
        max_target_tokens=4,
        anti_echo_ratio=args.echo_anti_ratio,
        contrast_ratio=args.echo_contrast_ratio,
        tokenizer=tok,
    )
    echo_eps: List[Dict] = []
    for i, (q, a, cat) in enumerate(echo_pairs):
        ep = create_echo_episode(q, a, i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["operator"] = "ANTI_ECHO" if "anti_echo" in cat else "ECHO"
        ep["_meta"]["category"] = cat
        ep["_meta"]["phase"] = "phase2_5_echo_antiecho"
        echo_eps.append(ep)
    echo_file = out_dir / "echo_antiecho_train.jsonl"
    _write_jsonl(echo_file, echo_eps)
    print(f"Echo/anti:    {len(echo_eps):,} train")

    # 3) Replay sample from current operator train set
    replay_path = (PROJECT_ROOT / args.replay_file).resolve()
    replay_eps = _read_jsonl(replay_path)
    rng = random.Random(args.seed + 333)
    replay_n = max(1, int(len(replay_eps) * args.replay_ratio))
    replay_sample = rng.sample(replay_eps, replay_n) if replay_n < len(replay_eps) else replay_eps
    for ep in replay_sample:
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_replay"
    replay_file = out_dir / "operator_replay_20pct.jsonl"
    _write_jsonl(replay_file, replay_sample)
    print(f"Replay:       {len(replay_sample):,} sampled from {len(replay_eps):,}")

    # 4) Mix all train sources
    mixed = []
    mixed.extend(wrap_train)
    mixed.extend(echo_eps)
    mixed.extend(replay_sample)
    if args.shuffle:
        random.Random(args.seed + 444).shuffle(mixed)
    mixed_file = out_dir / "phase2_5_mixed.jsonl"
    _write_jsonl(mixed_file, mixed)
    print(f"Mixed total:  {len(mixed):,} -> {mixed_file}")

    # Metadata
    meta = {
        "seed": args.seed,
        "sources": {
            "wrap_focus_train": str(wrap_train_file),
            "wrap_focus_val": str(wrap_val_file),
            "echo_antiecho_train": str(echo_file),
            "operator_replay_20pct": str(replay_file),
            "phase2_5_mixed": str(mixed_file),
        },
        "counts": {
            "wrap_focus_train": len(wrap_train),
            "wrap_focus_val": len(wrap_val),
            "echo_antiecho_train": len(echo_eps),
            "operator_replay_20pct": len(replay_sample),
            "phase2_5_mixed": len(mixed),
        },
        "replay_ratio": args.replay_ratio,
        "wrap_meta": wrap_meta,
        "echo_config": {
            "max_examples": args.echo_max_examples,
            "anti_echo_ratio": args.echo_anti_ratio,
            "contrast_ratio": args.echo_contrast_ratio,
            "gibberish_mode": args.echo_gibberish_mode,
        },
    }
    meta_file = out_dir / "phase2_5_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Metadata:     {meta_file}")

    print("\nNext step:")
    print(
        "  python scripts/sft/prepare_chat_sft.py "
        "--input data/sft_phase2_5_intermediate/phase2_5_mixed.jsonl "
        "--output_dir data/sft_phase2_5_wrap_antiecho "
        "--val_file data/sft_phase2_5_intermediate/wrap_focus_val.jsonl "
        "--no_system_prompt --enable_packing --pack_block_size 4096 --pack_by_field \"_meta.operator\""
    )


if __name__ == "__main__":
    main()
