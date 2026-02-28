#!/usr/bin/env python3
"""
Prepare Phase 2.8 bridge dataset (50% broad replay / 50% specialized correction).

Design goals:
- fast bridge run for A/B/C/E recovery
- strict unseen-eval exclusion for known eval prompts/payloads/templates
- reuse existing generators where possible
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.sft.generate_echo_dataset import generate_echo_pairs, create_episode as create_echo_episode
from scripts.sft.generate_operator_dataset import create_episode as create_operator_episode
from core.dataset_lineage import iso_now, merge_lineage, write_lineage_sidecar


KNOWN_EVAL_USER_TEXT = {
    "Say Hello.",
    "Say OK.",
    "Say yes.",
    "Repeat: Hello world",
    "Echo: Testing",
    "Sag Hallo.",
    "Say Banana.",
    "Say 42.",
    'What is "Blurpix"?',
    'Is "Zanthor" a real word?',
    'What does "Quexling" mean?',
    'Does "Flimzap" have a meaning?',
}

KNOWN_EVAL_PAYLOADS = {
    "xylophone", "quantum", "nebula", "twilight", "cascade",
    "phoenix", "marble", "crystal", "thunder", "horizon",
    "emerald", "silver", "dragon", "galaxy", "zenith",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _user_text(ep: Dict[str, Any]) -> str:
    for m in ep.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def _payload(ep: Dict[str, Any]) -> str:
    return str(ep.get("_meta", {}).get("payload", "")).strip().lower()


def _is_unseen_safe(ep: Dict[str, Any]) -> bool:
    u = _user_text(ep)
    if u in KNOWN_EVAL_USER_TEXT:
        return False
    p = _payload(ep)
    if p and p in KNOWN_EVAL_PAYLOADS:
        return False
    return True


def _build_specialized(rng: random.Random, n: int, seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    echo_pairs = generate_echo_pairs(
        max_examples=max(1000, n),
        seed=seed + 17,
        gibberish_mode="include",
        anti_echo_ratio=0.45,
        contrast_ratio=0.25,
    )
    for i, (q, a, cat) in enumerate(echo_pairs):
        ep = create_echo_episode(q, a, i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_8_specialized"
        ep["_meta"]["source_stream"] = "echo_anti_echo"
        out.append(ep)
        if len(out) >= int(n * 0.7):
            break

    # operator correction with quote-boundary variants
    quote_payloads = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    ops = [("COPY", 'Repeat exactly: "{}"', "{}"), ("EXTRACT", 'Return text between quotes: "{}"', "{}")]
    idx = 0
    while len(out) < n:
        p = rng.choice(quote_payloads) + str(idx % 7)
        op, q_tpl, a_tpl = rng.choice(ops)
        question = q_tpl.format(p)
        answer = a_tpl.format(p)
        ep = create_operator_episode(question, answer, "en")
        ep["_meta"] = {"operator": op, "payload": p, "expected": answer, "phase": "phase2_8_specialized", "source_stream": "operators"}
        out.append(ep)
        idx += 1
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare phase2.8 echo rebalance bridge dataset")
    p.add_argument("--output_dir", type=str, default="data/sft_phase2_8_intermediate")
    p.add_argument("--replay_file", type=str, required=True, help="Broad replay source jsonl")
    p.add_argument("--target_train_size", type=int, default=40000)
    p.add_argument("--val_size", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    replay_rows = _read_jsonl(Path(args.replay_file))
    replay_rows = [r for r in replay_rows if _is_unseen_safe(r)]
    half = args.target_train_size // 2
    replay_take = replay_rows if len(replay_rows) <= half else rng.sample(replay_rows, half)
    for ep in replay_take:
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_8_replay"
        ep["_meta"]["source_stream"] = "replay"

    specialized = _build_specialized(rng, args.target_train_size - len(replay_take), args.seed)
    specialized = [e for e in specialized if _is_unseen_safe(e)]
    # top-up specialized if filtering removed rows
    while len(specialized) < (args.target_train_size - len(replay_take)):
        specialized.extend(_build_specialized(rng, 256, args.seed + len(specialized)))
        specialized = [e for e in specialized if _is_unseen_safe(e)]
    specialized = specialized[: args.target_train_size - len(replay_take)]

    train_rows = replay_take + specialized
    if args.shuffle:
        rng.shuffle(train_rows)
    val_rows = train_rows[: args.val_size]
    train_rows = train_rows[args.val_size:]

    train_file = out_dir / "phase2_8_mixed_train.jsonl"
    val_file = out_dir / "phase2_8_val.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(val_file, val_rows)

    stream_counts: Dict[str, int] = {}
    for ep in train_rows:
        s = ep.get("_meta", {}).get("source_stream", "other")
        stream_counts[s] = stream_counts.get(s, 0) + 1

    lineage = merge_lineage(
        inputs=[{
            "path": str(Path(args.replay_file).resolve()),
            "sampled_rows": stream_counts.get("replay", 0),
            "effective_ratio": stream_counts.get("replay", 0) / max(1, len(train_rows)),
        }],
        output_rows=len(train_rows),
        creation_context={
            "timestamp": iso_now(),
            "script": "scripts/sft/prepare_phase2_8_echo_rebalance.py",
            "args": vars(args),
            "specialized_rows": stream_counts.get("echo_anti_echo", 0) + stream_counts.get("operators", 0),
            "unseen_eval_exclusion": True,
        },
    )

    meta = {
        "seed": args.seed,
        "target_train_size": args.target_train_size,
        "val_size": args.val_size,
        "sources": {"replay_file": args.replay_file},
        "counts": {
            "train_total": len(train_rows),
            "val_total": len(val_rows),
            "stream_counts": stream_counts,
        },
        "lineage": lineage,
    }
    meta_file = out_dir / "phase2_8_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    lineage_path = write_lineage_sidecar(train_file, lineage)

    print("=== Phase 2.8 bridge dataset ready ===")
    print(f"Train:   {train_file} ({len(train_rows):,})")
    print(f"Val:     {val_file} ({len(val_rows):,})")
    print(f"Meta:    {meta_file}")
    print(f"Lineage: {lineage_path}")


if __name__ == "__main__":
    main()

