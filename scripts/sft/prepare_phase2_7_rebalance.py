#!/usr/bin/env python3
"""
Build Phase 2.7 rebalance dataset (fresh run from 2.5 checkpoint).

Target train composition:
- 40% anti-echo
- 40% operators replay
- 20% code chat (optional if a compatible JSONL source exists)

If no code source is found/provided, ratios are renormalized across anti/operator
so training still proceeds (effectively 50/50).

Outputs:
- <output_dir>/phase2_7_mixed.jsonl
- <output_dir>/phase2_7_meta.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import GPTConfig, Tokenizer
from scripts.sft.generate_echo_dataset import (
    generate_echo_pairs,
    create_episode as create_echo_episode,
)


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


def _is_chat_episode(ep: Dict) -> bool:
    msgs = ep.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    has_user = any(m.get("role") == "user" and str(m.get("content", "")).strip() for m in msgs if isinstance(m, dict))
    has_asst = any(m.get("role") == "assistant" and str(m.get("content", "")).strip() for m in msgs if isinstance(m, dict))
    return has_user and has_asst


def _sample(rows: List[Dict], n: int, seed: int) -> List[Dict]:
    if n <= 0:
        return []
    if len(rows) >= n:
        rng = random.Random(seed)
        return rng.sample(rows, n)
    # With replacement when source is smaller than requested.
    rng = random.Random(seed)
    out: List[Dict] = []
    for _ in range(n):
        out.append(rows[rng.randrange(len(rows))])
    return out


def _detect_code_file(explicit_code_file: Optional[str]) -> Optional[Path]:
    if explicit_code_file:
        p = Path(explicit_code_file)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p if p.exists() else None

    # Auto-detect likely chat-JSONL code sources.
    patterns = [
        "data/**/*code*.jsonl",
        "data/**/*program*.jsonl",
        "data/**/*coding*.jsonl",
    ]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(PROJECT_ROOT.glob(pat))
    # Prefer "sft" intermediate data if present.
    candidates = sorted(set(candidates), key=lambda p: ("sft" not in str(p).lower(), len(str(p))))
    for p in candidates:
        try:
            rows = _read_jsonl(p)
        except Exception:
            continue
        if any(_is_chat_episode(ep) for ep in rows[:200]):
            return p
    return None


def _build_anti_echo_episodes(target_n: int, seed: int) -> List[Dict]:
    if target_n <= 0:
        return []
    tok = Tokenizer(GPTConfig(vocab_size=50304), "gpt2")
    # Over-generate to avoid shortages after filtering.
    pairs = generate_echo_pairs(
        max_examples=max(target_n * 3, 30000),
        seed=seed,
        gibberish_mode="include",
        bpe_safe=True,
        max_target_tokens=4,
        anti_echo_ratio=0.95,
        contrast_ratio=0.0,
        tokenizer=tok,
    )
    anti_pairs = [p for p in pairs if "anti_echo" in str(p[2])]
    if not anti_pairs:
        raise RuntimeError("No anti-echo pairs generated.")

    episodes: List[Dict] = []
    for i, (q, a, cat) in enumerate(anti_pairs):
        ep = create_echo_episode(q, a, i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_7_rebalance"
        ep["_meta"]["source_stream"] = "anti_echo"
        ep["_meta"]["operator"] = "ANTI_ECHO"
        episodes.append(ep)
        if len(episodes) >= target_n:
            break

    if len(episodes) < target_n:
        # Deterministic upsample when generation pool is too small.
        episodes = _sample(episodes, target_n, seed + 41)
    return episodes


def _prepare_operator_rows(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for ep in rows:
        if not _is_chat_episode(ep):
            continue
        e = dict(ep)
        meta = dict(e.get("_meta", {}))
        meta["phase"] = "phase2_7_rebalance"
        meta["source_stream"] = "operators"
        e["_meta"] = meta
        out.append(e)
    return out


def _prepare_code_rows(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for ep in rows:
        if not _is_chat_episode(ep):
            continue
        e = dict(ep)
        meta = dict(e.get("_meta", {}))
        meta["phase"] = "phase2_7_rebalance"
        meta["source_stream"] = "code"
        e["_meta"] = meta
        out.append(e)
    return out


def _normalized_ratios(use_code: bool, anti_ratio: float, operator_ratio: float, code_ratio: float) -> Tuple[float, float, float]:
    if use_code:
        s = anti_ratio + operator_ratio + code_ratio
        if s <= 0:
            raise ValueError("Ratios must sum to a positive value.")
        return anti_ratio / s, operator_ratio / s, code_ratio / s
    # No code source: keep anti/operator balance equalized.
    s = anti_ratio + operator_ratio
    if s <= 0:
        return 0.5, 0.5, 0.0
    return anti_ratio / s, operator_ratio / s, 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Phase 2.7 rebalance dataset JSONL")
    p.add_argument("--output_dir", type=str, default="data/sft_phase2_7_intermediate")
    p.add_argument("--operators_file", type=str, default="data/sft_phase2_intermediate/operators/operator_train.jsonl")
    p.add_argument("--code_file", type=str, default=None, help="Optional code chat JSONL. If omitted, auto-detect.")
    p.add_argument("--target_train_size", type=int, default=60000)
    p.add_argument("--anti_ratio", type=float, default=0.40)
    p.add_argument("--operator_ratio", type=float, default=0.40)
    p.add_argument("--code_ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=2701)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    args = p.parse_args()

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    operators_path = Path(args.operators_file)
    if not operators_path.is_absolute():
        operators_path = (PROJECT_ROOT / operators_path).resolve()
    if not operators_path.exists():
        raise FileNotFoundError(f"operators_file not found: {operators_path}")

    code_path = _detect_code_file(args.code_file)
    use_code = code_path is not None
    r_anti, r_op, r_code = _normalized_ratios(
        use_code=use_code,
        anti_ratio=args.anti_ratio,
        operator_ratio=args.operator_ratio,
        code_ratio=args.code_ratio,
    )

    n_total = args.target_train_size
    n_anti = int(round(n_total * r_anti))
    n_op = int(round(n_total * r_op))
    n_code = n_total - n_anti - n_op

    print("=== Phase 2.7 dataset build ===")
    print(f"Output dir:      {out_dir}")
    print(f"Operators file:  {operators_path}")
    print(f"Code file:       {code_path if code_path else '(none found)'}")
    print(f"Target train:    {n_total:,}")
    print(f"Ratios effective anti/op/code: {r_anti:.3f} / {r_op:.3f} / {r_code:.3f}")
    print(f"Counts target anti/op/code:    {n_anti:,} / {n_op:,} / {n_code:,}")

    anti_rows = _build_anti_echo_episodes(n_anti, args.seed + 11)

    operator_rows = _prepare_operator_rows(_read_jsonl(operators_path))
    if not operator_rows:
        raise RuntimeError("No valid operator chat episodes found.")
    operator_rows = _sample(operator_rows, n_op, args.seed + 23)

    code_rows: List[Dict] = []
    if n_code > 0 and code_path is not None:
        code_source = _prepare_code_rows(_read_jsonl(code_path))
        if code_source:
            code_rows = _sample(code_source, n_code, args.seed + 37)
        else:
            print("WARNING: code file exists but no valid chat episodes found; reallocating to anti/operator.")
            extra_anti = n_code // 2
            extra_op = n_code - extra_anti
            anti_rows.extend(_sample(anti_rows, extra_anti, args.seed + 41))
            operator_rows.extend(_sample(operator_rows, extra_op, args.seed + 43))

    mixed = anti_rows + operator_rows + code_rows
    if args.shuffle:
        rng = random.Random(args.seed + 101)
        rng.shuffle(mixed)

    out_file = out_dir / "phase2_7_mixed.jsonl"
    _write_jsonl(out_file, mixed)

    stream_counts = {"anti_echo": 0, "operators": 0, "code": 0, "other": 0}
    for ep in mixed:
        stream = ep.get("_meta", {}).get("source_stream", "other")
        stream_counts[stream if stream in stream_counts else "other"] += 1

    meta = {
        "phase": "phase2_7_rebalance",
        "seed": args.seed,
        "target_train_size": n_total,
        "sources": {
            "operators_file": str(operators_path),
            "code_file": str(code_path) if code_path else None,
        },
        "ratios_requested": {
            "anti_ratio": args.anti_ratio,
            "operator_ratio": args.operator_ratio,
            "code_ratio": args.code_ratio,
        },
        "ratios_effective": {
            "anti_ratio": r_anti,
            "operator_ratio": r_op,
            "code_ratio": r_code,
        },
        "counts": stream_counts,
        "output_file": str(out_file),
    }
    meta_path = out_dir / "phase2_7_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWritten: {out_file}")
    print(f"Counts: anti={stream_counts['anti_echo']:,}, op={stream_counts['operators']:,}, code={stream_counts['code']:,}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()

