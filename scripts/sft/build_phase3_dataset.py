#!/usr/bin/env python3
"""
Build Phase 3 mixed dataset with explicit composition targets.

Targets (defaults aligned with plan):
- operators maintenance: 5-10% (default 8%)
- anti-echo maintenance: 2-5% (default 4%)
- grounded context-only QA: 10-20% (default 16%)
- remainder: strict/checkable instruction data
- open-ended chat capped to avoid obedience drift
- short-first multi-turn: limit 3-4 turn share
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from core.dataset_lineage import iso_now, merge_lineage, write_lineage_sidecar


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


def _is_chat_episode(ep: Dict[str, Any]) -> bool:
    msgs = ep.get("messages")
    return isinstance(msgs, list) and len(msgs) >= 2


def _assistant_turns(ep: Dict[str, Any]) -> int:
    return sum(1 for m in ep.get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant")


def _is_grounded(ep: Dict[str, Any]) -> bool:
    msgs = ep.get("messages", [])
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user" and str(m.get("context", "")).strip():
            for j in range(i + 1, len(msgs)):
                n = msgs[j]
                if isinstance(n, dict) and n.get("role") == "assistant":
                    return True
    return False


def _is_anti_echo(ep: Dict[str, Any]) -> bool:
    meta = ep.get("_meta", {})
    cat = str(meta.get("category", "")).lower()
    if "anti_echo" in cat:
        return True
    for m in ep.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = str(m.get("content", "")).lower()
            if c in {"unknown.", "unknown", "no.", "no", "unbekannt.", "unbekannt", "nein.", "nein"}:
                return True
    return False


def _extract_first_quoted(text: str) -> Optional[str]:
    for q in ['"', "'"]:
        i = text.find(q)
        if i >= 0:
            j = text.find(q, i + 1)
            if j > i + 1:
                return text[i + 1:j].strip()
    return None


def _is_clean_anti_echo(ep: Dict[str, Any]) -> bool:
    """Keep anti-echo samples whose assistant output does not leak forbidden token."""
    if not _is_anti_echo(ep):
        return False
    user_text = ""
    asst_text = ""
    for m in ep.get("messages", []):
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user" and not user_text:
            user_text = str(m.get("content", ""))
        if m.get("role") == "assistant" and not asst_text:
            asst_text = str(m.get("content", ""))
    forbidden = _extract_first_quoted(user_text)
    if forbidden and forbidden.lower() in asst_text.lower():
        return False
    return True


def _is_operator(ep: Dict[str, Any]) -> bool:
    op = str(ep.get("_meta", {}).get("operator", "")).upper()
    return op in {"COPY", "WRAP", "EXTRACT"}


def _add_source(rows: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if not _is_chat_episode(r):
            continue
        rr = dict(r)
        rr["mix_source"] = source_name
        out.append(rr)
    return out


def _sample(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if not rows:
        return []
    rng = random.Random(seed)
    if len(rows) >= n:
        return rng.sample(rows, n)
    return [rows[rng.randrange(len(rows))] for _ in range(n)]


def _with_turn_cap(rows: List[Dict[str, Any]], max_multiturn_ratio: float, seed: int) -> List[Dict[str, Any]]:
    single = [r for r in rows if _assistant_turns(r) <= 1]
    multi = [r for r in rows if _assistant_turns(r) > 1]
    if not multi:
        return rows
    total = len(rows)
    cap_n = int(total * max_multiturn_ratio)
    rng = random.Random(seed)
    if len(multi) > cap_n:
        multi = rng.sample(multi, cap_n)
    merged = single + multi
    rng.shuffle(merged)
    return merged


def _count(rows: List[Dict[str, Any]], pred) -> int:
    return sum(1 for r in rows if pred(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 3 mixed dataset with explicit composition policy")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--meta_output", type=str, default=None)
    p.add_argument("--target_size", type=int, default=80000)
    p.add_argument("--seed", type=int, default=3201)

    p.add_argument("--precision_file", type=str, required=True)
    p.add_argument("--grounded_file", type=str, required=True, help="RAG/context dataset JSONL")
    p.add_argument("--operators_file", type=str, required=True)
    p.add_argument("--anti_echo_file", type=str, required=True)
    p.add_argument("--open_chat_files", nargs="*", default=[], help="HF/open-ended chat JSONL files")

    p.add_argument("--operators_ratio", type=float, default=0.08)
    p.add_argument("--anti_echo_ratio", type=float, default=0.02)
    p.add_argument("--grounded_ratio", type=float, default=0.16)
    p.add_argument("--open_chat_cap_ratio", type=float, default=0.20)
    p.add_argument("--multiturn_cap_ratio", type=float, default=0.22)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out_path = Path(args.output)
    meta_path = Path(args.meta_output) if args.meta_output else out_path.with_suffix(".meta.json")

    precision_rows = _add_source(_read_jsonl(Path(args.precision_file)), "phase3_precision")
    grounded_rows = _add_source(_read_jsonl(Path(args.grounded_file)), "phase3_grounded")
    operators_rows = _add_source(_read_jsonl(Path(args.operators_file)), "phase3_operators_replay")
    anti_rows = _add_source(_read_jsonl(Path(args.anti_echo_file)), "phase3_anti_echo_replay")
    open_rows: List[Dict[str, Any]] = []
    for f in args.open_chat_files:
        p = Path(f)
        open_rows.extend(_add_source(_read_jsonl(p), p.stem))

    # Filters for semantic intent.
    grounded_rows = [r for r in grounded_rows if _is_grounded(r)]
    operators_rows = [r for r in operators_rows if _is_operator(r)]
    anti_rows = [r for r in anti_rows if _is_clean_anti_echo(r)]

    n_total = args.target_size
    n_op = int(round(n_total * args.operators_ratio))
    n_anti = int(round(n_total * args.anti_echo_ratio))
    n_grounded = int(round(n_total * args.grounded_ratio))
    n_open_cap = int(round(n_total * args.open_chat_cap_ratio))

    fixed = n_op + n_anti + n_grounded
    if fixed > n_total:
        raise ValueError("operators_ratio + anti_echo_ratio + grounded_ratio exceed 100%")
    n_remaining = n_total - fixed
    n_open = min(n_open_cap, n_remaining)
    n_precision = n_remaining - n_open

    mixed: List[Dict[str, Any]] = []
    mixed.extend(_sample(operators_rows, n_op, args.seed + 11))
    mixed.extend(_sample(anti_rows, n_anti, args.seed + 17))
    mixed.extend(_sample(grounded_rows, n_grounded, args.seed + 23))
    mixed.extend(_sample(open_rows, n_open, args.seed + 29))
    mixed.extend(_sample(precision_rows, n_precision, args.seed + 31))

    mixed = _with_turn_cap(mixed, args.multiturn_cap_ratio, args.seed + 41)
    rng.shuffle(mixed)

    _write_jsonl(out_path, mixed)

    source_counts: Dict[str, int] = {}
    for r in mixed:
        s = str(r.get("mix_source", "unknown"))
        source_counts[s] = source_counts.get(s, 0) + 1

    meta = {
        "target_size": n_total,
        "actual_size": len(mixed),
        "requested_ratios": {
            "operators": args.operators_ratio,
            "anti_echo": args.anti_echo_ratio,
            "grounded": args.grounded_ratio,
            "open_chat_cap": args.open_chat_cap_ratio,
        },
        "allocated_counts": {
            "operators": n_op,
            "anti_echo": n_anti,
            "grounded": n_grounded,
            "open_chat": n_open,
            "precision": n_precision,
        },
        "actual_coverage": {
            "operators_rate": round(_count(mixed, _is_operator) * 100.0 / max(1, len(mixed)), 2),
            "anti_echo_rate": round(_count(mixed, _is_anti_echo) * 100.0 / max(1, len(mixed)), 2),
            "grounded_rate": round(_count(mixed, _is_grounded) * 100.0 / max(1, len(mixed)), 2),
            "multiturn_rate": round(sum(1 for r in mixed if _assistant_turns(r) > 1) * 100.0 / max(1, len(mixed)), 2),
            "avg_assistant_turns": round(sum(_assistant_turns(r) for r in mixed) / max(1, len(mixed)), 3),
        },
        "source_counts": source_counts,
        "inputs": {
            "precision_file": args.precision_file,
            "grounded_file": args.grounded_file,
            "operators_file": args.operators_file,
            "anti_echo_file": args.anti_echo_file,
            "open_chat_files": args.open_chat_files,
        },
    }
    lineage_inputs = []
    for src, key in [
        (args.precision_file, "phase3_precision"),
        (args.grounded_file, "phase3_grounded"),
        (args.operators_file, "phase3_operators_replay"),
        (args.anti_echo_file, "phase3_anti_echo_replay"),
    ]:
        c = source_counts.get(key, 0)
        lineage_inputs.append({
            "path": str(Path(src).resolve()),
            "sampled_rows": int(c),
            "effective_ratio": c / max(1, len(mixed)),
        })
    for f in args.open_chat_files:
        stem = Path(f).stem
        c = source_counts.get(stem, 0)
        lineage_inputs.append({
            "path": str(Path(f).resolve()),
            "sampled_rows": int(c),
            "effective_ratio": c / max(1, len(mixed)),
        })
    lineage = merge_lineage(
        inputs=lineage_inputs,
        output_rows=len(mixed),
        creation_context={
            "timestamp": iso_now(),
            "script": "scripts/sft/build_phase3_dataset.py",
            "args": vars(args),
        },
    )
    meta["lineage"] = lineage

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    lineage_path = write_lineage_sidecar(out_path, lineage)

    print("=" * 60)
    print("Phase 3 dataset built")
    print("=" * 60)
    print(f"Output: {out_path}")
    print(f"Meta:   {meta_path}")
    print(f"Lineage:{lineage_path}")
    print(f"Episodes: {len(mixed):,}")
    print("Actual coverage:")
    for k, v in meta["actual_coverage"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()

