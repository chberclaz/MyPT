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
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path (supports direct script execution via py.exe)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


def _make_echo_episode(question: str, answer: str, category: str = "echo_targeted") -> Dict[str, Any]:
    ep = create_echo_episode(question, answer, 0, category)
    ep.setdefault("_meta", {})
    ep["_meta"]["phase"] = "phase2_8_specialized"
    ep["_meta"]["source_stream"] = "echo_anti_echo"
    ep["_meta"]["operator"] = "ECHO"
    ep["_meta"]["payload"] = answer
    return ep


def _build_targeted_echo_set(rng: random.Random, n: int, split: str) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    payload_words = [
        "Hello", "OK", "yes", "Banana", "Hallo", "42", "7", "19",
        "blue ocean", "dark forest", "red mountain peak", "golden sunset sky",
    ]
    train_templates = [
        "Say {}.", "Repeat exactly: {}", "Echo: {}", "Output only: {}",
        "Return only this token: {}", "Sag {}.", "Antworte nur mit: {}",
    ]
    val_templates = [
        "Reply with only: {}", "State exactly: {}", "Gib exakt aus: {}",
        "Nur dies zurueckgeben: {}", "Write exactly this payload: {}",
        "No extra words. Return: {}", "Return raw text only: {}",
        "Produce only token: {}", "Return payload unchanged: {}",
        "Answer strictly with: {}", "Echo verbatim: {}", "Return exactly this: {}",
    ]
    templates = train_templates if split == "train" else val_templates
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < n:
        payload = payload_words[i % len(payload_words)]
        tpl = templates[i % len(templates)]
        q = tpl.format(payload)
        out.append(_make_echo_episode(q, payload, category=f"echo_targeted_{split}"))
        i += 1
    return out[:n]


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
    meta_payload = str(ep.get("_meta", {}).get("payload", "")).strip()
    if meta_payload:
        return meta_payload.lower()
    user = _user_text(ep)
    q = re.search(r'"([^"]+)"', user)
    if q:
        return q.group(1).strip().lower()
    c = re.search(r':\s*(.+)$', user)
    if c:
        return c.group(1).strip().lower()
    return ""


def _template_signature(ep: Dict[str, Any]) -> str:
    user = _user_text(ep)
    payload = _payload(ep)
    if payload and payload in user.lower():
        # best-effort payload replacement compatible with prepare_chat_sft semantics
        sig = re.sub(r'"[^"]+"', '"{PAYLOAD}"', user)
        sig = re.sub(re.escape(payload), "{PAYLOAD}", sig, flags=re.IGNORECASE)
    else:
        sig = user
    return " ".join(sig.split())


def _operator(ep: Dict[str, Any]) -> str:
    return str(ep.get("_meta", {}).get("operator", "")).strip().upper() or "UNKNOWN"


def _pair_key(ep: Dict[str, Any]) -> str:
    p = _payload(ep)
    if not p:
        return ""
    return f"{_operator(ep)}::{p}"


def _prefix_user(ep: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out = json.loads(json.dumps(ep))
    for msg in out.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            msg["content"] = f"{prefix}{msg.get('content', '')}"
            break
    return out


def _is_unseen_safe(ep: Dict[str, Any]) -> bool:
    u = _user_text(ep)
    if u in KNOWN_EVAL_USER_TEXT:
        return False
    p = _payload(ep)
    if p and p in KNOWN_EVAL_PAYLOADS:
        return False
    return True


def _build_specialized(rng: random.Random, n: int, seed: int, split: str = "train") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    # 1) Explicit echo-targeted correction for known weak patterns.
    echo_target_n = max(1, int(n * 0.35))
    out.extend(_build_targeted_echo_set(rng, echo_target_n, split=split))

    # 2) Mixed echo/anti-echo generation.
    echo_pairs = generate_echo_pairs(
        max_examples=max(1000, n),
        seed=seed + (17 if split == "train" else 17017),
        gibberish_mode="include",
        anti_echo_ratio=0.35,
        contrast_ratio=0.20,
    )
    for i, (q, a, cat) in enumerate(echo_pairs):
        ep = create_echo_episode(q, a, i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_8_specialized"
        ep["_meta"]["source_stream"] = "echo_anti_echo"
        out.append(ep)
        if len(out) >= int(n * 0.75):
            break

    # 3) Operator correction with WRAP edge-case emphasis.
    quote_payloads = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"] if split == "train" else ["valpha", "vbravo", "vcharlie", "vdelta", "vecho", "vfoxtrot"]
    if split == "train":
        # WRAP-heavy in train to recover residual failures on delimiters.
        ops = [
            ("COPY", 'Repeat exactly: "{}"', "{}"),
            ("EXTRACT", 'Return text between quotes: "{}"', "{}"),
            ("WRAP", 'Wrap this in []: {}', "[{}]"),
            ("WRAP", "Wrap this in '' : {}", "'{}'"),
            ("WRAP", "Return {} wrapped by {{ and }}.", "{{{{{}}}}}"),
        ]
    else:
        ops = [
            ("COPY", 'Reply with only "{}"', "{}"),
            ("EXTRACT", 'Return plain text from "{}"', "{}"),
            ("WRAP", "Enclose exactly in []: {}", "[{}]"),
            ("WRAP", "Return {} wrapped by {{ and }}.", "{{{{{}}}}}"),
            ("WRAP", "Wrap this in '' : {}", "'{}'"),
        ]
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


def _make_val_operator_episode(op: str, payload: str) -> Dict[str, Any]:
    if op == "COPY":
        q = f'VALCHECK: Output only payload: "{payload}"'
        a = payload
    elif op == "EXTRACT":
        q = f'VALCHECK: Dequote and return plain text from "{payload}"'
        a = payload
    else:  # WRAP
        q = f"VALCHECK: Enclose exactly in []: {payload}"
        a = f"[{payload}]"
    ep = create_operator_episode(q, a, "en")
    ep["_meta"] = {
        "operator": op,
        "payload": payload,
        "expected": a,
        "phase": "phase2_8_specialized",
        "source_stream": "operators",
    }
    return ep


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare phase2.8 echo rebalance bridge dataset")
    p.add_argument("--output_dir", type=str, default="data/sft_phase2_8_intermediate")
    p.add_argument("--replay_file", type=str, required=True, help="Broad replay source jsonl")
    p.add_argument("--target_train_size", type=int, default=40000)
    p.add_argument("--val_size", type=int, default=3000)
    p.add_argument("--min_val_per_operator", type=int, default=6, help="Target minimum val examples per operator family (COPY/WRAP/EXTRACT)")
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
    order = list(range(len(replay_rows)))
    rng.shuffle(order)
    replay_take_idx = set(order[: min(half, len(order))])
    replay_take = [replay_rows[i] for i in replay_take_idx]
    replay_holdout = [replay_rows[i] for i in range(len(replay_rows)) if i not in replay_take_idx]
    for ep in replay_take:
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_8_replay"
        ep["_meta"]["source_stream"] = "replay"
    for ep in replay_holdout:
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = "phase2_8_replay_val_pool"
        ep["_meta"]["source_stream"] = "replay"

    specialized = _build_specialized(rng, args.target_train_size - len(replay_take), args.seed, split="train")
    specialized = [e for e in specialized if _is_unseen_safe(e)]
    # top-up specialized if filtering removed rows
    while len(specialized) < (args.target_train_size - len(replay_take)):
        specialized.extend(_build_specialized(rng, 256, args.seed + len(specialized), split="train"))
        specialized = [e for e in specialized if _is_unseen_safe(e)]
    specialized = specialized[: args.target_train_size - len(replay_take)]

    train_rows = replay_take + specialized
    if args.shuffle:
        rng.shuffle(train_rows)

    # Build val from disjoint pools (NEVER slice from train)
    train_payloads = {_payload(e) for e in train_rows if _payload(e)}
    train_templates = {_template_signature(e) for e in train_rows}
    train_pairs = {_pair_key(e) for e in train_rows if _pair_key(e)}

    val_rows: List[Dict[str, Any]] = []

    def can_add_to_val(ep: Dict[str, Any]) -> bool:
        pld = _payload(ep)
        sig = _template_signature(ep)
        pr = _pair_key(ep)
        if pld and pld in train_payloads:
            return False
        if sig in train_templates:
            return False
        if pr and pr in train_pairs:
            return False
        return True

    # Reserve a minimum operator coverage in val first.
    min_per_op = max(2, int(args.min_val_per_operator))
    required_ops = ["COPY", "WRAP", "EXTRACT"]
    val_op_counts = {k: 0 for k in required_ops}
    op_idx = 0
    for op in required_ops:
        attempts = 0
        while val_op_counts[op] < min_per_op and attempts < 20000:
            attempts += 1
            payload = f"valop_{op.lower()}_{args.seed}_{op_idx}"
            op_idx += 1
            ep = _make_val_operator_episode(op, payload)
            if not _is_unseen_safe(ep):
                continue
            if can_add_to_val(ep):
                val_rows.append(ep)
                val_op_counts[op] += 1
        if val_op_counts[op] < 2:
            raise RuntimeError(
                f"Could not satisfy minimum val operator floor for {op}: "
                f"{val_op_counts[op]} < 2 after {attempts} attempts."
            )

    # First try replay holdout rows for broadness
    for ep in replay_holdout:
        if len(val_rows) >= args.val_size:
            break
        if can_add_to_val(ep):
            val_rows.append(ep)

    # Top up with val-specialized generator until full
    gen_attempt = 0
    while len(val_rows) < args.val_size and gen_attempt < 200:
        gen_attempt += 1
        batch = _build_specialized(rng, max(256, args.val_size - len(val_rows)), args.seed + 5000 + gen_attempt, split="val")
        for ep in batch:
            if len(val_rows) >= args.val_size:
                break
            if not _is_unseen_safe(ep):
                continue
            if can_add_to_val(ep):
                val_rows.append(ep)

    if len(val_rows) < args.val_size:
        raise RuntimeError(
            f"Could not build disjoint val set: got {len(val_rows)} / {args.val_size}. "
            "Increase source diversity or reduce val_size."
        )

    # Compute val uniqueness metrics for reporting.
    val_payloads = {_payload(e) for e in val_rows if _payload(e)}
    val_templates = {_template_signature(e) for e in val_rows}
    val_pairs = {_pair_key(e) for e in val_rows if _pair_key(e)}
    for ep in val_rows:
        op = _operator(ep)
        if op in val_op_counts:
            val_op_counts[op] = val_op_counts.get(op, 0) + 0  # keep declared keys explicit
    val_op_measured = {k: 0 for k in required_ops}
    for ep in val_rows:
        op = _operator(ep)
        if op in val_op_measured:
            val_op_measured[op] += 1

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
            "phase2_8b_bias_fix": True,
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
            "train_payloads": len(train_payloads),
            "val_payloads": len(val_payloads),
            "train_templates": len(train_templates),
            "val_templates": len(val_templates),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "val_operator_counts": val_op_measured,
            "val_operator_floor_target": min_per_op,
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

