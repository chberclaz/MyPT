#!/usr/bin/env python3
"""
Build a unified Phase-2 dataset from all Phase-2 lineage sources (2.0-2.8).

Goal:
- optimize ABCE (format/echo/anti-echo/operators) for 100% gate pass
- improve regression_basic trend without drifting into over-refusal
- enforce strict train/val disjointness and validation diversity floors
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sft.generate_echo_dataset import create_episode as create_echo_episode
from scripts.sft.generate_operator_dataset import create_episode as create_operator_episode
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


def _sample(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if not rows:
        return []
    if len(rows) >= n:
        return rng.sample(rows, n)
    # With replacement
    return [rows[rng.randrange(len(rows))] for _ in range(n)]


def _user_text(ep: Dict[str, Any]) -> str:
    for m in ep.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def _extract_payload_from_text(user_msg: str) -> Optional[str]:
    quote_match = re.search(r'"([^"]+)"', user_msg)
    if quote_match:
        return quote_match.group(1)
    colon_match = re.search(r':\s*(.+)$', user_msg)
    if colon_match:
        return colon_match.group(1).strip()
    return None


def _payload(ep: Dict[str, Any]) -> str:
    # Align exactly with prepare_chat_sft.py extraction semantics.
    meta = ep.get("_meta", {})
    if "payload" in meta:
        v = meta["payload"]
        return str(v) if v is not None else ""
    user = _user_text(ep)
    inferred = _extract_payload_from_text(user)
    return inferred if inferred is not None else ""


def _operator(ep: Dict[str, Any]) -> str:
    op = str(ep.get("_meta", {}).get("operator", "")).strip().upper()
    if op:
        return op
    u = _user_text(ep).lower()
    if "wrap" in u or "[" in u or "{" in u or "enclose" in u:
        return "WRAP"
    if "extract" in u or "quote" in u:
        return "EXTRACT"
    if "say " in u or "echo" in u or "repeat" in u:
        return "COPY"
    return "UNKNOWN"


def _pair_key(ep: Dict[str, Any]) -> str:
    p = _payload(ep)
    return f"{_operator(ep)}::{p}" if p else ""


def _template_sig(ep: Dict[str, Any]) -> str:
    # Align exactly with prepare_chat_sft.py get_template_signature().
    user = _user_text(ep)
    if not user:
        return ""
    payload = _payload(ep)
    if payload and payload in user:
        sig = user.replace(f'"{payload}"', '"{PAYLOAD}"')
        sig = sig.replace(payload, "{PAYLOAD}")
        return " ".join(sig.split())
    return " ".join(user.split())


def _mark_source(rows: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ep in rows:
        e = json.loads(json.dumps(ep))
        e.setdefault("_meta", {})
        e["_meta"]["unified_source"] = source_name
        out.append(e)
    return out


def _make_echo_exact_rows(n: int, split: str) -> List[Dict[str, Any]]:
    payloads = [
        "Hello", "OK", "yes", "Banana", "Hallo", "42", "17",
        "blue ocean", "dark forest", "red mountain peak",
    ]
    train_tpl = ["Say {}.", "Echo: {}", "Repeat exactly: {}", "Output only: {}", "Sag {}."]
    val_tpl = [
        "Reply with only: {}", "State exactly: {}", "Gib exakt aus: {}",
        "No extra words. Return: {}", "Return exactly this: {}",
        "Answer strictly with: {}", "Echo verbatim: {}",
    ]
    tpls = train_tpl if split == "train" else val_tpl
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        p = payloads[i % len(payloads)]
        q = tpls[i % len(tpls)].format(p)
        ep = create_echo_episode(q, p, i, f"echo_exact_{split}")
        ep.setdefault("_meta", {})
        ep["_meta"]["operator"] = "ECHO"
        ep["_meta"]["payload"] = p
        ep["_meta"]["unified_source"] = f"synthetic_echo_exact_{split}"
        rows.append(ep)
    return rows


def _make_wrap_edge_rows(n: int, split: str) -> List[Dict[str, Any]]:
    payloads = ["cascade", "ancient stone bridge", "cold winter night", "twilight", "dragon"]
    tpls_train = [
        ("WRAP", "Wrap this in '' : {}", "'{}'"),
        ("WRAP", "Return {} wrapped by {{ and }}.", "{{{{{}}}}}"),
        ("WRAP", "Wrap this in []: {}", "[{}]"),
        ("WRAP", "Wrap this in \"\" : {}", "\"{}\""),
    ]
    tpls_val = [
        ("WRAP", "Enclose exactly in '' : {}", "'{}'"),
        ("WRAP", "Return {} wrapped by {{ and }}.", "{{{{{}}}}}"),
        ("WRAP", "Enclose exactly in []: {}", "[{}]"),
        ("WRAP", "Wrap in double quotes: {}", "\"{}\""),
    ]
    tpls = tpls_train if split == "train" else tpls_val
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        payload = payloads[i % len(payloads)] + (f"_{i}" if i % 5 == 0 else "")
        op, q_tpl, a_tpl = tpls[i % len(tpls)]
        q = q_tpl.format(payload)
        a = a_tpl.format(payload)
        ep = create_operator_episode(q, a, "en")
        ep["_meta"] = {
            "operator": op,
            "payload": payload,
            "expected": a,
            "unified_source": f"synthetic_wrap_edge_{split}",
        }
        rows.append(ep)
    return rows


def _make_boundary_rows(n: int, split: str) -> List[Dict[str, Any]]:
    # Near-neighbor echo vs anti-echo boundaries.
    tokens = ["Blurpix", "Zanthor", "Quexling", "Flimzap", "Norbex"]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        tok = tokens[i % len(tokens)] + (f"_{i}" if i % 7 == 0 else "")
        if i % 2 == 0:
            q = f'Say "{tok}".'
            a = tok
            cat = f"boundary_echo_{split}"
        else:
            q = f'What is "{tok}"?'
            a = "Unknown."
            cat = f"boundary_antiecho_{split}"
        ep = create_echo_episode(q, a, i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["operator"] = "BOUNDARY"
        ep["_meta"]["payload"] = tok
        ep["_meta"]["unified_source"] = f"synthetic_boundary_{split}"
        rows.append(ep)
    return rows


def _make_antiecho_hard_rows(n: int, split: str) -> List[Dict[str, Any]]:
    # Focused anti-echo negatives that should never be copied.
    gibberish = ["Blurpix", "Zanthor", "Qevlor", "Nimvrax", "Trandok", "Vexlune"]
    train_tpl = [
        'What does "{}" mean?',
        'Define "{}".',
        'Is "{}" a real word?',
        'Explain "{}" in one word.',
        'Translate "{}" to German.',
    ]
    val_tpl = [
        'Give the meaning of "{}".',
        'Provide a short definition for "{}".',
        'What is "{}"?',
        'Is "{}" meaningful?',
        'Translate "{}" to English.',
    ]
    tpls = train_tpl if split == "train" else val_tpl
    abstain = ["Unknown.", "No.", "I don't know."]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        tok = gibberish[i % len(gibberish)] + (f"_{i}" if i % 3 == 0 else "")
        q = tpls[i % len(tpls)].format(tok)
        a = abstain[i % len(abstain)]
        ep = create_echo_episode(q, a, i, f"antiecho_hard_{split}")
        ep.setdefault("_meta", {})
        ep["_meta"]["operator"] = "ANTI_ECHO"
        ep["_meta"]["payload"] = tok
        ep["_meta"]["unified_source"] = f"synthetic_antiecho_hard_{split}"
        rows.append(ep)
    return rows


def _make_regression_rows(n: int, split: str) -> List[Dict[str, Any]]:
    qa = [
        ("What is 5 + 7?", "12"),
        ("What is 2 + 2?", "4"),
        ("What is 10 - 3?", "7"),
        ("Capital of Germany?", "Berlin"),
        ("Capital of France?", "Paris"),
        ("Is water wet?", "Yes"),
        ("Is fire cold?", "No"),
    ]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        q, a = qa[i % len(qa)]
        if split == "val":
            q = f"Answer exactly: {q}"
        ep = {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "_meta": {
                "operator": "REGRESSION",
                "payload": q.lower(),
                "expected": a,
                "unified_source": f"synthetic_regression_{split}",
            },
        }
        rows.append(ep)
    return rows


def _make_val_operator_episode(op: str, idx: int) -> Dict[str, Any]:
    payload = f"val_unseen_{op.lower()}_{idx}"
    if op == "COPY":
        q = f"Return exactly this payload: {payload}"
        a = payload
    elif op == "WRAP":
        q = f"Wrap this payload in square brackets: {payload}"
        a = f"[{payload}]"
    elif op == "EXTRACT":
        q = f'Return only the content inside quotes: "{payload}"'
        a = payload
    else:
        q = f"Repeat exactly: {payload}"
        a = payload
    ep = create_operator_episode(q, a, "en")
    ep["_meta"] = {
        "operator": op,
        "payload": payload,
        "expected": a,
        "unified_source": "synthetic_operator_floor_val",
    }
    return ep


def _take_disjoint(
    pool: List[Dict[str, Any]],
    n: int,
    train_payloads: set,
    train_templates: set,
    train_pairs: set,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    out: List[Dict[str, Any]] = []
    for i in idx:
        if len(out) >= n:
            break
        ep = pool[i]
        p = _payload(ep)
        t = _template_sig(ep)
        pr = _pair_key(ep)
        if p and p in train_payloads:
            continue
        if t and t in train_templates:
            continue
        if pr and pr in train_pairs:
            continue
        out.append(ep)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare unified Phase-2 rebuild dataset")
    p.add_argument("--output_dir", type=str, default="data/sft_phase2_unified_intermediate")
    p.add_argument("--phase2_file", type=str, default="data/sft_phase2_intermediate/operators/operator_train.jsonl")
    p.add_argument("--phase2_5_file", type=str, default="data/sft_phase2_5_intermediate/phase2_5_mixed.jsonl")
    p.add_argument("--phase2_6_file", type=str, default="data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl")
    p.add_argument("--phase2_7_file", type=str, default="data/sft_phase2_7_intermediate/phase2_7_mixed.jsonl")
    p.add_argument("--phase2_8_file", type=str, default="data/sft_phase2_8b_intermediate/phase2_8_mixed_train.jsonl")
    p.add_argument("--phase1_replay_file", type=str, default="data/sft_phase1_intermediate/phase1_mixed.jsonl")
    p.add_argument("--target_train_size", type=int, default=160000)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2901)
    p.add_argument("--allow_missing_sources", action="store_true", default=False)
    p.add_argument("--min_val_operator_floor", type=int, default=20)
    p.add_argument("--min_val_template_diversity", type=int, default=80)
    args = p.parse_args()

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    src_specs = [
        ("phase2_base", args.phase2_file, 0.20),
        ("phase2_5", args.phase2_5_file, 0.15),
        ("phase2_6", args.phase2_6_file, 0.08),
        ("phase2_7", args.phase2_7_file, 0.15),
        ("phase2_8", args.phase2_8_file, 0.10),
        ("phase1_replay", args.phase1_replay_file, 0.02),
    ]
    synth_specs = [
        ("synthetic_echo_exact", 0.12),
        ("synthetic_wrap_edge", 0.08),
        ("synthetic_antiecho_hard", 0.05),
        ("synthetic_boundary", 0.05),
        ("synthetic_regression", 0.05),
    ]

    source_rows: Dict[str, List[Dict[str, Any]]] = {}
    missing = []
    for name, rel_path, _ in src_specs:
        path = Path(rel_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            missing.append((name, str(path)))
            continue
        rows = _mark_source(_read_jsonl(path), name)
        source_rows[name] = rows

    if missing and not args.allow_missing_sources:
        miss_msg = "\n".join([f"- {n}: {p}" for n, p in missing])
        raise FileNotFoundError(f"Missing required phase2 source files:\n{miss_msg}\nUse --allow_missing_sources to continue with renormalization.")

    # Renormalize available source weights.
    available_specs = [(n, p, w) for (n, p, w) in src_specs if n in source_rows]
    w_sum = sum(w for _, _, w in available_specs) + sum(w for _, w in synth_specs)
    if w_sum <= 0:
        raise RuntimeError("No data sources available for unified phase2 build.")

    train_target = args.target_train_size
    train_rows: List[Dict[str, Any]] = []
    composition_counts: Dict[str, int] = {}

    for name, _, w in available_specs:
        n = int(round(train_target * (w / w_sum)))
        picked = _sample(source_rows[name], n, rng)
        train_rows.extend(picked)
        composition_counts[name] = len(picked)

    for name, w in synth_specs:
        n = int(round(train_target * (w / w_sum)))
        if name == "synthetic_echo_exact":
            rows = _make_echo_exact_rows(n, split="train")
        elif name == "synthetic_wrap_edge":
            rows = _make_wrap_edge_rows(n, split="train")
        elif name == "synthetic_antiecho_hard":
            rows = _make_antiecho_hard_rows(n, split="train")
        elif name == "synthetic_boundary":
            rows = _make_boundary_rows(n, split="train")
        else:
            rows = _make_regression_rows(n, split="train")
        train_rows.extend(rows)
        composition_counts[name] = len(rows)

    if len(train_rows) > train_target:
        train_rows = _sample(train_rows, train_target, rng)
    elif len(train_rows) < train_target:
        topup = _sample(train_rows, train_target - len(train_rows), rng)
        train_rows.extend(topup)

    rng.shuffle(train_rows)

    # Build validation from all sources + synthetic val variants, disjoint against train.
    train_payloads = {_payload(e) for e in train_rows if _payload(e)}
    train_templates = {_template_sig(e) for e in train_rows if _template_sig(e)}
    train_pairs = {_pair_key(e) for e in train_rows if _pair_key(e)}

    val_pool: List[Dict[str, Any]] = []
    for name in source_rows:
        val_pool.extend(_mark_source(source_rows[name], f"{name}_valpool"))
    val_pool.extend(_make_echo_exact_rows(args.val_size, split="val"))
    val_pool.extend(_make_wrap_edge_rows(args.val_size, split="val"))
    val_pool.extend(_make_antiecho_hard_rows(args.val_size, split="val"))
    val_pool.extend(_make_boundary_rows(args.val_size, split="val"))
    val_pool.extend(_make_regression_rows(args.val_size, split="val"))

    val_rows = _take_disjoint(
        val_pool,
        n=args.val_size,
        train_payloads=train_payloads,
        train_templates=train_templates,
        train_pairs=train_pairs,
        rng=rng,
    )
    if len(val_rows) < args.val_size:
        raise RuntimeError(f"Could not build val_size={args.val_size} with strict disjointness; got {len(val_rows)}")

    # Val quality floors.
    val_templates = {_template_sig(e) for e in val_rows if _template_sig(e)}
    if len(val_templates) < args.min_val_template_diversity:
        raise RuntimeError(
            f"Validation template diversity too low: {len(val_templates)} < {args.min_val_template_diversity}"
        )
    op_counts = {"COPY": 0, "WRAP": 0, "EXTRACT": 0}
    for ep in val_rows:
        op = _operator(ep)
        if op in op_counts:
            op_counts[op] += 1

    # Top-up missing operator classes with synthetic unseen val episodes.
    floor_cursor = 0
    for op in ("COPY", "WRAP", "EXTRACT"):
        while op_counts[op] < args.min_val_operator_floor:
            cand = _make_val_operator_episode(op, floor_cursor)
            floor_cursor += 1
            p = _payload(cand)
            t = _template_sig(cand)
            pr = _pair_key(cand)
            if p in train_payloads or t in train_templates or pr in train_pairs:
                continue
            val_rows.append(cand)
            op_counts[op] += 1
            if len(val_rows) > args.val_size:
                # keep deterministic size after top-up
                drop_idx = rng.randrange(0, len(val_rows) - 1)
                dropped = val_rows.pop(drop_idx)
                dop = _operator(dropped)
                if dop in op_counts:
                    op_counts[dop] = max(0, op_counts[dop] - 1)

    # Recompute template diversity after top-up/drop balancing.
    val_templates = {_template_sig(e) for e in val_rows if _template_sig(e)}
    if len(val_templates) < args.min_val_template_diversity:
        raise RuntimeError(
            f"Validation template diversity too low after top-up: {len(val_templates)} < {args.min_val_template_diversity}"
        )
    for op in ("COPY", "WRAP", "EXTRACT"):
        if op_counts[op] < args.min_val_operator_floor:
            raise RuntimeError(
                f"Validation operator floor failed: {op}={op_counts[op]} < {args.min_val_operator_floor}"
            )

    train_file = out_dir / "phase2_unified_train.jsonl"
    val_file = out_dir / "phase2_unified_val.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(val_file, val_rows)

    lineage_inputs = []
    for name, path, _ in available_specs:
        lineage_inputs.append({
            "path": str((PROJECT_ROOT / path).resolve()) if not Path(path).is_absolute() else str(Path(path).resolve()),
            "sampled_rows": composition_counts.get(name, 0),
            "effective_ratio": composition_counts.get(name, 0) / max(1, len(train_rows)),
        })
    lineage = merge_lineage(
        inputs=lineage_inputs,
        output_rows=len(train_rows),
        creation_context={
            "timestamp": iso_now(),
            "script": "scripts/sft/prepare_phase2_unified_rebuild.py",
            "args": vars(args),
            "mix_policy_version": "phase2_unified_v2_more_targeted_data",
        },
    )

    meta = {
        "phase": "phase2_unified_rebuild",
        "seed": args.seed,
        "target_train_size": args.target_train_size,
        "val_size": args.val_size,
        "composition_counts": composition_counts,
        "missing_sources": missing,
        "train_metrics": {
            "payloads": len({_payload(e) for e in train_rows if _payload(e)}),
            "templates": len({_template_sig(e) for e in train_rows if _template_sig(e)}),
            "pairs": len({_pair_key(e) for e in train_rows if _pair_key(e)}),
        },
        "val_metrics": {
            "payloads": len({_payload(e) for e in val_rows if _payload(e)}),
            "templates": len(val_templates),
            "pairs": len({_pair_key(e) for e in val_rows if _pair_key(e)}),
            "operator_counts": op_counts,
            "operator_floor": args.min_val_operator_floor,
            "template_diversity_floor": args.min_val_template_diversity,
        },
        "output": {
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
        "lineage": lineage,
    }
    meta_file = out_dir / "phase2_unified_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    lineage_path = write_lineage_sidecar(train_file, lineage)

    print("=== Phase2 unified rebuild ready ===")
    print(f"Train:   {train_file} ({len(train_rows):,})")
    print(f"Val:     {val_file} ({len(val_rows):,})")
    print(f"Meta:    {meta_file}")
    print(f"Lineage: {lineage_path}")
    print("Composition:")
    for k, v in sorted(composition_counts.items(), key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * v / max(1, len(train_rows))
        print(f"  - {k}: {v:,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()

