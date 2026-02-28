#!/usr/bin/env python3
"""
Audit Phase 3 JSONL datasets for schema quality and curriculum coverage.

This script is read-only and reports per-source and global metrics:
- schema compliance
- turn-depth distribution
- checkable vs open-ended task ratio (heuristic)
- conflict / hierarchy / injection coverage (heuristic)
- abstention coverage
- strict formatting-constraint coverage
- maintenance coverage (operators / anti-echo / grounded QA)
- context->cite linkage
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


CHECKABLE_RE = re.compile(
    r"(return only|extract|classify|convert|summari[sz]e|answer using only|json|output exactly|reply with only|no markdown)",
    re.IGNORECASE,
)
CONFLICT_RE = re.compile(
    r"(ignore (all|previous|prior)|system (forbids|says)|conflict|do not|must not|forbidden|prompt injection)",
    re.IGNORECASE,
)
ABSTAIN_RE = re.compile(
    r"(don't have enough information|do not have enough information|not enough information|unknown|cannot determine|can't determine|unclear|conflicting passages)",
    re.IGNORECASE,
)
STRICT_FMT_RE = re.compile(
    r"(return only|no quotes|no punctuation|no markdown|json|exactly|output only|schema)",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _source_name(ep: Dict[str, Any]) -> str:
    return str(ep.get("mix_source") or ep.get("source") or "unknown")


def _analyze_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    system = ep.get("system", "")
    messages = ep.get("messages", [])
    valid_messages = isinstance(messages, list) and len(messages) > 0

    has_user = False
    has_assistant = False
    user_turns = 0
    assistant_turns = 0
    has_context = False
    has_cite = False
    has_think = False
    context_turns = 0
    context_turns_with_cite = 0

    checkable_hits = 0
    conflict_hits = 0
    abstain_hits = 0
    strict_fmt_hits = 0
    anti_echo_hits = 0
    operator_hits = 0
    grounded_hits = 0

    if not valid_messages:
        return {
            "schema_ok": False,
            "assistant_turns": 0,
            "multi_turn_class": "invalid",
            "user_turns": 0,
            "has_context": False,
            "has_cite": False,
            "has_think": False,
            "checkable": False,
            "conflict": False,
            "abstain": False,
            "strict_fmt": False,
            "anti_echo": False,
            "operators": False,
            "grounded_qa": False,
            "context_turns": 0,
            "context_turns_with_cite": 0,
        }

    # top-level schema
    schema_ok = isinstance(system, str) and bool(system.strip()) and valid_messages

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            schema_ok = False
            continue
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "user":
            has_user = True
            user_turns += 1
            if str(msg.get("context", "")).strip():
                has_context = True
                context_turns += 1
                # Next assistant has cite?
                for j in range(idx + 1, len(messages)):
                    nxt = messages[j]
                    if not isinstance(nxt, dict):
                        continue
                    if nxt.get("role") != "assistant":
                        continue
                    if str(nxt.get("cite", "")).strip():
                        context_turns_with_cite += 1
                    break
            if CHECKABLE_RE.search(content):
                checkable_hits += 1
            if CONFLICT_RE.search(content):
                conflict_hits += 1
            if STRICT_FMT_RE.search(content):
                strict_fmt_hits += 1
            if re.search(r'".+?"', content) and re.search(r"\b(what|mean|real word)\b", content, re.IGNORECASE):
                anti_echo_hits += 1
        elif role == "assistant":
            has_assistant = True
            assistant_turns += 1
            if str(msg.get("cite", "")).strip():
                has_cite = True
                grounded_hits += 1
            if str(msg.get("think", "")).strip():
                has_think = True
            if ABSTAIN_RE.search(content):
                abstain_hits += 1
            if re.search(r"^\[.*\]$|^<.*>$|^'.*'$|^\".*\"$", content.strip()):
                operator_hits += 1
        elif role == "assistant_context":
            # allowed but masked role
            pass
        else:
            schema_ok = False

    if not has_user or not has_assistant:
        schema_ok = False

    if assistant_turns <= 1:
        multi_turn_class = "2turn_or_less"
    elif assistant_turns <= 2:
        multi_turn_class = "3to4_turn_dialogue"
    else:
        multi_turn_class = "5plus_turn_dialogue"

    return {
        "schema_ok": schema_ok,
        "assistant_turns": assistant_turns,
        "multi_turn_class": multi_turn_class,
        "user_turns": user_turns,
        "has_context": has_context,
        "has_cite": has_cite,
        "has_think": has_think,
        "checkable": checkable_hits > 0,
        "conflict": conflict_hits > 0,
        "abstain": abstain_hits > 0,
        "strict_fmt": strict_fmt_hits > 0,
        "anti_echo": anti_echo_hits > 0,
        "operators": operator_hits > 0 or bool(ep.get("_meta", {}).get("operator")),
        "grounded_qa": grounded_hits > 0 and has_context,
        "context_turns": context_turns,
        "context_turns_with_cite": context_turns_with_cite,
    }


def _empty_stats() -> Dict[str, Any]:
    return {
        "episodes": 0,
        "schema_ok": 0,
        "assistant_turns_total": 0,
        "turn_depth": Counter(),
        "checkable": 0,
        "open_ended": 0,
        "conflict": 0,
        "abstain": 0,
        "strict_fmt": 0,
        "anti_echo": 0,
        "operators": 0,
        "grounded_qa": 0,
        "has_context": 0,
        "has_cite": 0,
        "has_think": 0,
        "context_turns": 0,
        "context_turns_with_cite": 0,
    }


def _accumulate(stats: Dict[str, Any], features: Dict[str, Any]) -> None:
    stats["episodes"] += 1
    stats["schema_ok"] += 1 if features["schema_ok"] else 0
    stats["assistant_turns_total"] += int(features["assistant_turns"])
    stats["turn_depth"][features["multi_turn_class"]] += 1
    stats["checkable"] += 1 if features["checkable"] else 0
    stats["open_ended"] += 1 if not features["checkable"] else 0
    stats["conflict"] += 1 if features["conflict"] else 0
    stats["abstain"] += 1 if features["abstain"] else 0
    stats["strict_fmt"] += 1 if features["strict_fmt"] else 0
    stats["anti_echo"] += 1 if features["anti_echo"] else 0
    stats["operators"] += 1 if features["operators"] else 0
    stats["grounded_qa"] += 1 if features["grounded_qa"] else 0
    stats["has_context"] += 1 if features["has_context"] else 0
    stats["has_cite"] += 1 if features["has_cite"] else 0
    stats["has_think"] += 1 if features["has_think"] else 0
    stats["context_turns"] += int(features["context_turns"])
    stats["context_turns_with_cite"] += int(features["context_turns_with_cite"])


def _finalize(stats: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, stats["episodes"])
    cturns = max(1, stats["context_turns"])
    return {
        **stats,
        "schema_ok_rate": round(stats["schema_ok"] * 100.0 / n, 2),
        "checkable_rate": round(stats["checkable"] * 100.0 / n, 2),
        "open_ended_rate": round(stats["open_ended"] * 100.0 / n, 2),
        "conflict_rate": round(stats["conflict"] * 100.0 / n, 2),
        "abstain_rate": round(stats["abstain"] * 100.0 / n, 2),
        "strict_fmt_rate": round(stats["strict_fmt"] * 100.0 / n, 2),
        "operators_rate": round(stats["operators"] * 100.0 / n, 2),
        "anti_echo_rate": round(stats["anti_echo"] * 100.0 / n, 2),
        "grounded_qa_rate": round(stats["grounded_qa"] * 100.0 / n, 2),
        "context_rate": round(stats["has_context"] * 100.0 / n, 2),
        "cite_rate": round(stats["has_cite"] * 100.0 / n, 2),
        "think_rate": round(stats["has_think"] * 100.0 / n, 2),
        "avg_assistant_turns": round(stats["assistant_turns_total"] / n, 3),
        "context_to_cite_rate": round(stats["context_turns_with_cite"] * 100.0 / cturns, 2),
        "turn_depth": dict(stats["turn_depth"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 3 JSONL datasets")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL (typically phase3_mixed.jsonl)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON report")
    parser.add_argument("--top_sources", type=int, default=15, help="Top N sources to print")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    episodes = _read_jsonl(in_path)
    global_stats = _empty_stats()
    source_stats: Dict[str, Dict[str, Any]] = defaultdict(_empty_stats)

    for ep in episodes:
        f = _analyze_episode(ep)
        _accumulate(global_stats, f)
        _accumulate(source_stats[_source_name(ep)], f)

    final_global = _finalize(global_stats)
    final_sources = {k: _finalize(v) for k, v in source_stats.items()}
    sorted_sources = sorted(final_sources.items(), key=lambda kv: kv[1]["episodes"], reverse=True)

    print("=" * 70)
    print("Phase 3 Dataset Audit")
    print("=" * 70)
    print(f"Input: {in_path}")
    print(f"Episodes: {final_global['episodes']:,}")
    print(f"Schema OK: {final_global['schema_ok_rate']:.2f}%")
    print(
        f"Checkable/Open: {final_global['checkable_rate']:.2f}% / "
        f"{final_global['open_ended_rate']:.2f}%"
    )
    print(
        f"Conflict={final_global['conflict_rate']:.2f}% | "
        f"Abstain={final_global['abstain_rate']:.2f}% | "
        f"StrictFmt={final_global['strict_fmt_rate']:.2f}%"
    )
    print(
        f"Maintenance: operators={final_global['operators_rate']:.2f}% | "
        f"anti_echo={final_global['anti_echo_rate']:.2f}% | "
        f"grounded_qa={final_global['grounded_qa_rate']:.2f}%"
    )
    print(
        f"Context/Cite/Think: {final_global['context_rate']:.2f}% / "
        f"{final_global['cite_rate']:.2f}% / {final_global['think_rate']:.2f}%"
    )
    print(f"context->cite linkage: {final_global['context_to_cite_rate']:.2f}%")
    print(f"Turn depth: {final_global['turn_depth']}")
    print(f"Avg assistant turns: {final_global['avg_assistant_turns']:.3f}")

    print("\nTop sources:")
    for src, st in sorted_sources[: args.top_sources]:
        print(
            f"- {src}: n={st['episodes']:,}, checkable={st['checkable_rate']:.1f}%, "
            f"conflict={st['conflict_rate']:.1f}%, abstain={st['abstain_rate']:.1f}%, "
            f"strict={st['strict_fmt_rate']:.1f}%, ctx->cite={st['context_to_cite_rate']:.1f}%"
        )

    report = {
        "input_file": str(in_path),
        "global": final_global,
        "by_source": final_sources,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()

