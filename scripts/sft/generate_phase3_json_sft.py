#!/usr/bin/env python3
"""
Generate strict JSON-output episodes for Phase 3.1 corrective training.

Focus:
- exact JSON output without prose/markdown
- stable schema patterns (label/score and related variants)
- EN/DE prompt variants
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_lineage import iso_now, write_lineage_sidecar
from core.system_prompts import CHAT_SYSTEM_PROMPTS


def _episode(system: str, user: str, assistant_json: str, language: str, category: str) -> Dict:
    return {
        "system": system,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_json},
        ],
        "language": language,
        "source": "phase3_json_strict",
        "_meta": {
            "category": category,
            "source_stream": "phase3_json",
        },
    }


SCHEMA_SUFFIX_EN = [
    "Return JSON only. No markdown. No explanation.",
    "Output only the JSON object.",
    "Do not add any extra text.",
]

SCHEMA_SUFFIX_DE = [
    "Gib nur JSON aus. Kein Markdown. Keine Erklaerung.",
    "Nur das JSON-Objekt ausgeben.",
    "Keinen zusaetzlichen Text ausgeben.",
]


def _sample_label_score(rng: random.Random, language: str) -> Tuple[str, str, str]:
    rows = [
        ("A", 1),
        ("B", 0),
        ("A", 3),
        ("C", 2),
    ]
    label, score = rng.choice(rows)
    if language == "de":
        user = (
            f"Erstelle ein JSON-Objekt mit den Schluesseln label (String) und score (Zahl). "
            f"label={label}, score={score}. {rng.choice(SCHEMA_SUFFIX_DE)}"
        )
    else:
        user = (
            f"Create a JSON object with keys label (string) and score (number). "
            f"label={label}, score={score}. {rng.choice(SCHEMA_SUFFIX_EN)}"
        )
    assistant = json.dumps({"label": label, "score": score}, ensure_ascii=False, separators=(",", ":"))
    return user, assistant, "json_label_score"


def _sample_bool_reason(rng: random.Random, language: str) -> Tuple[str, str, str]:
    rows = [
        (True, "safe"),
        (False, "needs_review"),
        (True, "compliant"),
        (False, "missing_data"),
    ]
    allowed, reason = rng.choice(rows)
    if language == "de":
        user = (
            f"Erzeuge JSON mit allowed (Boolean) und reason (String). "
            f"allowed={str(allowed).lower()}, reason={reason}. {rng.choice(SCHEMA_SUFFIX_DE)}"
        )
    else:
        user = (
            f"Produce JSON with allowed (boolean) and reason (string). "
            f"allowed={str(allowed).lower()}, reason={reason}. {rng.choice(SCHEMA_SUFFIX_EN)}"
        )
    assistant = json.dumps({"allowed": allowed, "reason": reason}, ensure_ascii=False, separators=(",", ":"))
    return user, assistant, "json_allowed_reason"


def _sample_list_extract(rng: random.Random, language: str) -> Tuple[str, str, str]:
    pools = [
        ["tcp", "syn", "ack"],
        ["copy", "wrap", "extract"],
        ["alpha", "beta", "gamma"],
    ]
    items = rng.choice(pools)
    csv = ",".join(items)
    if language == "de":
        user = (
            f"Konvertiere in JSON mit key items (Liste von Strings). "
            f"Eingabe: {csv}. {rng.choice(SCHEMA_SUFFIX_DE)}"
        )
    else:
        user = (
            f"Convert to JSON with key items (array of strings). "
            f"Input: {csv}. {rng.choice(SCHEMA_SUFFIX_EN)}"
        )
    assistant = json.dumps({"items": items}, ensure_ascii=False, separators=(",", ":"))
    return user, assistant, "json_items"


def _sample_triplet(rng: random.Random, language: str) -> Tuple[str, str, str]:
    rows = [
        ("A", 1, "safe"),
        ("B", 0, "unsafe"),
        ("C", 2, "review"),
    ]
    label, score, reason = rng.choice(rows)
    if language == "de":
        user = (
            "Erzeuge genau ein JSON-Objekt mit den Schluesseln label (String), score (Zahl), reason (String). "
            f"Daten: label={label}, score={score}, reason={reason}. {rng.choice(SCHEMA_SUFFIX_DE)}"
        )
    else:
        user = (
            "Return exactly one JSON object with keys label (string), score (number), reason (string). "
            f"Data: label={label}, score={score}, reason={reason}. {rng.choice(SCHEMA_SUFFIX_EN)}"
        )
    assistant = json.dumps(
        {"label": label, "score": score, "reason": reason},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return user, assistant, "json_triplet"


def _sample_schema_contract(rng: random.Random, language: str) -> Tuple[str, str, str]:
    rows = [
        ("network", "pass", 0.92),
        ("policy", "fail", 0.11),
        ("safety", "pass", 0.78),
    ]
    check, status, confidence = rng.choice(rows)
    if language == "de":
        user = (
            "Formatiere als JSON mit exakt diesen Schluesseln: check, status, confidence. "
            "confidence muss eine Zahl sein. "
            f"Werte: check={check}, status={status}, confidence={confidence}. {rng.choice(SCHEMA_SUFFIX_DE)}"
        )
    else:
        user = (
            "Format as JSON with exactly these keys: check, status, confidence. "
            "confidence must be numeric. "
            f"Values: check={check}, status={status}, confidence={confidence}. {rng.choice(SCHEMA_SUFFIX_EN)}"
        )
    assistant = json.dumps(
        {"check": check, "status": status, "confidence": confidence},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return user, assistant, "json_schema_contract"


def generate_rows(num_examples: int, seed: int, de_ratio: float) -> List[Dict]:
    rng = random.Random(seed)
    systems = CHAT_SYSTEM_PROMPTS
    builders = [
        _sample_label_score,
        _sample_bool_reason,
        _sample_list_extract,
        _sample_triplet,
        _sample_schema_contract,
    ]
    out: List[Dict] = []
    for _ in range(num_examples):
        language = "de" if rng.random() < de_ratio else "en"
        user, assistant, category = rng.choice(builders)(rng, language)
        out.append(_episode(rng.choice(systems), user, assistant, language, category))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate strict JSON-output episodes for Phase 3.1")
    ap.add_argument("--output", type=str, default="data/sft_phase3_intermediate/phase3_json_strict.jsonl")
    ap.add_argument("--num_examples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=3311)
    ap.add_argument("--de_ratio", type=float, default=0.35)
    args = ap.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(args.num_examples, args.seed, args.de_ratio)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "created_at": iso_now(),
        "output": str(out_path),
        "num_examples": len(rows),
        "seed": args.seed,
        "de_ratio": args.de_ratio,
        "source": "phase3_json_strict",
        "categories": {
            "json_label_score": sum(1 for r in rows if r["_meta"]["category"] == "json_label_score"),
            "json_allowed_reason": sum(1 for r in rows if r["_meta"]["category"] == "json_allowed_reason"),
            "json_items": sum(1 for r in rows if r["_meta"]["category"] == "json_items"),
            "json_triplet": sum(1 for r in rows if r["_meta"]["category"] == "json_triplet"),
            "json_schema_contract": sum(1 for r in rows if r["_meta"]["category"] == "json_schema_contract"),
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    lineage = {
        "direct_inputs": [],
        "recursive_origins": [{"origin_path": "synthetic://generate_phase3_json_sft", "rows": len(rows)}],
        "flattened_contributions": [
            {
                "origin_path": "synthetic://generate_phase3_json_sft",
                "effective_rows": len(rows),
                "effective_percent": 100.0,
            }
        ],
        "creation_context": {
            "timestamp": iso_now(),
            "script": "scripts/sft/generate_phase3_json_sft.py",
            "args": vars(args),
        },
    }
    write_lineage_sidecar(out_path, lineage)
    print(f"Wrote {len(rows):,} rows -> {out_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
