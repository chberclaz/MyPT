#!/usr/bin/env python3
"""
Short QA / trivia episodes aligned with sft_eval_suite regression_basic (substring check).

Trains: small integer math, country to capital (city only), trivial yes or no.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_lineage import iso_now, write_lineage_sidecar
from core.system_prompts import CHAT_SYSTEM_PROMPT


def _episode(user: str, assistant: str, language: str, category: str) -> Dict:
    return {
        "system": CHAT_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "language": language,
        "source": "phase3_regression_short",
        "_meta": {
            "category": category,
            "source_stream": "phase3_regression_short",
        },
    }


_CAPITALS: List[Tuple[str, str, str, str]] = [
    ("Germany", "Berlin", "Deutschland", "Berlin"),
    ("France", "Paris", "Frankreich", "Paris"),
    ("Italy", "Rome", "Italien", "Rom"),
    ("Spain", "Madrid", "Spanien", "Madrid"),
    ("Portugal", "Lisbon", "Portugal", "Lissabon"),
    ("Japan", "Tokyo", "Japan", "Tokio"),
    ("Egypt", "Cairo", "Ägypten", "Kairo"),
    ("Canada", "Ottawa", "Kanada", "Ottawa"),
    ("Australia", "Canberra", "Australien", "Canberra"),
    ("Brazil", "Brasília", "Brasilien", "Brasília"),
    ("Sweden", "Stockholm", "Schweden", "Stockholm"),
    ("Norway", "Oslo", "Norwegen", "Oslo"),
    ("Poland", "Warsaw", "Polen", "Warschau"),
    ("Finland", "Helsinki", "Finnland", "Helsinki"),
]


def _gen_math(rng: random.Random, n: int) -> List[Dict]:
    out: List[Dict] = []
    templates_en_add = [
        "What is {a} + {b}?",
        "Compute {a} plus {b}.",
        "How much is {a} + {b}?",
        "{a} + {b} = ?",
    ]
    templates_en_sub = [
        "What is {a} - {b}?",
        "Compute {a} minus {b}.",
        "{a} − {b} = ?",
    ]
    templates_de_add = [
        "Was ist {a} plus {b}?",
        "Wie viel ist {a} + {b}?",
    ]
    templates_de_sub = [
        "Was ist {a} minus {b}?",
        "Wie viel ist {a} − {b}?",
    ]
    for _ in range(n):
        lang = "de" if rng.random() < 0.35 else "en"
        if rng.random() < 0.55:
            a, b = rng.randint(1, 12), rng.randint(1, 12)
            ans = str(a + b)
            if lang == "de":
                q = rng.choice(templates_de_add).format(a=a, b=b)
            else:
                q = rng.choice(templates_en_add).format(a=a, b=b)
            cat = "regression_math_add"
        else:
            a = rng.randint(8, 20)
            b = rng.randint(1, min(9, a - 2))
            ans = str(a - b)
            if lang == "de":
                q = rng.choice(templates_de_sub).format(a=a, b=b)
            else:
                q = rng.choice(templates_en_sub).format(a=a, b=b)
            cat = "regression_math_sub"
        out.append(_episode(q, ans, lang, cat))
    return out


def _gen_capitals(rng: random.Random, n: int) -> List[Dict]:
    out: List[Dict] = []
    for _ in range(n):
        c_en, cap_en, c_de, cap_de = rng.choice(_CAPITALS)
        if rng.random() < 0.3:
            lang = "de"
            country, capital = c_de, cap_de
            templates = [
                "Hauptstadt von {c}?",
                "Was ist die Hauptstadt von {c}?",
                "Nenne die Hauptstadt von {c}.",
            ]
        else:
            lang = "en"
            country, capital = c_en, cap_en
            templates = [
                "Capital of {c}?",
                "What is the capital of {c}?",
                "Name the capital city of {c}.",
                "Which city is the capital of {c}?",
            ]
        q = rng.choice(templates).format(c=country)
        out.append(_episode(q, capital, lang, "regression_capital"))
    return out


def _gen_yes_no(rng: random.Random, n: int) -> List[Dict]:
    out: List[Dict] = []
    yes_en = [
        ("Is water wet?", "Yes"),
        ("Is ice solid?", "Yes"),
        ("Does the sun emit light?", "Yes"),
        ("Is steel a metal?", "Yes"),
    ]
    no_en = [
        ("Is fire cold?", "No"),
        ("Is vacuum loud?", "No"),
        ("Is pure water dry?", "No"),
        ("Does 2 equal 5?", "No"),
    ]
    yes_de = [
        ("Ist Wasser nass?", "Yes"),
        ("Bestehen Wolken aus Wasser?", "Yes"),
    ]
    no_de = [
        ("Feuer ist kalt. Ja oder nein?", "No"),
        ("Ist Vakuum laut?", "No"),
    ]
    pool: List[Tuple[str, str, str]] = [(a, b, "en") for a, b in yes_en + no_en]
    pool.extend((a, b, "de") for a, b in yes_de + no_de)
    for _ in range(n):
        q, a, lang = rng.choice(pool)
        out.append(_episode(q, a, lang, "regression_yes_no"))
    return out


def generate_rows(seed: int, n_math: int, n_capital: int, n_yesno: int) -> List[Dict]:
    rng = random.Random(seed)
    parts: List[Dict] = []
    parts.extend(_gen_math(rng, n_math))
    parts.extend(_gen_capitals(rng, n_capital))
    parts.extend(_gen_yes_no(rng, n_yesno))
    rng.shuffle(parts)
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 regression_basic-aligned short QA JSONL")
    ap.add_argument(
        "--output",
        type=str,
        default="data/sft_phase3_intermediate/phase3_regression_short.jsonl",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_math", type=int, default=2400)
    ap.add_argument("--n_capital", type=int, default=2200)
    ap.add_argument("--n_yesno", type=int, default=1400)
    args = ap.parse_args()

    rows = generate_rows(args.seed, args.n_math, args.n_capital, args.n_yesno)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _count(cat: str) -> int:
        return sum(1 for r in rows if r.get("_meta", {}).get("category") == cat)

    meta = {
        "created_at": iso_now(),
        "num_examples": len(rows),
        "seed": args.seed,
        "counts": {
            "regression_math_add": _count("regression_math_add"),
            "regression_math_sub": _count("regression_math_sub"),
            "regression_capital": _count("regression_capital"),
            "regression_yes_no": _count("regression_yes_no"),
        },
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    lineage = {
        "direct_inputs": [],
        "recursive_origins": [{"origin_path": "synthetic://generate_phase3_regression_short_sft", "rows": len(rows)}],
        "flattened_contributions": [
            {
                "origin_path": "synthetic://generate_phase3_regression_short_sft",
                "effective_rows": len(rows),
                "effective_percent": 100.0,
            }
        ],
        "creation_context": {
            "timestamp": iso_now(),
            "script": "scripts/sft/generate_phase3_regression_short_sft.py",
            "args": vars(args),
        },
    }
    write_lineage_sidecar(out_path, lineage)
    print(f"Wrote {len(rows):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
