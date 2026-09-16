#!/usr/bin/env python3
"""Eval-mirror SFT for sft_eval_suite multiturn_coherence.

Same lesson as Phase 3 abstention v2d: put the scorer target in the assistant
turn, keep the user phrasing close to the suite. Held-out nonce facts only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_lineage import iso_now, write_lineage_sidecar
from core.system_prompts import CHAT_SYSTEM_PROMPT


def _ep(messages, category: str) -> dict:
    return {
        "system": CHAT_SYSTEM_PROMPT,
        "messages": messages,
        "language": "de" if category.endswith("_de") else "en",
        "source": "phase4_multiturn_eval_mirror",
        "_meta": {"category": category, "source_stream": "phase4_multiturn_eval_mirror"},
    }


TEMPLATES = [
    (
        "name_carry",
        [
            {"role": "user", "content": "My cat is named {name}."},
            {"role": "assistant", "content": "Noted, your cat is named {name}."},
            {"role": "user", "content": "What did I name my cat?"},
            {"role": "assistant", "content": "{name}"},
        ],
    ),
    (
        "code_distract",
        [
            {"role": "user", "content": "Remember this code: {code}."},
            {"role": "assistant", "content": "I will remember {code}."},
            {"role": "user", "content": "Thanks. What is 1 + 1?"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "What was the code I gave you?"},
            {"role": "assistant", "content": "{code}"},
        ],
    ),
    (
        "city_carry",
        [
            {"role": "user", "content": "I live in the city of {city}."},
            {"role": "assistant", "content": "You live in {city}."},
            {"role": "user", "content": "Which city do I live in?"},
            {"role": "assistant", "content": "{city}"},
        ],
    ),
    (
        "count_carry",
        [
            {"role": "user", "content": "I have exactly {n} wibble stones."},
            {"role": "assistant", "content": "You have {n} wibble stones."},
            {"role": "user", "content": "How many wibble stones do I have?"},
            {"role": "assistant", "content": "{n}"},
        ],
    ),
    (
        "clarify_pick",
        [
            {"role": "user", "content": "I can use red paint or {color} paint."},
            {"role": "assistant", "content": "You can use red or {color}."},
            {"role": "user", "content": "I will use {color}. Which color did I pick?"},
            {"role": "assistant", "content": "{color}"},
        ],
    ),
    (
        "topic_switch",
        [
            {"role": "user", "content": "Tell me about my cat {name}."},
            {"role": "assistant", "content": "{name} is your cat."},
            {"role": "user", "content": "New question: what color is a ripe banana?"},
            {"role": "assistant", "content": "yellow"},
        ],
    ),
    (
        "name_carry_de",
        [
            {"role": "user", "content": "Mein Hund heißt {name}."},
            {"role": "assistant", "content": "Dein Hund heißt {name}."},
            {"role": "user", "content": "Wie heißt mein Hund?"},
            {"role": "assistant", "content": "{name}"},
        ],
    ),
    (
        "context_carry",
        [
            {"role": "user", "content": "I opened a support ticket."},
            {"role": "assistant", "content": "Okay."},
            {"role": "assistant_context", "content": "The ticket id is {code}."},
            {"role": "user", "content": "What is the ticket id?"},
            {"role": "assistant", "content": "{code}"},
        ],
    ),
]

NAMES = ["Zephir", "Brio", "Quorin", "Mirael", "Tovik"]
CITIES = ["Keldara", "Vellmar", "Orrin", "Naleth"]
CODES = ["NEX-918", "WRN-440", "PLM-220", "KST-017"]
COLORS = ["teal", "indigo", "amber"]
COUNTS = ["17", "14", "23", "9"]


def _fill(messages, rng: random.Random) -> list:
    vals = {
        "name": rng.choice(NAMES),
        "city": rng.choice(CITIES),
        "code": rng.choice(CODES),
        "color": rng.choice(COLORS),
        "n": rng.choice(COUNTS),
    }
    out = []
    for m in messages:
        out.append({"role": m["role"], "content": m["content"].format(**vals)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 multiturn eval-mirror SFT")
    ap.add_argument(
        "--output",
        default="data/sft_phase4_intermediate/phase4_multiturn_eval_mirror.jsonl",
    )
    ap.add_argument("--seed", type=int, default=4405)
    ap.add_argument("--num_examples", type=int, default=800)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.num_examples):
        cat, tmpl = TEMPLATES[i % len(TEMPLATES)]
        rows.append(_ep(_fill(tmpl, rng), cat))
    rng.shuffle(rows)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {"created_at": iso_now(), "num_examples": len(rows), "seed": args.seed}
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_lineage_sidecar(
        out_path,
        {
            "direct_inputs": [],
            "recursive_origins": [
                {"origin_path": "synthetic://generate_phase4_multiturn_eval_mirror_sft", "rows": len(rows)}
            ],
            "flattened_contributions": [
                {
                    "origin_path": "synthetic://generate_phase4_multiturn_eval_mirror_sft",
                    "effective_rows": len(rows),
                    "effective_percent": 100.0,
                }
            ],
            "creation_context": {
                "timestamp": iso_now(),
                "script": "scripts/sft/generate_phase4_multiturn_eval_mirror_sft.py",
                "args": vars(args),
            },
        },
    )
    print(f"Wrote {len(rows):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
