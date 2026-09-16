#!/usr/bin/env python3
"""
Generate strict abstention episodes for Phase 3.2 corrective.
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


ABSTAIN_EN = "I don't have enough information."
ABSTAIN_DE = "Ich habe nicht genug Informationen."

CONTEXTS: List[Tuple[str, str, str]] = [
    ("The passage discusses apples and pears.", "What is the GDP of Germany?", "gdp"),
    ("This section explains TCP handshake basics.", "Who won the world cup in 2030?", "sports"),
    ("The document covers Python syntax.", "What is the population of Mars colony A?", "mars"),
    ("The text is about database indexing.", "What is the current inflation rate in France?", "inflation"),
]


def _episode(user: str, context: str, assistant: str, language: str) -> Dict:
    return {
        "system": "You are MyPT. Answer only from provided context. If context is insufficient, abstain exactly.",
        "messages": [
            {"role": "user", "content": user, "context": context},
            {"role": "assistant", "content": assistant, "cite": "no_source"},
        ],
        "language": language,
        "source": "phase3_abstention_strict",
        "_meta": {
            "category": "abstention_context_strict",
            "source_stream": "phase3_abstention_strict",
        },
    }


def generate_rows(num_examples: int, seed: int, de_ratio: float) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []
    for _ in range(num_examples):
        context, question, _ = rng.choice(CONTEXTS)
        de = rng.random() < de_ratio
        if de:
            user = (
                f"{question}\n"
                "Beantworte nur mit dem bereitgestellten Kontext. "
                f"Wenn Informationen fehlen, antworte exakt: \"{ABSTAIN_DE}\""
            )
            answer = ABSTAIN_DE
            lang = "de"
        else:
            user = (
                f"{question}\n"
                "Answer using only the provided context. "
                f"If information is missing, answer exactly: \"{ABSTAIN_EN}\""
            )
            answer = ABSTAIN_EN
            lang = "en"
        out.append(_episode(user, context, answer, lang))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate strict abstention corrective episodes")
    ap.add_argument("--output", type=str, default="data/sft_phase3_intermediate/phase3_abstention_strict.jsonl")
    ap.add_argument("--num_examples", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=3327)
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
        "num_examples": len(rows),
        "seed": args.seed,
        "de_ratio": args.de_ratio,
        "category": "abstention_context_strict",
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    lineage = {
        "direct_inputs": [],
        "recursive_origins": [{"origin_path": "synthetic://generate_phase3_abstention_sft", "rows": len(rows)}],
        "flattened_contributions": [
            {"origin_path": "synthetic://generate_phase3_abstention_sft", "effective_rows": len(rows), "effective_percent": 100.0}
        ],
        "creation_context": {
            "timestamp": iso_now(),
            "script": "scripts/sft/generate_phase3_abstention_sft.py",
            "args": vars(args),
        },
    }
    write_lineage_sidecar(out_path, lineage)
    print(f"Wrote {len(rows):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
