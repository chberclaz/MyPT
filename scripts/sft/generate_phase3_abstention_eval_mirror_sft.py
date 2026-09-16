#!/usr/bin/env python3
"""Abstention episodes aligned with sft_eval_suite ABSTAIN_CONTEXT_PROMPTS.

Eval user shape: "<question> Answer using ONLY context." plus a user_context
block that does not contain the answer. Scorer accepts:
  don't have enough information / do not have enough information /
  not enough information / unknown / cannot determine

The older generate_phase3_abstention_sft.py puts the target sentence in the
user turn ("answer exactly: ..."). That does not transfer to the suite.
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

_SUITE_VERBATIM: List[Tuple[str, str]] = [
    (
        "What is the GDP of Germany? Answer using ONLY context.",
        "This passage explains apples and oranges.",
    ),
    (
        "Who won the 2030 world cup? Answer using ONLY context.",
        "This text is about TCP handshake basics.",
    ),
]

_PAIRS: List[Tuple[str, str]] = [
    ("What is the GDP of Germany? Answer using ONLY context.", "This passage explains apples and oranges."),
    ("Who won the 2030 world cup? Answer using ONLY context.", "This text is about TCP handshake basics."),
    ("What is the GDP of France? Answer using ONLY context.", "The notes describe river lengths."),
    ("Who won the 2028 olympics? Answer using ONLY the context.", "This page is about fruit storage."),
    ("What is the inflation rate in Japan? Answer using ONLY context.", "The excerpt covers baking bread."),
    ("How many people live on Mars colony A? Use ONLY the context.", "The document is about Python lists."),
    ("What is the capital of Atlantis? Answer using ONLY context.", "This section explains hashing."),
    ("When did the 2040 election end? Answer using ONLY context.", "The text is about TCP handshake basics."),
]

_ANSWERS = [
    "I don't have enough information.",
    "I do not have enough information.",
    "Not enough information.",
    "I cannot determine that from the context.",
    "Unknown.",
]


def _episode(user: str, context: str, assistant: str) -> Dict:
    return {
        "system": CHAT_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": user, "context": context},
            {"role": "assistant", "content": assistant},
        ],
        "language": "en",
        "source": "phase3_abstention_eval_mirror",
        "_meta": {
            "category": "abstention_eval_mirror",
            "source_stream": "phase3_abstention_eval_mirror",
        },
    }


def generate_rows(num_examples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    out: List[Dict] = []
    for i in range(num_examples):
        if i < len(_SUITE_VERBATIM) or rng.random() < 0.22:
            user, ctx = rng.choice(_SUITE_VERBATIM)
        else:
            user, ctx = rng.choice(_PAIRS)
        out.append(_episode(user, ctx, rng.choice(_ANSWERS)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate eval-aligned abstention SFT")
    ap.add_argument("--output", type=str, default="data/sft_phase3_intermediate/phase3_abstention_eval_mirror.jsonl")
    ap.add_argument("--num_examples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=3377)
    args = ap.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_rows(args.num_examples, args.seed)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "created_at": iso_now(),
        "num_examples": len(rows),
        "seed": args.seed,
        "category": "abstention_eval_mirror",
    }
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    lineage = {
        "direct_inputs": [],
        "recursive_origins": [
            {"origin_path": "synthetic://generate_phase3_abstention_eval_mirror_sft", "rows": len(rows)}
        ],
        "flattened_contributions": [
            {
                "origin_path": "synthetic://generate_phase3_abstention_eval_mirror_sft",
                "effective_rows": len(rows),
                "effective_percent": 100.0,
            }
        ],
        "creation_context": {
            "timestamp": iso_now(),
            "script": "scripts/sft/generate_phase3_abstention_eval_mirror_sft.py",
            "args": vars(args),
        },
    }
    write_lineage_sidecar(out_path, lineage)
    print(f"Wrote {len(rows):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
