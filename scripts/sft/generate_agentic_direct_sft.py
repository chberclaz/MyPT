#!/usr/bin/env python3
"""Think-free direct answers under AGENTIC_STANDARD_PROMPT.

Phase 5 eval uses the agentic preset on format/echo. Tool episodes with
<myPT_think> fail those buckets. These episodes teach: tools are available,
but 'Say OK' is still just OK.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.system_prompts import AGENTIC_STANDARD_PROMPT

PAIRS = [
    ("Say OK.", "OK"),
    ("Say OK", "OK"),
    ("Please say OK.", "OK"),
    ("Say Hello.", "Hello"),
    ("Say Hello", "Hello"),
    ("Say hello.", "hello"),
    ("Say yes.", "yes"),
    ("Say Yes.", "Yes"),
    ("Say no.", "no"),
    ("Repeat: Hello world", "Hello world"),
    ("Repeat: ready", "ready"),
    ("Echo: Testing", "Testing"),
    ("Echo: alpha", "alpha"),
    ("Sag Hallo.", "Hallo"),
    ("Sag OK.", "OK"),
    ("Say Banana.", "Banana"),
    ("Say 42.", "42"),
    ("Say 7.", "7"),
    ("Status?", "OK"),
    ("Ready?", "Yes"),
    ("Output only: SAFE", "SAFE"),
    ("Reply with exactly: DONE", "DONE"),
    ("Print the word cat", "cat"),
    ("Say the number 100", "100"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_examples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=5601)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.num_examples):
        user, ans = PAIRS[i % len(PAIRS)] if i < len(PAIRS) else rng.choice(PAIRS)
        rows.append({
            "system": AGENTIC_STANDARD_PROMPT,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": ans},
            ],
        })
    rng.shuffle(rows)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} agentic-direct episodes -> {out}")


if __name__ == "__main__":
    main()
