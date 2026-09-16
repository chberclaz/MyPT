#!/usr/bin/env python3
"""Abstain / cite-from-context under AGENTIC_STANDARD_PROMPT, no tools.

P5 eval still injects the agentic preset. These episodes teach: if the user
turn already has context, do not search — cite it or abstain.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.system_prompts import AGENTIC_STANDARD_PROMPT

ABSTAIN = [
    (
        "This passage explains apples and oranges.",
        "What is the GDP of Germany? Answer using ONLY context.",
        "I don't have enough information.",
    ),
    (
        "This text is about TCP handshake basics.",
        "Who won the 2030 world cup? Answer using ONLY context.",
        "I don't have enough information.",
    ),
    (
        "The note covers office Wi-Fi passwords.",
        "What is the capital of Mars? Answer using ONLY context.",
        "I don't have enough information.",
    ),
    (
        "Der Text beschreibt nur Apfelkuchen.",
        "Was ist das BIP von Deutschland? Nur Kontext.",
        "Ich habe nicht genug Informationen.",
    ),
]

CITE = [
    (
        "[CID:doc_alpha#c001] The passage describes Python as a programming language.",
        "Answer briefly: what does the passage describe? Include citation tag.",
        "The passage describes Python as a programming language.",
        "doc_alpha#c001",
    ),
    (
        "[CID:net_spec#c007] This section discusses TCP three-way handshake.",
        "Answer briefly: what protocol is discussed? Include citation tag.",
        "This section discusses TCP three-way handshake.",
        "net_spec#c007",
    ),
    (
        "[CID:install_md#c002] Install covers prerequisites, setup, and verify.",
        "What does install cover? Include citation tag.",
        "Install covers prerequisites, setup, and verify.",
        "install_md#c002",
    ),
    (
        "[CID:guide_de#c010] Der Abschnitt erklärt die RoPE-Skalierung.",
        "Worum geht es? Mit Zitat-Tag.",
        "Der Abschnitt erklärt die RoPE-Skalierung.",
        "guide_de#c010",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_examples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=5701)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.num_examples):
        if i % 2 == 0:
            ctx, user, ans = ABSTAIN[i // 2 % len(ABSTAIN)] if i < 40 else rng.choice(ABSTAIN)
            rows.append({
                "system": AGENTIC_STANDARD_PROMPT,
                "messages": [
                    {"role": "user", "content": user, "context": ctx},
                    {"role": "assistant", "content": ans},
                ],
            })
        else:
            ctx, user, ans, cite = CITE[i // 2 % len(CITE)] if i < 40 else rng.choice(CITE)
            rows.append({
                "system": AGENTIC_STANDARD_PROMPT,
                "messages": [
                    {"role": "user", "content": user, "context": ctx},
                    {"role": "assistant", "content": ans, "cite": cite},
                ],
            })
    rng.shuffle(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} grounded episodes -> {out}")


if __name__ == "__main__":
    main()
