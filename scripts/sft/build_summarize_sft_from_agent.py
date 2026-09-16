#!/usr/bin/env python3
"""Build summarize SFT episodes from agent-written summaries.

toolresult  = the agent summary (what the tool returns)
cite        = full indexed extract of that doc (provenance)
assistant   = short restatement of the summary
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.system_prompts import AGENTIC_STANDARD_PROMPT
from core.workspace.summarize import (
    SUMMARIZE_CITE_BUDGET,
    SUMMARIZE_EPISODE_BUDGET,
    clip_indexed_for_context,
    n_tokens,
)

Q_EN = [
    "Summarize {filename}.",
    "Give me a short summary of {filename}.",
    "What is {filename} about? Summarize it.",
]
Q_DE = [
    "Fasse {filename} zusammen.",
    "Kurze Zusammenfassung von {filename}.",
    "Worum geht es in {filename}? Bitte zusammenfassen.",
]
A_EN = [
    "Summary of {filename}:\n\n{summary}",
    "Here is a short overview of {filename}:\n\n{summary}",
]
A_DE = [
    "Zusammenfassung von {filename}:\n\n{summary}",
    "Kurzer Überblick zu {filename}:\n\n{summary}",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracts", default="data/sft_phase5_intermediate/doc_extracts.jsonl")
    ap.add_argument("--summaries", default="data/sft_phase5_intermediate/doc_agent_summaries.jsonl")
    ap.add_argument("--output", default="data/sft_phase5_intermediate/phase5_summarize_agent.jsonl")
    ap.add_argument("--repeats", type=int, default=8, help="Template variants per doc per language")
    ap.add_argument("--seed", type=int, default=5511)
    ap.add_argument("--cite_max_tokens", type=int, default=SUMMARIZE_CITE_BUDGET)
    ap.add_argument("--episode_max_tokens", type=int, default=SUMMARIZE_EPISODE_BUDGET)
    args = ap.parse_args()

    def _p(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    extracts = {}
    with _p(args.extracts).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                extracts[r["doc_id"]] = r

    summaries = []
    with _p(args.summaries).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                summaries.append(json.loads(line))

    rng = random.Random(args.seed)
    rows = []
    for s in summaries:
        ext = extracts.get(s["doc_id"])
        if not ext:
            print(f"skip missing extract {s.get('doc_id')}", file=sys.stderr)
            continue
        cite = clip_indexed_for_context(ext.get("indexed_text") or "", args.cite_max_tokens)
        filename = s.get("filename") or ext["filename"]
        doc_id = s["doc_id"]
        for lang, qs, ans, key in (
            ("en", Q_EN, A_EN, "summary_en"),
            ("de", Q_DE, A_DE, "summary_de"),
        ):
            summary = (s.get(key) or s.get("summary") or "").strip()
            if len(summary) < 20:
                continue
            for _ in range(args.repeats):
                q = rng.choice(qs).format(filename=filename)
                a = rng.choice(ans).format(filename=filename, summary=summary)
                rows.append({
                    "system": AGENTIC_STANDARD_PROMPT,
                    "messages": [
                        {"role": "user", "content": q},
                        {
                            "role": "assistant_toolcall",
                            "name": "workspace.summarize",
                            "arguments": {"doc_id": doc_id},
                        },
                        {
                            "role": "toolresult",
                            "name": "workspace.summarize",
                            "content": {"summary": summary},
                        },
                        {
                            "role": "assistant",
                            "content": a,
                            "cite": cite,
                        },
                    ],
                    "language": lang,
                    "source": "phase5_summarize_agent",
                    "_meta": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "source_stream": "phase5_summarize_agent",
                    },
                })

    rng.shuffle(rows)
    out = _p(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    over = 0
    worst = 0
    for r in rows:
        blob = r["system"] + "".join(
            (m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")))
            + (m.get("cite") or "")
            for m in r["messages"]
        )
        nt = n_tokens(blob)
        worst = max(worst, nt)
        if nt > args.episode_max_tokens:
            over += 1
    print(f"Wrote {len(rows)} episodes -> {out}")
    print(f"Token check (gpt2): worst={worst} budget={args.episode_max_tokens} over={over}")


if __name__ == "__main__":
    main()
