#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6.3 SFT: copy facts from the get_doc/summarize toolresult into the answer.

Live 6.2 GOLD binds rank-1 doc_id, then emits the think template
("Based on the document content, I can provide a grounded answer about {topic}")
instead of the toolresult body. 6.2 gold think and gold answer shared that
phrasing, so greedy copies think after </myPT_think>.

These episodes keep rank-1 bind, but:
- think after get_doc is short and lexically disjoint from the answer
- gold answer is a verbatim span copied from the toolresult text
- validator requires a 6-word n-gram from the toolresult in the answer
  and rejects think-template leaks

Usage:
    python scripts/sft/generate_phase6_3_ground_sft.py \\
        --workspace_dir workspace/ \\
        --docs_dir workspace/docs \\
        --output data/sft_phase6_3_intermediate/phase6_3_ground.jsonl \\
        --num_examples 3000 --seed <SEED> --language mixed
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sft"))

from core.system_prompts import AGENTIC_STANDARD_PROMPT
from core.workspace.summarize import clip_indexed_for_context, structured_summary
from generate_phase6_2_docbind_sft import (
    GETDOC_CITE_TOKENS,
    load_workspace_docs,
    pick_distractors,
    _dumps,
    _hit,
    _topic_from_doc,
)

ANSWER_COPY_CHARS = 720
_TOKEN = re.compile(r"[A-Za-z0-9ÄÖÜäöüß]{2,}")
THINK_LEAK = re.compile(
    r"(based on the document content|"
    r"i can provide a grounded answer|"
    r"now i have the information to answer|"
    r"i'll generate a concise answer|"
    r"i'll retrieve the document content|"
    r"let me summarize|"
    r"ich hole den dokumentinhalt|"
    r"lass mich ihn zusammenfassen|"
    r"fundierte antwort|"
    r"basierend auf dem dokumentinhalt)",
    re.I,
)

Q_EN = [
    "Who or what is {topic}?",
    "What do our documents say about {topic}?",
    "Read the workspace file about {topic} and tell me the facts.",
    "After you retrieve {topic}, quote what the document says.",
]
Q_DE = [
    "Wer oder was ist {topic}?",
    "Was sagen unsere Dokumente zu {topic}?",
    "Lies die Workspace-Datei zu {topic} und nenne die Fakten.",
    "Nachdem du {topic} geholt hast, zitiere was das Dokument sagt.",
]
THINK_SEARCH_EN = [
    "The user wants to know about {topic}. Let me search the workspace.",
    "I should search for documents related to {topic} first.",
]
THINK_SEARCH_DE = [
    "Der Benutzer moechte etwas ueber {topic} wissen. Lass mich den Workspace durchsuchen.",
    "Ich sollte zuerst nach Dokumenten zu {topic} suchen.",
]
THINK_GET_EN = [
    "I'll retrieve the document content.",
]
THINK_GET_DE = [
    "Ich hole den Dokumentinhalt.",
]
THINK_SUM_EN = [
    "The top hit is enough. Let me summarize it.",
]
THINK_SUM_DE = [
    "Der oberste Treffer reicht. Lass mich ihn zusammenfassen.",
]
# Must not share a prefix with THINK_GET / THINK_SUM.
THINK_COPY_EN = [
    "Copying the facts from the tool result.",
]
THINK_COPY_DE = [
    "Ich kopiere die Fakten aus dem Tool-Ergebnis.",
]
THINK_MISS_EN = [
    "The search returned no matching documents.",
    "Nothing in the catalog matched the query.",
]
THINK_MISS_DE = [
    "Die Suche hat keine passenden Dokumente gefunden.",
    "Im Katalog gibt es keinen Treffer zur Anfrage.",
]
ANS_MISS_EN = [
    "I don't have enough information in the workspace about {topic}.",
    "There is not enough information in the workspace about {topic}.",
]
ANS_MISS_DE = [
    "Ich habe nicht genug Informationen im Workspace zu {topic}.",
    "Im Workspace gibt es not enough information zu {topic}.",
]
MISS_TOPICS = [
    "Emanuela Kyritsis",
    "Piedpiper",
    "Zorblax Nine",
    "Kieferngasse millwright",
    "coiled-hare foundry",
    "Wexmere brassworks",
]


def _copy_span(text: str, max_chars: int = ANSWER_COPY_CHARS) -> str:
    blob = (text or "").strip()
    if not blob:
        return ""
    paras = [p.strip() for p in re.split(r"\n\s*\n", blob) if p.strip()]
    if not paras:
        paras = [blob]
    out: List[str] = []
    n = 0
    for p in paras:
        out.append(p)
        n += len(p)
        if n >= max_chars:
            break
    s = "\n\n".join(out).strip()
    if len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
    return s


def _has_ngram(src: str, dst: str, n: int = 6) -> bool:
    words = _TOKEN.findall(src or "")
    if len(words) < n:
        return False
    hay = " ".join(_TOKEN.findall(dst or "")).lower()
    for i in range(len(words) - n + 1):
        span = " ".join(words[i : i + n]).lower()
        if span in hay:
            return True
    return False


def validate_episode(
    ep: Dict[str, Any],
    gold_id: str,
    distractor_ids: List[str],
    source_text: str,
) -> bool:
    msgs = ep.get("messages") or []
    if not msgs:
        return False
    gold_in_tool = False
    for m in msgs:
        if m.get("role") not in ("assistant_toolcall", "assistant_frozen"):
            continue
        name = m.get("name") or ""
        args = m.get("arguments") or {}
        if name in ("workspace.get_doc", "workspace.summarize"):
            did = args.get("doc_id") or ""
            if did == gold_id:
                gold_in_tool = True
            if did in distractor_ids:
                return False
    if any(
        m.get("role") in ("assistant_toolcall", "assistant_frozen")
        and m.get("name") in ("workspace.get_doc", "workspace.summarize")
        for m in msgs
    ):
        if not gold_in_tool:
            return False
    ans = next((m for m in msgs if m.get("role") == "assistant"), None)
    if not ans or not (ans.get("content") or "").strip():
        return False
    content = ans.get("content") or ""
    think = ans.get("think") or ""
    blob = content + " " + (ans.get("cite") or "")
    for did in distractor_ids:
        if did and did in blob:
            return False
    if THINK_LEAK.search(content) or THINK_LEAK.search(think):
        return False
    if think and think.lower() in content.lower():
        return False
    if not _has_ngram(source_text, content, n=6):
        return False
    return True


def make_miss_episode(lang: str, pretty: bool, rng: random.Random) -> Optional[Dict[str, Any]]:
    topic = rng.choice(MISS_TOPICS)
    if lang == "de":
        q_user = rng.choice(Q_DE).format(topic=topic)
        t_search = rng.choice(THINK_SEARCH_DE).format(topic=topic)
        t_miss = rng.choice(THINK_MISS_DE)
        answer = rng.choice(ANS_MISS_DE).format(topic=topic)
    else:
        q_user = rng.choice(Q_EN).format(topic=topic)
        t_search = rng.choice(THINK_SEARCH_EN).format(topic=topic)
        t_miss = rng.choice(THINK_MISS_EN)
        answer = rng.choice(ANS_MISS_EN).format(topic=topic)
    empty = _dumps({"documents": [], "total": 0}, pretty)
    msgs: List[Dict[str, Any]] = [
        {"role": "user", "content": q_user},
        {
            "role": "assistant_toolcall",
            "name": "workspace.search",
            "arguments": {"query": topic, "top_k": 3},
            "think": t_search,
        },
        {"role": "toolresult", "name": "workspace.search", "content": empty},
        {
            "role": "assistant",
            "content": answer,
            "think": t_miss,
        },
    ]
    ep = {
        "system": AGENTIC_STANDARD_PROMPT,
        "messages": msgs,
        "language": lang,
        "_meta": {
            "phase": "6.3",
            "kind": "miss_abstain",
            "gold_id": "",
            "n_hits": 0,
            "pretty": pretty,
        },
    }
    if not validate_episode(ep, "", [], answer):
        return None
    return ep


def make_episode(
    gold: Dict[str, Any],
    distractors: List[Dict[str, Any]],
    topic: str,
    lang: str,
    kind: str,
    pretty: bool,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    hits = [_hit(1, gold, topic)]
    for i, d in enumerate(distractors, start=2):
        hits.append(_hit(i, d, topic))
    search_result = _dumps({"documents": hits, "total": len(hits)}, pretty)
    dist_ids = [d["doc_id"] for d in distractors]
    gold_id = gold["doc_id"]
    filename = gold["filename"]

    if lang == "de":
        q_user = rng.choice(Q_DE)
        t_search = rng.choice(THINK_SEARCH_DE)
        t_get = rng.choice(THINK_GET_DE)
        t_sum = rng.choice(THINK_SUM_DE)
        t_copy = rng.choice(THINK_COPY_DE)
    else:
        q_user = rng.choice(Q_EN)
        t_search = rng.choice(THINK_SEARCH_EN)
        t_get = rng.choice(THINK_GET_EN)
        t_sum = rng.choice(THINK_SUM_EN)
        t_copy = rng.choice(THINK_COPY_EN)

    msgs: List[Dict[str, Any]] = [
        {"role": "user", "content": q_user.format(topic=topic)},
        {
            "role": "assistant_toolcall",
            "name": "workspace.search",
            "arguments": {"query": topic, "top_k": max(3, len(hits))},
            "think": t_search.format(topic=topic),
        },
        {"role": "toolresult", "name": "workspace.search", "content": search_result},
    ]

    clipped = clip_indexed_for_context(gold["text"], GETDOC_CITE_TOKENS)
    getdoc_result = {
        "doc_id": gold_id,
        "title": gold["title"],
        "text": clipped,
        "length": len(gold["text"]),
    }

    source_text = ""
    if kind == "get_doc_eval":
        # Exact eval prefix: retrieve-think + get_doc are history (frozen).
        # Only the copy turn after get_doc JSON is trained.
        source_text = _copy_span(clipped)
        msgs = [
            {"role": "user", "content": q_user.format(topic=topic)},
            {
                "role": "assistant_frozen",
                "name": "workspace.search",
                "arguments": {"query": topic, "top_k": max(3, len(hits))},
                "think": t_search.format(topic=topic),
            },
            {"role": "toolresult", "name": "workspace.search", "content": search_result},
            {
                "role": "assistant_frozen",
                "name": "workspace.get_doc",
                "arguments": {"doc_id": gold_id},
                "think": t_get,
            },
            {
                "role": "toolresult",
                "name": "workspace.get_doc",
                "content": _dumps(getdoc_result, pretty),
            },
            {
                "role": "assistant",
                "content": source_text,
                "think": t_copy,
                "cite": filename,
            },
        ]
    elif kind in ("get_doc", "copy_turn"):
        source_text = _copy_span(clipped)
        msgs.append({
            "role": "assistant_toolcall",
            "name": "workspace.get_doc",
            "arguments": {"doc_id": gold_id},
            "think": t_get,
        })
        msgs.append({
            "role": "toolresult",
            "name": "workspace.get_doc",
            "content": _dumps(getdoc_result, pretty),
        })
        msgs.append({
            "role": "assistant",
            "content": source_text,
            "think": t_copy,
            "cite": filename,
        })
    elif kind == "get_doc_final":
        # Eval-matched: prompt already ends after get_doc JSON. Only the copy
        # turn is trained, so retrieve-think is not a valid next token.
        source_text = _copy_span(clipped)
        msgs = [
            {"role": "user", "content": q_user.format(topic=topic)},
            {
                "role": "toolresult",
                "name": "workspace.get_doc",
                "content": _dumps(getdoc_result, pretty),
            },
            {
                "role": "assistant",
                "content": source_text,
                "think": t_copy,
                "cite": filename,
            },
        ]
    elif kind == "search_answer":
        snippet = hits[0]["snippet"]
        source_text = snippet
        msgs.append({
            "role": "assistant",
            "content": snippet,
            "think": t_copy,
            "cite": filename,
        })
    elif kind == "summarize":
        summary = structured_summary(gold["text"], 400, lang)
        source_text = summary
        sum_result = {
            "summary": summary,
            "source": "doc_id",
            "original_length": len(gold["text"]),
        }
        msgs.append({
            "role": "assistant_toolcall",
            "name": "workspace.summarize",
            "arguments": {"doc_id": gold_id},
            "think": t_sum,
        })
        msgs.append({
            "role": "toolresult",
            "name": "workspace.summarize",
            "content": _dumps(sum_result, pretty),
        })
        msgs.append({
            "role": "assistant",
            "content": summary,
            "think": t_copy,
            "cite": filename,
        })
    else:
        source_text = _copy_span(clipped)
        msgs.append({
            "role": "assistant_toolcall",
            "name": "workspace.get_doc",
            "arguments": {"doc_id": gold_id},
            "think": t_get,
        })
        msgs.append({
            "role": "toolresult",
            "name": "workspace.get_doc",
            "content": _dumps(getdoc_result, pretty),
        })
        msgs.append({
            "role": "assistant",
            "content": source_text,
            "think": t_copy,
            "cite": filename,
        })

    if len(_TOKEN.findall(source_text)) < 8:
        return None

    ep = {
        "system": AGENTIC_STANDARD_PROMPT,
        "messages": msgs,
        "language": lang,
        "_meta": {
            "phase": "6.3",
            "kind": kind,
            "gold_id": gold_id,
            "n_hits": len(hits),
            "pretty": pretty,
        },
    }
    if not validate_episode(ep, gold_id, dist_ids, source_text):
        return None
    return ep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase 6.3 toolresult-ground SFT")
    p.add_argument("--workspace_dir", default="workspace/")
    p.add_argument("--docs_dir", default="workspace/docs")
    p.add_argument(
        "--output",
        default="data/sft_phase6_3_intermediate/phase6_3_ground.jsonl",
    )
    p.add_argument("--num_examples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--language", choices=["en", "de", "mixed"], default="mixed")
    p.add_argument("--de_ratio", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    docs = load_workspace_docs(Path(args.workspace_dir), Path(args.docs_dir))
    if len(docs) < 3:
        print(f"Need at least 3 documents, got {len(docs)}")
        sys.exit(1)
    print(f"Loaded {len(docs)} workspace documents")

    # Iter-1 spine (from62 restart): full chain only. Do not add copy_turn /
    # get_doc_final / frozen — those washed copy when stacked on 6.3 gold.
    kinds = (
        ["get_doc"] * 55
        + ["summarize"] * 12
        + ["search_answer"] * 13
        + ["miss_abstain"] * 20
    )

    episodes: List[Dict[str, Any]] = []
    fails = 0
    attempts = 0
    max_attempts = args.num_examples * 20
    while len(episodes) < args.num_examples and attempts < max_attempts:
        attempts += 1
        lang = (
            ("de" if rng.random() < args.de_ratio else "en")
            if args.language == "mixed"
            else args.language
        )
        kind = rng.choice(kinds)
        pretty = rng.random() < 0.50
        if kind == "miss_abstain":
            ep = make_miss_episode(lang, pretty, rng)
            if ep is None:
                fails += 1
                continue
            episodes.append(ep)
            continue
        gold = rng.choice(docs)
        topic = _topic_from_doc(gold["filename"], gold["text"])
        n_hits = 3 if rng.random() < 0.30 else 2
        distractors = pick_distractors(gold, docs, topic, n_hits - 1, rng)
        if len(distractors) < n_hits - 1:
            fails += 1
            continue
        ep = make_episode(gold, distractors, topic, lang, kind, pretty, rng)
        if ep is None:
            fails += 1
            continue
        episodes.append(ep)

    if len(episodes) < args.num_examples:
        print(f"WARNING: only {len(episodes)}/{args.num_examples} after {attempts} attempts")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    kinds_c: Dict[str, int] = {}
    hits_c: Dict[int, int] = {}
    for ep in episodes:
        meta = ep.get("_meta") or {}
        kinds_c[meta.get("kind", "?")] = kinds_c.get(meta.get("kind", "?"), 0) + 1
        hits_c[int(meta.get("n_hits") or 0)] = hits_c.get(int(meta.get("n_hits") or 0), 0) + 1
    print(f"Wrote {len(episodes)} -> {out}")
    print(f"  kinds={kinds_c} n_hits={hits_c} validation_fails={fails}")


if __name__ == "__main__":
    main()
