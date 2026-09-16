#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6.2 SFT: multi-hit search catalogs with rank-1 doc_id bind.

Live 750M attends to hit 1's content then copies hit 2's hex id (off-by-one).
P6 search→get_doc episodes had total=1. These episodes put 2–3 live-shaped
hits in the toolresult and always label rank 1 as the next get_doc/summarize id.

Usage:
    python scripts/sft/generate_phase6_2_docbind_sft.py \\
        --workspace_dir workspace/ \\
        --docs_dir workspace/docs \\
        --output data/sft_phase6_2_intermediate/phase6_2_docbind.jsonl \\
        --num_examples 3000 --seed <SEED> --language mixed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.system_prompts import AGENTIC_STANDARD_PROMPT
from core.workspace.summarize import clip_indexed_for_context, structured_summary

SEARCH_SNIPPET_CHARS = 280
GETDOC_CITE_TOKENS = 480
ANSWER_CONTENT_TOKENS = 220
_WORD = re.compile(r"[a-zA-Z0-9]{4,}")

Q_EN = [
    "Who or what is {topic}?",
    "Find {topic} in the workspace and read the matching document.",
    "Search for {topic} and open the top result.",
    "What do our docs say about {topic}? Get the full document.",
    "Look up {topic} and retrieve the relevant file.",
]
Q_DE = [
    "Wer oder was ist {topic}?",
    "Finde {topic} im Workspace und lies das passende Dokument.",
    "Suche nach {topic} und oeffne das oberste Ergebnis.",
    "Was sagen unsere Docs zu {topic}? Hol das volle Dokument.",
    "Schlag {topic} nach und rufe die relevante Datei ab.",
]
Q_ANSWER_EN = [
    "What information do we have about {topic}?",
    "Search for {topic} and tell me what you find.",
]
Q_ANSWER_DE = [
    "Welche Informationen haben wir ueber {topic}?",
    "Suche nach {topic} und sag mir was du findest.",
]
THINK_SEARCH_EN = [
    "The user wants to know about {topic}. Let me search the workspace.",
    "I should search for documents related to {topic} first.",
]
THINK_SEARCH_DE = [
    "Der Benutzer moechte etwas ueber {topic} wissen. Lass mich den Workspace durchsuchen.",
    "Ich sollte zuerst nach Dokumenten zu {topic} suchen.",
]
THINK_GENERIC_EN = [
    "I'll retrieve the document content.",
    "I'll generate a concise answer.",
    "Let me read the top result.",
]
THINK_GENERIC_DE = [
    "Ich hole den Dokumentinhalt.",
    "Ich formuliere eine knappe Antwort.",
    "Lass mich das oberste Ergebnis lesen.",
]
THINK_TITLED_EN = [
    "The search found a relevant document: {title}. Let me get the full text.",
    "I found {title} in the search results. I'll retrieve the complete document.",
]
THINK_TITLED_DE = [
    "Die Suche hat ein relevantes Dokument gefunden: {title}. Lass mich den vollen Text holen.",
    "Ich habe {title} in den Suchergebnissen gefunden. Ich rufe das vollstaendige Dokument ab.",
]
THINK_SUM_EN = [
    "I'll generate a concise answer.",
    "The top hit is enough. Let me summarize it.",
]
THINK_SUM_DE = [
    "Ich formuliere eine knappe Antwort.",
    "Der oberste Treffer reicht. Lass mich ihn zusammenfassen.",
]
THINK_ANSWER_EN = [
    "Now I have the information to answer the user's question about {topic}.",
    "Based on the document content, I can provide a grounded answer about {topic}.",
]
THINK_ANSWER_DE = [
    "Jetzt habe ich die Informationen um die Frage zu {topic} zu beantworten.",
    "Basierend auf dem Dokumentinhalt kann ich eine fundierte Antwort zu {topic} geben.",
]
ANS_GETDOC_EN = [
    "Based on {filename}:\n\n{content}",
    "From {filename}:\n\n{content}",
]
ANS_GETDOC_DE = [
    "Basierend auf {filename}:\n\n{content}",
    "Aus {filename}:\n\n{content}",
]
ANS_SUM_EN = [
    "Summary of {filename}: {summary}",
]
ANS_SUM_DE = [
    "Zusammenfassung von {filename}: {summary}",
]
ANS_SEARCH_EN = [
    "Search found {filename}: {snippet}",
]
ANS_SEARCH_DE = [
    "Die Suche fand {filename}: {snippet}",
]


def _engine_doc_id(workspace_dir: Path, path: str) -> str:
    """Match WorkspaceEngine._generate_doc_id (md5 of path relative to workspace)."""
    base = Path(workspace_dir)
    p = Path(path)
    try:
        rel = p.relative_to(base)
    except ValueError:
        try:
            rel = p.resolve().relative_to(base.resolve())
        except ValueError:
            rel = p
    return hashlib.md5(str(rel).encode()).hexdigest()[:12]


def _title_of(filename: str) -> str:
    return os.path.splitext(filename)[0]


def _snippet(text: str, query: str, max_chars: int = SEARCH_SNIPPET_CHARS) -> str:
    blob = " ".join((text or "").split())
    if not blob:
        return ""
    q_words = [w.lower() for w in query.split() if len(w) > 2]
    low = blob.lower()
    pos = 0
    for w in q_words:
        i = low.find(w)
        if i >= 0:
            pos = max(0, i - 40)
            break
    cut = blob[pos : pos + max_chars]
    if pos > 0:
        cut = cut.split(" ", 1)[-1] if " " in cut else cut
    if pos + max_chars < len(blob):
        cut = cut.rsplit(" ", 1)[0] + "..."
    return cut


def _query_words(query: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(query) if len(w) > 3]


def _overlap(text: str, words: List[str]) -> int:
    low = (text or "").lower()
    return sum(1 for w in words if w in low)


def _topic_from_doc(filename: str, text: str) -> str:
    stem = _title_of(filename).replace("_", " ").replace("-", " ").strip()
    first = ""
    for line in (text or "").splitlines():
        s = line.strip().lstrip("#").strip()
        if s and 3 <= len(s) <= 60:
            first = s
            break
    return first or stem or filename


def _hit(rank: int, doc: Dict[str, Any], query: str) -> Dict[str, Any]:
    return {
        "rank": rank,
        "title": doc["title"],
        "filename": doc["filename"],
        "snippet": _snippet(doc["text"], query),
        "doc_id": doc["doc_id"],
    }


def _dumps(obj: Any, pretty: bool) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False)


def load_workspace_docs(workspace_dir: Path, docs_dir: Path) -> List[Dict[str, Any]]:
    from core.document import DocumentLoader

    loader = DocumentLoader()
    raw = loader.load_directory(str(docs_dir))
    out: List[Dict[str, Any]] = []
    for d in raw:
        text = (d.text or "").strip()
        if len(text) < 80:
            continue
        fn = d.filename
        out.append({
            "doc_id": _engine_doc_id(workspace_dir, d.source),
            "title": _title_of(fn),
            "filename": fn,
            "text": text,
            "source": d.source,
        })
    # Dedup by doc_id
    seen = set()
    uniq = []
    for d in out:
        if d["doc_id"] in seen:
            continue
        seen.add(d["doc_id"])
        uniq.append(d)
    return uniq


def pick_distractors(
    gold: Dict[str, Any],
    docs: List[Dict[str, Any]],
    query: str,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    words = _query_words(query)
    others = [d for d in docs if d["doc_id"] != gold["doc_id"]]
    scored = []
    for d in others:
        scored.append((_overlap(d["text"], words), rng.random(), d))
    scored.sort(key=lambda x: (x[0], x[1]))
    picked = [t[2] for t in scored[:k]]
    if len(picked) < k:
        rest = [d for d in others if d not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: k - len(picked)])
    return picked[:k]


def validate_episode(ep: Dict[str, Any], gold_id: str, distractor_ids: List[str]) -> bool:
    msgs = ep.get("messages") or []
    if not msgs:
        return False
    gold_in_tool = False
    for m in msgs:
        if m.get("role") != "assistant_toolcall":
            continue
        name = m.get("name") or ""
        args = m.get("arguments") or {}
        if name in ("workspace.get_doc", "workspace.summarize"):
            did = args.get("doc_id") or ""
            if did == gold_id:
                gold_in_tool = True
            if did in distractor_ids:
                return False
    if any(m.get("role") == "assistant_toolcall" and m.get("name") in (
        "workspace.get_doc", "workspace.summarize"
    ) for m in msgs):
        if not gold_in_tool:
            return False
    ans = next((m for m in msgs if m.get("role") == "assistant"), None)
    if not ans or not (ans.get("content") or "").strip():
        return False
    content = (ans.get("content") or "") + " " + (ans.get("cite") or "")
    for did in distractor_ids:
        if did and did in content:
            return False
    return True


def make_episode(
    gold: Dict[str, Any],
    distractors: List[Dict[str, Any]],
    query: str,
    topic: str,
    lang: str,
    kind: str,
    pretty: bool,
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    hits = [_hit(1, gold, query)]
    for i, d in enumerate(distractors, start=2):
        hits.append(_hit(i, d, query))
    search_result = _dumps({"documents": hits, "total": len(hits)}, pretty)
    dist_ids = [d["doc_id"] for d in distractors]
    gold_id = gold["doc_id"]
    title = gold["title"]
    filename = gold["filename"]

    if lang == "de":
        q_search = rng.choice(Q_DE)
        q_ans_only = rng.choice(Q_ANSWER_DE)
        t_search = rng.choice(THINK_SEARCH_DE)
        t_gen = rng.choice(THINK_GENERIC_DE)
        t_tit = rng.choice(THINK_TITLED_DE)
        t_sum = rng.choice(THINK_SUM_DE)
        t_ans = rng.choice(THINK_ANSWER_DE)
        a_get = rng.choice(ANS_GETDOC_DE)
        a_sum = rng.choice(ANS_SUM_DE)
        a_srch = rng.choice(ANS_SEARCH_DE)
    else:
        q_search = rng.choice(Q_EN)
        q_ans_only = rng.choice(Q_ANSWER_EN)
        t_search = rng.choice(THINK_SEARCH_EN)
        t_gen = rng.choice(THINK_GENERIC_EN)
        t_tit = rng.choice(THINK_TITLED_EN)
        t_sum = rng.choice(THINK_SUM_EN)
        t_ans = rng.choice(THINK_ANSWER_EN)
        a_get = rng.choice(ANS_GETDOC_EN)
        a_sum = rng.choice(ANS_SUM_EN)
        a_srch = rng.choice(ANS_SEARCH_EN)

    user_q = (q_ans_only if kind == "search_answer" else q_search).format(topic=topic)
    msgs: List[Dict[str, Any]] = [
        {"role": "user", "content": user_q},
        {
            "role": "assistant_toolcall",
            "name": "workspace.search",
            "arguments": {"query": topic, "top_k": max(3, len(hits))},
            "think": t_search.format(topic=topic),
        },
        {"role": "toolresult", "name": "workspace.search", "content": search_result},
    ]

    snippet = hits[0]["snippet"]
    if kind == "search_answer":
        msgs.append({
            "role": "assistant",
            "content": a_srch.format(filename=filename, snippet=snippet),
            "think": t_ans.format(topic=topic),
            "cite": filename,
        })
    elif kind == "summarize":
        summary = structured_summary(gold["text"], 400, lang)
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
            "content": a_sum.format(filename=filename, summary=summary),
            "think": t_ans.format(topic=topic),
            "cite": filename,
        })
    else:
        # get_doc (generic or titled think)
        think = t_tit.format(title=title) if kind == "get_doc_titled" else t_gen
        clipped = clip_indexed_for_context(gold["text"], GETDOC_CITE_TOKENS)
        getdoc_result = {
            "doc_id": gold_id,
            "title": title,
            "text": clipped,
            "length": len(gold["text"]),
        }
        content = clip_indexed_for_context(gold["text"], ANSWER_CONTENT_TOKENS)
        msgs.append({
            "role": "assistant_toolcall",
            "name": "workspace.get_doc",
            "arguments": {"doc_id": gold_id},
            "think": think,
        })
        msgs.append({
            "role": "toolresult",
            "name": "workspace.get_doc",
            "content": _dumps(getdoc_result, pretty),
        })
        msgs.append({
            "role": "assistant",
            "content": a_get.format(filename=filename, content=content),
            "think": t_ans.format(topic=topic),
            "cite": filename,
        })

    ep = {
        "system": AGENTIC_STANDARD_PROMPT,
        "messages": msgs,
        "language": lang,
        "_meta": {
            "phase": "6.2",
            "kind": kind,
            "gold_id": gold_id,
            "n_hits": len(hits),
            "pretty": pretty,
        },
    }
    if not validate_episode(ep, gold_id, dist_ids):
        return None
    return ep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase 6.2 rank-1 doc_id bind SFT")
    p.add_argument("--workspace_dir", default="workspace/")
    p.add_argument("--docs_dir", default="workspace/docs")
    p.add_argument(
        "--output",
        default="data/sft_phase6_2_intermediate/phase6_2_docbind.jsonl",
    )
    p.add_argument("--num_examples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--language", choices=["en", "de", "mixed"], default="mixed")
    p.add_argument("--de_ratio", type=float, default=0.4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    workspace_dir = Path(args.workspace_dir)
    docs_dir = Path(args.docs_dir)
    docs = load_workspace_docs(workspace_dir, docs_dir)
    if len(docs) < 3:
        print(f"Need at least 3 documents, got {len(docs)}")
        sys.exit(1)
    print(f"Loaded {len(docs)} workspace documents")

    # kind weights: 50% generic get_doc, 20% titled get_doc, 20% summarize, 10% search-answer
    kinds = (
        ["get_doc_generic"] * 50
        + ["get_doc_titled"] * 20
        + ["summarize"] * 20
        + ["search_answer"] * 10
    )

    episodes: List[Dict[str, Any]] = []
    fails = 0
    attempts = 0
    max_attempts = args.num_examples * 20
    while len(episodes) < args.num_examples and attempts < max_attempts:
        attempts += 1
        gold = rng.choice(docs)
        topic = _topic_from_doc(gold["filename"], gold["text"])
        n_hits = 3 if rng.random() < 0.30 else 2
        distractors = pick_distractors(gold, docs, topic, n_hits - 1, rng)
        if len(distractors) < n_hits - 1:
            fails += 1
            continue
        lang = (
            ("de" if rng.random() < args.de_ratio else "en")
            if args.language == "mixed"
            else args.language
        )
        kind = rng.choice(kinds)
        pretty = rng.random() < 0.5
        ep = make_episode(gold, distractors, topic, topic, lang, kind, pretty, rng)
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
