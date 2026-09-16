"""Local structured summaries. No API.

Turns a workspace doc into a short abstract: title + topics from headings +
1–2 lead sentences. Train and runtime must use this so SFT matches the tool.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Packed SFT and generate() both use 4096. Leave headroom for system, user,
# toolcall, summary, and the assistant restatement.
CONTEXT_LIMIT = 4096
SUMMARIZE_EPISODE_BUDGET = 1400
SUMMARIZE_CITE_BUDGET = 480
SUMMARIZE_MODEL_INPUT_BUDGET = 900


def gpt2_encode(text: str) -> List[int]:
    try:
        from core.gpt2_encoding import get_gpt2_encoding
        return get_gpt2_encoding().encode(text or "")
    except Exception:
        # ~4 chars/token fallback if tiktoken is missing
        return list(range(max(1, len(text or "") // 4)))


def n_tokens(text: str) -> int:
    return len(gpt2_encode(text))


def clip_indexed_for_context(text: str, max_tokens: int = SUMMARIZE_CITE_BUDGET) -> str:
    """Keep a 4k-safe indexed extract: headings first, then head, then a tail.

    Drops the middle of long docs. That is the cut when summarize is called —
    the index still has the full file; the window only sees this extract.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if n_tokens(text) <= max_tokens:
        return text

    headings = []
    body = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if _HEADING.match(line.strip()):
            headings.append(line.strip())
        else:
            body.append(line)
    outline = "\n".join(headings[:12])
    outline_tok = n_tokens(outline)
    remain = max_tokens - outline_tok - 8
    if remain < 80:
        # Outline alone is too big — keep first heading lines only
        kept = []
        used = 0
        for h in headings:
            t = n_tokens(h) + 1
            if used + t > max_tokens:
                break
            kept.append(h)
            used += t
        return "\n".join(kept)

    body_text = "\n".join(body).strip()
    head_budget = int(remain * 0.75)
    tail_budget = remain - head_budget
    head = _take_tokens(body_text, head_budget)
    tail = _take_tokens_from_end(body_text, tail_budget)
    parts = [p for p in (outline, head) if p]
    if tail and tail not in head:
        parts.append("…")
        parts.append(tail)
    out = "\n".join(parts).strip()
    # Hard clamp
    while n_tokens(out) > max_tokens and len(out) > 40:
        out = out[: int(len(out) * 0.9)].rsplit("\n", 1)[0]
    return out


def _take_tokens(text: str, max_tokens: int) -> str:
    if n_tokens(text) <= max_tokens:
        return text
    ids = gpt2_encode(text)
    try:
        from core.gpt2_encoding import get_gpt2_encoding
        return get_gpt2_encoding().decode(ids[:max_tokens])
    except Exception:
        return text[: max_tokens * 4]


def _take_tokens_from_end(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or n_tokens(text) <= max_tokens:
        return text
    ids = gpt2_encode(text)
    try:
        from core.gpt2_encoding import get_gpt2_encoding
        return get_gpt2_encoding().decode(ids[-max_tokens:])
    except Exception:
        return text[-max_tokens * 4 :]

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ]|[0-9])")


def _clean_line(line: str) -> str:
    return re.sub(r"[#*_`]+", "", line).strip()


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE.split(text)
    skip_start = ("http", "|", "-", "*", "$", "git ", "pip ", "python ", "cd ")
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 30 or len(p) > 220:
            continue
        low = p.lower()
        if p.startswith("http") or p.startswith("|"):
            continue
        if any(low.startswith(s) for s in skip_start):
            continue
        if low.startswith("minimum:") or low.startswith("recommended:"):
            continue
        out.append(p.rstrip(" .") + ".")
    return out


def parse_doc(text: str) -> Tuple[str, List[str], List[str]]:
    """Return (title, headings, body_sentences)."""
    title = ""
    headings: List[str] = []
    body_chunks: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _HEADING.match(line)
        if m:
            level = len(m.group(1))
            name = _clean_line(m.group(2))
            if not name:
                continue
            if level == 1 and not title:
                title = name
            elif name.lower() != (title or "").lower():
                headings.append(name)
            continue
        if line.startswith("```") or line.startswith("---"):
            continue
        body_chunks.append(_clean_line(line))
    if not title:
        for h in headings:
            if h:
                title = h
                break
    if not title:
        sents0 = _sentences(" ".join(body_chunks[:2]))
        title = sents0[0][:80].rstrip(".") if sents0 else "This document"
    sentences = _sentences(" ".join(body_chunks))
    # de-dupe headings, keep order
    seen = set()
    uniq_h = []
    for h in headings:
        key = h.lower()
        if key in seen or key == title.lower():
            continue
        seen.add(key)
        uniq_h.append(h)
    return title, uniq_h, sentences


def _join_topics(topics: List[str], lang: str) -> str:
    topics = topics[:4]
    if not topics:
        return ""
    if lang == "de":
        if len(topics) == 1:
            return topics[0]
        return ", ".join(topics[:-1]) + " und " + topics[-1]
    if len(topics) == 1:
        return topics[0]
    return ", ".join(topics[:-1]) + ", and " + topics[-1]


def structured_summary(text: str, max_length: int = 400, lang: str = "en") -> str:
    """Build a 2–4 sentence abstract. Deterministic, no model/API."""
    title, headings, sentences = parse_doc(text)
    topics = headings[:4]
    parts: List[str] = []
    if lang == "de":
        topic_s = _join_topics(topics, "de")
        if topic_s:
            parts.append(f"{title} behandelt {topic_s}.")
        else:
            parts.append(f"{title} fasst das Thema knapp zusammen.")
    else:
        topic_s = _join_topics(topics, "en")
        if topic_s:
            parts.append(f"{title} covers {topic_s}.")
        else:
            parts.append(f"{title} is a short overview of the topic.")
    # Extra sentence only when the doc has almost no headings (otherwise
    # title+topics already is the abstract and body leads are often junk).
    if len(topics) < 2:
        for s in sentences[:3]:
            if s.lower().startswith(title.lower()):
                continue
            if any(ch in s for ch in ("`", "✅", "[", "{")):
                continue
            parts.append(s)
            break
    out = " ".join(parts).strip()
    if len(out) > max_length:
        out = out[: max_length - 3].rsplit(" ", 1)[0] + "..."
    return out or (text[:max_length].strip() if text else "")
