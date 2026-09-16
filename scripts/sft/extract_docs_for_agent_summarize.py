#!/usr/bin/env python3
"""Extract indexed workspace docs for a separate agent to summarize.

Writes two files:
  - extracts.jsonl  one row per doc: id, filename, full indexed chunk text
  - agent_pack.md   compact pack you paste/send to the summarizing agent

After the agent returns summaries, write them as
  data/sft_phase5_intermediate/doc_agent_summaries.jsonl
with keys: doc_id, filename, summary_en, summary_de
then run build_summarize_sft_from_agent.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.document import DocumentLoader, TextChunker


def _doc_id(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract indexed docs for agent summarize")
    ap.add_argument("--docs_dir", default="workspace/docs")
    ap.add_argument("--output_dir", default="data/sft_phase5_intermediate")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--chunk_overlap", type=int, default=100)
    ap.add_argument("--pack_chars", type=int, default=2200,
                    help="Max indexed chars shown per doc in the agent pack")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_absolute():
        docs_dir = PROJECT_ROOT / docs_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DocumentLoader()
    documents = loader.load_directory(str(docs_dir))
    if not documents:
        print("No documents found", file=sys.stderr)
        sys.exit(1)

    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    by_source: dict = {}
    for ch in chunks:
        src_meta = ch.source if isinstance(ch.source, dict) else {}
        src = src_meta.get("file") or src_meta.get("filename") or ""
        by_source.setdefault(src, []).append(ch.text or "")

    extracts = []
    pack_lines = [
        "# Agent summarize pack",
        "",
        "For each DOC, write a 2–4 sentence abstract (not a prefix dump).",
        "Return JSONL lines: {\"doc_id\", \"filename\", \"summary_en\", \"summary_de\"}",
        "",
    ]

    for doc in documents:
        src = doc.source
        did = _doc_id(src)
        indexed = "\n\n".join(t for t in by_source.get(src, []) if t)
        if not indexed:
            indexed = doc.text
        row = {
            "doc_id": did,
            "filename": doc.filename,
            "source": src,
            "n_chars": len(doc.text),
            "n_chunks": len(by_source.get(src, [])),
            "indexed_text": indexed,
        }
        extracts.append(row)
        excerpt = indexed[: args.pack_chars]
        if len(indexed) > args.pack_chars:
            excerpt += "\n…[truncated for pack; full text is in extracts.jsonl]"
        pack_lines.append(f"## DOC {did}  `{doc.filename}`  ({len(indexed)} indexed chars)")
        pack_lines.append("")
        pack_lines.append(excerpt)
        pack_lines.append("")
        pack_lines.append("---")
        pack_lines.append("")

    ext_path = out_dir / "doc_extracts.jsonl"
    with ext_path.open("w", encoding="utf-8") as f:
        for r in extracts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pack_path = out_dir / "doc_agent_pack.md"
    pack_path.write_text("\n".join(pack_lines), encoding="utf-8")
    print(f"Wrote {len(extracts)} extracts -> {ext_path}")
    print(f"Agent pack -> {pack_path}")
    print("Summarize the pack, then write doc_agent_summaries.jsonl and run")
    print("  python scripts/sft/build_summarize_sft_from_agent.py")


if __name__ == "__main__":
    main()
