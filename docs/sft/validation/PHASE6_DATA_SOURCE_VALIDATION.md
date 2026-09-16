# Phase 6 Datasource Validation

> **Objective:** Validate every Phase 6 source against agentic-RAG roles.  
> **Rule:** Each source must cover at least 2 critical roles.  
> **Spine:** `generate_sft_tool_episodes.py` + Phase 5 maintenance replay.

---

## 1) Critical roles

1. Multi-step tool chaining (search → get_doc / list → get → summarize)
2. Retrieval-then-cite
3. Workspace tool selection (`search` / `get_doc` / `list_docs` / `summarize`)
4. Tool-result incorporation into the next call
5. Tool vs no-tool (format lock under `AGENTIC_STANDARD_PROMPT`)
6. Grounded abstain / cite-from-context (no search when context is given)
7. 4k-safe summarize / get_doc cites (clip, not full file)
8. Prior-phase maintenance (P4 multiturn, P5 single-step tools)

Known 700M limit: do not chase 3+ hop perfection. Gate is `agentic_chain` ≥ 40%.

---

## 2) Mix

Sources: `agentic_episodes.jsonl`, Phase 5 tool/summarize/direct/grounded replay, Phase 4 replay. Set output-relative weights yourself. This project's mix table is not published.

No HF Dolci tools. No prefix-extract summarize. Episode `system` is hoisted to `AGENTIC_STANDARD_PROMPT`.

---

## 3) 4k lock

- `workspace.get_doc` text: `clip_indexed_for_context` (480 gpt2 tokens)
- `list_docs`: at most 10 entries
- `workspace.summarize` toolresult: agent abstract when present, else `structured_summary`
- Pack: `--enable_packing --pack_block_size 4096`
