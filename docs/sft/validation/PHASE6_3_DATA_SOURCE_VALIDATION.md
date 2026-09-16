# Phase 6.3 Datasource Validation

Canonical phase spec: **[PHASE6_3_GROUND.md](../phases/PHASE6_3_GROUND.md)**.

This file is the short §7 audit extract (each source ≥2 critical roles).

> **Objective:** After get_doc, copy facts from the toolresult into the answer.  
> **Rule:** Each source must cover at least 2 critical roles.  
> **Spine:** `generate_phase6_3_ground_sft.py` + 6.2 bind / P6 / P5 / P4 maintenance.

## 1) Critical roles

1. Verbatim span from get_doc `text` / summarize `summary` in the final answer
2. Think after retrieve is disjoint from the answer (no “grounded answer about {topic}”)
3. Rank-1 bind maintenance (gold is always rank 1)
4. Live JSON shape (search catalog + get_doc dump)
5. Empty search catalog → abstain (no get_doc on unrelated neighbors)
6. Multi-step tool chaining (search → get_doc / summarize → answer)
7. Prior-phase maintenance (6.2 bind, P6 agentic, P5 tools, P4 multiturn)

Known 700M limit: do not chase math (`regression_basic` is structural). Gate is `--phase 8` with `toolresult_ground` ≥ 70% and `search_miss` ≥ 60%.

## 2) Mix

Spine `phase6_3_ground.jsonl` plus 6.2 / P6 / P5 / P4 maintenance. Set output-relative weights yourself. This project's mix table is not published.

Episode `system` is hoisted to `AGENTIC_STANDARD_PROMPT`. No HF. No RAG extractive-answer fallback.

## 3) 4k lock

- `workspace.get_doc` text: `clip_indexed_for_context` (480 gpt2 tokens)
- Answer copy: first paragraphs, ~720 chars
- Pack: `--enable_packing --pack_block_size 4096`

## 4) Remediation

- Fail `toolresult_ground` only → raise the ground slice, or switch the spine to **get_doc_final** (copy-only after get_doc JSON). Init from the last better GOLD, not a worse remake. Measured iter outcomes are not published.
