# Phase 6.2 Datasource Validation

Canonical phase spec: **[PHASE6_2_DOCBIND.md](../phases/PHASE6_2_DOCBIND.md)**.

This file is the short §7 audit extract (each source ≥2 critical roles).

> **Objective:** Teach rank-1 `doc_id` bind from a 2–3 hit search catalog.  
> **Rule:** Each source must cover at least 2 critical roles.  
> **Spine:** `generate_phase6_2_docbind_sft.py` + Phase 6/5 maintenance replay.

## 1) Critical roles

1. Multi-hit catalog bind (gold is always rank 1)
2. Generic-think get_doc / summarize (no title in think — the live failure)
3. Distractor rejection (do not copy hit 2/3 hex ids)
4. Live JSON shape (`rank`, `title`, `filename`, `snippet`, `doc_id` last; compact and pretty)
5. Multi-step tool chaining maintenance (search → get_doc / summarize → cite)
6. Prior-phase maintenance (P6 agentic, P5 tools, P4 multiturn, grounded abstain)

Known 700M limit: do not chase math (`regression_basic` is structural). Gate is `--phase 7` with `docid_bind` ≥ 70%.

## 2) Mix

Spine `phase6_2_docbind.jsonl` plus P6/P5/P4 maintenance. Set output-relative weights yourself. This project's mix table is not published.

Episode `system` is hoisted to `AGENTIC_STANDARD_PROMPT`. No HF. No “get_doc rank 2 when think names rank 2.”

## 3) 4k lock

- `workspace.get_doc` text: `clip_indexed_for_context` (480 gpt2 tokens)
- Search snippets: 280 chars
- Pack: `--enable_packing --pack_block_size 4096`

## 4) Remediation

- Fail `docid_bind` only → upweight bind, shorter retrain from `phase6_2_docbind`.
- Fail `toolcall_basic` / `agentic_chain` → more P6 replay.
- Never remediate math. Never resume 3.2 / dead P4 / prefix-extract P5.
