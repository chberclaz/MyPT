# Phase 6 — Agentic RAG

**Gate:** `run_regression_gate.py --phase 6` (adds `agentic_chain`)  
**Prompt:** `AGENTIC_STANDARD_PROMPT`  
**Commands:** [SFT_PIPELINE_GUIDE.md](../SFT_PIPELINE_GUIDE.md) §9 · packer `scripts/sft/run_phase6_mix_pack.ps1`  
**Config:** `configs/sft/phase6_agentic_rag.json`

## What this phase teaches

Multi-step tool chains (search → get_doc → answer), reasoning, and error recovery.

`--phase 7` / `--phase 8` are subsets of Phase 6 (bind / ground), not new curriculum numbers.

Measured mix weights, val loss, and the narrative of how this project's run was tuned are not published.
