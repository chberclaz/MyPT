# Phase 6.3 — Toolresult answer ground

**Gate:** `run_regression_gate.py --phase 8` (adds `toolresult_ground`, `search_miss`)  
**Parent:** [PHASE6_AGENTIC.md](PHASE6_AGENTIC.md) — not a new curriculum number  
**Packer:** `scripts/sft/run_phase6_3_mix_pack.ps1`  
**Config:** `configs/sft/phase6_3_ground.json`

## What this phase teaches

After `workspace.get_doc`, copy facts from the toolresult body. Empty search catalog → abstain. Do **not** stitch the answer in the RAG controller.

Measured mix weights, copy scores, and the narrative of how this project's run was tuned are not published.
