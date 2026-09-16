# Phase 6.2 — Search catalog doc-id bind

**Gate:** `run_regression_gate.py --phase 7` (adds `docid_bind`)  
**Parent:** [PHASE6_AGENTIC.md](PHASE6_AGENTIC.md) — not a new curriculum number  
**Packer:** `scripts/sft/run_phase6_2_mix_pack.ps1`  
**Config:** `configs/sft/phase6_2_docbind.json`

## What this phase teaches

After a multi-hit search catalog, bind `workspace.get_doc` to the rank-1 `doc_id`. Gold is always rank 1.

Measured mix weights, seeds, and packed counts for this project's run are not published.
