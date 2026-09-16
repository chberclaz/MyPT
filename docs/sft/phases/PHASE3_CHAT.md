# Phase 3 — Chat SFT

Curriculum: [SFT README](../README.md) · stage 3 of SFT (after format-lock and operators).

**Gate:** `run_regression_gate.py --phase 3`  
**Prompt:** `CHAT_SYSTEM_PROMPT`  
**Commands:** [SFT_PIPELINE_GUIDE.md](../SFT_PIPELINE_GUIDE.md) §6 / Phase 3.1 restart  
**Config:** `configs/sft/phase3_1_restart.json`

## What this phase teaches

Ordinary bilingual chat, system-over-user conflicts, abstention, citation, and short checkable answers after format-lock and operators.

## Eval-aligned synthetics

The regression suite is substring-scored. Dedicated synthetic slices exist so those tokens appear in gold (hierarchy exact `OK` / `No`, injection refusals that never echo the forbidden token, abstention phrases the suite accepts, literal JSON). Set those slot ratios yourself. This project's percents and the shipped-run narrative are not published.

Do **not** stitch answers in the RAG controller. Do not resume the Phase 3.2 “all `No.`” track.
