# Supervised fine-tuning

SFT starts **after** base pretrain, domain adaptation, and context extension. That order: [../training/README.md](../training/README.md).

**Resume here:** [CURRENT_STATE.md](CURRENT_STATE.md)  
**Commands and gates:** [SFT_PIPELINE_GUIDE.md](SFT_PIPELINE_GUIDE.md)  
**Unattended loop:** [AUTOPILOT.md](AUTOPILOT.md) · [autopilot_agent.md](../../autopilot_agent.md)  
**Tags:** [TAG_NESTING_REFERENCE.md](TAG_NESTING_REFERENCE.md)  
**Later (not started):** [SCALE_1_4B.md](SCALE_1_4B.md)

Init SFT from the Phase 1b context-extension checkpoint (`rope_scale` 4.0, 4096), not from the 1024 pretrain GOLD.

Tuned mix weights, measured eval results, and run history are not published.

---

## Phases (in order)

Phases 1–2 have no separate narrative file; the pipeline guide is the spec. Phases 3–6.3 are short public stubs plus (private) mix-role audits.

| Stage | What it teaches | Spec | Gate |
| --- | --- | --- | --- |
| **1 Format lock** | Special-tag envelope, echo, loss masking | [SFT_PIPELINE_GUIDE.md](SFT_PIPELINE_GUIDE.md) §5 | (format / echo suites in pipeline) |
| **2 Operators** | Short checkable operators | [SFT_PIPELINE_GUIDE.md](SFT_PIPELINE_GUIDE.md) §6 | (operator suites in pipeline) |
| **3 Chat** | Bilingual chat, hierarchy, abstention, citation | [phases/PHASE3_CHAT.md](phases/PHASE3_CHAT.md) | `--phase 3` |
| **4 Multi-turn** | Turn boundaries, context carryover | [phases/PHASE4_MULTITURN.md](phases/PHASE4_MULTITURN.md) | `--phase 4` |
| **5 Toolcall** | Single-step tool use, JSON, grounded answers | [phases/PHASE5_TOOLCALL.md](phases/PHASE5_TOOLCALL.md) | `--phase 5` |
| **6 Agentic RAG** | Multi-step search → get_doc → answer | [phases/PHASE6_AGENTIC.md](phases/PHASE6_AGENTIC.md) | `--phase 6` |
| **6.2 Doc-id bind** | Rank-1 `doc_id` after a multi-hit catalog | [phases/PHASE6_2_DOCBIND.md](phases/PHASE6_2_DOCBIND.md) | `--phase 7` (subset of 6) |
| **6.3 Toolresult ground** | Copy facts from `get_doc` toolresult | [phases/PHASE6_3_GROUND.md](phases/PHASE6_3_GROUND.md) | `--phase 8` (subset of 6) |

6.2 and 6.3 are **not** new curriculum numbers. They are subsets of Phase 6.

Fallback (not the current path): [phases/PHASE6_UNIFIED.md](phases/PHASE6_UNIFIED.md).

## Mix-role audits

Public role lists for HF converters. Slot percents for this project’s run are not published.

- [validation/PHASE3_DATA_SOURCE_VALIDATION.md](validation/PHASE3_DATA_SOURCE_VALIDATION.md)
- [validation/PHASE6_DATA_SOURCE_VALIDATION.md](validation/PHASE6_DATA_SOURCE_VALIDATION.md)
- [validation/PHASE6_2_DATA_SOURCE_VALIDATION.md](validation/PHASE6_2_DATA_SOURCE_VALIDATION.md)
- [validation/PHASE6_3_DATA_SOURCE_VALIDATION.md](validation/PHASE6_3_DATA_SOURCE_VALIDATION.md)

## Archive

Retired SFT guides live under `docs/sft/archive/` (not published).
