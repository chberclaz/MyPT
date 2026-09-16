# MyPT Documentation

**Tagline:** On-Premise governed AI Foundry  
On-Premise governed Training and Inference System.

Active docs are listed first. Legacy/historical docs remain under archive folders.

## Start Here

| Guide | Description |
| --- | --- |
| [sft/CURRENT_STATE.md](sft/CURRENT_STATE.md) | **Resume point** — GOLD status, what not to restart |
| [training/README.md](training/README.md) | Pretrain order: unified → domain → context |
| [sft/README.md](sft/README.md) | SFT order: format-lock → … → 6.3 |
| [sft/SFT_PIPELINE_GUIDE.md](sft/SFT_PIPELINE_GUIDE.md) | SFT commands, gates, architecture |
| [sft/AUTOPILOT.md](sft/AUTOPILOT.md) | Unattended SFT loop |
| [autopilot_agent.md](../autopilot_agent.md) | Operator + agent RunPod runbook |
| [setup/INSTALL.md](setup/INSTALL.md) | Environment setup |
| [guides/GETTING_STARTED.md](guides/GETTING_STARTED.md) | First (small) training run |
| [guides/QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md) | Command cheat sheet |
| [sft/SCALE_1_4B.md](sft/SCALE_1_4B.md) | Future 1.4B (not started) |

---

## Curriculum (in order)

Do not skip stages. SFT assumes a 4096-context checkpoint from stage 3.

### 1–3. Base model — [training/README.md](training/README.md)

| Stage | Document |
| --- | --- |
| 1. Unified from-scratch pretrain | [training/01_UNIFIED_FROM_SCRATCH.md](training/01_UNIFIED_FROM_SCRATCH.md) |
| 2. Domain corpus | [training/02_DOMAIN_CORPUS.md](training/02_DOMAIN_CORPUS.md) |
| 2. Domain adaptation (continued pretrain) | [training/02_DOMAIN_ADAPTATION.md](training/02_DOMAIN_ADAPTATION.md) |
| 3. Context extension 1024 → 4096 | [training/03_CONTEXT_EXTENSION.md](training/03_CONTEXT_EXTENSION.md) |

Mechanics (shards, configs, lineage): see the table in [training/README.md](training/README.md).

### 4. Supervised fine-tuning — [sft/README.md](sft/README.md)

| Stage | Document |
| --- | --- |
| 1 Format lock / 2 Operators | [sft/SFT_PIPELINE_GUIDE.md](sft/SFT_PIPELINE_GUIDE.md) §5–6 |
| 3 Chat | [sft/phases/PHASE3_CHAT.md](sft/phases/PHASE3_CHAT.md) |
| 4 Multi-turn | [sft/phases/PHASE4_MULTITURN.md](sft/phases/PHASE4_MULTITURN.md) |
| 5 Toolcall | [sft/phases/PHASE5_TOOLCALL.md](sft/phases/PHASE5_TOOLCALL.md) |
| 6 Agentic RAG | [sft/phases/PHASE6_AGENTIC.md](sft/phases/PHASE6_AGENTIC.md) |
| 6.2 Doc-id bind (gate `--phase 7`) | [sft/phases/PHASE6_2_DOCBIND.md](sft/phases/PHASE6_2_DOCBIND.md) |
| 6.3 Toolresult ground (gate `--phase 8`) | [sft/phases/PHASE6_3_GROUND.md](sft/phases/PHASE6_3_GROUND.md) |
| Fallback unified P6 | [sft/phases/PHASE6_UNIFIED.md](sft/phases/PHASE6_UNIFIED.md) |

Mix-role audits: [sft/validation/PHASE3_DATA_SOURCE_VALIDATION.md](sft/validation/PHASE3_DATA_SOURCE_VALIDATION.md) (and siblings). Tags: [sft/TAG_NESTING_REFERENCE.md](sft/TAG_NESTING_REFERENCE.md).

---

## By Topic

### Setup
- [setup/INSTALL.md](setup/INSTALL.md)
- [setup/DEPENDENCIES.md](setup/DEPENDENCIES.md)
- [setup/DOCKER.md](setup/DOCKER.md)
- [setup/PROJECT_STRUCTURE.md](setup/PROJECT_STRUCTURE.md)

### Guides
- [guides/GETTING_STARTED.md](guides/GETTING_STARTED.md)
- [guides/MODEL_SELECTION_GUIDE.md](guides/MODEL_SELECTION_GUIDE.md)
- [guides/QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md)
- [guides/TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)

### Model
- [model/CHECKPOINT_FORMAT.md](model/CHECKPOINT_FORMAT.md)
- [model/GENERATION_GUIDE.md](model/GENERATION_GUIDE.md)
- [model/SPECIAL_TOKENS.md](model/SPECIAL_TOKENS.md)
- [model/TOKENIZATION_COMPARISON.md](model/TOKENIZATION_COMPARISON.md)
- [model/SHARDED_DATASET_IMPLEMENTATION.md](model/SHARDED_DATASET_IMPLEMENTATION.md)

### Webapp
- [webapp/WEBAPP_GUIDE.md](webapp/WEBAPP_GUIDE.md)
- [webapp/AUTHENTICATION.md](webapp/AUTHENTICATION.md)
- [webapp/workspace_api.md](webapp/workspace_api.md)
- [webapp/DOCUMENT_FORMATS.md](webapp/DOCUMENT_FORMATS.md)

### Compliance
- [compliance/AUDIT_COMPLIANCE.md](compliance/AUDIT_COMPLIANCE.md)
- [compliance/PYTORCH_SECURITY_FIX.md](compliance/PYTORCH_SECURITY_FIX.md)

### Reference
- [reference/CONFIG_PRESETS.md](reference/CONFIG_PRESETS.md)
- [reference/CONFIG_PRESETS_SUMMARY.md](reference/CONFIG_PRESETS_SUMMARY.md)
- [reference/WHERE_TO_SEE_PARAMETERS.md](reference/WHERE_TO_SEE_PARAMETERS.md)

### Export / GGUF
- [export/README.md](../export/README.md) — convert GOLD → GGUF; run it in the RAG web UI or on the console
- [export/TOKENIZER_NOTES.md](../export/TOKENIZER_NOTES.md) — `pat_str`, CONTROL tags, trust model

---

## Legacy Docs

Historical material is kept for traceability and reproduction of older runs:

- `docs/archive/`
- `docs/sft/archive/`
- `docs/training/legacy/` — including the old 256-vs-1024 “high context” presets

Commands in those folders may reference retired paths/configs.
