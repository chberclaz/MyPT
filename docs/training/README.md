# Training curriculum

How a MyPT base model is built, in order. Do not skip stages. Supervised fine-tuning is a **later** tree: [docs/sft/README.md](../sft/README.md).

```
1. Unified from-scratch pretrain     01_UNIFIED_FROM_SCRATCH.md
2. Domain corpus + domain adaptation 02_DOMAIN_CORPUS.md → 02_DOMAIN_ADAPTATION.md
3. Context extension 1024 → 4096     03_CONTEXT_EXTENSION.md
4. SFT phases 1–6 (+ 6.2 / 6.3)      ../sft/README.md
```

Resume / GOLD status: [../sft/CURRENT_STATE.md](../sft/CURRENT_STATE.md).

---

## 1. Unified from-scratch pretrain

Train the LLaMA-2-style ~750M model from random init on a mixed general corpus (code for induction heads, extractive Q&A for retrieval heads, plus general text).

- Spec: [01_UNIFIED_FROM_SCRATCH.md](01_UNIFIED_FROM_SCRATCH.md)
- Config: `configs/base/750M_unified_v1.json`
- Future 1.4B token floor: [../sft/SCALE_1_4B.md](../sft/SCALE_1_4B.md)

## 2. Domain

Build a domain corpus, then continued-pretrain the unified checkpoint on it (replay / mixing so general capability does not collapse).

- Corpus builder: [02_DOMAIN_CORPUS.md](02_DOMAIN_CORPUS.md)
- Adaptation (continued pretrain, catastrophic-forgetting eval): [02_DOMAIN_ADAPTATION.md](02_DOMAIN_ADAPTATION.md)

This is **not** SFT Phase 2 (operators). Domain adaptation is continued pretrain, before context extension.

## 3. Context extension (1024 → 4096)

Position interpolation (`rope_scale` 4.0) so RAG prompts, tool traces, and multi-turn chat fit. Init from the domain (or unified) GOLD, not from random init.

- Spec: [03_CONTEXT_EXTENSION.md](03_CONTEXT_EXTENSION.md)

## 4. Supervised fine-tuning

Format-lock → operators → chat → multi-turn → toolcall → agentic RAG (6.2 bind, 6.3 ground). Commands and gates: [../sft/SFT_PIPELINE_GUIDE.md](../sft/SFT_PIPELINE_GUIDE.md). Phase list: [../sft/README.md](../sft/README.md).

---

## Mechanics (not a stage)

These apply across the curriculum. They are not extra phases.

| Document | Topic |
| --- | --- |
| [LARGE_DATASET_TRAINING.md](LARGE_DATASET_TRAINING.md) | Sharded corpora, `dataset_dir` |
| [TRAINING_CONFIG.md](TRAINING_CONFIG.md) | Config fields, GOLD checkpoint |
| [DATA_PERSISTENCE.md](DATA_PERSISTENCE.md) | Dataset / checkpoint layout |
| [DATASET_LINEAGE_STANDARD.md](DATASET_LINEAGE_STANDARD.md) | Lineage schema |
| [DATASET_COVERAGE_ANALYSIS.md](DATASET_COVERAGE_ANALYSIS.md) | Coverage checks |
| [PARAMETER_CALCULATION.md](PARAMETER_CALCULATION.md) | Parameter counting |
| [TOKEN_ACCURACY_SATURATION.md](TOKEN_ACCURACY_SATURATION.md) | Saturation detector |

SFT mix-role audits live next to the SFT phases, e.g. [../sft/validation/PHASE3_DATA_SOURCE_VALIDATION.md](../sft/validation/PHASE3_DATA_SOURCE_VALIDATION.md).

## Legacy

Historical 256-vs-1024 presets and retired data-script notes live under `docs/training/legacy/` (not published). Current context target is **4096** via stage 3, not the old 1024 “high context” configs.
