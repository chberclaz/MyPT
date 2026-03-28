# MyPT Project Structure

## Overview

This document reflects the current repository layout used by training, SFT, evaluation, and webapp workflows.

## Top-Level Layout

```text
MyPT/
├── README.md
├── train.py
├── generate.py
├── core/                  # Model, loaders, checkpointing, tokens, lineage, eval guards
├── scripts/               # Operational scripts grouped by domain
├── configs/               # base, sft, sft_eval, audit configs
├── docs/                  # Active docs + archive docs
├── webapp/                # FastAPI app, auth, routers
├── tools/                 # Corpus builders and converters
├── tests/                 # Unit/integration tests
├── data/                  # Local datasets/artifacts (workspace data)
├── checkpoints/           # Local model checkpoints
└── sources/               # Raw source corpora inputs
```

## `scripts/` Structure

```text
scripts/
├── data_prep/             # Dataset build/tokenization/mixing utilities
├── sft/                   # SFT dataset generation, preparation, validation, analysis
├── eval/                  # Eval suites and gates
├── model/                 # Model utilities (inspect, params, migration, embedding tools)
├── unified_build/         # Unified pretraining data pipeline
├── utils/                 # General helpers (e.g. show_configs)
├── debug/                 # Debug/diagnostic helpers
├── translation/           # Translation-related data tooling
├── workspace_chat.py
└── build_rag_index.py
```

## `configs/` Structure

```text
configs/
├── base/                  # Pretraining/base model configs
│   ├── 750M_unified_v1.json
│   ├── 750M_phase1_5_induction*.json
│   └── archiv/            # Historical base presets
├── sft/                   # Active SFT phase configs + sft archive
├── sft_eval/              # Prompt/eval config packs
└── audit/                 # Audit/compliance config
```

## `docs/` Structure

- `docs/README.md` is the documentation index.
- Active references live under:
  - `docs/setup/`
  - `docs/guides/`
  - `docs/training/`
  - `docs/sft/`
  - `docs/model/`
  - `docs/webapp/`
  - `docs/compliance/`
  - `docs/reference/`
  - `docs/specs/`
- Historical material remains in:
  - `docs/archive/`
  - `docs/sft/archive/`
  - `docs/training/legacy/`

## Conventions

- User-facing entry points:
  - `train.py`
  - `generate.py`
- Prefer script paths by subgroup:
  - `scripts/data_prep/...`
  - `scripts/sft/...`
  - `scripts/model/...`
  - `scripts/utils/...`
- Prefer config paths by domain:
  - `configs/base/...`
  - `configs/sft/...`


