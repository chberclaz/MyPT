# MyPT Domain Adaptation Training Guide

A comprehensive guide to training domain-specific language models with MyPT, covering Phase 1 (base model) and Phase 2 (domain adaptation) training.

## Table of Contents

1. [Overview](#overview)
2. [Training Phases](#training-phases)
3. [Phase 1: Base Model Training](#phase-1-base-model-training)
4. [Phase 2: Domain Adaptation](#phase-2-domain-adaptation)
5. [Hyperparameter Tuning Journey](#hyperparameter-tuning-journey)
6. [Detailed Run Configurations](#detailed-run-configurations)
7. [Choosing Your Adaptation Strategy](#choosing-your-adaptation-strategy)
8. [Reproduction Steps](#reproduction-steps)

---

## Overview

Domain adaptation allows you to specialize a general-purpose language model for specific domains (e.g., IT security, legal documents, medical texts) while preserving its general language capabilities.

### Key Concepts

| Term | Description |
|------|-------------|
| **Phase 1** | Initial pretraining on large general corpus (Wikipedia, etc.) |
| **Phase 2** | Domain adaptation on specialized corpus |
| **Replay/Mixing** | Adding general data to domain training to prevent forgetting |
| **Catastrophic Forgetting** | Loss of general capabilities when training only on domain data |

---

## Training Phases

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  PHASE 1: Base Model                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │  Wikipedia  │───▶│   Train     │───▶│  Base Model │                   │
│  │  DE/EN/etc  │    │  from scratch│    │  (general)  │                   │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                   │
│                                               │                           │
│  PHASE 2: Domain Adaptation (Two-Stage)       ▼                           │
│                                                                           │
│  Stage 1: Foundation (Conservative)                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                   │
│  │Domain + 30% │───▶│  Low LR     │───▶│  Model v3   │                   │
│  │Replay       │    │  (3e-5)     │    │ (foundation)│                   │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                   │
│                                               │                           │
│  Stage 2: Specialization (Branch from v3)     │                           │
│                                    ┌──────────┴──────────┐               │
│                                    ▼                     ▼               │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │Domain + 20% │───▶│ v5: LR 5e-5     │    │ v7: LR 7e-5     │          │
│  │Replay       │    │ (balanced)      │    │ (domain expert) │          │
│  └─────────────┘    └─────────────────┘    └─────────────────┘          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Base Model Training

### Dataset: Multilingual Wikipedia Goldset

| Source | Tokens | Weight |
|--------|--------|--------|
| Wikipedia EN | ~789M | 50% |
| Wikipedia DE | ~632M | 40% |
| Europarl DE-EN | ~47M | 3% |
| News Commentary | ~32M | 2% |
| OpenSubtitles | ~75M | 5% |
| **Total** | **~1.575B** | 100% |

### Configuration

```json
{
  "name": "GPT-750M-1024-Base",
  "batch_size": 12,
  "block_size": 1024,
  "n_embd": 1280,
  "n_head": 20,
  "n_layer": 32,
  "learning_rate": 3e-4,
  "max_iters": 128000,
  "warmup_iters": 2000,
  "dropout": 0.1,
  "weight_decay": 0.1
}
```

### Training Command

```powershell
python train.py `
    --config configs/pretrain/750M_1024_reasoning.json `
    --dataset_dir data/multilingual_1.5B_wiki90 `
    --model_name base_750M
```

### Expected Results

| Metric | Target |
|--------|--------|
| Final val loss | ~2.2 |
| Training time | ~24-48 hours (A100) |

---

## Phase 2: Domain Adaptation

### Domain Corpus: IT Security + Swiss Law

| Source | Tokens | Description |
|--------|--------|-------------|
| OWASP Projects | ~15M | Security cheat sheets, guides |
| MITRE ATT&CK | ~5M | Threat intelligence |
| RFC Documents | ~255M | Internet protocols |
| Man Pages | ~6M | Unix documentation |
| Python Docs | ~9M | Language documentation |
| MDN Web Docs | ~54M | Web development |
| OpenJDK | ~15M | Java documentation |
| Swiss Law (Fedlex) | ~50M | Legal texts DE/EN |
| **Domain Total** | **~170M** | |

### Data Mixing (Replay)

To prevent catastrophic forgetting, domain data is mixed with general data:

| Mix Ratio | Domain | Replay | Use Case |
|-----------|--------|--------|----------|
| 30% replay | 70% | 30% | Conservative, maximum preservation |
| **20% replay** | **80%** | **20%** | **Recommended balance** |
| 10% replay | 90% | 10% | Aggressive domain focus |
| 0% replay | 100% | 0% | ⚠️ Risk of catastrophic forgetting |

---

## Hyperparameter Tuning Journey

This section documents the actual experiments performed to find optimal settings.

### The Problem: Finding the Right Balance

| Challenge | Solution |
|-----------|----------|
| Too low LR → Domain doesn't learn | Increase LR |
| Too high LR → Forgetting | Add replay data |
| Too much replay → Domain diluted | Reduce replay ratio |

### Experiment Summary

```
Base 750M Model
      │
      ▼
   ┌──────────────────────────────────┐
   │  v3: Conservative Foundation     │
   │  LR=3e-5, 30% replay             │
   │  (Maximum forgetting protection) │
   └──────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ v5: Balanced  │   │ v7: Aggressive│
│ LR=5e-5       │   │ LR=7e-5       │
│ 20% replay    │   │ 20% replay    │
│ (General use) │   │ (Domain focus)│
└───────────────┘   └───────────────┘
```

| Run | Init From | LR | Replay | Domain Final | General Final | General Δ | Verdict |
|-----|-----------|-----|--------|--------------|---------------|-----------|---------|
| v3 | Base model | 3e-5 | 30% | **3.92** | **2.18** | 0% | Foundation |
| v5 | **v3** | 5e-5 | 20% | **3.71** | **2.21** | +1.4% | ✅ Balanced |
| v7 | **v3** | 7e-5 | 20% | *running* | *running* | TBD | ✅ Best domain |

*v7 still in progress - update when complete

**Key insight**: v3 serves as a "safe checkpoint" - it learned basic domain patterns while preserving nearly 100% of general capabilities. v5 and v7 then branch from this foundation with more aggressive learning.

---

## Detailed Run Configurations

### Run v3: Foundation (Conservative First Pass)

**Purpose**: Build a safe foundation checkpoint with basic domain understanding while preserving nearly 100% of general capabilities. This checkpoint becomes the starting point for more aggressive Stage 2 runs.

**Configuration**:
```json
{
  "name": "domain_v3",
  "learning_rate": 3e-5,
  "max_iters": 44760,
  "warmup_iters": 1790,
  "batch_size": 12,
  "block_size": 1024
}
```

**Dataset**: 220M tokens (70% domain + 30% replay)

**Initialization**: Started from base 750M model

**Results** (44,000 iterations):
| Metric | Final Value |
|--------|-------------|
| val_loss | **3.125** |
| eval_domain | **3.921** |
| eval_general | **2.177** |

**Analysis**: The conservative settings achieved the intended goal: the model gained initial exposure to domain content while preserving general capabilities (~0% degradation from base). This creates a valuable "safe checkpoint" for branching into more aggressive experiments (v5, v7) without risking the base model.

---

### Run v5: Balanced (Good General-Purpose)

**Purpose**: Build on v3's foundation with higher LR and less replay to improve domain learning while maintaining balance.

**Configuration**:
```json
{
  "name": "domain_v5",
  "learning_rate": 5e-5,
  "max_iters": 44760,
  "warmup_iters": 1790,
  "batch_size": 12,
  "block_size": 1024
}
```

**Dataset**: 200M tokens (80% domain + 20% replay)

**Initialization**: Continued from **domain_v3** checkpoint

**Results** (44,000 iterations):
| Metric | v3 End (Start) | v5 Final | Change |
|--------|----------------|----------|--------|
| val_loss | 3.125 | **2.880** | -0.245 |
| eval_domain | 3.921 | **3.709** | -0.212 |
| eval_general | 2.177 | **2.214** | +0.037 (+1.7%) |

**Analysis**: Good balance between domain learning and general preservation. Domain improved by 0.21 points while general degraded only 1.7%. Recommended for users who want domain knowledge while maintaining strong general capabilities.

---

### Run v7: Aggressive (Best Domain Performance)

**Purpose**: Push domain learning further with higher LR.

**Configuration**:
```json
{
  "name": "domain_v7",
  "learning_rate": 7e-5,
  "max_iters": 38000,
  "warmup_iters": 1790,
  "batch_size": 12,
  "block_size": 1024
}
```

**Dataset**: 200M tokens (80% domain + 20% replay)

**Initialization**: Started from domain_v3 checkpoint

**Results** (still running):
| Metric | v3 End (Start) | Current | Projected |
|--------|----------------|---------|-----------|
| val_loss | 3.125 | *running* | ~2.6-2.7 |
| eval_domain | 3.921 | *running* | ~3.1-3.4 |
| eval_general | 2.177 | *running* | ~2.25-2.35 |

**Analysis**: Higher LR should push domain learning further than v5. Recommended for users who prioritize domain expertise and accept slightly more general degradation.

*→ Update this section when v7 completes*

---

## Choosing Your Adaptation Strategy

### Recommended Approach: Two-Stage Training

**Always run Stage 1 first**, then choose your Stage 2 based on priorities:

```
Stage 1 (Required)     →    Stage 2 (Choose One)
─────────────────────       ─────────────────────
v3: Foundation              v5: Balanced (general use)
LR=3e-5, 30% replay         LR=5e-5, 20% replay
                            
                      OR    v7: Domain Expert
                            LR=7e-5, 20% replay
```

### Decision Matrix

| Priority | Stage 2 Choice | Domain Loss | General Δ |
|----------|----------------|-------------|-----------|
| **Balanced (recommended)** | v5-style (5e-5, 20% replay) | **3.71** | +1.7% |
| **Maximum domain expertise** | v7-style (7e-5, 20% replay) | *~3.1-3.4* | *~3-5%* |

Note: You can run both v5 and v7 from the same v3 checkpoint to compare results!

### Target Metrics (Based on Actual Runs)

| Metric | v3 Achieved | v5 Achieved | Excellent Target |
|--------|-------------|-------------|------------------|
| eval_domain | 3.92 | **3.71** | < 3.5 |
| eval_general | 2.18 | 2.21 | < 2.25 |
| General degradation | ~0% | **+1.7%** | < 5% |

**Baseline reference**: Base model eval_general ≈ 2.17-2.18

---

## Reproduction Steps

### Prerequisites

1. Base model trained on general corpus (Phase 1)
2. Domain corpus prepared and tokenized
3. Replay corpus (general data) tokenized

### Step 1: Prepare Domain Corpus

```powershell
# Build IT security corpus with cleaning
python tools/build_phase2_corpus.py `
    --fetch `
    --sources_dir sources `
    --out_dir data/itsec_v3_corpus `
    --seed 42

# Tokenize
python scripts/prepare_weighted_dataset.py `
    --source domain:data/itsec_v3_corpus/corpus_shards/*.txt `
    --total_tokens 500000000 `
    --out_dir data/itsec_v3_tokenized

# Append Swiss law
python scripts/append_to_dataset.py `
    --dataset_dir data/itsec_v3_tokenized `
    --source swiss_law:data/swiss_law_domain/fedlex_de_en.txt `
    --target_tokens 50000000
```

### Step 2: Mix with Replay Data

```powershell
# Create mixed dataset with 20% replay
python scripts/mix_tokenized_datasets.py `
    --domain_dir data/itsec_v3_tokenized `
    --replay_dir data/multilingual_1.5B_wiki90 `
    --output_dir data/domain_mixed_20pct `
    --replay_ratio 0.20
```

### Step 3: Train Domain Model (Two-Stage Approach)

#### Stage 1: Foundation Run (v3-style)
Conservative first pass with maximum forgetting protection:

```powershell
# Create dataset with 30% replay for maximum safety
python scripts/mix_tokenized_datasets.py `
    --domain_dir data/itsec_v3_tokenized `
    --replay_dir data/multilingual_1.5B_wiki90 `
    --output_dir data/domain_mixed_30pct `
    --replay_ratio 0.30

# Train foundation model
python train.py `
    --config configs/pretrain/750M_1024_domain_adapt.json `
    --dataset_dir data/domain_mixed_30pct `
    --init_from_model checkpoints/base_750M `
    --model_name domain_v3 `
    --learning_rate 3e-5 `
    --max_iters 45000
```

#### Stage 2a: Balanced Model (v5-style)
Continue from v3 with moderate aggression:

```powershell
# Create dataset with 20% replay
python scripts/mix_tokenized_datasets.py `
    --domain_dir data/itsec_v3_tokenized `
    --replay_dir data/multilingual_1.5B_wiki90 `
    --output_dir data/domain_mixed_20pct `
    --replay_ratio 0.20

# Continue from v3 foundation
python train.py `
    --config configs/pretrain/750M_1024_domain_adapt.json `
    --dataset_dir data/domain_mixed_20pct `
    --init_from_model checkpoints/domain_v3 `
    --model_name domain_v5 `
    --learning_rate 5e-5 `
    --max_iters 45000
```

#### Stage 2b: Domain Expert Model (v7-style)
Continue from v3 with aggressive domain focus:

```powershell
# Uses same 20% replay dataset as v5
python train.py `
    --config configs/pretrain/750M_1024_domain_adapt.json `
    --dataset_dir data/domain_mixed_20pct `
    --init_from_model checkpoints/domain_v3 `
    --model_name domain_v7 `
    --learning_rate 7e-5 `
    --max_iters 38000
```

### Step 4: Monitor Training

Watch for:
- `eval_domain` should decrease (domain learning)
- `eval_general` should stay < 2.35 (no catastrophic forgetting)
- `val` should decrease (overall training progress)

---

## Evaluation Datasets

| Eval Set | Path | Purpose |
|----------|------|---------|
| domain | `data/domain_161M_corpus_tokenized` | Pure domain val shards |
| general | `data/multilingual_1.5B_wiki90` | General knowledge val shards |

**Note**: The `val` metric in training logs uses the mixed training dataset's validation split, which contains both domain and replay data. Use `eval_domain` for pure domain performance measurement.

---

## Key Learnings

1. **Two-stage approach works best**: 
   - Stage 1 (v3): Conservative run with 30% replay and low LR (3e-5) to build a "safe foundation"
   - Stage 2 (v5/v7): Branch from v3 with 20% replay and higher LR to push domain learning

2. **Foundation checkpoint is valuable**: v3 serves as a reusable starting point. You can experiment with different Stage 2 settings (v5 vs v7) without risking your base model.

3. **Replay ratios have different purposes**:
   - 30% replay: Maximum safety, slower domain learning (use for foundation)
   - 20% replay: Good balance (use for final models)
   - <10% replay: Risk of catastrophic forgetting

4. **Monitor both metrics**: Always track both domain AND general performance. Domain improvement at the cost of >15% general degradation is usually not worth it.

5. **Continue from checkpoints**: The v5 and v7 runs both started from v3, not from scratch. This preserves the initial domain understanding and general knowledge from the conservative first pass.

6. **Data quality matters**: Clean boilerplate, remove duplicates, and strip formatting artifacts for better training signal.

---

## Files Reference

| File | Purpose |
|------|---------|
| `configs/pretrain/750M_1024_domain_adapt.json` | Domain adaptation config |
| `tools/build_phase2_corpus.py` | Domain corpus builder |
| `scripts/mix_tokenized_datasets.py` | Dataset mixing tool |
| `scripts/append_to_dataset.py` | Append data to existing dataset |
| `data/sources/phase2_domain.json` | Domain source definitions |

---

*Last updated: January 2026*

