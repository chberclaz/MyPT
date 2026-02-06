# Phase 3a SFT Complete Guide

**Last updated:** January 2026  
**Status:** Active  
**Consolidates:** PHASE3A_TRAINING_PIPELINE.md, phase3a_training_regime.md, phase3a1_checks_*.md, phase3a1_operator_*.md

---

## Table of Contents

1. [Overview & Goals](#overview--goals)
2. [Dataset Generation Scripts](#dataset-generation-scripts)
3. [Dataset Preparation (with Packing)](#dataset-preparation-with-packing)
4. [Training Phases](#training-phases)
5. [Operator Learning (Abstraction Test)](#operator-learning-abstraction-test)
6. [Evaluation Gates](#evaluation-gates)
7. [Troubleshooting & Red Lines](#troubleshooting--red-lines)
8. [Quick Reference](#quick-reference)

---

## Overview & Goals

Phase 3a teaches the model conversational behavior through Supervised Fine-Tuning (SFT):

| Phase | Goal | Key Learning |
|-------|------|--------------|
| **3a-1α** | Core format lock | Chat tags, basic copy, short answers |
| **3a-1β** | Echo expansion | Diverse echo templates, anti-echo |
| **3a-1γ** | BPE-safe gibberish | True copy with novel words |
| **3a-2** | Knowledge training | Math, facts, Q&A patterns |
| **Operators** | Abstraction test | COPY, WRAP, EXTRACT operators |

**Prerequisites:**
- Base model: `domain_v5_sft_ready` (with pre-weighted special tags)
- All scripts verified for CLI arguments

---

## Dataset Generation Scripts

### generate_format_lock_dataset.py

Generates Q&A pairs for format locking.

```bash
python scripts/generate_format_lock_dataset.py \
    --mode minimal \
    --math exclude \
    --output_dir data/sft_format_lock_alpha
```

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--output_dir` | path | `data/sft_format_lock` | Output directory |
| `--mode` | `full`, `minimal` | `full` | Dataset size (~1000 vs ~11000) |
| `--math` | `include`, `exclude`, `only`, `minimal` | `include` | Math content mode |

### generate_echo_dataset.py

Generates echo/repeat instruction datasets.

```bash
python scripts/generate_echo_dataset.py \
    --gibberish exclude \
    --anti_echo_ratio 0.2 \
    --contrast_ratio 0.2 \
    --output_dir data/sft_echo_beta
```

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--output_dir` | path | `data/sft_echo` | Output directory |
| `--gibberish` | `include`, `exclude`, `only` | `exclude` | Gibberish mode |
| `--bpe_safe` | flag | True | Filter gibberish by BPE token count |
| `--max_target_tokens` | int | 4 | Max tokens for BPE-safe filtering |
| `--anti_echo_ratio` | float | 0.2 | Fraction of anti-echo examples |
| `--contrast_ratio` | float | 0.2 | Fraction of contrast pairs |
| `--max_examples` | int | None | Optional cap on examples |

### generate_operator_dataset.py

Generates operator learning dataset with unique payloads and template splits.

```bash
python scripts/generate_operator_dataset.py \
    --output_dir data/sft_operator_v2 \
    --n_train 9000 \
    --n_val 1000 \
    --max_words 4 \
    --max_tokens 12
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `data/sft_operator` | Output directory |
| `--n_train` | 5000 | Training examples |
| `--n_val` | 500 | Validation examples |
| `--max_words` | 4 | Max words per payload (1-4) |
| `--max_tokens` | 12 | Max BPE tokens per payload |
| `--word_dist` | `0.35,0.30,0.20,0.15` | Word count distribution |
| `--seed_train` | 42 | Train seed |
| `--seed_val` | 12345 | Val seed (MUST differ) |
| `--no_german` | flag | Exclude German variants |

**Output:**
- `operator_train.jsonl` — Unique payloads, train templates
- `operator_val.jsonl` — Unique payloads, DIFFERENT templates

### mix_sft_jsonl.py

Mix multiple datasets with explicit weights.

```bash
python scripts/mix_sft_jsonl.py \
    --inputs data/file1.jsonl data/file2.jsonl data/file3.jsonl \
    --weights 0.70 0.20 0.10 \
    --output data/mixed.jsonl \
    --shuffle
```

| Argument | Description |
|----------|-------------|
| `--inputs` | Space-separated list of input JSONL files |
| `--weights` | Explicit weights for each input |
| `--output` | Output JSONL file path |
| `--shuffle` | Shuffle the output |
| `--seed` | Random seed for shuffling |

---

## Dataset Preparation (with Packing)

### Basic Preparation (No Packing)

```bash
python scripts/prepare_chat_sft.py \
    --input data/my_dataset.jsonl \
    --output_dir data/sft_ready/my_dataset \
    --val_split 0.1
```

### Separate Train/Val Files (Strict Separation)

For operator datasets where train/val must have no overlap:

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_operator/operator_train.jsonl \
    --val_file data/sft_operator/operator_val.jsonl \
    --output_dir data/sft_operator/prepared
```

**Automatic validation:**
- Hard fails on payload overlap
- Hard fails on template overlap

### PACKED Preparation (Optimized for Short Episodes)

**The Problem:** Short episodes (like operator learning) have ~13% mask ratio. 87% of compute is wasted on padding.

**The Solution:** Pack multiple episodes into 1024-token sequences.

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_operator/operator_train.jsonl \
    --val_file data/sft_operator/operator_val.jsonl \
    --output_dir data/sft_operator/packed \
    --enable_packing \
    --pack_block_size 1024 \
    --pack_by_field "_meta.operator"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_packing` | flag | Enable sequence packing |
| `--pack_block_size` | 1024 | Target packed sequence length |
| `--pack_by_field` | None | Group episodes by this metadata field before packing |
| `--pack_shuffle` | True | Shuffle episodes before packing |
| `--no_pack_shuffle` | flag | Don't shuffle before packing |

**Key properties:**
- No cross-episode splitting (complete episodes only)
- Episodes grouped by category before packing
- `<myPT_eot>` already in each episode acts as separator
- Padding with mask=0 at end if needed

**Expected improvement:**

| Metric | Before (unpacked) | After (packed) |
|--------|------------------|----------------|
| Mask ratio | ~13% | ~50-70% |
| Supervised tokens/step | ~300 | ~2000-3500 |
| Training efficiency | 1x | 4-5x |

### Inspect Prepared Dataset

```bash
python scripts/inspect_sft_dataset.py \
    --dataset_dir data/sft_operator/packed \
    --show_samples 3
```

Shows:
- Basic statistics (episodes, tokens, mask ratio)
- Packing info (if enabled)
- Supervised token counts
- Recommended max_iters based on masked-token passes
- Decoded sample sequences

---

## Training Phases

### Phase 3a-1α: Core Format Lock

**Goal:** Teach chat structure, special tokens, basic copy (~1000 examples)

```bash
# 1. Generate dataset
python scripts/generate_format_lock_dataset.py \
    --mode minimal \
    --math exclude \
    --output_dir data/sft_format_lock_alpha

# 2. Prepare dataset
python scripts/prepare_chat_sft.py \
    --input data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --output_dir data/sft_ready/phase3a1_alpha \
    --val_split 0.1

# 3. Train
python train.py \
    --model_name phase3a1_alpha \
    --dataset_dir data/sft_ready/phase3a1_alpha \
    --init_from_model domain_v5_sft_ready \
    --config_file configs/sft1/750M_phase3a1_alpha.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json

# 4. Eval Gate (MANDATORY)
python scripts/sft_eval_suite.py --model phase3a1_alpha -v
python scripts/check_tie_weights.py --model phase3a1_alpha

# 5. Manual Test
python generate.py --model phase3a1_alpha \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say Hello.</myPT_user><myPT_assistant>" \
    --temperature 0
```

**Expected:** `Hello.`

---

### Phase 3a-1β: Echo Expansion

**Goal:** Diverse echo templates + anti-echo to prevent blind copying

```bash
# 1. Generate dataset
python scripts/generate_echo_dataset.py \
    --gibberish exclude \
    --anti_echo_ratio 0.2 \
    --contrast_ratio 0.2 \
    --output_dir data/sft_echo_beta

# 2. Mix with alpha replay
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --weights 0.85 0.15 \
    --output data/sft_mixed/phase3a1_beta_mixed.jsonl \
    --shuffle

# 3. Prepare
python scripts/prepare_chat_sft.py \
    --input data/sft_mixed/phase3a1_beta_mixed.jsonl \
    --output_dir data/sft_ready/phase3a1_beta \
    --val_split 0.1

# 4. Train (from alpha)
python train.py \
    --model_name phase3a1_beta \
    --dataset_dir data/sft_ready/phase3a1_beta \
    --init_from_model phase3a1_alpha \
    --config_file configs/sft1/750M_phase3a1_beta.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json

# 5. Eval Gate
python scripts/sft_eval_suite.py --model phase3a1_beta -v
```

---

### Phase 3a-1γ: BPE-Safe Gibberish (True Copy)

**Goal:** Learn true copy with BPE-safe nonsense words

**What is BPE-safe gibberish?** Random gibberish creates rare BPE sub-tokens that are unlearnable. BPE-safe gibberish is filtered to max 4 tokens per word.

```bash
# 1. Generate BPE-safe gibberish dataset
python scripts/generate_echo_dataset.py \
    --gibberish only \
    --bpe_safe \
    --max_target_tokens 4 \
    --anti_echo_ratio 0.2 \
    --contrast_ratio 0.2 \
    --output_dir data/sft_echo_gamma

# 2. Mix: 35% gibberish + 50% beta + 15% alpha
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_echo_gamma/mypt_echo_diverse.jsonl \
             data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --weights 0.35 0.50 0.15 \
    --output data/sft_mixed/phase3a1_gamma_mixed.jsonl \
    --shuffle

# 3. Prepare
python scripts/prepare_chat_sft.py \
    --input data/sft_mixed/phase3a1_gamma_mixed.jsonl \
    --output_dir data/sft_ready/phase3a1_gamma \
    --val_split 0.1

# 4. Train (from beta)
python train.py \
    --model_name phase3a1_gamma \
    --dataset_dir data/sft_ready/phase3a1_gamma \
    --init_from_model phase3a1_beta \
    --config_file configs/sft1/750M_phase3a1_gamma.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json

# 5. Eval Gate
python scripts/sft_eval_suite.py --model phase3a1_gamma -v
```

---

### Phase 3a-2: Knowledge Training

**Goal:** Math, facts, programming knowledge

```bash
# 1. Generate full format_lock with math
python scripts/generate_format_lock_dataset.py \
    --mode full \
    --math include \
    --output_dir data/sft_format_lock_3a2

# 2. Mix with 30% replay
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_format_lock_3a2/mypt_format_lock_v1.jsonl \
             data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --weights 0.70 0.20 0.10 \
    --output data/sft_mixed/phase3a2_mixed.jsonl \
    --shuffle

# 3. Prepare
python scripts/prepare_chat_sft.py \
    --input data/sft_mixed/phase3a2_mixed.jsonl \
    --output_dir data/sft_ready/phase3a2 \
    --val_split 0.1

# 4. Train (from gamma)
python train.py \
    --model_name phase3a2 \
    --dataset_dir data/sft_ready/phase3a2 \
    --init_from_model phase3a1_gamma \
    --config_file configs/sft1/750M_phase3a2.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json

# 5. Eval Gate
python scripts/sft_eval_suite.py --model phase3a2 -v
```

---

## Operator Learning (Abstraction Test)

**Goal:** Test whether model can learn abstract operators (COPY, WRAP, EXTRACT) vs just memorizing patterns.

### Why Operators Matter

- **Unique payloads**: No memorization possible
- **Template split**: Val uses unseen phrasings
- **Exact-match metric**: Pass or fail, no "looks OK"

### Complete Pipeline with Packing

```bash
# 1. Generate multi-word operator dataset
python scripts/generate_operator_dataset.py \
    --output_dir data/sft_operator_v2 \
    --n_train 9000 \
    --n_val 1000 \
    --max_words 4 \
    --max_tokens 12

# 2. Prepare with PACKING (critical for efficiency)
python scripts/prepare_chat_sft.py \
    --input data/sft_operator_v2/operator_train.jsonl \
    --val_file data/sft_operator_v2/operator_val.jsonl \
    --output_dir data/sft_operator_v2/packed \
    --enable_packing \
    --pack_block_size 1024 \
    --pack_by_field "_meta.operator"

# 3. Inspect the packed dataset
python scripts/inspect_sft_dataset.py \
    --dataset_dir data/sft_operator_v2/packed \
    --show_samples 2

# 4. Train (use recommended max_iters from inspector)
python train.py \
    --model_name phase3a_operator_v2 \
    --init_from_model phase3a1_alpha_v2 \
    --dataset_dir data/sft_operator_v2/packed \
    --config_file configs/sft1/750M_phase3a_operator.json \
    --eval_prompts_file configs/sft_eval/phase3a_operator_eval_prompts.json \
    --learning_rate 7e-5

# 5. Evaluate
python scripts/sft_eval_suite.py --model phase3a_operator_v2 -v
```

### Training Parameters (Operator Learning)

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | 7e-5 | Sweet spot |
| Coverage | 2.0-3.5x masked-token passes | Higher = overfitting |
| Batch size | 16 | Standard |
| Warmup | 100 iters | ~5% of training |

### Expected Results

| Operator | Target | Notes |
|----------|--------|-------|
| WRAP | ≥80% | Easiest (structural signal) |
| EXTRACT | ≥80% | Quote markers help |
| COPY | ≥60% | Hardest (no structural hint) |
| Overall | ≥75% | Validates abstraction learning |

---

## Evaluation Gates

### MANDATORY after every phase:

```bash
# Full eval suite
python scripts/sft_eval_suite.py --model <model_name> -v

# Tie weights check
python scripts/check_tie_weights.py --model <model_name>

# Train/eval parity (optional)
python scripts/debug_train_eval_parity.py --model <model_name> --dataset_dir <dataset_dir>
```

### Eval Suite Buckets

| Bucket | Tests | Pass Criteria |
|--------|-------|---------------|
| **A: format_strict** | Valid tags, proper structure | 100% |
| **B: echo_basic** | "Say Hello" → "Hello" | ≥80% |
| **C: anti_echo** | Don't blindly copy quoted text | ≥80% |
| **D: regression_basic** | Math/facts, no mode collapse | ≥50% |
| **E: operators** | COPY/WRAP/EXTRACT exact-match | ≥75% |

### Manual Testing

```bash
python generate.py --model <model_name> \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say Hello.</myPT_user><myPT_assistant>" \
    --temperature 0 \
    --top_k 0 \
    --top_p 1.0 \
    --repetition_penalty 1.0
```

---

## Troubleshooting & Red Lines

### Red Lines (Restart Required)

1. **Simple task = random garbage** after 2000+ iters on small data
2. **Loss stuck** at ~10+ and not decreasing
3. **Special tokens broken** - malformed tags
4. **Mode collapse** - same output regardless of input
5. **Embedding corruption** - lost basic language ability
6. **Eval suite fails ALL buckets**

### NOT a Restart Situation

- Generalization issues (case sensitivity, template variation)
- Occasional wrong answers on sparse content
- Loss decreasing but slowly
- Anti-echo bucket weak (can be trained)

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Crossref" spam output | Untrained model | Train more iterations |
| Mode collapse to "Yes." | Too little data diversity | Add anti-echo, contrast pairs |
| Overfitting (val loss up) | Too many iterations | Stop earlier, use 2-3x coverage |
| Low mask ratio (~13%) | Short episodes not packed | Enable `--enable_packing` |
| Cache mismatch warnings | Often false positive | Check if output is still correct |

### Coverage Guidelines

**Unpacked datasets (episode-based):**
- Minimum: 2x episode coverage
- Optimal: 3-3.5x episode coverage
- Maximum: 5x (risk of overfitting)

**Packed datasets (masked-token-based):**
- Minimum: 2x masked-token passes
- Optimal: 3-3.5x masked-token passes
- Maximum: 5x (risk of overfitting)

Use `inspect_sft_dataset.py` to get recommended max_iters.

---

## Quick Reference

### Mix Ratios

| Phase | Primary | Replay 1 | Replay 2 |
|-------|---------|----------|----------|
| 3a-1α | 100% format_lock_minimal | - | - |
| 3a-1β | 85% echo_beta | 15% alpha | - |
| 3a-1γ | 35% gibberish | 50% beta | 15% alpha |
| 3a-2 | 70% format_lock_full | 20% beta | 10% alpha |

### Estimated Dataset Sizes

| Phase | Mode | Examples |
|-------|------|----------|
| 3a-1α | format_lock minimal | ~1,000 |
| 3a-1β | echo (no gibberish) | ~3,000-5,000 |
| 3a-1γ | echo (gibberish only) | ~2,000-3,000 |
| 3a-2 | format_lock full | ~11,000 |
| Operators | unique payloads | ~9,000 train / 1,000 val |

### Config Files

| Phase | Config |
|-------|--------|
| 3a-1α | `configs/sft1/750M_phase3a1_alpha.json` |
| 3a-1β | `configs/sft1/750M_phase3a1_beta.json` |
| 3a-1γ | `configs/sft1/750M_phase3a1_gamma.json` |
| 3a-2 | `configs/sft1/750M_phase3a2.json` |
| Operators | `configs/sft1/750M_phase3a_operator.json` |

### Key Scripts

| Script | Purpose |
|--------|---------|
| `generate_format_lock_dataset.py` | Format lock Q&A pairs |
| `generate_echo_dataset.py` | Echo/repeat with anti-echo |
| `generate_operator_dataset.py` | Operator learning dataset |
| `mix_sft_jsonl.py` | Mix datasets with weights |
| `prepare_chat_sft.py` | Tokenize and pack datasets |
| `inspect_sft_dataset.py` | Visualize prepared datasets |
| `sft_eval_suite.py` | Evaluation gates |
| `check_tie_weights.py` | Weight tying validation |

---

## Archive Note

This document consolidates and replaces:
- `PHASE3A_TRAINING_PIPELINE.md`
- `phase3a_training_regime.md`
- `phase3a1_checks_2.md`
- `phase3a1_checks_3.md`
- `phase3a1_operator_2.md`
- `phase3a1_inference_checks.md`
- `phase3a1_operators_optimization.md`

See `OPERATOR_TRAINING_RESULTS.md` for detailed results from the operator learning experiment.
