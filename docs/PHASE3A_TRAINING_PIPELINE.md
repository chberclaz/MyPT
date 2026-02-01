# Phase 3a Training Pipeline

Complete CLI commands for each training phase. **All commands verified against actual script arguments.**

## Prerequisites

Base model: `domain_v5_sft_ready` (with pre-weighted special tags)

---

## MANDATORY: Evaluation Gates

**Run after EVERY phase before proceeding:**

```bash
# Eval suite (must pass all buckets)
python scripts/sft_eval_suite.py --model <phase_model_name> --output logs/eval/<phase>.json

# Tie weights check
python scripts/check_tie_weights.py --model <phase_model_name>

# Train/eval parity (optional but recommended)
python scripts/debug_train_eval_parity.py --model <phase_model_name> --dataset_dir <dataset_dir>
```

**Eval Suite Buckets:**
- A) `format_strict`: Valid tags, proper structure
- B) `echo_basic`: "Say Hello" → "Hello"
- C) `anti_echo`: Model doesn't blindly copy quoted text
- D) `regression_basic`: Simple math/facts, no mode collapse

---

## Phase 3a-1α: Core Format Lock

**Goal:** Teach chat structure, special tokens, basic copy (~1000 examples)

### Step 1: Generate Dataset

```bash
python scripts/generate_format_lock_dataset.py \
    --mode minimal \
    --math exclude \
    --output_dir data/sft_format_lock_alpha
```

### Step 2: Prepare Dataset

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --output_dir data/sft_ready/phase3a1_alpha \
    --val_split 0.1
```

### Step 3: Train

```bash
python train.py \
    --model_name phase3a1_alpha \
    --dataset_dir data/sft_ready/phase3a1_alpha \
    --init_from_model domain_v5_sft_ready \
    --config_file configs/sft1/750M_phase3a1_alpha.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json
```

### Step 4: Eval Gate (MANDATORY)

```bash
python scripts/sft_eval_suite.py --model phase3a1_alpha --output logs/eval/phase3a1_alpha.json
python scripts/check_tie_weights.py --model phase3a1_alpha
```

### Step 5: Manual Test

```bash
python generate.py --model phase3a1_alpha \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say Hello.</myPT_user><myPT_assistant>" \
    --temperature 0
```

**Expected:** `Hello.`

---

## Phase 3a-1β: Echo Expansion

**Goal:** Diverse echo templates + anti-echo to prevent blind copying

### Step 1: Generate Dataset

```bash
# Echo dataset with anti-echo (no gibberish yet)
python scripts/generate_echo_dataset.py \
    --gibberish exclude \
    --anti_echo_ratio 0.2 \
    --contrast_ratio 0.2 \
    --output_dir data/sft_echo_beta

# Mix with precise ratios: 70% echo + 20% alpha replay + 10% (built-in anti-echo)
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --weights 0.85 0.15 \
    --output data/sft_mixed/phase3a1_beta_mixed.jsonl \
    --shuffle
```

### Step 2: Prepare Dataset

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_mixed/phase3a1_beta_mixed.jsonl \
    --output_dir data/sft_ready/phase3a1_beta \
    --val_split 0.1
```

### Step 3: Train (from alpha checkpoint)

```bash
python train.py \
    --model_name phase3a1_beta \
    --dataset_dir data/sft_ready/phase3a1_beta \
    --init_from_model phase3a1_alpha \
    --config_file configs/sft1/750M_phase3a1_beta.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json
```

### Step 4: Eval Gate (MANDATORY)

```bash
python scripts/sft_eval_suite.py --model phase3a1_beta --output logs/eval/phase3a1_beta.json
```

---

## Phase 3a-1γ: BPE-Safe Gibberish (True Copy)

**Goal:** Learn true copy/echo with BPE-safe nonsense words (filtered by token count)

**What is BPE-safe gibberish?**
Random gibberish creates rare BPE sub-tokens that are unlearnable. The model minimizes loss by outputting frequent "safe" tokens (mode collapse). BPE-safe gibberish is filtered to max 4 tokens per word, ensuring learnability.

### Step 1: Generate Dataset

```bash
# BPE-safe gibberish only (with anti-echo to prevent blind copying)
python scripts/generate_echo_dataset.py \
    --gibberish only \
    --bpe_safe \
    --max_target_tokens 4 \
    --anti_echo_ratio 0.2 \
    --contrast_ratio 0.2 \
    --output_dir data/sft_echo_gamma

# Mix: 35% gibberish + 50% beta echo + 15% alpha format
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_echo_gamma/mypt_echo_diverse.jsonl \
             data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --weights 0.35 0.50 0.15 \
    --output data/sft_mixed/phase3a1_gamma_mixed.jsonl \
    --shuffle
```

### Step 2: Prepare Dataset

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_mixed/phase3a1_gamma_mixed.jsonl \
    --output_dir data/sft_ready/phase3a1_gamma \
    --val_split 0.1
```

### Step 3: Train (from beta checkpoint)

```bash
python train.py \
    --model_name phase3a1_gamma \
    --dataset_dir data/sft_ready/phase3a1_gamma \
    --init_from_model phase3a1_beta \
    --config_file configs/sft1/750M_phase3a1_gamma.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json
```

### Step 4: Eval Gate (MANDATORY)

```bash
python scripts/sft_eval_suite.py --model phase3a1_gamma --output logs/eval/phase3a1_gamma.json
```

---

## Phase 3a-2: Knowledge Training

**Goal:** Math, facts, programming knowledge (~8000+ examples)

### Step 1: Generate Dataset

```bash
# Full format_lock with math
python scripts/generate_format_lock_dataset.py \
    --mode full \
    --math include \
    --output_dir data/sft_format_lock_3a2

# Mix with 15-25% replay from previous phases
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_format_lock_3a2/mypt_format_lock_v1.jsonl \
             data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --weights 0.70 0.20 0.10 \
    --output data/sft_mixed/phase3a2_mixed.jsonl \
    --shuffle
```

### Step 2: Prepare Dataset

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_mixed/phase3a2_mixed.jsonl \
    --output_dir data/sft_ready/phase3a2 \
    --val_split 0.1
```

### Step 3: Train (from gamma checkpoint)

```bash
python train.py \
    --model_name phase3a2 \
    --dataset_dir data/sft_ready/phase3a2 \
    --init_from_model phase3a1_gamma \
    --config_file configs/sft1/750M_phase3a2.json \
    --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json
```

### Step 4: Eval Gate (MANDATORY)

```bash
python scripts/sft_eval_suite.py --model phase3a2 --output logs/eval/phase3a2.json
```

---

## Recommended Mix Ratios

| Phase | Primary | Replay 1 | Replay 2 | Notes |
|-------|---------|----------|----------|-------|
| **3a-1α** | 100% format_lock_minimal | - | - | Pure format learning |
| **3a-1β** | 85% echo_beta | 15% alpha | - | Anti-echo built into echo |
| **3a-1γ** | 35% gibberish | 50% beta | 15% alpha | BPE-safe only |
| **3a-2** | 70% format_lock_full | 20% beta | 10% alpha | Knowledge phase |

---

## Quick Reference: Script Arguments

### generate_format_lock_dataset.py
| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--output_dir` | path | `data/sft_format_lock` | Output directory |
| `--mode` | `full`, `minimal` | `full` | Dataset size mode |
| `--math` | `include`, `exclude`, `only`, `minimal` | `include` | Math content mode |

### generate_echo_dataset.py
| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--output_dir` | path | `data/sft_echo` | Output directory |
| `--gibberish` | `include`, `exclude`, `only` | `exclude` | Gibberish mode |
| `--bpe_safe` | flag | True | Filter gibberish by BPE token count |
| `--max_target_tokens` | int | 4 | Max tokens for BPE-safe filtering |
| `--anti_echo_ratio` | float | 0.2 | Fraction of anti-echo examples |
| `--contrast_ratio` | float | 0.2 | Fraction of contrast pairs |
| `--max_examples` | int | None | Optional cap on examples |

### mix_sft_jsonl.py
| Argument | Description |
|----------|-------------|
| `--inputs` | Space-separated list of input JSONL files |
| `--weights` | Explicit weights for each input (e.g., `--weights 0.7 0.2 0.1`) |
| `--output` | Output JSONL file path |
| `--shuffle` | Shuffle the output |

### sft_eval_suite.py
| Argument | Description |
|----------|-------------|
| `--model` | Model name to evaluate |
| `--output` | Output JSON file for results |
| `--verbose` | Show all results, not just failures |

### check_tie_weights.py
| Argument | Description |
|----------|-------------|
| `--model` | Model name to check |
| `--verbose` | Show additional details |

---

## Estimated Dataset Sizes

| Phase | Mode | Approx. Examples |
|-------|------|------------------|
| 3a-1α | format_lock `--mode minimal --math exclude` | ~1,000 |
| 3a-1β | echo `--gibberish exclude` (with anti-echo) | ~3,000-5,000 |
| 3a-1γ | echo `--gibberish only --bpe_safe` | ~2,000-3,000 |
| 3a-2 | format_lock `--mode full --math include` | ~11,000 |

---

## Red Line Criteria (When to Restart from Phase 1)

1. **Simple task = random garbage** after 2000+ iters on small data
2. **Loss stuck** at ~10+ and not decreasing
3. **Special tokens broken** - model generates malformed tags
4. **Mode collapse** - same output regardless of input (detected by eval suite)
5. **Embedding corruption** - lost basic language ability
6. **Eval suite fails all buckets** - fundamental capability missing

**NOT a restart situation:**
- Generalization issues (case sensitivity, template variation)
- Occasional wrong answers on sparse content
- Loss decreasing but slowly
- Anti-echo bucket weak (can be trained)
