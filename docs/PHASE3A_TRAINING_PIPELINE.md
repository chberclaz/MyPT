# Phase 3a Training Pipeline

Complete CLI commands for each training phase. **All commands verified against actual script arguments.**

## Prerequisites

Base model: `domain_v5_sft_ready` (with pre-weighted special tags)

---

## Phase 3a-1α: Core Format Lock

**Goal:** Teach chat structure, special tokens, basic copy (~1000 examples, ~56 exposures each)

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

### Step 4: Test

```bash
python generate.py --model phase3a1_alpha \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say hello.</myPT_user><myPT_assistant>" \
    --temperature 0

python generate.py --model phase3a1_alpha \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say Hello.</myPT_user><myPT_assistant>" \
    --temperature 0

python generate.py --model phase3a1_alpha \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Sag Hallo.</myPT_user><myPT_assistant>" \
    --temperature 0
```

**Expected outputs:** `hello.`, `Hello.`, `Hallo.`

---

## Phase 3a-1β: Echo Expansion

**Goal:** Diverse echo templates, more content (~2000-3000 examples)

### Step 1: Generate Dataset

```bash
# Echo dataset (no gibberish, capped)
python scripts/generate_echo_dataset.py \
    --gibberish exclude \
    --max_examples 2500 \
    --output_dir data/sft_echo_beta

# Optional: Mix with 15% replay from alpha
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_echo_beta/mypt_echo_diverse.jsonl \
             data/sft_format_lock_alpha/mypt_format_lock_v1.jsonl \
    --output data/sft_mixed/phase3a1_beta_mixed.jsonl \
    --shuffle
```

### Step 2: Prepare Dataset

```bash
# If using echo only:
python scripts/prepare_chat_sft.py \
    --input data/sft_echo_beta/mypt_echo_diverse.jsonl \
    --output_dir data/sft_ready/phase3a1_beta \
    --val_split 0.1

# If using mixed:
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

### Step 4: Test

```bash
python generate.py --model phase3a1_beta \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Repeat: Hello world</myPT_user><myPT_assistant>" \
    --temperature 0

python generate.py --model phase3a1_beta \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Echo: Testing 123</myPT_user><myPT_assistant>" \
    --temperature 0
```

---

## Phase 3a-1γ: Gibberish (True Copy)

**Goal:** Learn true copy/echo with nonsense words (~2000-3000 examples)

### Step 1: Generate Dataset

```bash
# Gibberish only
python scripts/generate_echo_dataset.py \
    --gibberish only \
    --output_dir data/sft_echo_gamma

# Mix with 15% replay from beta
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_echo_gamma/mypt_echo_diverse.jsonl \
             data/sft_echo_beta/mypt_echo_diverse.jsonl \
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

### Step 4: Test

```bash
python generate.py --model phase3a1_gamma \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say: Blurpix</myPT_user><myPT_assistant>" \
    --temperature 0

python generate.py --model phase3a1_gamma \
    --prompt "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Repeat: Zanthor quexling</myPT_user><myPT_assistant>" \
    --temperature 0
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

# Mix with replay from gamma
python scripts/mix_sft_jsonl.py \
    --inputs data/sft_format_lock_3a2/mypt_format_lock_v1.jsonl \
             data/sft_echo_beta/mypt_echo_diverse.jsonl \
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
| `--max_examples` | int | None (no cap) | Optional cap on examples |
| `--seed` | int | 42 | Random seed |

### mix_sft_jsonl.py
| Argument | Description |
|----------|-------------|
| `--inputs` | Space-separated list of input JSONL files |
| `--output` | Output JSONL file path |
| `--shuffle` | Shuffle the output |
| `--seed` | Random seed (default: 42) |

### prepare_chat_sft.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input JSONL file |
| `--output_dir` | required | Output directory for prepared data |
| `--val_split` | 0.1 | Validation split ratio |
| `--tokenization` | gpt2 | Tokenization scheme |

### train.py (relevant args)
| Argument | Description |
|----------|-------------|
| `--model_name` | Name for saved model |
| `--dataset_dir` | Prepared dataset directory |
| `--init_from_model` | Checkpoint to initialize from |
| `--config_file` | Config JSON file |
| `--eval_prompts_file` | Eval prompts JSON for training-time inference |

---

## Estimated Dataset Sizes

| Phase | Mode | Approx. Examples |
|-------|------|------------------|
| 3a-1α | format_lock `--mode minimal --math exclude` | ~1,000 |
| 3a-1β | echo `--gibberish exclude --max_examples 2500` | ~2,500 |
| 3a-1γ | echo `--gibberish only` | ~3,500 |
| 3a-2 | format_lock `--mode full --math include` | ~11,000 |

---

## Red Line Criteria (When to Restart from Phase 1)

1. **Simple task = random garbage** after 2000+ iters on small data
2. **Loss stuck** at ~10+ and not decreasing
3. **Special tokens broken** - model generates malformed tags
4. **Mode collapse** - same output regardless of input
5. **Embedding corruption** - lost basic language ability

**NOT a restart situation:**
- Generalization issues (case sensitivity, template variation)
- Occasional wrong answers on sparse content
- Loss decreasing but slowly
