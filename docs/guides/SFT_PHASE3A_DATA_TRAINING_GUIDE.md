# SFT Phase 3a: Comprehensive Data & Training Guide

This guide covers the complete Phase 3a SFT (Supervised Fine-Tuning) pipeline for MyPT, from gold episode creation through training and validation.

**Last updated:** January 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Gold Episode Format](#2-gold-episode-format)
3. [Creating Quality Gold Episodes](#3-creating-quality-gold-episodes)
4. [Bilingual Dataset Workflow](#4-bilingual-dataset-workflow)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Optional: Mixing with General Data](#6-optional-mixing-with-general-data)
7. [Training Configuration](#7-training-configuration)
8. [Running Training](#8-running-training)
9. [Monitoring & Validation](#9-monitoring--validation)
10. [Troubleshooting](#10-troubleshooting)
11. [Complete Workflow Checklist](#11-complete-workflow-checklist)

---

## 1. Overview

### What is Phase 3a?

Phase 3a is the **Chat SFT** stage where we teach the domain-adapted model (from Phase 2) to:

- Follow conversational patterns
- Respond appropriately to user instructions
- Generate assistant-style responses
- Handle denials, limitations, and uncertainty

### Training Pipeline Position

```
Phase 1: Base Pretraining (general corpus)
    ↓
Phase 2: Domain Adaptation (IT-Sec, Coding, Swiss Law)
    ↓
Phase 3a: Chat SFT ← YOU ARE HERE
    ↓
Phase 3b: Tool-Calling SFT (future)
```

### Key Concepts

| Concept                  | Description                                                          |
| ------------------------ | -------------------------------------------------------------------- |
| **Loss Masking**         | Train ONLY on assistant responses (mask=1), not user/system (mask=0) |
| **Episode-Indexed**      | Each conversation is a discrete unit, never crossed during sampling  |
| **Deterministic Epochs** | Same seed → same training order (reproducibility)                    |
| **Special Tokens**       | `<myPT_user>`, `<myPT_assistant>`, etc. structure the conversation   |

### Complete SFT Pipeline Diagram

```
                    ┌─────────────────────────────┐
                    │     JSONL Gold Episodes     │
                    │ (mypt_phase3a_gold_en.jsonl)│
                    └─────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     │                     ▼
     ┌────────────┐               │              ┌────────────┐
     │ Translation│               │              │ Translation│
     │    (EN)    │◄──────────────┘──────────────►│    (DE)    │
     │  KEPT AS-IS│                              │  via DeepL │
     └─────┬──────┘                              └─────┬──────┘
           │                                          │
           └────────────────┬─────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ merge_bilingual│
                   │   _episodes.py │
                   └────────┬───────┘
                            │
                            ▼
               ┌───────────────────────┐
               │ Bilingual JSONL       │
               │ (~416 episodes)       │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  prepare_chat_sft.py  │
               │ - Tokenize (GPT-2)    │
               │ - Create loss masks   │
               │ - Build episode index │
               └───────────┬───────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │  Episode-Indexed Dataset                     │
    │  ├── train/                                  │
    │  │   ├── tokens.bin    (uint32)              │
    │  │   ├── mask.bin      (uint8, 0/1)          │
    │  │   └── episodes.idx  (uint64 pairs)        │
    │  ├── val/                                    │
    │  └── dataset_metadata.json                   │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │      train.py         │
               │                       │
               │  Auto-detects:        │
               │  → GPTEpisodeDataLoader│
               │  → use_loss_mask=True │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  model.fit()          │
               │  - Loss masking       │
               │  - Epoch-based        │
               │  - Deterministic      │
               └───────────────────────┘
```

---

## 2. Gold Episode Format

### JSONL Schema

Each line in the gold episode file is a complete conversation:

```json
{
  "system": "You are MyPT, a helpful AI assistant specialized in IT security, coding, and Swiss law.",
  "context": "episode_id: 0042",
  "messages": [
    { "role": "user", "content": "What is SQL injection?" },
    {
      "role": "assistant",
      "content": "SQL injection is a code injection technique..."
    }
  ]
}
```

### Multi-Turn Conversations

````json
{
  "system": "You are MyPT...",
  "context": "episode_id: mt_001",
  "messages": [
    { "role": "user", "content": "How do I read a file in Python?" },
    {
      "role": "assistant",
      "content": "Use the `open()` function with a context manager:\n\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"
    },
    { "role": "user", "content": "What if the file doesn't exist?" },
    {
      "role": "assistant",
      "content": "Wrap it in a try-except block:\n\n```python\ntry:\n    with open('file.txt', 'r') as f:\n        content = f.read()\nexcept FileNotFoundError:\n    print('File not found')\n```"
    }
  ]
}
````

### Field Reference

| Field                | Required | Description                                     |
| -------------------- | -------- | ----------------------------------------------- |
| `system`             | Optional | System prompt (masked out during training)      |
| `context`            | Optional | Metadata like episode_id, language (masked out) |
| `messages`           | Required | Array of role/content pairs                     |
| `messages[].role`    | Required | `"user"` or `"assistant"`                       |
| `messages[].content` | Required | Message text                                    |

### Serialized Format (What the Model Sees)

```
<myPT_system>You are MyPT...</myPT_system>
<myPT_user_context>episode_id: 0042</myPT_user_context>
<myPT_user>What is SQL injection?</myPT_user>
<myPT_assistant>SQL injection is a code injection technique...</myPT_assistant>
<myPT_eot>
```

### Loss Mask Alignment

```
Token:  <myPT_system> You are... </myPT_system> <myPT_user> What is... </myPT_user> <myPT_assistant> SQL injection... </myPT_assistant> <myPT_eot>
Mask:        0          0    0          0            0         0    0         0            1              1      1              1           0
```

**Important:** The `<myPT_assistant>` and `</myPT_assistant>` tags themselves have **mask=1** (trained on). This teaches the model to:

1. Generate the opening tag when it should respond
2. Generate the closing tag when the response is complete
3. Learn the correct conversation structure

**Mask summary:**
| Token Type | Mask | Reason |
|------------|------|--------|
| `<myPT_system>`, `</myPT_system>` | 0 | Don't predict system prompts |
| System content | 0 | Context only |
| `<myPT_user>`, `</myPT_user>` | 0 | Don't predict user messages |
| User content | 0 | Context only |
| `<myPT_assistant>` | **1** | Model must learn to generate this |
| Assistant content | **1** | Core training signal |
| `</myPT_assistant>` | **1** | Model must learn to close responses |
| `<myPT_eot>` | 0 | End marker (not generated) |

---

## 3. Creating Quality Gold Episodes

### Episode Categories

Target a balanced mix across these categories:

| Category               | Target % | Description                                  |
| ---------------------- | -------- | -------------------------------------------- |
| **Technical Q&A**      | 40%      | Domain knowledge (IT-Sec, Coding, Swiss Law) |
| **How-To / Tutorials** | 20%      | Step-by-step instructions with code          |
| **Multi-Turn**         | 15%      | Follow-up questions, clarifications          |
| **Denials**            | 10%      | Refusing inappropriate requests              |
| **Limitations**        | 10%      | "I can't do X because..."                    |
| **Uncertainty**        | 5%       | "It depends on...", "I'm not sure, but..."   |

### Quality Guidelines

#### ✅ Good Episode Characteristics

- **Specific and accurate** - No placeholder text like "[specific example]"
- **Natural language** - Varied phrasing, not templated
- **Appropriate length** - Concise but complete
- **Code examples** - Working, tested code with comments
- **Citations when needed** - "According to Swiss OR Art. 32..."

#### ❌ Bad Episode Characteristics

- Templated responses ("Explain X like I'm a senior developer...")
- Factually incorrect information
- Overly long responses that exceed context
- Near-duplicate episodes
- Missing code examples where expected

### Token Count Constraint

**Each episode MUST fit within the 1024 token context window.**

```python
# Validate episode length
from core.tokenizer import Tokenizer
from core.model import GPTConfig

config = GPTConfig(vocab_size=50304)
tokenizer = Tokenizer(config, 'gpt2')

serialized = serialize_episode(episode)  # Your serialization function
tokens = tokenizer.encode(serialized)

if len(tokens) > 1024:
    print(f"WARNING: Episode too long ({len(tokens)} tokens)")
```

### Diversity Check

Use `scripts/analyze_episode_diversity.py` to detect:

- Duplicate user messages
- Templated patterns
- Vocabulary overlap

```bash
python scripts/analyze_episode_diversity.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl
```

---

## 4. Bilingual Dataset Workflow

### Overview

```
English Gold Episodes
        │
        ├──────────────────────────────────┐
        │                                  │
        ▼                                  ▼
  Extract for Translation           Keep English
        │
        ▼
  translate_deepl.py (DeepL API)
        │
        ▼
  Manual Review & Fixes
        │
        ▼
  recombine_translations.py
        │
        ▼
  German Episodes
        │
        └──────────────────────────────────┐
                                           │
                                           ▼
                              merge_bilingual_episodes.py
                                           │
                                           ▼
                              Bilingual JSONL (~416 episodes)
```

### Step 1: Extract Messages for Translation

```bash
python scripts/extract_for_translation.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl \
    --output_dir data/temp
```

Creates:

- `data/temp/user_messages_en.txt`
- `data/temp/assistant_messages_en.txt`
- `data/temp/episode_ids.txt`

### Step 2: Translate with DeepL

```bash
# Set up API key
echo "DEEPL_API_KEY=your_key_here" > .env

# Run translation
python scripts/translate_deepl.py
```

Creates:

- `data/temp/user_messages_de.txt`
- `data/temp/assistant_messages_de.txt`

### Step 3: Manual Review

Review and fix German translations, especially:

- Technical terms that shouldn't be translated
- Code snippets (keep in English)
- Swiss-specific terminology

### Step 4: Recombine Translations

```bash
python scripts/recombine_translations.py \
    --original data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl \
    --user_translated data/temp/user_messages_de.txt \
    --assistant_translated data/temp/assistant_messages_de.txt \
    --output data/sft_conversation_goldset/mypt_phase3a_gold_de.jsonl \
    --language de
```

### Step 5: Merge Bilingual

```bash
python scripts/merge_bilingual_episodes.py \
    --english data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl \
    --german data/sft_conversation_goldset/mypt_phase3a_gold_de.jsonl \
    --output data/sft_conversation_goldset/mypt_phase3a_gold_bilingual.jsonl \
    --shuffle
```

---

## 5. Dataset Preparation

### Converting JSONL to Episode-Indexed Format

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_bilingual.jsonl \
    --output_dir data/sft_conversation_goldset_prepared \
    --tokenization gpt2 \
    --val_split 0.1 \
    --verbose
```

### Output Structure

```
data/sft_conversation_goldset_prepared/
├── train/
│   ├── tokens.bin        # uint32 token IDs
│   ├── mask.bin          # uint8 loss mask (0/1)
│   └── episodes.idx      # uint64 pairs (start, length)
├── val/
│   ├── tokens.bin
│   ├── mask.bin
│   └── episodes.idx
├── tokenizer_state.json  # Tokenizer config for reproducibility
└── dataset_metadata.json # Stats and provenance
```

### Verify Preparation

Check `dataset_metadata.json`:

```json
{
  "schema": "episode_indexed_sft_v1",
  "has_loss_mask": true,
  "num_train_episodes": 374,
  "num_val_episodes": 42,
  "num_train_tokens": 85432,
  "num_val_tokens": 9876,
  "train_mask_ratio": 0.42, // ~42% of tokens are assistant (trained on)
  "tokenization": "gpt2"
}
```

**Expected mask ratio:** 30-50% (assistant tokens)

---

## 6. Optional: Mixing with General Data

To prevent catastrophic forgetting, mix SFT episodes with general domain data:

```bash
python scripts/mix_general_with_sft.py \
    --sft_dir data/sft_conversation_goldset_prepared \
    --general_dir data/domain_v8_prepared \
    --output_dir data/sft_mixed_prepared \
    --general_ratio 0.2 \
    --chunk_size 512 \
    --shuffle
```

### What This Does

| Component      | Mask                       | Training Behavior           |
| -------------- | -------------------------- | --------------------------- |
| SFT episodes   | Selective (assistant only) | Learn conversation patterns |
| General chunks | All 1s                     | Maintain domain knowledge   |

### Recommended Ratios

| Scenario               | SFT % | General % |
| ---------------------- | ----- | --------- |
| Strong SFT alignment   | 80%   | 20%       |
| Balanced               | 70%   | 30%       |
| Knowledge preservation | 60%   | 40%       |

---

## 7. Training Configuration

### SFT Config Template

Create `configs/sft1/750M_1024_chat_sft_phase3a.json`:

```json
{
  "name": "750M Phase 3a Chat SFT",
  "description": "SFT on bilingual gold episodes with loss masking",

  "n_layer": 32,
  "n_embd": 1280,
  "n_head": 20,
  "block_size": 1024,
  "vocab_size": 50304,
  "dropout": 0.15,
  "bias": false,

  "use_loss_mask": true,
  "batch_sampling_mode": "epoch",
  "epoch_seed": 3301,
  "epoch_shuffle": true,
  "epoch_drop_last": true,
  "episode_min_tokens": 10,

  "learning_rate": 5e-5,
  "max_iters": 5000,
  "eval_interval": 500,
  "eval_iters": 50,
  "warmup_iters": 150,
  "weight_decay": 0.05,
  "grad_clip": 1.0
}
```

### Key Parameters Explained

| Parameter       | Value | Rationale                                                     |
| --------------- | ----- | ------------------------------------------------------------- |
| `learning_rate` | 5e-5  | **Lower than pretraining** (was 1.5e-4) to prevent forgetting |
| `dropout`       | 0.15  | Slightly higher for small dataset regularization              |
| `max_iters`     | 5000  | ~10-20 epochs over ~400 episodes                              |
| `warmup_iters`  | 150   | 3% warmup for fine-tuning stability                           |
| `use_loss_mask` | true  | **Critical** - train only on assistant responses              |
| `epoch_seed`    | 3301  | Phase 3a seed for reproducibility                             |

### Architecture Must Match Base Model

```
Base model (domain_v8):  32 layers, 1280 embd, 20 heads
SFT config:              32 layers, 1280 embd, 20 heads  ✓
```

---

## 8. Running Training

### Command

```bash
python train.py \
    --model_name phase3a_chat_sft \
    --config_file configs/sft1/750M_1024_chat_sft_phase3a.json \
    --dataset_dir data/sft_conversation_goldset_prepared \
    --init_from_model checkpoints/domain_v8 \
    --tokenization gpt2
```

### What Happens

1. ✅ Loads `domain_v8` checkpoint (your domain-adapted model)
2. ✅ Auto-detects episode-indexed dataset → uses `GPTEpisodeDataLoader`
3. ✅ Enables loss masking (from config)
4. ✅ Trains with deterministic epoch-based sampling
5. ✅ Saves checkpoints every `eval_interval` steps

### Expected Console Output

```
📦 Loaded episode-indexed dataset:
   Schema: episode_indexed_sft_v1
   Episodes: 374 train, 42 val
   Tokens: 85,432 train, 9,876 val
   [train] 1 shard(s), 374 episodes, 85,432 tokens
   [val] 1 shard(s), 42 episodes, 9,876 tokens
   Using loss masking (assistant-only training)

Detected: Episode-indexed dataset (SFT mode)
Sampling mode: epoch
Epoch seed: 3301 (for reproducibility)

step 0: val 8.2341
step 500: val 4.1523
step 1000: val 3.2145
...
```

---

## 9. Monitoring & Validation

### Loss Expectations

| Step | Train Loss | Val Loss | Status                            |
| ---- | ---------- | -------- | --------------------------------- |
| 0    | ~8-10      | ~8-10    | Initial (random assistant tokens) |
| 500  | ~4-6       | ~4-6     | Learning patterns                 |
| 1000 | ~3-4       | ~3.5-4.5 | Converging                        |
| 2000 | ~2-3       | ~3-3.5   | Good fit                          |
| 5000 | ~1.5-2     | ~2.5-3   | Target (stop if val diverges)     |

### Signs of Overfitting

```
step 2000: train=1.8, val=2.5   ✓ OK (gap < 1.0)
step 3000: train=1.2, val=3.5   ⚠️ Warning (gap > 2.0)
step 4000: train=0.8, val=4.2   ❌ Stop! Overfitting
```

**Actions:**

- Stop training early
- Increase `dropout` (0.15 → 0.2)
- Increase `weight_decay` (0.05 → 0.1)
- Add more training data

### Test Generation

After training, test the model:

```bash
python generate.py \
    --model checkpoints/phase3a_chat_sft \
    --prompt "<myPT_system>You are a helpful assistant.</myPT_system>
<myPT_user>What is SQL injection?</myPT_user>
<myPT_assistant>" \
    --max_new_tokens 200 \
    --temperature 0.7
```

**Expected:** Coherent, on-topic assistant response.

### Compare Base vs SFT

```bash
# Base model (domain_v8)
python generate.py --model checkpoints/domain_v8 --prompt "..."

# SFT model
python generate.py --model checkpoints/phase3a_chat_sft --prompt "..."
```

The SFT model should:

- Follow instructions better
- Generate conversational responses
- Stay on topic
- Use appropriate tone

---

## 10. Troubleshooting

### Issue: "No episodes loaded"

**Cause:** Dataset format not detected correctly.

**Fix:** Verify `episodes.idx` exists in `train/`:

```bash
ls data/sft_conversation_goldset_prepared/train/
# Should show: tokens.bin, mask.bin, episodes.idx
```

### Issue: Loss stays high (~8-10)

**Causes:**

1. Loss mask not applied
2. Wrong tokenization
3. Learning rate too low

**Fixes:**

1. Check config: `"use_loss_mask": true`
2. Verify: `dataset_metadata.json` shows `"tokenization": "gpt2"`
3. Try `learning_rate: 7.5e-5`

### Issue: Model generates gibberish

**Causes:**

1. Learning rate too high (catastrophic forgetting)
2. Architecture mismatch
3. Wrong tokenizer

**Fixes:**

1. Lower LR to 3e-5
2. Verify config matches base model exactly
3. Ensure `--tokenization gpt2` matches base model

### Issue: Val loss much higher than train

**Cause:** Overfitting on small dataset.

**Fixes:**

1. Stop training earlier
2. Increase dropout: 0.15 → 0.2
3. Add more training data or mix with general data

### Issue: OOM (Out of Memory)

**Fixes:**

1. Reduce `batch_size`: 10 → 8 → 6
2. Enable gradient checkpointing: `"use_checkpoint": true`
3. Use gradient accumulation (manual edit to train.py)

---

## 11. Complete Workflow Checklist

### Pre-Training Checklist

- [ ] **Gold episodes created** in `data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl`
- [ ] **Episode diversity verified** with `analyze_episode_diversity.py`
- [ ] **Token counts validated** (all episodes < 1024 tokens)
- [ ] **Translation completed** (if bilingual)
- [ ] **Bilingual merge done** with `merge_bilingual_episodes.py`
- [ ] **Dataset prepared** with `prepare_chat_sft.py`
- [ ] **Metadata verified** (check `dataset_metadata.json`)
- [ ] **Config created** in `configs/sft1/`
- [ ] **Base model available** in `checkpoints/domain_v8/`

### Training Commands

```bash
# 1. Prepare dataset
python scripts/prepare_chat_sft.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_bilingual.jsonl \
    --output_dir data/sft_conversation_goldset_prepared \
    --tokenization gpt2 \
    --val_split 0.1

# 2. (Optional) Mix with general data
python scripts/mix_general_with_sft.py \
    --sft_dir data/sft_conversation_goldset_prepared \
    --general_dir data/domain_v8_prepared \
    --output_dir data/sft_mixed_prepared \
    --general_ratio 0.2 \
    --shuffle

# 3. Train
python train.py \
    --model_name phase3a_chat_sft \
    --config_file configs/sft1/750M_1024_chat_sft_phase3a.json \
    --dataset_dir data/sft_conversation_goldset_prepared \
    --init_from_model checkpoints/domain_v8

# 4. Test
python generate.py \
    --model checkpoints/phase3a_chat_sft \
    --prompt "<myPT_system>You are helpful.</myPT_system><myPT_user>Hello!</myPT_user><myPT_assistant>"
```

### Post-Training Checklist

- [ ] **Final loss reasonable** (train ~2, val ~3)
- [ ] **No overfitting** (val-train gap < 1.0)
- [ ] **Generation test passed** (coherent responses)
- [ ] **Comparison test** (SFT better than base)
- [ ] **Checkpoint saved** to `checkpoints/phase3a_chat_sft/`

---

## Appendix: File Reference

### Key Files

| Path                                                              | Description                 |
| ----------------------------------------------------------------- | --------------------------- |
| `data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl`     | English gold episodes       |
| `data/sft_conversation_goldset/mypt_phase3a_gold_bilingual.jsonl` | EN+DE merged                |
| `data/sft_conversation_goldset_prepared/`                         | Prepared binary dataset     |
| `configs/sft1/750M_1024_chat_sft_phase3a.json`                    | Training config             |
| `checkpoints/domain_v8/`                                          | Base model (domain-adapted) |
| `checkpoints/phase3a_chat_sft/`                                   | Output SFT model            |

### Key Scripts

| Script                                 | Purpose                                |
| -------------------------------------- | -------------------------------------- |
| `scripts/prepare_chat_sft.py`          | Convert JSONL → episode-indexed binary |
| `scripts/translate_deepl.py`           | Translate via DeepL API                |
| `scripts/merge_bilingual_episodes.py`  | Combine EN+DE                          |
| `scripts/mix_general_with_sft.py`      | Add replay data                        |
| `scripts/analyze_episode_diversity.py` | Check episode quality                  |
| `scripts/diversify_user_messages.py`   | Reduce templated patterns              |

### Key Modules

| Module                        | Purpose                      |
| ----------------------------- | ---------------------------- |
| `core/episode_data_loader.py` | Episode-indexed data loading |
| `core/data_loader.py`         | Token-stream data loading    |
| `core/model.py`               | GPT model with loss masking  |
| `core/special_tokens.py`      | Token definitions            |
| `train.py`                    | Training entry point         |

---

## See Also

- [SFT Loss Masking](../sft/SFT_LOSS_MASKING.md) - Detailed loss masking explanation
- [Episode-Indexed SFT](../sft/EPISODE_INDEXED_SFT.md) - Data loader documentation
- [Special Tokens Guide](../model/SPECIAL_TOKENS.md) - Token definitions
- [Domain Adaptation Guide](DOMAIN_ADAPTATION_GUIDE.md) - Phase 2 reference
- [Quick Reference](QUICK_REFERENCE.md) - Command cheat sheet

---

**Questions?** Check the troubleshooting section or review the test suite in `tests/test_episode_data_loader.py`.
