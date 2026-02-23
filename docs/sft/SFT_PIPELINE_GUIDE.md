# MyPT SFT Pipeline Guide

Complete reference for Supervised Fine-Tuning the MyPT 750M LLaMA-2 base model
into an agentic RAG assistant. Covers architecture, data flow, every phase,
exact commands, and success criteria.

**Last updated:** February 2026
**Base model:** `checkpoints/phase1b_context_ext` (LLaMA-2 style, 750M params, 4096 context via PI)
**Architecture:** RoPE + SwiGLU + RMSNorm, tie_weights=true, 1280d/20h/32L, rope_scale=4.0

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1b: Context Extension (1024 → 4096)](#2-phase-1b-context-extension)
3. [Data Flow: Generator to Training](#3-data-flow-generator-to-training)
4. [Special Tokens and Loss Masking](#4-special-tokens-and-loss-masking)
5. [Phase 1: Format Lock](#5-phase-1-format-lock)
6. [Phase 2: Operators](#6-phase-2-operators)
7. [Phase 3: Chat SFT](#7-phase-3-chat-sft)
8. [Phase 4: Multi-turn Boundaries](#8-phase-4-multi-turn-boundaries)
9. [Phase 5: Simple Toolcall](#9-phase-5-simple-toolcall)
10. [Phase 6: Agentic RAG](#10-phase-6-agentic-rag)
11. [HuggingFace Dataset Integration](#11-huggingface-dataset-integration)
12. [System Prompt Strategy](#12-system-prompt-strategy-loss-mask-optimization)
13. [Anti-Forgetting Strategies](#13-anti-forgetting-strategies)
14. [NEFTune Embedding Noise](#14-neftune-embedding-noise)
15. [Weighted Loss Masking](#15-weighted-loss-masking)
16. [Scripts Reference](#16-scripts-reference)
17. [Configs Reference](#17-configs-reference)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Architecture Overview

Every SFT phase trains the same architecture. The model config is loaded from the
checkpoint; the SFT config only needs to match so `train.py` can reconstruct the model.

```
Model:   GPT-750M (LLaMA-2 style)
Params:  ~700M
Context: 4096 tokens (extended from 1024 via Position Interpolation in Phase 1b)
Vocab:   50304 (50257 base GPT-2 + 19 myPT special tokens, padded to 64)

Architecture fields (must be in every SFT config):
  n_embd: 1280
  n_head: 20
  n_layer: 32
  bias: false
  tie_weights: true
  pos_encoding: "rope"
  mlp_type: "swiglu"
  norm_type: "rmsnorm"
  rope_theta: 10000.0
  rope_scale: 4.0
```

SFT configs live in `configs/sft/` -- one file per phase (`phase1_format_lock.json` through `phase6_agentic_rag.json`).

---

## 2. Phase 1b: Context Extension (1024 → 4096)

**Runs BEFORE any SFT.** Extends the pre-trained model's context window from 1024 to 4096 using Position Interpolation (PI).

### Why

At 1024 tokens, the model cannot fit a meaningful RAG episode: system prompt + retrieved passages + question + answer + tool calls exceed 1024 in nearly all production scenarios. At 4096, all SFT phases and production inference have room for multi-passage retrieval, multi-step tool chains, and detailed answers.

### Method: Position Interpolation (PI)

PI compresses position indices by the extension factor: positions `[0, 1, ..., 4095]` are mapped to `[0, 0.25, 0.5, ..., 1023.75]`. This lets the model interpolate between positions it learned during pre-training. Implemented as `rope_scale: 4.0` in the config, which divides the position vector by 4.0 in `precompute_rope_frequencies()`.

### Dataset

Two-part dataset (~800M tokens, no myPT tags -- the model has no tag knowledge at this point):

**QA episodes (40%, ~320M tokens):** 6 HuggingFace sources for structured retrieval training:

| Source | Tokens | Role |
|:--|:--|:--|
| HotpotQA (distractor) | ~108M | Long multi-passage, position-varied gold paragraphs |
| MS MARCO v2.1 | ~100M | 10-passage search results, position-varied, real Bing queries |
| TriviaQA (evidence) | ~78M | Medium-long grounded trivia |
| SQuAD v2 | ~32M | Short extractive EN QA (incl. unanswerable) |
| MuSiQue | ~20M | Hard multi-hop (2-4 hops) |
| GermanQuAD | ~1.5M | Short extractive DE QA |

**General text (60%, ~480M tokens):** Sampled from pre-training shards. Short documents concatenated into ~4096-token mega-episodes for full-length attention spans during RoPE adaptation.

Built by `scripts/data_prep/build_context_extension_dataset.py` (QA), tokenized and combined with general text by `scripts/data_prep/prepare_context_extension.py`. Uses episode-indexed format with greedy bin-packing and diversity interleaving. Padding masked out, all real tokens are loss targets.

### Config and Training

```bash
# Build QA dataset
python scripts/data_prep/build_context_extension_dataset.py

# Tokenize QA + sample general text, pack into episode-indexed format
python scripts/data_prep/prepare_context_extension.py \
  --general_shards_dir data/unified_6B \
  --general_target_tokens 480000000

# Train
python train.py \
  --model_name phase1b_context_ext \
  --config_file configs/phase1b_context_extension.json \
  --dataset_dir data/context_extension \
  --init_from_model GOLD_unified_v1
```

Config: `configs/phase1b_context_extension.json` -- B=4, T=4096, grad_accum=4, LR=3e-5, 12K iters, rope_scale=4.0, episode-indexed with epoch sampling. ~800M tokens (40% QA, 60% general).

After this phase, ALL subsequent SFT phases use `block_size: 4096` and `rope_scale: 4.0`.

See `docs/training/PHASE1B_CONTEXT_EXTENSION.md` for the full training document.

---

## 3. Data Flow: Generator to Training

All SFT data follows the same pipeline, regardless of phase:

```
Step 1: GENERATE
  Generator script (or HF converter)
       |
       v
  data/intermediate/episodes.jsonl    <-- human-readable JSONL
       |
Step 2: MIX (optional)
  mix_sft_jsonl.py
       |
       v
  data/intermediate/mixed.jsonl       <-- combined, shuffled JSONL
       |
Step 3: TOKENIZE
  prepare_chat_sft.py   (Phase 1-4)
  prepare_tool_sft.py   (Phase 5-6)
       |
       v
  data/sft_phaseN/                    <-- binary, ready for train.py
    train/
      tokens.bin          uint32 token IDs
      mask.bin            uint8 loss mask (0=skip, 1=train)
      episodes.idx        uint64 (start, length) pairs
    val/
      tokens.bin
      mask.bin
      episodes.idx
    tokenizer_state.json
    dataset_metadata.json
       |
Step 4: TRAIN
  python train.py \
    --model_name phaseN \
    --config_file configs/sftX/config.json \
    --dataset_dir data/sft_phaseN \
    --init_from_model checkpoints/PREVIOUS_PHASE
```

### JSONL Format (All Phases)

Every generator outputs the same JSONL schema:

```json
{
  "system": "You are MyPT.",
  "messages": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "A programming language."}
  ],
  "language": "en"
}
```

For tool-calling episodes (Phase 5-6):

```json
{
  "system": "You are MyPT. Answer questions using workspace tools...",
  "messages": [
    {"role": "user", "content": "Find ML docs"},
    {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": "ML"}},
    {"role": "toolresult", "name": "workspace.search", "content": {"documents": [...]}},
    {"role": "assistant", "content": "Found ML documentation.", "cite": "ml-101"}
  ]
}
```

Optional fields on messages:
- `"context"` on user messages -- becomes `<myPT_user_context>` (Phase 5-6)
- `"think"` on assistant messages -- becomes `<myPT_think>` (Phase 3+)
- `"cite"` on assistant messages -- becomes `<myPT_cite>` (Phase 3+)

### Tokenization: Chat vs Tool

- **`prepare_chat_sft.py`** -- For Phase 1-4 (no toolcall/toolresult roles). Uses token-ID-based
  masking: assistant content = train, everything else = mask. Supports packing (multiple short
  episodes per block) for efficiency.

- **`prepare_tool_sft.py`** -- For Phase 5-6 (has toolcall/toolresult roles). Handles all 19 tags
  including think, cite, user_context, assistant_context. Char-level masking converted to token-level.
  Does NOT support packing yet (TODO for Phase 5).

Both produce the same binary format (tokens.bin + mask.bin + episodes.idx).

### Episode Packing

Without packing, each episode is padded to `block_size` (1024). A 30-token format lock
episode wastes 97% of compute on padding. Packing fills each 1024-token block with
multiple episodes back-to-back, dramatically increasing supervised tokens per training step.

Cross-episode attention is isolated via `segment_ids` -- the attention mask prevents
episodes within the same packed sequence from attending to each other.

**When to use `--enable_packing`:**

| Phase | Avg Episode | Episodes/Block | Efficiency Gain | Pack? |
|-------|-------------|----------------|-----------------|-------|
| 1     | ~30 tokens  | 25-50          | 25-50x          | YES   |
| 2     | ~45 tokens  | 17-34          | 17-34x          | YES   |
| 3     | ~250 tokens | 2-10           | 2-10x           | YES   |
| 4     | ~500 tokens | 1-5            | 1.3-5x          | YES   |
| 5     | ~400 tokens | 1-5            | 1.5-5x          | N/A (prepare_tool_sft.py, no packing yet) |
| 6     | ~800 tokens | 1-2            | ~1x             | NO (episodes fill the window) |

**Phase 1-2 are where packing is transformative** -- without it, training is 17-50x slower
because almost every token in the 1024 window is wasted padding.

```bash
# Packing flag (add to prepare_chat_sft.py calls):
--enable_packing --pack_block_size 1024

# Optional: group by field before packing (keeps similar episodes together):
--pack_by_field "_meta.operator"
```

---

## 3. Special Tokens and Loss Masking

19 special tokens (IDs 50257-50275). Full reference: `docs/sft/TAG_NESTING_REFERENCE.md`.

**Loss rule:** Everything inside `<myPT_assistant>...</myPT_assistant>` is trained (mask=1).
Everything else (system, user, context, toolresult) is masked (mask=0).
`<myPT_eot>` is trained (model learns to stop).

Quick reference of what the model generates vs what the system injects:

```
SYSTEM INJECTS (mask=0):        MODEL GENERATES (mask=1):
  <myPT_system>                   <myPT_assistant> content </myPT_assistant>
  <myPT_user>                     <myPT_think> reasoning </myPT_think>
  <myPT_user_context>             <myPT_toolcall> JSON </myPT_toolcall>
  <myPT_assistant_context>        <myPT_cite> source </myPT_cite>
  <myPT_toolresult>               <myPT_eot>
```

### Phase-by-Phase Tag Introduction

| Phase | New Tags Introduced                        |
|-------|--------------------------------------------|
| 1     | system, user, assistant, eot               |
| 2     | (none new)                                 |
| 3     | think, cite, user_context                  |
| 4     | assistant_context, multi-turn eot patterns |
| 5     | toolcall, toolresult                       |
| 6     | (none new -- multi-step tool chains)       |

---

## 4. Phase 1: Format Lock

**Goal:** Model learns to generate inside assistant tags and STOP.

**What it teaches:**
- `<myPT_system>...<myPT_user>...<myPT_assistant>RESPONSE</myPT_assistant><myPT_eot>` skeleton
- Ultra-short responses (1-5 tokens)
- Basic echo/repeat instructions (EN + DE)
- Anti-echo contrast (do NOT blindly copy)

### Automated Pipeline

```bash
# One command does everything (generate + mix + tokenize):
python scripts/sft/prepare_phase1_format_lock.py

# Or with options:
python scripts/sft/prepare_phase1_format_lock.py \
    --output_dir data/sft_phase1_format_lock \
    --format_lock_mode full \
    --format_lock_math include \
    --echo_gibberish exclude \
    --format_lock_ratio 0.7 \
    --echo_ratio 0.3

# Dry run (see commands without executing):
python scripts/sft/prepare_phase1_format_lock.py --dry_run
```

### Manual Step-by-Step

```bash
# 1. Generate format lock dataset (EN + DE combinatorial templates)
python scripts/sft/generate_format_lock_dataset.py \
    --output_dir data/sft_phase1_intermediate/format_lock \
    --mode full --math include

# 2. Generate echo dataset
python scripts/sft/generate_echo_dataset.py \
    --output_dir data/sft_phase1_intermediate/echo \
    --gibberish exclude

# 3. Mix 70% format_lock + 30% echo
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase1_intermediate/format_lock/mypt_format_lock_v1.jsonl:0.7 \
             data/sft_phase1_intermediate/echo/mypt_echo_diverse.jsonl:0.3 \
    --output data/sft_phase1_intermediate/phase1_mixed.jsonl \
    --shuffle

# 4. Tokenize with loss masking + PACKING (critical for short episodes)
#    Episodes are ~30 tokens each. Without packing: 97% wasted padding.
#    With packing: ~30 episodes per 1024 block = 25-50x more efficient.
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase1_intermediate/phase1_mixed.jsonl \
    --output_dir data/sft_phase1_format_lock \
    --val_split 0.05 \
    --enable_packing --pack_block_size 1024
```

### Train

```bash
python train.py \
    --model_name phase1_format_lock \
    --config_file configs/sft/phase1_format_lock.json \
    --dataset_dir data/sft_phase1_format_lock \
    --init_from_model checkpoints/GOLD_unified_v1
```

### Validate & Inspect

```bash
# Inspect tokenized dataset
python scripts/sft/inspect_sft_dataset.py \
    --dataset_dir data/sft_phase1_format_lock --show_samples 5

# Validate loss masks
python scripts/sft/validate_sft_dataset.py \
    --dataset data/sft_phase1_format_lock

# After training: test generation
python generate.py --model phase1_format_lock \
    --prompt "<myPT_system>You are MyPT.</myPT_system><myPT_user>Say hello.</myPT_user><myPT_assistant>"
```

### Success Gate

- Model generates `</myPT_assistant><myPT_eot>` within 20 tokens for simple prompts
- No runaway generation (infinite text without stopping)
- Responds in both EN and DE

---

## 5. Phase 2: Operators

**Goal:** Model generalizes abstract operators (COPY, WRAP, EXTRACT) to unseen payloads.

### Generate & Prepare

```bash
# 1. Generate operator dataset (contrastive design)
python scripts/sft/generate_operator_dataset.py \
    --output_dir data/sft_phase2_intermediate/operators

# 2. Mix 80% operators + 20% Phase 1 replay
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase2_intermediate/operators/operators.jsonl:0.8 \
             data/sft_phase1_intermediate/phase1_mixed.jsonl:0.2 \
    --output data/sft_phase2_intermediate/phase2_mixed.jsonl --shuffle

# 3. Tokenize with PACKING (critical for short operator episodes)
#    Operator episodes average ~45 tokens. Without packing: 96% wasted padding.
#    With packing: ~20 episodes per 1024 block = 17-34x more efficient.
#
#    The --enable_packing flag concatenates multiple episodes into each 1024-token
#    block, separated by segment_ids to prevent cross-episode attention bleed.
#    This is the single most impactful flag for Phase 1-2 training speed.
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase2_intermediate/phase2_mixed.jsonl \
    --output_dir data/sft_phase2_operators \
    --enable_packing --pack_block_size 1024
```

### Train

```bash
python train.py \
    --model_name phase2_operators \
    --config_file configs/sft/phase2_operators.json \
    --dataset_dir data/sft_phase2_operators \
    --init_from_model checkpoints/phase1_format_lock
```

### Success Gate

```bash
python scripts/eval/eval_operator.py --model phase2_operators -v
# Target: exact-match on unseen payloads > 80%
```

---

## 6. Phase 3: Chat SFT

**Goal:** Natural conversation, bilingual (DE/EN), system prompt adherence, basic think + cite.

### Data Sources

1. **Gold episodes** (existing) -- bilingual conversations
2. **HuggingFace** -- OASST2 (DE+EN), alpaca-gpt4_de, Dolci-Instruct
3. **Augmented** -- paraphrased variants of gold episodes
4. **Replay** -- 15% from Phase 1-2

### Generate & Prepare

```bash
# 1. Convert HuggingFace datasets
python scripts/sft/convert_hf_dataset.py \
    --dataset OpenAssistant/oasst2 \
    --output data/sft_hf/oasst2.jsonl \
    --languages en de --max_examples 10000

python scripts/sft/convert_hf_dataset.py \
    --dataset mayflowergmbh/alpaca-gpt4_de \
    --output data/sft_hf/alpaca_de.jsonl \
    --max_examples 5000

# 2. Generate RAG chat episodes (user_context + think + cite, EN+DE)
python scripts/sft/generate_rag_chat_sft.py \
    --docs_dir workspace/docs \
    --output data/sft_phase3_intermediate/rag_chat.jsonl \
    --num_examples 2000 --language mixed

# 3. Augment gold episodes
python scripts/sft/augment_episodes_paraphrase.py \
    --input data/gold_episodes/gold_bilingual.jsonl \
    --output data/sft_phase3_intermediate/gold_augmented.jsonl \
    --target_count 1000

# 4. Mix all sources + replay
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_hf/oasst2.jsonl:1.0 \
             data/sft_hf/alpaca_de.jsonl:1.0 \
             data/sft_phase3_intermediate/rag_chat.jsonl:1.0 \
             data/sft_phase3_intermediate/gold_augmented.jsonl:1.0 \
             data/sft_phase1_intermediate/phase1_mixed.jsonl:0.15 \
    --output data/sft_phase3_intermediate/phase3_mixed.jsonl --shuffle

# 5. Tokenize with packing (episodes average ~250 tokens, 2-10x efficiency gain)
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase3_intermediate/phase3_mixed.jsonl \
    --output_dir data/sft_phase3_chat \
    --enable_packing --pack_block_size 1024
```

### Train

```bash
python train.py \
    --model_name phase3_chat \
    --config_file configs/sft/phase3_chat_sft.json \
    --dataset_dir data/sft_phase3_chat \
    --init_from_model checkpoints/phase2_operators
```

### Success Gate

```bash
python scripts/eval/sft_eval_suite.py --model phase3_chat -v
# Passes: format, echo, anti-echo, regression gates
# Model responds in user's language
# Citations present when context is provided
```

---

## 7. Phase 4: Multi-turn Boundaries

**Goal:** Multi-turn conversations (2-4 turns), clean turn boundaries, context carryover.

### Generate & Prepare

```bash
# 1. Generate multi-turn episodes (followup, clarification, topic switch, context)
python scripts/sft/generate_multiturn_sft.py \
    --docs_dir workspace/docs \
    --output data/sft_phase4_intermediate/multiturn_synthetic.jsonl \
    --num_examples 3000 --language mixed

# 2. Convert OASST2 multi-turn trees (depth >= 2)
python scripts/sft/convert_hf_dataset.py \
    --dataset OpenAssistant/oasst2 \
    --output data/sft_hf/oasst2_multiturn.jsonl \
    --languages en de --max_examples 8000

# 3. Mix 70% multi-turn + 15% context injection + 15% Phase 3 replay
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase4_intermediate/multiturn_synthetic.jsonl:1.0 \
             data/sft_hf/oasst2_multiturn.jsonl:0.85 \
             data/sft_phase3_intermediate/phase3_mixed.jsonl:0.15 \
    --output data/sft_phase4_intermediate/phase4_mixed.jsonl --shuffle

# 4. Tokenize with packing (multi-turn episodes average ~500 tokens, 1.3-5x gain)
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase4_intermediate/phase4_mixed.jsonl \
    --output_dir data/sft_phase4_multiturn \
    --enable_packing --pack_block_size 1024
```

### Train

```bash
python train.py \
    --model_name phase4_multiturn \
    --config_file configs/sft/phase4_multiturn.json \
    --dataset_dir data/sft_phase4_multiturn \
    --init_from_model checkpoints/phase3_chat
```

### Success Gate

- Model maintains topic and language across 3+ turns
- Clean `</myPT_assistant><myPT_eot>` after each turn (no bleed-through)

---

## 8. Phase 5: Simple Toolcall

**Goal:** Single-step tool use: when to call, JSON format, reading results, grounded answers.

### Generate & Prepare

```bash
# 1. Generate tool-calling episodes
python scripts/sft/generate_agent_sft.py \
    --docs_dir workspace/docs \
    --output data/sft_phase5_intermediate/tool_episodes.jsonl \
    --num_examples 5000

# 2. Convert HuggingFace tool-use datasets
python scripts/sft/convert_hf_dataset.py \
    --dataset allenai/Dolci-Instruct-SFT-Tool-Use \
    --output data/sft_hf/dolci_tools.jsonl \
    --max_examples 5000

# 3. Mix: 40% our tools + 30% HF tools + 20% NO_TOOL + 10% replay
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase5_intermediate/tool_episodes.jsonl:1.0 \
             data/sft_hf/dolci_tools.jsonl:0.6 \
             data/sft_phase3_intermediate/phase3_mixed.jsonl:0.1 \
    --output data/sft_phase5_intermediate/phase5_mixed.jsonl --shuffle

# 4. Tokenize with TOOL serializer (handles toolcall/toolresult roles)
#    NOTE: prepare_tool_sft.py does NOT support --enable_packing yet.
#    Phase 5 episodes (~400 tokens avg) would benefit from packing (1.5-5x gain).
#    TODO: port packing logic from prepare_chat_sft.py to prepare_tool_sft.py
python scripts/sft/prepare_tool_sft.py \
    --input data/sft_phase5_intermediate/phase5_mixed.jsonl \
    --output_dir data/sft_phase5_toolcall
```

### Train

```bash
python train.py \
    --model_name phase5_toolcall \
    --config_file configs/sft/phase5_simple_toolcall.json \
    --dataset_dir data/sft_phase5_toolcall \
    --init_from_model checkpoints/phase4_multiturn
```

### Success Gate

- Model calls correct tool with valid JSON > 90%
- NO_TOOL accuracy > 80% (answers directly when no tool needed)
- Grounded answers cite source via `<myPT_cite>`

---

## 9. Phase 6: Agentic RAG

**Goal:** Multi-step tool chains (search -> get_doc -> answer), full reasoning, error recovery.

### Generate & Prepare

```bash
# 1. Generate multi-step agentic episodes (think+cite, validated, EN+DE)
#    Patterns: SEARCH_ANSWER, SEARCH_GETDOC_ANSWER, LIST_SELECT_SUMMARIZE, ERROR_RECOVERY, NO_TOOL
python scripts/sft/generate_sft_tool_episodes.py \
    --workspace_dir workspace/ \
    --output data/sft_phase6_intermediate/agentic_episodes.jsonl \
    --num_examples 5000 --language mixed

# 2. Mix: 30% multi-step + 20% single-step review + 20% HF tools + 20% NO_TOOL + 10% replay
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase6_intermediate/agentic_episodes.jsonl:1.0 \
             data/sft_phase5_intermediate/tool_episodes.jsonl:0.2 \
             data/sft_hf/dolci_tools.jsonl:0.2 \
             data/sft_phase3_intermediate/phase3_mixed.jsonl:0.1 \
    --output data/sft_phase6_intermediate/phase6_mixed.jsonl --shuffle

# 3. Tokenize
python scripts/sft/prepare_tool_sft.py \
    --input data/sft_phase6_intermediate/phase6_mixed.jsonl \
    --output_dir data/sft_phase6_agentic
```

### Train

```bash
python train.py \
    --model_name phase6_agentic \
    --config_file configs/sft/phase6_agentic_rag.json \
    --dataset_dir data/sft_phase6_agentic \
    --init_from_model checkpoints/phase5_toolcall
```

### Success Gate

- Completes 2-3 step tool chains correctly (no infinite loops)
- Final answer grounded in retrieved content
- Uses `<myPT_think>` to reason about which tool to call next
- Uses `<myPT_cite>` to attribute sources

---

## 10. HuggingFace Dataset Integration

The converter script maps external datasets to our JSONL format:

```bash
python scripts/sft/convert_hf_dataset.py \
    --dataset <HF_PATH> \
    --output <OUTPUT.jsonl> \
    --languages en de \
    --max_examples 10000
```

### Recommended Datasets by Phase

**Phase 3 (Chat):**

| Dataset                        | Command                                                                | Notes              |
|--------------------------------|------------------------------------------------------------------------|--------------------|
| OASST2 (EN+DE)                | `--dataset OpenAssistant/oasst2 --languages en de`                     | Native German!     |
| Alpaca-GPT4 DE                | `--dataset mayflowergmbh/alpaca-gpt4_de`                               | 50K German         |
| Dolci-Instruct                 | `--dataset allenai/Dolci-Instruct-SFT --max_examples 10000`           | 2.15M, 70+ langs   |
| OpenSchnabeltier               | `--dataset LeoLM/OpenSchnabeltier`                                     | 21.7K German       |

**Phase 4 (Multi-turn):**

| Dataset                        | Command                                                                | Notes              |
|--------------------------------|------------------------------------------------------------------------|--------------------|
| OASST2 multi-turn             | `--dataset OpenAssistant/oasst2 --languages en de`                     | Tree depth >= 2    |
| Ultra-Chat DE                  | `--dataset mayflowergmbh/ultra-chat_de --max_examples 10000`          | 208K multi-turn    |

**Phase 5-6 (Tool Calling):**

| Dataset                        | Command                                                                | Notes              |
|--------------------------------|------------------------------------------------------------------------|--------------------|
| Dolci Tool-Use                 | `--dataset allenai/Dolci-Instruct-SFT-Tool-Use --max_examples 10000`  | XML -> JSON mapped |
| German Function Calling        | `--dataset flozi00/german-function-calling`                            | 1.33K German       |
| German RAG SFT                 | `--dataset avemio/German-RAG-SFT-ShareGPT-HESSIAN-AI`                 | 200K+ with RAG     |

All datasets have permissive licenses (ODC-BY, Apache-2.0, CC-BY-SA).

---

## 11. System Prompt Strategy (Loss Mask Optimization)

System prompt tokens are **always masked** (loss=0). In short episodes (Phase 1-2),
a long system prompt dramatically reduces the fraction of supervised tokens per
sequence. We use a three-tier strategy to maximize loss mask %:

| Phases | Prompt | ~Tokens | Rationale |
|:--|:--|--:|:--|
| 1-2 (Format Lock, Operators) | `"You are MyPT."` | 4 | Episodes are ultra-short; every masked token hurts |
| 3-4 (Chat, Multi-turn) | `CHAT_SYSTEM_PROMPT` | 15-20 | Episodes are long enough to absorb it |
| 5-6 (Toolcall, Agentic) | `AGENTIC_STANDARD_PROMPT` | ~80 | Tool episodes are long; need to list available tools |

All prompts are defined in `core/system_prompts.py`. Phase 3-4 uses 4 short
variants (via `CHAT_SYSTEM_PROMPTS` list) for surface diversity without token
bloat. Phase 5-6 uses 8 variants (in `generate_agent_sft.py`'s `SYSTEM_MESSAGES`).

**Rule:** Do NOT increase system prompt length for early phases. If you need more
context for the model, put it in the training data, not the system prompt.

---

## 12. Anti-Forgetting Strategies

### Problem

SFT phases are sequential: each phase trains exclusively on its own data. Without
mitigation, the model gradually forgets pre-training knowledge (math, facts, German)
and skills from earlier SFT phases (format compliance, operators).

### Strategy 1: Pre-training Data Replay (1-5%)

Mix a small fraction of raw pre-training data into each SFT phase. Research shows
that even 1% replay prevents catastrophic forgetting of base knowledge.

```bash
# Step 1: Generate the replay buffer (one-time)
python scripts/sft/generate_pretrain_replay.py \
    --shard_dirs data/unified_tokenized/fineweb_edu \
                 data/unified_tokenized/stackexchange_qa \
                 data/unified_tokenized/code_python \
                 data/multilingual_1.5B_wiki90 \
    --output data/sft_replay/pretrain_replay.jsonl \
    --num_episodes 2000 --max_tokens 256

# Step 2: Mix 5% replay into any SFT phase
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_replay/pretrain_replay.jsonl:1.0 \
             data/sft_phase3_intermediate/episodes.jsonl:19.0 \
    --output data/sft_phase3_intermediate/mixed_with_replay.jsonl \
    --shuffle
```

Replay episodes have `{"_replay": true, "text": "..."}` format. Both
`prepare_chat_sft.py` and `prepare_tool_sft.py` automatically detect these
and emit full loss (mask=1 on all tokens) instead of assistant-only masking.

### Strategy 2: Cross-Phase Replay (5-10%)

Carry forward a small sample of episodes from each completed phase into the
next phase's training mix. This prevents "phase forgetting" where Phase 4
overwrites what Phase 2 taught.

**Mandatory cross-phase replay schedule:**

| Training Phase | Replay Sources | Ratios |
|:--|:--|:--|
| Phase 2 (Operators) | Phase 1 format lock | 5% Phase 1 + 95% Phase 2 |
| Phase 3 (Chat SFT) | Phase 1 + Phase 2 | 3% P1 + 3% P2 + 94% P3 |
| Phase 4 (Multi-turn) | Phase 1-3 | 2% P1 + 2% P2 + 3% P3 + 93% P4 |
| Phase 5 (Toolcall) | Phase 1-4 | 2% P1 + 2% P2 + 2% P3 + 2% P4 + 92% P5 |
| Phase 6 (Agentic) | Phase 1-5 | 1% each P1-P5 + 95% P6 |

Plus 5% pre-training replay in every phase (included in the ratios above as
part of the phase-specific data).

Example for Phase 3:
```bash
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_replay/pretrain_replay.jsonl:1.0 \
             data/sft_phase1_intermediate/format_lock_mixed.jsonl:0.6 \
             data/sft_phase2_intermediate/operator_train.jsonl:0.6 \
             data/sft_phase3_intermediate/rag_chat_episodes.jsonl:19.0 \
    --output data/sft_phase3_intermediate/mixed_with_all_replay.jsonl \
    --shuffle
```

### Strategy 3: Regression Gating

After each phase, run `scripts/eval/run_regression_gate.py` to verify that
previous skills have not regressed. See Section 15 (Scripts Reference) for usage.

### Strategy 4: OOD Generalization Eval

Held-out evaluation prompts using **novel phrasings not seen during training**
detect overfitting to synthetic templates (audit item A4). Four JSONL files in
`data/eval_ood/` cover Phases 3-6:

| File | Phase | What it tests |
|:--|:--|:--|
| `phase3_chat_ood.jsonl` | 3 | RAG-context answering with novel question styles |
| `phase4_multiturn_ood.jsonl` | 4 | Multi-turn follow-up with unseen phrasings |
| `phase5_toolcall_ood.jsonl` | 5 | Tool selection from unfamiliar request forms |
| `phase6_agentic_ood.jsonl` | 6 | Multi-step tool chaining with novel phrasing |

The regression gate automatically picks up these files for phases >= 3 and
reports pass rates (currently as warnings, not hard failures).

### Strategy 5: Contrastive / Negative Tool Examples

8% of Phase 5+ episodes are **wrong-tool-then-correction** examples. The model
first calls the wrong tool, observes an unhelpful result, then self-corrects
with a think block explaining the mistake, and finally calls the correct tool.
This teaches tool-selection discrimination and self-correction.

---

## 13. NEFTune Embedding Noise

NEFTune (Noisy Embedding Fine-Tuning, arXiv:2310.05914) adds uniform noise to
token embeddings during training. This simple regularization technique shows
10-115% improvement on instruction-following benchmarks.

### How It Works

After the embedding lookup and before the transformer blocks, noise is added:
```
embed += uniform(-alpha, alpha) / sqrt(seq_len * embed_dim)
```

This is ONLY active during training (`model.training=True`). Inference is
completely unaffected.

### Configuration

Add `neftune_alpha` to the SFT config JSON:
```json
{
  "neftune_alpha": 5.0
}
```

Recommended values per phase:

| Phase | neftune_alpha | Rationale |
|:--|:--|:--|
| Phase 1 (Format Lock) | 5.0 | Light noise, don't interfere with tag learning |
| Phase 2 (Operators) | 5.0 | Light noise for abstract patterns |
| Phase 3 (Chat SFT) | 10.0 | Moderate, improves response diversity |
| Phase 4 (Multi-turn) | 10.0 | Moderate, prevents template memorization |
| Phase 5 (Toolcall) | 5.0 | Light, JSON structure must be precise |
| Phase 6 (Agentic) | 5.0 | Light, multi-step chains need precision |

Set to `0.0` to disable (default).

---

## 14. Weighted Loss Masking

Standard SFT uses binary loss masking: 0 (masked) or 1 (trained). Weighted
loss masking assigns higher importance to structural control tokens that
"steer" generation direction. Research shows 39-83% gains on reasoning benchmarks.

### Weight Scheme

| Token Type | Weight | Rationale |
|:--|:--|:--|
| System, user, toolresult | 0.0 | Never train |
| `<myPT_assistant>` (open) | 0.0 | Given in prompt |
| Normal assistant content | 1.0 | Standard training |
| `</myPT_assistant>` (close) | 2.0 | Critical stop signal |
| `<myPT_eot>` | 2.0 | Critical stop signal |
| `<myPT_think>` / `</myPT_think>` | 1.5 | Steering tokens |
| `<myPT_cite>` / `</myPT_cite>` | 1.5 | Steering tokens |
| `<myPT_toolcall>` / `</myPT_toolcall>` | 2.0 | Critical action triggers |

### Usage

Add `--weighted_mask` to the prepare scripts:
```bash
python scripts/sft/prepare_chat_sft.py \
    --input data/episodes.jsonl \
    --output_dir data/sft_weighted \
    --weighted_mask

python scripts/sft/prepare_tool_sft.py \
    --input data/tool_episodes.jsonl \
    --output_dir data/tool_sft_weighted \
    --weighted_mask
```

The loss computation in `core/model.py` already supports continuous float masks,
so no model changes are needed.

---

## 15. Scripts Reference

### Generators (produce JSONL)

| Script | Phase | What it does |
|--------|-------|-------------|
| `generate_format_lock_dataset.py` | 1 | Combinatorial Q&A (EN+DE), short answers |
| `generate_echo_dataset.py` | 1 | Echo/repeat instructions, anti-echo, gibberish |
| `generate_operator_dataset.py` | 2 | COPY/WRAP/EXTRACT with contrastive design |
| `generate_rag_chat_sft.py` | 3 | RAG episodes with user_context + think + cite (EN+DE) |
| `generate_multiturn_sft.py` | 4 | Multi-turn conversations: followup, clarification, topic switch |
| `generate_agent_sft.py` | 5 | Single-step tool-calling (EN+DE, think+cite, NO_TOOL) |
| `generate_sft_tool_episodes.py` | 6 | Multi-step agentic tool chains (validated, EN+DE) |
| `convert_hf_dataset.py` | 3-6 | Universal HuggingFace dataset converter |

### Pipeline (mix, tokenize)

| Script | What it does |
|--------|-------------|
| `prepare_phase1_format_lock.py` | Automated Phase 1 pipeline (generate + mix + tokenize) |
| `mix_sft_jsonl.py` | Mix multiple JSONL files with sampling ratios |
| `prepare_chat_sft.py` | Tokenize chat JSONL to binary (Phase 1-4) |
| `prepare_tool_sft.py` | Tokenize tool JSONL to binary (Phase 5-6) |

### Quality (inspect, validate, deduplicate)

| Script | What it does |
|--------|-------------|
| `inspect_sft_dataset.py` | Show samples, stats from tokenized dataset |
| `validate_sft_dataset.py` | Validate dataset integrity |
| `validate_sft_episode_masks.py` | Check mask alignment |
| `verify_loss_mask_direction.py` | Verify mask direction (assistant=1) |
| `verify_mask_alignment.py` | Token-level mask alignment check |
| `deduplicate_episodes.py` | Remove duplicate episodes |
| `deduplicate_by_user_message.py` | Deduplicate by user message content |
| `analyze_episode_diversity.py` | Analyze diversity metrics |

### Augmentation

| Script | What it does |
|--------|-------------|
| `augment_episodes_paraphrase.py` | Rule-based paraphrasing to expand datasets |
| `diversify_user_messages.py` | Vary user message templates |

### Evaluation

| Script | What it does |
|--------|-------------|
| `scripts/eval/sft_eval_suite.py` | Full evaluation (format, echo, anti-echo, regression) |
| `scripts/eval/eval_operator.py` | Operator exact-match evaluation |

### Translation (DE/EN)

| Script | What it does |
|--------|-------------|
| `scripts/translation/extract_for_translation.py` | Extract translatable strings |
| `scripts/translation/translate_deepl.py` | Translate via DeepL API |
| `scripts/translation/recombine_translations.py` | Recombine translated strings |
| `scripts/translation/merge_bilingual_episodes.py` | Merge EN + DE episodes |

---

## 16. Configs Reference

All SFT configs include the LLaMA-2 architecture fields. Key configs by phase:

| Phase | Config File | LR | Iters | Block |
|-------|-------------|-----|-------|-------|
| 1 | `configs/sft/phase1_format_lock.json` | 7e-5 | 2000 | 512 |
| 2 | `configs/sft/phase2_operators.json` | 3e-5 | 1200 | 1024 |
| 3 | `configs/sft/phase3_chat_sft.json` | 3e-5 | 5000 | 1024 |
| 4 | `configs/sft/phase4_multiturn.json` | 2.5e-5 | 3000 | 1024 |
| 5 | `configs/sft/phase5_simple_toolcall.json` | 2e-5 | 3000 | 1024 |
| 6 | `configs/sft/phase6_agentic_rag.json` | 1.5e-5 | 4000 | 1024 |

All configs use:
- `use_loss_mask: true`
- `batch_sampling_mode: "epoch"`
- `use_amp: true` / `amp_dtype: "bf16"`

---

## 17. Troubleshooting

### Model generates endless text (no stopping)

Phase 1 Format Lock was insufficient. Re-run with more iterations or higher format_lock ratio.
Check that `<myPT_eot>` has mask=1 in the tokenized dataset.

### Model copies user input instead of answering

Needs more anti-echo contrast in training. Add echo dataset with `--gibberish include`
and increase the anti-echo ratio.

### Model always calls tools (even when not needed)

Phase 5 needs more NO_TOOL episodes (target 20% of dataset).
Model should learn to answer directly for general knowledge questions.

### German responses are poor quality

Check DE/EN ratio in training data. Target at least 30% German.
Add more German sources from HuggingFace (alpaca_de, OpenSchnabeltier).

### Loss doesn't decrease during SFT

Check that `use_loss_mask: true` is in the config. Without it, the model
trains on system/user tokens too, which dilutes the signal.
Also verify `init_from_model` points to the correct checkpoint.

### Tokenized dataset is very small

Check that `prepare_chat_sft.py` is finding episodes in the JSONL.
Run `inspect_sft_dataset.py` to see episode count and token stats.

---

## Related Documents

- **Tag nesting rules:** `docs/sft/TAG_NESTING_REFERENCE.md`
- **Special tokens source:** `core/special_tokens.py`
- **System prompts:** `core/system_prompts.py`
- **SFT curriculum plan:** `.cursor/plans/sft_curriculum_plan_*.plan.md`
- **Archived docs:** `docs/sft/archive/` (historical reference)
