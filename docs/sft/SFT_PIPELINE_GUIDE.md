# MyPT SFT Pipeline Guide

Complete reference for Supervised Fine-Tuning the MyPT 750M LLaMA-2 base model
into an agentic RAG assistant. Covers architecture, data flow, every phase,
exact commands, and success criteria.

**Last updated:** March 2026 (Phase 3 pipeline section refreshed with a full build log)
**Base model:** `checkpoints/phase1b_context_ext` (LLaMA-2 style, 750M params, 4096 context via PI)
**Architecture:** RoPE + SwiGLU + RMSNorm, tie_weights=true, 1280d/20h/32L, rope_scale=4.0

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1b: Context Extension (1024 → 4096)](#2-phase-1b-context-extension)
3. [Data Flow: Generator to Training](#3-data-flow-generator-to-training)
4. [Special Tokens and Loss Masking](#4-special-tokens-and-loss-masking)
5. [Phase 1: Format Lock](#5-phase-1-format-lock)
6. [Phase 2: Operators](#5-phase-2-operators)
7. [Phase 3: Chat SFT](#6-phase-3-chat-sft)
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

| Source                | Tokens | Role                                                          |
| :-------------------- | :----- | :------------------------------------------------------------ |
| HotpotQA (distractor) | ~108M  | Long multi-passage, position-varied gold paragraphs           |
| MS MARCO v2.1         | ~100M  | 10-passage search results, position-varied, real Bing queries |
| TriviaQA (evidence)   | ~78M   | Medium-long grounded trivia                                   |
| SQuAD v2              | ~32M   | Short extractive EN QA (incl. unanswerable)                   |
| MuSiQue               | ~20M   | Hard multi-hop (2-4 hops)                                     |
| GermanQuAD            | ~1.5M  | Short extractive DE QA                                        |

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

After this phase, ALL SFT phases use `block_size: 4096` and `rope_scale: 4.0`.

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
    { "role": "user", "content": "What is Python?" },
    { "role": "assistant", "content": "A programming language." }
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

- `"context"` on user messages -- becomes `<myPT_user_context>` when `prepare_chat_sft.py` is run with `--enable_rag_tags`
- `"think"` on assistant messages -- becomes `<myPT_think>` when `prepare_chat_sft.py` is run with `--enable_rag_tags`
- `"cite"` on assistant messages -- becomes `<myPT_cite>` when `prepare_chat_sft.py` is run with `--enable_rag_tags`

### Tokenization: Chat vs Tool

- **`prepare_chat_sft.py`** -- For Phase 1-4 (no toolcall/toolresult roles). Uses token-ID-based
  masking: assistant content = train, everything else = mask. Supports packing (multiple short
  episodes per block) for efficiency. Optional RAG-tag serialization is gated behind
  `--enable_rag_tags` and includes an automatic dataset audit printout (context/think/cite by source).

- **`prepare_tool_sft.py`** -- For Phase 5-6 (has toolcall/toolresult roles). Handles all 19 tags
  including think, cite, user_context, assistant_context. Char-level masking converted to token-level.
  Does NOT support packing yet (TODO for Phase 5).

Both produce the same binary format (tokens.bin + mask.bin + episodes.idx).

### Episode Packing

Without packing, each episode is padded to `block_size` (4096). A 30-token format lock
episode wastes >99% of compute on padding. Packing fills each 4096-token block with
multiple episodes back-to-back, dramatically increasing supervised tokens per training step.

Cross-episode attention is isolated via `segment_ids` -- the attention mask prevents
episodes within the same packed sequence from attending to each other.

**When to use `--enable_packing`:**

| Phase | Avg Episode | Episodes/Block | Efficiency Gain | Pack?                                     |
| ----- | ----------- | -------------- | --------------- | ----------------------------------------- |
| 1     | ~30 tokens  | 25-50          | 25-50x          | YES                                       |
| 2     | ~45 tokens  | 17-34          | 17-34x          | YES                                       |
| 3     | ~250 tokens | 2-10           | 2-10x           | YES                                       |
| 4     | ~500 tokens | 1-5            | 1.3-5x          | YES                                       |
| 5     | ~400 tokens | 1-5            | 1.5-5x          | N/A (prepare_tool_sft.py, no packing yet) |
| 6     | ~800 tokens | 1-2            | ~1x             | NO (episodes fill the window)             |

**Phase 1-2 are where packing is transformative** -- without it, training is dramatically slower
because almost every token in the 4096 window is wasted padding.

```bash
# Packing flag (add to prepare_chat_sft.py calls):
--enable_packing --pack_block_size 4096

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
| ----- | ------------------------------------------ |
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

# 4A. Tokenize with loss masking + PACKING (default)
#     Episodes are short; packing improves token utilization.
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase1_intermediate/phase1_mixed.jsonl \
    --output_dir data/sft_phase1_format_lock \
    --val_split 0.05 \
    --enable_packing --pack_block_size 4096

# 4B. Tokenize with loss masking + NO PACKING (A/B variant)
#     Use a separate output dir so you can compare packed vs non-packed runs.
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase1_intermediate/phase1_mixed.jsonl \
    --output_dir data/sft_phase1_format_lock_nopack \
    --val_split 0.05
```

### Train

```bash
# Packed variant (default)
python train.py \
    --model_name phase1_format_lock \
    --config_file configs/sft/phase1_format_lock.json \
    --dataset_dir data/sft_phase1_format_lock \
    --init_from_model checkpoints/GOLD_unified_v1

# Non-packed variant (A/B)
# Uses a dedicated short-context config. Without packing, 4096 would waste too much padding.
# Expect more iterations than packed mode for similar token budget.
python train.py \
    --model_name phase1_format_lock_nopack \
    --config_file configs/sft/phase1_format_lock_nopack_shortctx.json \
    --dataset_dir data/sft_phase1_format_lock_nopack \
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

### Goal

Teach **COPY**, **WRAP**, and **EXTRACT** as abstract skills: same instruction pattern, **unseen payloads** in validation, and no shortcut via memorizing fixed answers. Phase 2 keeps **Phase 1 format tags** stable while the model learns precise assistant outputs.

### Canonical path (recommended): **Phase 2 Remix Existing** (`phase2_remix_existing`)

The **final** operator phase for this project is a **single SFT run** on data built only from **existing generators** — no separate “unified rebuild” synthesizer.

| Source | Role |
| ------ | ---- |
| **Original Phase 2** (`phase2_mixed.jsonl`) | Backbone: operator contrast, payload/template diversity, train/val segregation from `generate_operator_dataset.py`. Strongest signal for **abstraction**. |
| **Phase 2.5** (`phase2_5_mixed.jsonl`) | Stronger minority share (**0.3** in the canonical run): WRAP delimiter diversity + echo/anti-echo from `prepare_phase2_5_wrap_antiecho.py`. |
| **Phase 2.7** (`phase2_7_mixed.jsonl`) | Small weight: rebalance / anti-echo + operator replay from `prepare_phase2_7_rebalance.py`. |
| **Phase 2.8** | Not blended into **train** in this recipe; **validation** rows come from **Phase 2.8 val** (`phase2_8_val.jsonl`) so val covers echo/Wrap edge cases **without** overlapping train payloads/templates. |

**Why remix?** Sequential training (2 → 2.5 → 2.7 → 2.8) each moved specific skills forward; **remixing validated JSONLs** into one acquisition run preserved **original Phase 2–style abstraction** while adding **targeted** statistics from 2.5 and 2.7. A monolithic re-synthesized “unified Phase 2” mix did **not** match that behavior in practice.

**Train mix (canonical):** `1.0 : 0.3 : 0.15` in `mix_sft_jsonl.py` means, **by default**, “take **100%** of `phase2_mixed` rows, **30%** of `phase2_5_mixed` rows, **15%** of `phase2_7_mixed` rows” and **concatenate** — i.e. ratios are **per-source fractions**, **not** guaranteed shares of the final row count (output size depends on each file’s length). For a **fixed output size** and **normalized blend weights** (e.g. “~68.9% phase2 / ~20.7% phase2.5 / ~10.3% phase2.7 of **N** rows”), use **`--output_blend --target_size N`** with **`--weights 1.0 0.3 0.15`** and plain input paths (see script docstring).

**Validation:** Combine candidates from **operator val**, **wrap_focus val**, and **phase2_8 val**, then enforce **train/val disjointness** with `--exclude_from_train` and `--disjoint_keys payload,template,pair`, and cap with `--target_size` (e.g. 5000). Use the `phase2_8_val.jsonl` path that exists on your machine (`.../sft_phase2_8_intermediate/...` or `.../sft_phase2_8b_intermediate/...`).

#### Remix reassessment: COPY, echo, and anti-echo (bridge-aligned)

`eval_phase2_8_bridge.py` stresses **exact** operator outputs, **literal echo**, and **anti-echo** (assistant must **not** contain a quoted “forbidden” token). Phase 3 chat/RAG then **pulls** the policy toward paraphrase and long answers, so the **remix train** should carry **enough marginal mass** on those manifolds before you ever fine-tune chat.

| Bridge pain | What to reinforce | Best corpus leverage |
| ----------- | ----------------- | --------------------- |
| **COPY / WRAP / EXTRACT** exact match | Abstract operator + delimiter habits | **`phase2_mixed.jsonl`** (backbone); **2.5** adds **WRAP** style diversity |
| **Echo basic** (repeat exact string) | Literal copy under instruction | **`phase2_5_mixed`** (echo + contrast stream from `prepare_phase2_5_wrap_antiecho.py`); optional **`phase2_6_mixed_train`** (~20% of 2.6 is echo) |
| **Anti-echo** (e.g. must not say `Blurpix`) | Short safe answers; no leakage of quoted token | **`phase2_7_mixed`** (**40%** anti-echo design in `prepare_phase2_7_rebalance.py`); **2.5** (`--echo_anti_ratio` default **0.4**); **`phase2_6_mixed_train`** (**60%** anti-echo + **strict-safe** injections in `prepare_phase2_6_antiecho.py`) |

**Design rule:** Prefer **`mix_sft_jsonl.py --output_blend --target_size N`** so weights are **shares of the final row count**, not “30% of file B’s rows.” Legacy `path:0.3` mixing often **underweights** 2.5/2.7 relative to 2 once file sizes differ.

**Presets (weights are relative; script normalizes to `N`/`target_size` rows):**

1. **Balanced repair (default if bridge echo/anti-echo regressed after Phase 3):**  
   `--weights 1.0 0.45 0.22` on **(phase2_mixed, phase2_5_mixed, phase2_7_mixed)** → about **60% / 27% / 13%** of `N`. Keeps Phase 2 abstraction largest while **lifting** echo/anti-echo density vs canonical `1.0 / 0.3 / 0.15` *output* proportions (~69% / 21% / 10%).

2. **Strong bridge focus (recommended when anti-echo COPY/echo still fail):** add **`phase2_6_mixed_train.jsonl`** (run `prepare_phase2_6_antiecho.py` once if missing):  
   `--weights 1.0 0.35 0.15 0.12` on **(phase2, 2.5, 2.7, 2.6)** → about **62% / 22% / 9% / 7%**. The **~7%** from 2.6 is **dense** echo + anti-echo + operator replay (high ROI per row for gate-shaped errors).

3. **Aggressive:** e.g. **`0.85 0.50 0.28 0.15`** on the same four files (**~48% / 28% / 16% / 8%**) — use only if you accept less Phase-2-pure abstraction share in exchange for more contrast.

Pick **`N`** (`--target_size`) as the **exact episode count** in the merged train JSONL (e.g. **~110k**). After rebuilding **`phase2_remix_existing_train.jsonl`**, **re-tokenize** with **`--pack_block_size 4096`** (packed **shard** count will be lower than `N`; that is expected). **Retrain** `phase2_remix_existing`, then **rebuild Phase 3** as needed.

```bash
# Example: ~110k episodes, 4-way bridge-focused blend, 4096 packing in step 3
python scripts/sft/prepare_phase2_6_antiecho.py --output_dir data/sft_phase2_6_intermediate

python scripts/sft/mix_sft_jsonl.py --output_blend --target_size 110000 \
    --inputs data/sft_phase2_intermediate/phase2_mixed.jsonl \
             data/sft_phase2_5_intermediate/phase2_5_mixed.jsonl \
             data/sft_phase2_7_intermediate/phase2_7_mixed.jsonl \
             data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl \
    --weights 1.0 0.35 0.15 0.12 \
    --output data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
    --shuffle --seed 2907
```

#### Prerequisites (intermediate JSONLs)

1. **Phase 2 core** — `generate_operator_dataset.py` + `phase2_mixed.jsonl` (80% operators / 20% Phase 1 replay). See **§5.1** for the minimal recipe.
2. **Phase 2.5** — `data/sft_phase2_5_intermediate/phase2_5_mixed.jsonl` + `wrap_focus_val.jsonl` from `prepare_phase2_5_wrap_antiecho.py`.
3. **Phase 2.7** — `data/sft_phase2_7_intermediate/phase2_7_mixed.jsonl` from `prepare_phase2_7_rebalance.py`.
4. **Phase 2.8** — `phase2_8_val.jsonl` from `prepare_phase2_8_echo_rebalance.py`.
5. **(Optional, bridge-heavy remix)** — `data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl` from `prepare_phase2_6_antiecho.py` for **echo + anti-echo** density.

#### Build remix → tokenize → train

```bash
# 1) Training JSONL — legacy: 100% phase2 + 30% of phase2.5 file + 15% of phase2.7 file (source-relative)
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase2_intermediate/phase2_mixed.jsonl:1.0 \
             data/sft_phase2_5_intermediate/phase2_5_mixed.jsonl:0.3 \
             data/sft_phase2_7_intermediate/phase2_7_mixed.jsonl:0.15 \
    --output data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
    --shuffle --seed 2907

# Alternative — ~110k rows, 3-way output proportions 1 : 0.45 : 0.22 (normalized)
# python scripts/sft/mix_sft_jsonl.py --output_blend --target_size 110000 \
#     --inputs data/sft_phase2_intermediate/phase2_mixed.jsonl \
#              data/sft_phase2_5_intermediate/phase2_5_mixed.jsonl \
#              data/sft_phase2_7_intermediate/phase2_7_mixed.jsonl \
#     --weights 1.0 0.45 0.22 \
#     --output data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
#     --shuffle --seed 2907

# 2) Validation JSONL: broad sources, disjoint from train (adjust phase2_8 path if needed)
python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase2_intermediate/operators/operator_val.jsonl:1.0 \
             data/sft_phase2_5_intermediate/wrap_focus_val.jsonl:1.0 \
             data/sft_phase2_8_intermediate/phase2_8_val.jsonl:1.0 \
    --output data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_val.jsonl \
    --exclude_from_train data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
    --disjoint_keys payload,template,pair \
    --target_size 5000 \
    --shuffle --seed 2908

# 3) Tokenize + pack
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
    --output_dir data/sft_phase2_remix_existing \
    --val_file data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_val.jsonl \
    --no_system_prompt \
    --enable_packing --pack_block_size 4096 --pack_by_field "_meta.operator"

# 4) Train (GOLD may use external_gate — see Learnings below)
python train.py \
    --model_name phase2_remix_existing \
    --config_file configs/sft/phase2_remix_existing.json \
    --dataset_dir data/sft_phase2_remix_existing \
    --init_from_model checkpoints/phase1_format_lock_gold
```

**Windows:** use `py.exe -3` instead of `python` where needed.

### Learnings

- **Packing:** Episodes are short; **`--enable_packing`** with `--pack_block_size 4096` is essential. **`--pack_by_field "_meta.operator"`** keeps packs operator-coherent.
- **No eval leakage:** `core.eval_blacklist` in generators + **disjoint val** when mixing.
- **GOLD:** `configs/sft/phase2_remix_existing.json` uses **`gold_selection.strategy: external_gate`**: runs `scripts/eval/eval_phase2_8_bridge.py` on a schedule and can maximize a gate metric (e.g. `hard_avg_rate`) with loss guards. `require_pass: false` avoids freezing GOLD when the gate is imperfect.
- **Interpreting `sft_eval_suite`:** Strong **operators**, **format**, and **anti-echo** buckets are on-target for Phase 2. **Regression, hierarchy, injection, strict JSON, citation** are mostly **Phase 3+** — a global FAIL on the full suite does **not** by itself block Chat SFT if the operator checkpoint is strong.

### Success checks

```bash
python scripts/eval/eval_operator.py --model phase2_remix_existing_gold -v
python scripts/eval/sft_eval_suite.py --model phase2_remix_existing_gold --no_system_prompt -v
python scripts/eval/eval_phase2_8_bridge.py --model phase2_remix_existing_gold --no_system_prompt -v
python scripts/eval/eval_phase2_5_wrap_focus.py --model phase2_remix_existing_gold -v
```

---

### 5.1 Legacy: standalone Phase 2 (`phase2_operators`)

Minimal operator-only run (operators + 20% Phase 1 replay) — useful for baselines or reproducing older checkpoints. **Remix** (above) is the recommended path for new work.

```bash
python scripts/sft/generate_operator_dataset.py \
    --output_dir data/sft_phase2_intermediate/operators

python scripts/sft/mix_sft_jsonl.py \
    --inputs data/sft_phase2_intermediate/operators/operator_train.jsonl:0.8 \
             data/sft_phase1_intermediate/phase1_mixed.jsonl:0.2 \
    --output data/sft_phase2_intermediate/phase2_mixed.jsonl --shuffle

# Packing is critical (short episodes; segment isolation for packed blocks)
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase2_intermediate/phase2_mixed.jsonl \
    --output_dir data/sft_phase2_operators \
    --enable_packing --pack_block_size 4096 --pack_by_field "_meta.operator"
```

```bash
python train.py \
    --model_name phase2_operators \
    --config_file configs/sft/phase2_operators.json \
    --dataset_dir data/sft_phase2_operators \
    --init_from_model checkpoints/phase1_format_lock_gold
```

```bash
python scripts/eval/eval_operator.py --model phase2_operators_gold -v
```

```powershell
py.exe scripts/sft/generate_operator_dataset.py --output_dir data/sft_phase2_intermediate/operators
py.exe scripts/sft/mix_sft_jsonl.py --inputs data/sft_phase2_intermediate/operators/operator_train.jsonl:0.8 data/sft_phase1_intermediate/phase1_mixed.jsonl:0.2 --output data/sft_phase2_intermediate/phase2_mixed.jsonl --shuffle
py.exe scripts/sft/prepare_chat_sft.py --input data/sft_phase2_intermediate/phase2_mixed.jsonl --output_dir data/sft_phase2_operators --enable_packing --pack_block_size 4096 --pack_by_field "_meta.operator"
```

---

### 5.2 Appendix — Sequential bridge builds (2.5 → 2.7 → 2.8)

These scripts produce the **intermediate JSONLs** that remix combines. You do **not** have to train every bridge end-to-end if you only need the files for **§5** — but you **must** run the **prepare** steps (or have the artifacts) **before** mixing.

**2.5 — WRAP + anti-echo**

```bash
python scripts/sft/prepare_phase2_5_wrap_antiecho.py \
    --output_dir data/sft_phase2_5_intermediate \
    --replay_file data/sft_phase2_intermediate/operators/operator_train.jsonl \
    --replay_ratio 0.20 \
    --wrap_train_payloads 9000 \
    --wrap_val_payloads 800 \
    --wrap_reps_per_style 2 \
    --echo_max_examples 70000 \
    --echo_anti_ratio 0.40 \
    --echo_contrast_ratio 0.35
```

**2.7 — Rebalance**

```bash
python scripts/sft/prepare_phase2_7_rebalance.py \
    --output_dir data/sft_phase2_7_intermediate \
    --operators_file data/sft_phase2_intermediate/operators/operator_train.jsonl \
    --target_train_size 60000
```

**2.8 — Echo bridge (also emits `phase2_8_val.jsonl` for remix val)**

```bash
python scripts/sft/prepare_phase2_8_echo_rebalance.py \
    --output_dir data/sft_phase2_8_intermediate \
    --replay_file data/sft_phase2_7_intermediate/phase2_7_mixed.jsonl \
    --target_train_size 40000 \
    --val_size 3000 \
    --min_val_per_operator 6
```

**Optional full sequential training** (historical): tokenize each mix with `prepare_chat_sft.py` (`--no_system_prompt`, packing, `--pack_by_field "_meta.operator"`), then `train.py` with `configs/sft/phase2_5_wrap_antiecho.json`, `phase2_7_rebalance.json`, `phase2_8_echo_rebalance.json` and init checkpoints as in each config’s comments. Eval: `eval_phase2_5_wrap_focus.py`, `sft_eval_suite.py`, `eval_phase2_8_bridge.py`.

**2.6** — `prepare_phase2_6_antiecho.py` exists for reproduction; **not** required for remix.

**GOLD selection (bridges and remix):** `gold_selection` in JSON may use `val_loss`, `hybrid`, or `external_gate` (e.g. `eval_phase2_8_bridge.py`, metric `hard_avg_rate`). Logs include gate JSON.

---

## 6. Phase 3: Chat SFT

**Goal:** Natural conversation, bilingual (DE/EN), system prompt adherence, basic think + cite.

### Data Sources

1. **Gold episodes** (existing) -- bilingual conversations
2. **HuggingFace** -- OASST2 (DE+EN), alpaca-gpt4_de, Dolci-Instruct
3. **Augmented** -- paraphrased variants of gold episodes
4. **Replay** -- ~20% Phase 2 remix existing (see `build_phase3_dataset.py` defaults)

**Phase 3 mix vs Phase 2 remix (easy to confuse):**

| What you want | Where to do it |
| --------------- | -------------- |
| **Stronger copy / echo / anti-echo in the *Phase 3* chat dataset** (more replay from Phase 2–style JSONL, without necessarily retraining Phase 2) | **`build_phase3_dataset.py`**: increase `--remix_ratio` (target-fraction of `target_size`), optionally add `--anti_echo_file` + `--anti_echo_ratio`, `--operators_file` + `--operators_ratio`, and lower `--open_chat_cap_ratio` if you keep a fixed `--target_size`. These ratios apply to the **Phase 3 mix total**, not “% of a source file.” |
| **A different `phase2_remix_existing_train.jsonl`** (new blend, new `phase2_remix_existing_gold`) | **`mix_sft_jsonl.py`** + Phase 2 intermediates + **Phase 2 remix train** — a **separate** step. Only required if the replay **file** itself must change; otherwise you can leave it and only turn up **`--remix_ratio`** in Phase 3. |

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

# 3. Generate high-precision instruction episodes (checkable + conflicts + abstention)
python scripts/sft/generate_phase3_precision_sft.py \
    --output data/sft_phase3_intermediate/phase3_precision.jsonl \
    --num_examples 12000

# 4. Augment gold episodes (use your repo’s bilingual gold file)
python scripts/sft/augment_episodes_paraphrase.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_bilingual.jsonl \
    --output data/sft_phase3_intermediate/gold_augmented.jsonl \
    --target_count 1000 --no_model

#    (Alternate path if you keep gold under data/gold_episodes/:)
#    --input data/gold_episodes/gold_bilingual.jsonl

# 5. Build Phase 3 mix with explicit policy:
#    - phase2 remix replay: --remix_ratio is a fraction OF target_size (e.g. 0.28 = 28% of 80k rows from remix file)
#    - optional: --anti_echo_file + --anti_echo_ratio, --operators_file + --operators_ratio (copy/echo/anti-echo maintenance)
#    - grounded ~16%, open chat capped, remainder precision — adjust open_chat_cap_ratio if fixed buckets exceed target_size
python scripts/sft/build_phase3_dataset.py \
    --output data/sft_phase3_intermediate/phase3_mixed.jsonl \
    --target_size 80000 \
    --precision_file data/sft_phase3_intermediate/phase3_precision.jsonl \
    --grounded_file data/sft_phase3_intermediate/rag_chat.jsonl \
    --remix_train_file data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
    --remix_ratio 0.20 \
    --open_chat_files data/sft_hf/oasst2.jsonl data/sft_hf/alpaca_de.jsonl data/sft_phase3_intermediate/gold_augmented.jsonl

# Example — higher Phase 2 replay + maintenance (recreate Phase 3 chat data; no Phase 2 remix rebuild required
# if phase2_remix_existing_train.jsonl is unchanged). Tune ratios so remix + grounded + op + anti + open cap + precision <= 100%.
# python scripts/sft/build_phase3_dataset.py \
#     --output data/sft_phase3_intermediate/phase3_mixed.jsonl \
#     --target_size 80000 \
#     --precision_file data/sft_phase3_intermediate/phase3_precision.jsonl \
#     --grounded_file data/sft_phase3_intermediate/rag_chat.jsonl \
#     --remix_train_file data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl \
#     --remix_ratio 0.28 \
#     --anti_echo_file data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl \
#     --anti_echo_ratio 0.04 \
#     --operators_file data/sft_phase2_intermediate/operators/operator_train.jsonl \
#     --operators_ratio 0.02 \
#     --open_chat_cap_ratio 0.14 \
#     --open_chat_files data/sft_hf/oasst2.jsonl data/sft_hf/alpaca_de.jsonl data/sft_phase3_intermediate/gold_augmented.jsonl

# 6. Audit Phase 3 composition and schema coverage before tokenization
python scripts/sft/audit_phase3_dataset.py \
    --input data/sft_phase3_intermediate/phase3_mixed.jsonl \
    --output data/sft_phase3_intermediate/phase3_mixed.audit.json

# 7. (If needed) Normalize operator replay schema — see “Phase 3 build log” below.
# 8. Tokenize with packing + RAG tags + CHAT system prompt (not the default CONVERSATION prompt)
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase3_intermediate/phase3_mixed.jsonl \
    --output_dir data/sft_phase3_chat \
    --val_split 0.05 \
    --enable_packing --pack_block_size 4096 \
    --enable_rag_tags \
    --schema_validation_mode error \
    --system_prompt "You are MyPT, a helpful assistant. Answer based on the provided context when available."

# 9. (Optional) Tokenize Phase 2 remix **validation** JSONL for an extra eval set (val-only dir; no train split).
#    Run `normalize_phase3_inline_system.py` on the val JSONL first if episodes still have inline `role=system` in messages.
#    Use the **same** `--system_prompt` (and RAG flags) as step 8 so eval matches Phase 3 CHAT training.
python scripts/sft/prepare_chat_sft.py \
    --input data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_val.jsonl \
    --output_dir data/sft_phase3_eval_phase2_remix \
    --val_only \
    --enable_packing --pack_block_size 4096 \
    --enable_rag_tags \
    --schema_validation_mode error \
    --system_prompt "You are MyPT, a helpful assistant. Answer based on the provided context when available."
```

The `--system_prompt` string must match `CHAT_SYSTEM_PROMPT` in `core/system_prompts.py` (training serializes this; per-episode JSONL `"system"` is validated but not used as the system block unless you align generators separately).

`configs/sft/phase3_chat_sft.json` **adds** `eval_sets.phase2_remix_existing` → `data/sft_phase3_eval_phase2_remix` alongside existing eval dirs (Phase 2 full configs may list many `eval_sets`; Phase 3 only adds this remix eval path).

### Phase 3 dataset build log (March 2026)

End-to-end run that produced `data/sft_phase3_chat/` (packed, RAG tags, `CHAT_SYSTEM_PROMPT`). The command block above (steps 1–8) reflects this run, with these **measured outputs** where relevant:

| Step | Output notes |
|------|----------------|
| HF OASST2 | Requested 10k; after tree walk + language filter **~5.5k** episodes written (varies with HF snapshot). |
| HF alpaca_de | Requested 5k; **~4.9k** episodes after filtering. |
| RAG chat | **2000** episodes from `workspace/docs` markdown. |
| Precision | **12000** episodes. |
| Gold augment | **1000** episodes from `mypt_phase3a_gold_bilingual.jsonl` (rule-based `--no_model`). |
| Mix | **80000** episodes → `phase3_mixed.jsonl`. |
| Audit | `audit_phase3_dataset.py` → `phase3_mixed.audit.json` (e.g. schema OK ~92% before normalization — see below). |
| Tokenize | `data/sft_phase3_chat/` with **~10.9M** train tokens (exact counts depend on split/pack). |

**Code fixes applied in-repo for this build**

1. **`scripts/sft/convert_hf_dataset.py` (OASST2)** — Some rows have `rank: null`. `max(..., key=lambda ...)` must not compare `None` to `int`. Use a helper (e.g. `_oasst2_rank_key`) that treats `None` like a worst rank before `max()`.

2. **`scripts/sft/generate_rag_chat_sft.py`** — (a) `sys.path.insert(...)` **before** any `from core...` imports. (b) `generate_context_answer`: define `topic = rng.choice(doc["topics"])` before formatting questions/thinks. (c) `generate_insufficient_context`: ensure `distractor_topic` is defined where used. (d) Lineage metadata: use `args.docs_dir`, not a non-existent `workspace_docs` attribute.

3. **`scripts/sft/generate_phase3_precision_sft.py`** — Move `sys.path.insert(0, ...)` **above** `from core.dataset_lineage ...` so `py scripts/sft/generate_phase3_precision_sft.py` works from the repo root without `PYTHONPATH`. Until that edit is in your tree, Windows users can set `PYTHONPATH` to the project root.

**Schema normalization before tokenization (`--schema_validation_mode error`)**

- **`build_phase3_dataset.py`** samples **phase2 remix existing** replay (and optional operator/anti-echo rows when you pass those files). Those episodes can use **`messages[0].role == "system"`** with **no top-level `"system"`** field. `prepare_chat_sft` schema validation allows only `user` / `assistant` / `assistant_context` in `messages`, so **6400** such rows failed validation in one run.
- **Fix:** `scripts/sft/normalize_phase3_inline_system.py` — hoists the first `messages` entry when `role == "system"` into `episode["system"]` and removes it from `messages`. Run with `--backup` on `phase3_mixed.jsonl` (or on `phase2_remix_existing_val.jsonl` before step 9), then rerun `prepare_chat_sft.py`.

**Validation: do not use `validate_sft_dataset.py` as a hard gate on packed chat SFT**

- Packing uses **`myPT_eot` as the pad token ID**; padding positions are **mask 0**.
- `validate_sft_dataset.py` flags **every** `<myPT_eot>` with mask 0 as **`EOT_NOT_MASKED`**, which **floods false positives** on packed data (pad EOTs are not end-of-turn tokens).
- Prefer **`inspect_sft_dataset.py`** on `data/sft_phase3_chat`, or extend the validator to **ignore EOT where `segment_ids.bin` is 0** (padding), if present.

### Train

```bash
python train.py \
    --model_name phase3_chat \
    --config_file configs/sft/phase3_chat_sft.json \
    --dataset_dir data/sft_phase3_chat \
    --init_from_model checkpoints/phase2_5_wrap_antiecho_gold
```

`phase3_chat_sft.json` sets **`max_iters` ~763** to target **~3.5× epoch coverage** for a ~2.6k-episode train split (effective batch 12 episodes per step). For a **~110k-row** Phase 3 mix (≈3.6k packed train episodes at similar packing density), use **`configs/sft/phase3_chat_sft_110k.json`** (`max_iters` **1051**, separate log path). `train.py` prints episode coverage before training; rescale if your packed `num_train_episodes` differs.

### Success Gate

```bash
python scripts/eval/sft_eval_suite.py --model phase3_chat -v
python scripts/eval/run_regression_gate.py --model phase3_chat --phase 3 -v
# Required before Phase 4:
# - format strict still high
# - echo/anti-echo do not collapse
# - regression basics recover
# - operators do not collapse
# - instruction hierarchy + injection resistance pass
# - abstention/context and strict formatting buckets pass
# - context/citation linkage is present
```

If this gate fails: adjust Phase 3 data mix first (especially strict/checkable vs open chat ratio) before changing LR.

### Phase 3.1 corrective (echo/operators/anti-echo + JSON strict)

Use this when Phase 3 quality is good overall but gate buckets regress on exactness/control.

Windows one-liners:

```powershell
py.exe scripts/sft/generate_phase3_json_sft.py --output data/sft_phase3_intermediate/phase3_json_strict.jsonl --num_examples 6000 --seed 3311 --de_ratio 0.35
py.exe scripts/sft/build_phase3_dataset.py --output data/sft_phase3_intermediate/phase3_1_corrective_mixed.jsonl --target_size 80000 --seed 3331 --precision_file data/sft_phase3_intermediate/phase3_precision.jsonl --grounded_file data/sft_phase3_intermediate/rag_chat.jsonl --remix_train_file data/sft_phase2_remix_existing_intermediate/phase2_remix_existing_train.jsonl --remix_ratio 0.30 --operators_file data/sft_phase2_intermediate/operators/operator_train.jsonl --operators_ratio 0.05 --anti_echo_file data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl --anti_echo_ratio 0.05 --json_file data/sft_phase3_intermediate/phase3_json_strict.jsonl --json_ratio 0.08 --grounded_ratio 0.14 --open_chat_cap_ratio 0.10 --multiturn_cap_ratio 0.22 --open_chat_files data/sft_hf/oasst2.jsonl data/sft_hf/alpaca_de.jsonl data/sft_phase3_intermediate/gold_augmented.jsonl
py.exe scripts/sft/audit_phase3_dataset.py --input data/sft_phase3_intermediate/phase3_1_corrective_mixed.jsonl --output data/sft_phase3_intermediate/phase3_1_corrective_mixed.audit.json
py.exe scripts/sft/normalize_phase3_inline_system.py --input data/sft_phase3_intermediate/phase3_1_corrective_mixed.jsonl --backup
py.exe scripts/sft/prepare_chat_sft.py --input data/sft_phase3_intermediate/phase3_1_corrective_mixed.jsonl --output_dir data/sft_phase3_1_corrective_chat --val_split 0.05 --enable_packing --pack_block_size 4096 --enable_rag_tags --schema_validation_mode error --system_prompt "You are MyPT, a helpful assistant. Answer based on the provided context when available."
py.exe train.py --model_name phase3_1_corrective --config_file configs/sft/phase3_1_corrective.json --dataset_dir data/sft_phase3_1_corrective_chat --init_from_model phase3_chat_110k_gold
py.exe scripts/eval/sft_eval_suite.py --model phase3_1_corrective_gold -v
py.exe scripts/eval/run_regression_gate.py --model phase3_1_corrective_gold --phase 3 -v
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
    --enable_packing --pack_block_size 4096 \
    --enable_rag_tags
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

| Dataset          | Command                                                     | Notes            |
| ---------------- | ----------------------------------------------------------- | ---------------- |
| OASST2 (EN+DE)   | `--dataset OpenAssistant/oasst2 --languages en de`          | Native German!   |
| Alpaca-GPT4 DE   | `--dataset mayflowergmbh/alpaca-gpt4_de`                    | 50K German       |
| Dolci-Instruct   | `--dataset allenai/Dolci-Instruct-SFT --max_examples 10000` | 2.15M, 70+ langs |
| OpenSchnabeltier | `--dataset LeoLM/OpenSchnabeltier`                          | 21.7K German     |

**Phase 4 (Multi-turn):**

| Dataset           | Command                                                      | Notes           |
| ----------------- | ------------------------------------------------------------ | --------------- |
| OASST2 multi-turn | `--dataset OpenAssistant/oasst2 --languages en de`           | Tree depth >= 2 |
| Ultra-Chat DE     | `--dataset mayflowergmbh/ultra-chat_de --max_examples 10000` | 208K multi-turn |

**Phase 5-6 (Tool Calling):**

| Dataset                 | Command                                                              | Notes              |
| ----------------------- | -------------------------------------------------------------------- | ------------------ |
| Dolci Tool-Use          | `--dataset allenai/Dolci-Instruct-SFT-Tool-Use --max_examples 10000` | XML -> JSON mapped |
| German Function Calling | `--dataset flozi00/german-function-calling`                          | 1.33K German       |
| German RAG SFT          | `--dataset avemio/German-RAG-SFT-ShareGPT-HESSIAN-AI`                | 200K+ with RAG     |

All datasets have permissive licenses (ODC-BY, Apache-2.0, CC-BY-SA).

---

## 11. System Prompt Strategy (Loss Mask Optimization)

System prompt tokens are **always masked** (loss=0). In short episodes (Phase 1-2),
a long system prompt dramatically reduces the fraction of supervised tokens per
sequence. We use a three-tier strategy to maximize loss mask %:

| Phases                       | Prompt                    | ~Tokens | Rationale                                            |
| :--------------------------- | :------------------------ | ------: | :--------------------------------------------------- |
| 1-2 (Format Lock, Operators) | `"You are MyPT."`         |       4 | Episodes are ultra-short; every masked token hurts   |
| 3-4 (Chat, Multi-turn)       | `CHAT_SYSTEM_PROMPT`      |   15-20 | Episodes are long enough to absorb it                |
| 5-6 (Toolcall, Agentic)      | `AGENTIC_STANDARD_PROMPT` |     ~80 | Tool episodes are long; need to list available tools |

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

| Training Phase       | Replay Sources      | Ratios                                 |
| :------------------- | :------------------ | :------------------------------------- |
| Phase 2 (Operators)  | Phase 1 format lock | 5% Phase 1 + 95% Phase 2               |
| Phase 3 (Chat SFT)   | Phase 1 + Phase 2   | 3% P1 + 3% P2 + 94% P3                 |
| Phase 4 (Multi-turn) | Phase 1-3           | 2% P1 + 2% P2 + 3% P3 + 93% P4         |
| Phase 5 (Toolcall)   | Phase 1-4           | 2% P1 + 2% P2 + 2% P3 + 2% P4 + 92% P5 |
| Phase 6 (Agentic)    | Phase 1-5           | 1% each P1-P5 + 95% P6                 |

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

| File                         | Phase | What it tests                                    |
| :--------------------------- | :---- | :----------------------------------------------- |
| `phase3_chat_ood.jsonl`      | 3     | RAG-context answering with novel question styles |
| `phase4_multiturn_ood.jsonl` | 4     | Multi-turn follow-up with unseen phrasings       |
| `phase5_toolcall_ood.jsonl`  | 5     | Tool selection from unfamiliar request forms     |
| `phase6_agentic_ood.jsonl`   | 6     | Multi-step tool chaining with novel phrasing     |

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

| Phase                 | neftune_alpha | Rationale                                      |
| :-------------------- | :------------ | :--------------------------------------------- |
| Phase 1 (Format Lock) | 5.0           | Light noise, don't interfere with tag learning |
| Phase 2 (Operators)   | 5.0           | Light noise for abstract patterns              |
| Phase 3 (Chat SFT)    | 10.0          | Moderate, improves response diversity          |
| Phase 4 (Multi-turn)  | 10.0          | Moderate, prevents template memorization       |
| Phase 5 (Toolcall)    | 5.0           | Light, JSON structure must be precise          |
| Phase 6 (Agentic)     | 5.0           | Light, multi-step chains need precision        |

Set to `0.0` to disable (default).

---

## 14. Weighted Loss Masking

Standard SFT uses binary loss masking: 0 (masked) or 1 (trained). Weighted
loss masking assigns higher importance to structural control tokens that
"steer" generation direction. Research shows 39-83% gains on reasoning benchmarks.

### Weight Scheme

| Token Type                             | Weight | Rationale                |
| :------------------------------------- | :----- | :----------------------- |
| System, user, toolresult               | 0.0    | Never train              |
| `<myPT_assistant>` (open)              | 0.0    | Given in prompt          |
| Normal assistant content               | 1.0    | Standard training        |
| `</myPT_assistant>` (close)            | 2.0    | Critical stop signal     |
| `<myPT_eot>`                           | 2.0    | Critical stop signal     |
| `<myPT_think>` / `</myPT_think>`       | 1.5    | Steering tokens          |
| `<myPT_cite>` / `</myPT_cite>`         | 1.5    | Steering tokens          |
| `<myPT_toolcall>` / `</myPT_toolcall>` | 2.0    | Critical action triggers |

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

| Script                             | Phase | What it does                                                    |
| ---------------------------------- | ----- | --------------------------------------------------------------- |
| `generate_format_lock_dataset.py`  | 1     | Combinatorial Q&A (EN+DE), short answers                        |
| `generate_echo_dataset.py`         | 1     | Echo/repeat instructions, anti-echo, gibberish                  |
| `generate_operator_dataset.py`     | 2     | COPY/WRAP/EXTRACT with contrastive design                       |
| `generate_rag_chat_sft.py`         | 3     | RAG episodes with user_context + think + cite (EN+DE)           |
| `generate_phase3_precision_sft.py` | 3     | High-precision checkable tasks + conflicts + abstention         |
| `generate_multiturn_sft.py`        | 4     | Multi-turn conversations: followup, clarification, topic switch |
| `generate_agent_sft.py`            | 5     | Single-step tool-calling (EN+DE, think+cite, NO_TOOL)           |
| `generate_sft_tool_episodes.py`    | 6     | Multi-step agentic tool chains (validated, EN+DE)               |
| `convert_hf_dataset.py`            | 3-6   | Universal HuggingFace dataset converter                         |

### Pipeline (mix, tokenize)

| Script                               | What it does                                                         |
| ------------------------------------ | -------------------------------------------------------------------- |
| `prepare_phase1_format_lock.py`      | Automated Phase 1 pipeline (generate + mix + tokenize)               |
| `prepare_phase2_5_wrap_antiecho.py`  | Build 2.5 bridge intermediate dataset                                |
| `prepare_phase2_6_antiecho.py`       | Build 2.6 anti-echo micro-phase dataset                              |
| `prepare_phase2_7_rebalance.py`      | Build 2.7 rebalance dataset                                          |
| `prepare_phase2_8_echo_rebalance.py` | Build 2.8 replay+specialized bridge dataset                          |
| `mix_sft_jsonl.py`                   | Mix multiple JSONL files with sampling ratios                        |
| `build_phase3_dataset.py`            | Policy-driven Phase 3 mixer (maintenance + strict tasks + turn caps) |
| `normalize_phase3_inline_system.py`  | Hoist `messages[0]` `role=system` into `episode["system"]` before tokenize (Phase 3 operator replay) |
| `prepare_chat_sft.py`                | Tokenize chat JSONL to binary (Phase 1-4)                            |
| `prepare_tool_sft.py`                | Tokenize tool JSONL to binary (Phase 5-6)                            |

### Quality (inspect, validate, deduplicate)

| Script                           | What it does                                           |
| -------------------------------- | ------------------------------------------------------ |
| `inspect_sft_dataset.py`         | Show samples, stats from tokenized dataset             |
| `validate_sft_dataset.py`        | Validate mask/tag rules; **not packing-aware** (EOT pad false positives on packed data; see Phase 3 build log) |
| `validate_sft_episode_masks.py`  | Check mask alignment                                   |
| `verify_loss_mask_direction.py`  | Verify mask direction (assistant=1)                    |
| `verify_mask_alignment.py`       | Token-level mask alignment check                       |
| `deduplicate_episodes.py`        | Remove duplicate episodes                              |
| `deduplicate_by_user_message.py` | Deduplicate by user message content                    |
| `analyze_episode_diversity.py`   | Analyze diversity metrics                              |
| `audit_phase3_dataset.py`        | Phase 3 schema/composition audit (global + per-source) |

### Augmentation

| Script                           | What it does                               |
| -------------------------------- | ------------------------------------------ |
| `augment_episodes_paraphrase.py` | Rule-based paraphrasing to expand datasets |
| `diversify_user_messages.py`     | Vary user message templates                |

### Evaluation

| Script                                 | What it does                                          |
| -------------------------------------- | ----------------------------------------------------- |
| `scripts/eval/sft_eval_suite.py`       | Full evaluation (format, echo, anti-echo, regression) |
| `scripts/eval/eval_operator.py`        | Operator exact-match evaluation                       |
| `scripts/eval/eval_phase2_8_bridge.py` | Phase 2.8 ABCE hard gate (D report-only)              |

### Translation (DE/EN)

| Script                                            | What it does                 |
| ------------------------------------------------- | ---------------------------- |
| `scripts/translation/extract_for_translation.py`  | Extract translatable strings |
| `scripts/translation/translate_deepl.py`          | Translate via DeepL API      |
| `scripts/translation/recombine_translations.py`   | Recombine translated strings |
| `scripts/translation/merge_bilingual_episodes.py` | Merge EN + DE episodes       |

---

## 16. Configs Reference

All SFT configs include the LLaMA-2 architecture fields. Key configs by phase:

| Phase | Config File                               | LR     | Iters | Block |
| ----- | ----------------------------------------- | ------ | ----- | ----- |
| 1     | `configs/sft/phase1_format_lock.json`     | 7e-5   | 2000  | 4096  |
| 2     | `configs/sft/phase2_operators.json`       | 3e-5   | 1200  | 4096  |
| 3     | `configs/sft/phase3_chat_sft.json`        | 3e-5   | 763   | 4096  |
| 3b    | `configs/sft/phase3_chat_sft_110k.json`   | 3e-5   | 1051  | 4096  |
| 4     | `configs/sft/phase4_multiturn.json`       | 2.5e-5 | 3000  | 4096  |
| 5     | `configs/sft/phase5_simple_toolcall.json` | 2e-5   | 3000  | 4096  |
| 6     | `configs/sft/phase6_agentic_rag.json`     | 1.5e-5 | 4000  | 4096  |

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
