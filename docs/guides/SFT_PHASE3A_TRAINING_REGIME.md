# Phase 3a Training Regime: Chat SFT

## Overview

Phase 3a transforms a domain-adapted language model into a conversational assistant. Unlike Phase 1-2 (which trained on raw text), Phase 3a uses **sequential episode-indexed data** with **loss masking** to teach specific behaviors.

**Key Insight**: Small models (750M parameters) cannot learn everything at once. We use **curriculum learning** - teaching simple skills first, then building complexity.

---

## Training Philosophy

### Why Multi-Stage?

Early attempts at Phase 3a failed because we tried to teach everything simultaneously:
- Format (tags, stopping)
- Content (facts, reasoning)  
- Style (concise, verbose)
- Languages (English, German)

A 750M model seeing a diverse dataset gets **inconsistent signals** - each pattern appears too few times to learn reliably.

**Solution**: Break training into focused stages, each with a clear behavioral target.

### The Curriculum

| Stage | Name | Goal | Dataset Size |
|-------|------|------|--------------|
| 3a-1 | Format Lock | Stop at closing tag, respect roles | ~500 episodes |
| 3a-2 | Minimal Q&A | 1-sentence answers, constraints | ~1000 episodes |
| 3a-3 | Task Execution | Multi-step tasks, summarization | ~1500 episodes |
| 3a-4 | Structured Output | JSON schemas, tool-readiness | ~800 episodes |

Each stage **inherits from the previous** - train 3a-2 from the 3a-1 checkpoint.

---

## Stage Details

### Phase 3a-1: Format Lock

**Goal**: Model always produces correct MyPT tags and stops when it should.

**Dataset**: Ultra-short responses (1-5 tokens), single-turn, English only.

**Success Criteria**:
- Outputs closing tag and stops
- No tag leakage
- Val loss stabilizes ~3.0-3.5

**Config**: `configs/sft1/750M_phase3a1_format_lock.json`

---

### Phase 3a-2: Minimal Q&A Obedience  

**Goal**: Concise answers, length constraints, "I don't know" responses.

**Dataset**: 1-2 sentence responses, explicit constraints, bilingual.

**Success Criteria**:
- Respects length constraints
- Says "I don't know" appropriately
- No format regression

**Config**: `configs/sft1/750M_phase3a2_minimal_qa.json`
**Base model**: `phase3a1_format_lock`
**Replay**: 20% of 3a-1 data

---

### Phase 3a-3: Task Execution + Summarization

**Goal**: Multi-step instructions, information compression.

**Dataset**: Extract-then-summarize, schema outputs, query-focused summaries.

**Replay**: 20% Run-2 + 10% Run-1

---

### Phase 3a-4: Structured Output + Tool Readiness

**Goal**: Parse-safe outputs, tool-call muscle memory.

**Dataset**: JSON schemas, fake tool calls, decide tool vs direct answer.

---

## Replay Strategy

Each run includes replay from previous stages to prevent catastrophic forgetting:

| Run | Primary Data | Replay |
|-----|-------------|--------|
| 3a-1 | Format lock (100%) | None |
| 3a-2 | Minimal Q&A (80%) | 3a-1 (20%) |
| 3a-3 | Tasks + Summary (70%) | 3a-2 (20%) + 3a-1 (10%) |
| 3a-4 | Structured (70%) | 3a-3 (20%) + gold (10%) |

**Critical Rule**: Only replay sequential SFT data. Never mix in Phase 1-2 random text.

---

## Execution Commands

### Phase 3a-1: Format Lock
```bash
# 1. Generate dataset
python scripts/generate_format_lock_dataset.py

# 2. Prepare (tokenize)
python scripts/prepare_chat_sft.py \
  --input data/sft_format_lock/mypt_format_lock_v1.jsonl \
  --output_dir data/sft_format_lock_prepared \
  --val_split 0.1

# 3. Train
python train.py --model_name phase3a1_format_lock \
  --init_from_model domain_v5_sft_ready \
  --config_file configs/sft1/750M_phase3a1_format_lock.json \
  --dataset_dir data/sft_format_lock_prepared
```

### Phase 3a-2: Minimal Q&A (with Run 1 Replay)
```bash
# 1. Generate Run 2 dataset
python scripts/generate_run2_minimal_qa.py

# 2. (Optional) Translate to German for bilingual training
python scripts/extract_for_translation.py \
  --input data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl \
  --output_dir data/temp
python scripts/translate_deepl.py
python scripts/recombine_translations.py \
  --original data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl \
  --user_translated data/temp/user_messages_de.txt \
  --assistant_translated data/temp/assistant_messages_de.txt \
  --output data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1_de.jsonl
cat data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl \
    data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1_de.jsonl \
    > data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1_bilingual.jsonl

# 3. Mix with 20% Run 1 replay
python scripts/mix_sft_jsonl.py \
  --inputs data/sft_format_lock/mypt_format_lock_v1.jsonl:0.2 \
           data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1_bilingual.jsonl:1.0 \
  --output data/sft_run2_mixed/mypt_run2_with_replay.jsonl \
  --shuffle

# 4. Prepare mixed dataset
python scripts/prepare_chat_sft.py \
  --input data/sft_run2_mixed/mypt_run2_with_replay.jsonl \
  --output_dir data/sft_run2_mixed_prepared \
  --val_split 0.1

# 5. Train (config includes format_lock as additional eval set)
python train.py --model_name phase3a2_minimal_qa \
  --init_from_model phase3a1_format_lock \
  --config_file configs/sft1/750M_phase3a2_minimal_qa.json \
  --dataset_dir data/sft_run2_mixed_prepared
```

**Note**: The config includes `"eval_sets": {"format_lock": "data/sft_format_lock_prepared"}` to monitor Run 1 skills during Run 2 training.

---

## Success Validation

After each stage, test with:
```bash
python generate.py --model phase3aX_name \
  --prompt "<myPT_system>You are MyPT, a helpful assistant.</myPT_system>\n<myPT_user>YOUR_TEST</myPT_user>\n<myPT_assistant>"
```

**Note**: Tags use opening/closing pairs (`<myPT_system>...</myPT_system>`), NOT `<myPT_eot>` after each message. The `<myPT_eot>` only appears at the very end of a complete conversation.

**3a-1 Pass**: Model stops after 1-3 tokens, outputs closing tag.
**3a-2 Pass**: Answers in 1-2 sentences, respects "10 words" constraint.
**3a-3 Pass**: Produces KEY_FACTS + SUMMARY structure.
**3a-4 Pass**: Outputs valid JSON, decides tool vs direct correctly.

---

## Phase 3b Outlook: Tool Calls

Phase 3b builds on 3a-4 by adding **real tool execution**:

### What 3a Provides to 3b
- Tag discipline (from 3a-1)
- Output constraints (from 3a-2)
- Multi-step execution (from 3a-3)  
- Schema discipline (from 3a-4)
- Tool/no-tool decision (from 3a-4)

### What 3b Adds
- Real tool call syntax with `<myPT_tool>` tags
- Tool result handling
- Multi-turn tool chains
- Error recovery

### 3b Dataset Structure
```
User: Search for Python tutorials
Assistant: <myPT_tool>{"name": "web_search", "args": {"query": "Python tutorials"}}</myPT_tool>
Tool: {"results": [{"title": "...", "url": "..."}]}
Assistant: I found several Python tutorials...
```

### Transition Strategy
1. Complete 3a-4 with fake tool calls
2. Verify JSON output is 99%+ parse-safe
3. Create 3b dataset with real tool patterns
4. Train 3b from 3a-4 checkpoint with tool replay

---

## Common Issues

### High Initial Loss (>10)
- Check dataset format matches expected schema
- Verify special tokens are in vocabulary
- Ensure loss masking is enabled

### Format Regression
- Increase replay percentage
- Check that base model is correct stage (not domain_v5)

### Runaway Generation
- 3a-1 not converged - train longer or check dataset
- Closing tags not in vocab

### "House Style" Lock-in
- Dataset too homogeneous
- Add constraint variations (same Q, different lengths)

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/generate_format_lock_dataset.py` | Generate 3a-1 data |
| `scripts/generate_run2_minimal_qa.py` | Generate 3a-2 data |
| `scripts/mix_sft_jsonl.py` | Mix JSONL datasets with sampling ratios (for replay) |
| `scripts/extract_for_translation.py` | Extract messages for DeepL translation |
| `scripts/translate_deepl.py` | Translate EN→DE via DeepL API |
| `scripts/recombine_translations.py` | Recombine translations into JSONL |
| `configs/sft1/750M_phase3a1_format_lock.json` | 3a-1 config |
| `configs/sft1/750M_phase3a2_minimal_qa.json` | 3a-2 config (includes Run 1 eval set) |
| `docs/sft/SFT_LOSS_MASKING.md` | Loss masking explanation |
| `docs/sft/EPISODE_INDEXED_SFT.md` | Episode format details |

---

_Last updated: January 2026_
