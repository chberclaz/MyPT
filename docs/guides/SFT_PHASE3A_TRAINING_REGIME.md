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

### Phase 3a-1
```bash
python scripts/generate_format_lock_dataset.py
python scripts/prepare_chat_sft.py \
  --input data/sft_format_lock/mypt_format_lock_v1.jsonl \
  --output data/sft_format_lock_prepared --val_split 0.1
python train.py --model_name phase3a1_format_lock \
  --init_from_model domain_v5 \
  --config_file configs/sft1/750M_phase3a1_format_lock.json \
  --dataset_dir data/sft_format_lock_prepared
```

### Phase 3a-2
```bash
python scripts/generate_run2_minimal_qa.py
python scripts/prepare_chat_sft.py \
  --input data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl \
  --output data/sft_run2_prepared --val_split 0.1
python train.py --model_name phase3a2_minimal_qa \
  --init_from_model phase3a1_format_lock \
  --config_file configs/sft1/750M_phase3a2_minimal_qa.json \
  --dataset_dir data/sft_run2_prepared
```

---

## Success Validation

After each stage, test with:
```bash
python generate.py --model phase3aX_name \
  --prompt "<myPT_system>You are MyPT.<myPT_eot>\n<myPT_user>YOUR_TEST<myPT_eot>\n<myPT_assistant>"
```

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
| `configs/sft1/750M_phase3a1_format_lock.json` | 3a-1 config |
| `configs/sft1/750M_phase3a2_minimal_qa.json` | 3a-2 config |
| `docs/sft/SFT_LOSS_MASKING.md` | Loss masking explanation |
| `docs/sft/EPISODE_INDEXED_SFT.md` | Episode format details |

---

_Last updated: January 2026_
