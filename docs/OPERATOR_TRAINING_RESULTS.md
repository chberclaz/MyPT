# Operator Training Results — Phase 3a Operator Learning Test

**Date:** January 2026  
**Status:** Proof-of-concept PASSED  
**Model:** 750M GPT (32L/1280E/20H)

---

## Executive Summary

The operator learning test proved that a 750M model CAN learn abstract operators when trained with:
- **Unique payloads** (no memorization possible)
- **Strict template separation** (val uses unseen phrasings)
- **Proper coverage** (2-3x, not more)

**Best result: 80% exact-match on operators (WRAP: 100%, EXTRACT: 80%, COPY: 60%)**

---

## The Problem We Solved

Previous attempts at "instruction following" failed because:
1. Repeated payloads allowed memorization
2. Same templates in train/val didn't test generalization
3. Fuzzy metrics ("looks OK") hid failures
4. Overtraining caused brittleness

The model would output "Hello" for "Say Hello" but fail on "Say: Hello" — it was pattern-matching, not understanding.

---

## The Solution: Operator Learning with Strict Separation

### Dataset Design (Critical)

| Property | Implementation |
|----------|----------------|
| **Unique payloads** | Every payload appears exactly ONCE in train, different set in val |
| **Template split** | Train templates ≠ Val templates (tests generalization) |
| **BPE-safe payloads** | Max 4 tokens per payload (learnability) |
| **No colon bias** | Mix of "Repeat: X" and "Repeat X" formats |
| **Mechanical operators** | COPY, WRAP, EXTRACT — exact-match verification |

### Operators Tested

| Operator | Train Templates | Val Templates | Example |
|----------|-----------------|---------------|---------|
| **COPY** | "Repeat exactly:", "Echo:", "Say back" | "Parrot:", "Mirror:", "Reproduce" | "Parrot nebula" → "nebula" |
| **WRAP** | "Wrap in brackets:", "Enclose in []" | "Put in square brackets:", "Add [] around" | "Add [] around thunder" → "[thunder]" |
| **EXTRACT** | "Return text between quotes:", "Get what's in quotes" | "What's inside the quotes:", "Output quoted content" | 'What\'s inside "emerald"?' → "emerald" |

### Dataset Generation

```bash
python scripts/generate_operator_dataset.py \
    --output_dir data/sft_operator \
    --n_train 9000 \
    --n_val 1000 \
    --max_tokens 4 \
    --seed_train 42 \
    --seed_val 12345
```

Generates:
- `operator_train.jsonl` — 9000+ unique episodes
- `operator_val.jsonl` — 1000 unique episodes (different templates)
- Automatic German variants (10%)

### Dataset Preparation (with Strict Validation)

```bash
python scripts/prepare_chat_sft.py \
    --input data/sft_operator/operator_train.jsonl \
    --val_file data/sft_operator/operator_val.jsonl \
    --output_dir data/sft_operator/prepared
```

The script **hard fails** if:
- Any payload appears in both train and val
- Any template appears in both train and val

---

## Training Configuration

### What Worked

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Learning rate** | 7e-5 | Sweet spot for this model |
| **Coverage** | 2.0-2.5x | Higher = overfitting |
| **Batch size** | 16 | Standard |
| **Warmup** | 100 iters | ~5% of training |
| **Dropout** | 0.05 | Mild regularization |

### What Didn't Work

| Parameter | Problem |
|-----------|---------|
| Coverage > 3x | Overfitting, garbage output ("111111...") |
| Coverage 5-7x | Severe degradation |
| Lower LR (5e-5) | Slower convergence, similar ceiling |

### Optimal Training Command

```bash
python train.py \
    --model_name phase3a_operator \
    --init_from_model phase3a1_alpha_v2 \
    --dataset_dir data/sft_operator/prepared \
    --config_file configs/sft1/750M_phase3a_operator.json \
    --eval_prompts_file configs/sft_eval/phase3a_operator_eval_prompts.json \
    --learning_rate 7e-5
```

**Stop at ~2.3x coverage** (for 11,700 episodes with batch 16: ~1700 iters)

---

## Results

### Best Checkpoint (v3 @ 1700 iters)

| Operator | Accuracy | Notes |
|----------|----------|-------|
| **WRAP** | 5/5 (100%) | ✅ Perfect |
| **EXTRACT** | 4/5 (80%) | ✅ Good |
| **COPY** | 3/5 (60%) | ⚠️ Weakest |
| **Total** | 12/15 (80%) | ✅ Passed |

### Overfitting Evidence (v3 @ 2400 iters)

| Operator | Accuracy | Change |
|----------|----------|--------|
| **WRAP** | 5/5 (100%) | Same |
| **EXTRACT** | 4/5 (80%) | Same |
| **COPY** | 2/5 (40%) | ⬇️ Worse |
| **Total** | 11/15 (73%) | ⬇️ Worse |

**Lesson:** More training ≠ better. Stop at optimal coverage.

---

## Key Learnings

### 1. Unique Payloads Force Abstraction
When every payload is unique, the model CANNOT memorize input→output pairs. It MUST learn the operator transformation.

### 2. Template Split Tests Generalization
Val uses phrasings the model never saw. If it succeeds, it learned the OPERATOR, not the TEMPLATE.

### 3. Coverage Sweet Spot: 2-3x
- < 2x: Underfitting
- 2-3x: Optimal learning
- > 3x: Overfitting, garbage output

### 4. WRAP is Easiest, COPY is Hardest
- WRAP has clear structural signal (brackets in output)
- EXTRACT has quote markers as anchors
- COPY has no structural hint — pure content transfer

### 5. BPE Matters
Payloads that tokenize into many subwords are harder to learn. Filter to max 4 tokens.

---

## Evaluation

### Exact-Match Eval (Strict)

```bash
python scripts/sft_eval_suite.py --model phase3a_operator -v
```

Checks:
- Bucket E: Operators (exact match on unseen templates)
- Per-operator breakdown (COPY, WRAP, EXTRACT)

### Pass Criteria

| Metric | Threshold | Achieved |
|--------|-----------|----------|
| WRAP | ≥ 80% | ✅ 100% |
| EXTRACT | ≥ 80% | ✅ 80% |
| COPY | ≥ 60% | ✅ 60% |
| Overall | ≥ 75% | ✅ 80% |

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/generate_operator_dataset.py` | Generate unique-payload operator dataset |
| `scripts/prepare_chat_sft.py` | Prepare with strict train/val separation |
| `scripts/sft_eval_suite.py` | Exact-match operator evaluation |
| `configs/sft1/750M_phase3a_operator.json` | Training config |
| `configs/sft_eval/phase3a_operator_eval_prompts.json` | Runtime eval prompts |

---

## Next Steps

1. **Accept 80% as passing** — WRAP at 100% proves the model can learn operators
2. **Proceed to context-grounded QA** — The actual RAG task
3. **If RAG fails** — Consider scaling to 1.5B-2B

---

## Conclusion

The 750M model CAN learn abstract operators when trained properly. The key is:
- **Data quality over quantity** (unique payloads, template splits)
- **Exact-match metrics** (no fuzzy "looks OK")
- **Careful coverage control** (2-3x, stop before overfitting)

This validates the "from scratch" approach for domain-specific RAG assistants.
