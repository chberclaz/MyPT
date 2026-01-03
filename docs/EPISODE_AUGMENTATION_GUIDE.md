# Episode Augmentation Guide

This guide covers how to augment your gold conversation episodes through paraphrasing to expand your SFT training dataset from 560 to 1200+ episodes.

---

## Why Augment?

**Your situation**: 560 gold episodes (280 EN + 280 DE)  
**Target**: 1200 episodes  
**Method**: Paraphrasing with quality control

**Benefits over simple upsampling**:
- ✅ **More diversity**: Each episode has unique surface form
- ✅ **Reduced memorization**: Model learns patterns, not exact sequences
- ✅ **Better generalization**: Varied phrasings improve robustness
- ✅ **Maintained quality**: Human review ensures gold standard

---

## Augmentation Strategy

### Two Modes Available

#### 1. **Model-Assisted Paraphrasing** (Recommended)

Uses your trained 750M model to generate natural paraphrases:

```bash
python scripts/augment_episodes_paraphrase.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_augmented \
  --model checkpoints/750M_gold_2.22 \
  --target_count 1200 \
  --review_mode
```

**How it works**:
- ✅ Keeps system prompts unchanged (critical for behavior)
- ✅ Lightly varies user messages (simple paraphrasing)
- ✅ Uses model to paraphrase assistant responses (natural variation)
- ✅ Quality filters: rejects paraphrases that are too short/long

**Time**: ~30-60 minutes for 640 new episodes

#### 2. **Rule-Based Paraphrasing** (Faster, Lower Quality)

Simple template-based paraphrasing:

```bash
python scripts/augment_episodes_paraphrase.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_augmented \
  --target_count 1200 \
  --no_model \
  --review_mode
```

**Time**: ~5 minutes for 640 new episodes  
**Quality**: Lower - use only if you can't load the model

---

## Step-by-Step Process

### Step 1: Prepare Your Gold Dataset

Ensure your gold episodes are in JSON format:

```json
{
  "id": "ep_0001",
  "language": "en",
  "category": "question_answer",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

### Step 2: Run Augmentation

**For balanced EN/DE augmentation:**

```bash
python scripts/augment_episodes_paraphrase.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_augmented \
  --model checkpoints/750M_gold_2.22 \
  --target_count 1200 \
  --lang_filter both \
  --temperature 0.7 \
  --review_mode
```

**Parameters**:
- `--target_count 1200`: Final dataset size (560 original + 640 augmented)
- `--lang_filter both`: Augment both EN and DE equally
- `--temperature 0.7`: Controls generation diversity (0.5-0.9 recommended)
- `--review_mode`: Marks augmented episodes for human review

**Output**:
```
data/sft_conversation_goldset_augmented/
├── ep_0001.json                    # Original episodes (560)
├── ep_0002.json
├── ...
├── aug_00000.json                  # Augmented episodes (640)
├── aug_00001.json                  # Marked with "needs_review": true
├── ...
└── augmentation_metadata.json      # Statistics
```

### Step 3: Review Augmented Episodes

**Critical**: Review a sample of augmented episodes before training!

```python
import json

# Load an augmented episode
with open('data/sft_conversation_goldset_augmented/aug_00000.json') as f:
    episode = json.load(f)

# Check if it needs review
if episode.get('needs_review'):
    print("Source:", episode.get('source_id'))
    print("\nMessages:")
    for msg in episode['messages']:
        print(f"  {msg['role']}: {msg['content']}")
```

**What to check**:
- ✅ **Semantic preservation**: Meaning unchanged?
- ✅ **Naturalness**: Sounds human-like?
- ✅ **Grammar**: No errors introduced?
- ✅ **Consistency**: Conversation flows logically?

**Quality control**:
- Review **at least 50 augmented episodes** (random sample)
- If quality is poor, adjust temperature or use rule-based mode
- Delete bad augmentations, keep good ones

### Step 4: Prepare for Training

```bash
python scripts/prepare_chat_sft.py \
  --input data/sft_conversation_goldset_augmented \
  --output data/sft_conversation_goldset_augmented_prepared \
  --tokenization gpt2 \
  --val_split 0.1 \
  --verbose
```

**Expected output**:
```
Dataset preparation complete!
  Train episodes: ~1080
  Val episodes:   ~120
  Total tokens:   ~[depends on episode length]
```

### Step 5: Train with Augmented Dataset

Update your config to account for larger dataset:

```json
{
  "name": "GPT-750M-1024-Chat-SFT-Phase3a-Augmented",
  "max_iters": 8000,
  "eval_interval": 500,
  "comment": "560 original + 640 augmented = 1200 episodes total"
}
```

Train:

```bash
python train.py \
  --model 750M_gold_2.22 \
  --config configs/sft1/750M_1024_chat_sft_phase3a_augmented.json \
  --dataset data/sft_conversation_goldset_augmented_prepared \
  --output checkpoints/750M_phase3a_chat_sft_augmented
```

---

## Expected Results

### Dataset Growth

| Metric | Before | After | Growth |
|--------|--------|-------|--------|
| **Total episodes** | 560 | 1200 | +114% |
| **EN episodes** | 280 | ~600 | +114% |
| **DE episodes** | 280 | ~600 | +114% |
| **Diversity** | 1x | ~2x | Paraphrased variations |

### Training Impact

**With augmentation (1200 episodes)**:
```
Batch size: 10
Batches/epoch: 120
8000 iterations ≈ 66 epochs

Expected: Less overfitting, better generalization
```

**Without augmentation (560 episodes)**:
```
Batch size: 10
Batches/epoch: 56
5000 iterations ≈ 89 epochs

Risk: Higher memorization, potential overfitting
```

---

## Quality Control Strategies

### Strategy 1: Graduated Review

1. **Sample 10%**: Review 64 random augmented episodes
2. **Calculate quality score**: Good / Total
3. **Decision**:
   - Score > 90% → Accept all, proceed to training
   - Score 70-90% → Review all, fix issues
   - Score < 70% → Regenerate with different temperature

### Strategy 2: Category-Based Review

Focus review on critical categories:

```python
# Review priorities
high_priority = ['tool_calling', 'error_handling', 'reasoning']
medium_priority = ['question_answer', 'multi_turn']
low_priority = ['greeting', 'simple_qa']

# Review 100% of high-priority augmentations
# Review 50% of medium-priority
# Review 20% of low-priority
```

### Strategy 3: Automated Quality Checks

Add checks to the script:

```python
def quality_check(original, paraphrased):
    """Automated quality metrics."""
    # Length similarity
    len_ratio = len(paraphrased) / len(original)
    if not (0.5 <= len_ratio <= 2.0):
        return False
    
    # Semantic keywords preserved
    original_keywords = extract_keywords(original)
    para_keywords = extract_keywords(paraphrased)
    overlap = len(original_keywords & para_keywords) / len(original_keywords)
    if overlap < 0.5:  # At least 50% keyword overlap
        return False
    
    # No obvious errors
    if has_grammar_errors(paraphrased):
        return False
    
    return True
```

---

## Advanced: Selective Augmentation

Augment only specific categories that need more data:

```bash
# Example: Augment only reasoning episodes
python scripts/augment_episodes_paraphrase.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_reasoning_aug \
  --model checkpoints/750M_gold_2.22 \
  --category reasoning \
  --target_multiplier 3.0
```

---

## Troubleshooting

### Issue 1: Model Generates Nonsense

**Symptoms**: Paraphrased responses are incoherent

**Solutions**:
1. Lower temperature: `--temperature 0.5`
2. Use rule-based mode: `--no_model`
3. Check model quality (generate a few examples manually)

### Issue 2: Paraphrases Too Similar

**Symptoms**: Augmented episodes barely differ from originals

**Solutions**:
1. Increase temperature: `--temperature 0.9`
2. Add more variation to templates in script
3. Use multiple augmentation passes with different seeds

### Issue 3: Quality Inconsistent

**Symptoms**: Some augmentations great, others terrible

**Solutions**:
1. Enable review mode: `--review_mode`
2. Implement quality scoring
3. Generate 2x needed, then filter to best 50%

---

## Comparison: Augmentation vs Upsampling

| Aspect | Augmentation (1200) | Upsampling (560 × 2) | No Change (560) |
|--------|---------------------|---------------------|------------------|
| **Diversity** | High | Low | Medium |
| **Memorization risk** | Low | High | Medium |
| **Setup time** | 1-2 hours | 5 minutes | 0 minutes |
| **Quality control** | Required | None | N/A |
| **Training epochs** | ~66 @ 8K iters | ~178 @ 10K iters | ~89 @ 5K iters |
| **Expected quality** | Best | Medium | Good |

---

## Recommended Path

For your 560-episode bilingual dataset:

### Option A: Conservative (Recommended First Try)

```bash
# No augmentation, extended training
python train.py \
  --config configs/sft1/750M_1024_chat_sft_phase3a_extended.json \
  --max_iters 10000
```

**If overfitting occurs** → Try Option B

### Option B: Moderate Augmentation

```bash
# Augment to 900 episodes (560 + 340)
python scripts/augment_episodes_paraphrase.py --target_count 900

# Train with standard config
python train.py \
  --config configs/sft1/750M_1024_chat_sft_phase3a.json \
  --max_iters 6000
```

### Option C: Full Augmentation

```bash
# Augment to 1200 episodes (560 + 640)
python scripts/augment_episodes_paraphrase.py --target_count 1200

# Train with augmented config
python train.py \
  --config configs/sft1/750M_1024_chat_sft_phase3a_augmented.json \
  --max_iters 8000
```

---

## Summary

✅ **Script provided**: `augment_episodes_paraphrase.py`  
✅ **Target**: 560 → 1200 episodes  
✅ **Method**: Model-assisted paraphrasing  
✅ **Quality**: Human review recommended  
✅ **Time**: ~1-2 hours total (generation + review)  
✅ **Benefit**: Better generalization, less overfitting  

**Next step**: Run the script and review output quality!

