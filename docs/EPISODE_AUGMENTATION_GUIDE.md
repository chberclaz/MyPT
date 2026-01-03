# Episode Augmentation Guide

Expand your gold conversation dataset through paraphrasing to improve training diversity and reduce memorization.

---

## Overview

| Metric       | Your Dataset | After Augmentation      |
| ------------ | ------------ | ----------------------- |
| **Episodes** | ~255         | 600+                    |
| **Format**   | JSONL        | JSONL                   |
| **Method**   | -            | Rule-based paraphrasing |

**Benefits**:

- More surface-level diversity → less memorization
- Varied phrasings → better generalization
- Same semantic content → maintained quality

---

## Quick Start

```bash
python scripts/augment_episodes_paraphrase.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_en_verified.jsonl \
    --output data/sft_augmented.jsonl \
    --target_count 600 \
    --review_mode
```

Output: Single JSONL file with original + augmented episodes.

---

## Input Format

The script expects JSONL files where each line is a valid JSON episode:

```json
{
  "system": "You are MyPT, an offline, privacy-first assistant...",
  "context": "episode_id: 0001",
  "messages": [
    {
      "role": "user",
      "content": "Explain idempotency like I'm a senior developer."
    },
    {
      "role": "assistant",
      "content": "idempotency in practice:\n\n- **What it is:** ..."
    }
  ]
}
```

**Required fields**:

- `system`: System prompt (preserved unchanged during augmentation)
- `messages`: Array of user/assistant turns

**Optional fields**:

- `context`: Metadata (episode ID, category, etc.)
- `language`: Language code (en/de)

---

## Command Reference

```bash
python scripts/augment_episodes_paraphrase.py \
    --input INPUT_PATH \
    --output OUTPUT_PATH \
    [--target_count N] \
    [--lang_filter LANG] \
    [--temperature T] \
    [--model CHECKPOINT] \
    [--no_model] \
    [--review_mode]
```

### Arguments

| Argument         | Required | Default | Description                                          |
| ---------------- | -------- | ------- | ---------------------------------------------------- |
| `--input`        | ✅       | -       | Input JSONL file or directory                        |
| `--output`       | ✅       | -       | Output JSONL file (auto-appends `.jsonl` if missing) |
| `--target_count` | -        | 1200    | Target total episode count                           |
| `--lang_filter`  | -        | both    | Filter: `en`, `de`, or `both`                        |
| `--temperature`  | -        | 0.7     | Generation temperature (0.5-0.9 recommended)         |
| `--model`        | -        | None    | Model checkpoint for model-assisted paraphrasing     |
| `--no_model`     | -        | False   | Use rule-based paraphrasing only                     |
| `--review_mode`  | -        | False   | Mark augmented episodes with `needs_review: true`    |

---

## Paraphrasing Modes

### 1. Rule-Based (Default)

Simple template-based paraphrasing. Fast and predictable.

```bash
python scripts/augment_episodes_paraphrase.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_en_verified.jsonl \
    --output data/sft_augmented.jsonl \
    --target_count 600
```

**What it does**:

- Adds filler words, rephrases sentences
- Swaps synonyms where safe
- Preserves meaning and structure

**Time**: ~1-2 minutes for 300 new episodes

### 2. Model-Assisted (Higher Quality)

Uses your trained model to generate more natural paraphrases.

```bash
python scripts/augment_episodes_paraphrase.py \
    --input data/sft_conversation_goldset/mypt_phase3a_gold_en_verified.jsonl \
    --output data/sft_augmented.jsonl \
    --model checkpoints/750M_gold_2.22 \
    --target_count 600 \
    --temperature 0.7
```

**What it does**:

- Model rewrites assistant responses
- More natural variation
- Quality filters reject bad generations

**Time**: ~30-60 minutes for 300 new episodes

---

## Output Format

The script outputs a single JSONL file:

```json
{"system": "...", "context": "episode_id: 0001", "messages": [...]}
{"system": "...", "context": "episode_id: 0002", "messages": [...]}
{"system": "...", "id": "aug_0001", "source_id": "0001", "augmented": true, "messages": [...]}
{"system": "...", "id": "aug_0002", "source_id": "0002", "augmented": true, "messages": [...]}
```

**Augmented episodes include**:

- `id`: Unique augmented ID (e.g., `aug_0001`)
- `source_id`: Original episode ID
- `augmented`: `true` flag
- `needs_review`: `true` if `--review_mode` was used

A metadata file `*_metadata.json` is also created with statistics.

---

## Workflow

### Step 1: Prepare Gold Dataset

Ensure your episodes are in JSONL format and deduplicated:

```bash
# Deduplicate first
python scripts/deduplicate_by_user_message.py \
    --input data/raw_episodes.jsonl \
    --output data/gold_unique.jsonl
```

### Step 2: Augment

```bash
python scripts/augment_episodes_paraphrase.py \
    --input data/gold_unique.jsonl \
    --output data/gold_augmented.jsonl \
    --target_count 600 \
    --review_mode
```

### Step 3: Review (Optional but Recommended)

Check a sample of augmented episodes:

```python
import json

with open('data/gold_augmented.jsonl') as f:
    for line in f:
        ep = json.loads(line)
        if ep.get('needs_review'):
            print(f"Source: {ep.get('source_id')}")
            for msg in ep['messages']:
                print(f"  {msg['role']}: {msg['content'][:100]}...")
            print()
            input("Press Enter for next...")
```

### Step 4: Prepare for Training

```bash
python scripts/prepare_chat_sft.py \
    --input data/gold_augmented.jsonl \
    --output data/sft_prepared \
    --val_split 0.1
```

### Step 5: Train

```bash
python train.py \
    --model checkpoints/750M_base \
    --config configs/sft1/750M_1024_chat_sft_phase3a.json \
    --dataset data/sft_prepared
```

---

## Quality Control

### What to Check

When reviewing augmented episodes:

| Check                 | Pass                | Fail                 |
| --------------------- | ------------------- | -------------------- |
| **Meaning preserved** | Same intent         | Contradicts original |
| **Natural language**  | Sounds human        | Awkward/robotic      |
| **Grammar**           | Correct             | Errors introduced    |
| **Flow**              | Logical progression | Non-sequiturs        |

### Sampling Strategy

For 300 augmented episodes:

- **Minimum**: Review 30 (10%)
- **Recommended**: Review 60 (20%)
- **If quality < 80%**: Regenerate with lower temperature

---

## Troubleshooting

### "No episodes found"

**Cause**: Input file format not recognized  
**Fix**: Ensure input is valid JSONL (one JSON object per line)

### Augmented text too similar to original

**Cause**: Temperature too low  
**Fix**: Increase `--temperature 0.8` or `--temperature 0.9`

### Augmented text is nonsense

**Cause**: Temperature too high or model issues  
**Fix**: Lower `--temperature 0.5` or use `--no_model`

### Invalid filename characters (Windows)

**Cause**: Episode IDs contain special characters  
**Fix**: Already handled - script sanitizes filenames automatically

---

## Best Practices

1. **Start small**: Augment to 1.5x first, evaluate, then go to 2x
2. **Review before training**: Bad augmentations hurt more than no augmentation
3. **Keep originals identifiable**: Use `--review_mode` to track what's augmented
4. **Preserve system prompts**: Never augment system messages
5. **Match language**: Use `--lang_filter` for bilingual datasets

---

## Summary

| Task                    | Command                                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| **Basic augmentation**  | `py scripts/augment_episodes_paraphrase.py --input gold.jsonl --output aug.jsonl --target_count 600` |
| **With review markers** | Add `--review_mode`                                                                                  |
| **Model-assisted**      | Add `--model checkpoints/your_model`                                                                 |
| **English only**        | Add `--lang_filter en`                                                                               |
| **Higher diversity**    | Add `--temperature 0.9`                                                                              |
