# Loss Masking for Supervised Fine-Tuning (SFT)

This document explains the loss masking feature in MyPT, designed for supervised fine-tuning (SFT) scenarios, particularly for training chat/assistant-style models.

---

## Table of Contents

1. [What is Loss Masking?](#what-is-loss-masking)
2. [Why Use Loss Masking?](#why-use-loss-masking)
3. [How It Works](#how-it-works)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Creating Masked Datasets](#creating-masked-datasets)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## What is Loss Masking?

**Loss masking** is a technique where you selectively compute loss on only certain tokens during training, while still using all tokens as context for prediction.

In chat/assistant training scenarios, you typically have conversational data like:

```
<User>What is the capital of France?</User>
<Assistant>The capital of France is Paris.</Assistant>
```

With loss masking enabled (`use_loss_mask: true`), the model:

- ✅ **Sees** the entire conversation (user + assistant)
- ✅ **Learns from context** provided by user tokens
- ✅ **Only computes loss** on assistant tokens
- ❌ **Doesn't try to predict** user tokens

This focuses training on generating high-quality assistant responses while still understanding user context.

---

## Why Use Loss Masking?

### The Problem: Training Without Masks

If you train a chat model **without** loss masking:

```python
# Sequence: [USER_TOKEN, "What", "is", "Paris", "?", ASSISTANT_TOKEN, "Paris", "is", ...]
# Loss computed on ALL tokens
```

**Issues:**

1. 🔴 **Model learns to generate user questions** - Not useful for an assistant
2. 🔴 **Low-quality user text dilutes signal** - Typos, informal language treated as targets
3. 🔴 **Model may hallucinate user turns** - Starts generating `<User>` in responses
4. 🔴 **Wastes gradient updates** - Computing loss on tokens you don't care about

### The Solution: Training With Masks

With loss masking:

```python
# Sequence:  [USER, "What", "is", "Paris", "?", ASST,    "Paris", "is", ...]
# Loss mask: [  0  ,   0   ,  0  ,   0    ,  0 ,  0  ,      1    ,  1   , ...]
#            ↑ No loss on user side        ↑ Loss only on assistant tokens
```

**Benefits:**

1. ✅ **Focused learning** - Only supervise assistant-quality responses
2. ✅ **Better quality** - No contamination from user-side text
3. ✅ **Prevents role confusion** - Model won't try to generate user questions
4. ✅ **Efficient training** - Gradients only flow where they matter

---

## How It Works

### Conceptual Overview

At each time step `t`, the transformer:

1. Takes tokens `[0..t-1]` as input context
2. Predicts token `t`
3. Computes loss if `loss_mask[t] == 1`, otherwise skips

**Example:**

```
Input:  [<User>, "Hi", ",", "who", "are", "you", "?", <Asst>]
Predict:         "Hi", ",", "who", "are", "you", "?", <Asst>, "I'm"
Mask:             0  ,  0  ,  0  ,   0  ,   0  ,  0  ,   0   ,  1
                 ↑ No loss computed              ↑ Loss computed here
```

When predicting `"I'm"` (the first assistant token):
- **Input context:** All previous tokens including user message
- **Prediction target:** `"I'm"`
- **Loss mask:** `1` (compute loss)

**Gradient Flow:**

Even though user tokens have `mask=0`, their embeddings and attention weights **still receive gradients** because they're part of the context used to predict masked tokens.

This is key: **the model learns to use user context effectively**, it just doesn't learn to generate user-style text.

### Mathematical Details

**Standard cross-entropy (no masking):**

```
loss = -1/N × Σ log P(y_i | x_{<i})
```

**Masked cross-entropy:**

```
loss = -1/M × Σ mask_i × log P(y_i | x_{<i})

where M = Σ mask_i (number of masked positions)
```

**Normalization:** We divide by the number of masked positions (`M`), not total positions (`N`), so loss scale is comparable.

---

## Implementation Details

### 1. Config Field: `use_loss_mask`

Add to any config file:

```json
{
  "name": "GPT-200M-Chat-SFT",
  "n_layer": 16,
  "n_embd": 896,
  "use_loss_mask": true  // ← Enable loss masking
}
```

This field is saved/loaded with checkpoints, so the behavior is preserved.

### 2. Model Forward Pass

The `GPT.forward()` method now accepts an optional `loss_mask` parameter:

```python
def forward(self, idx, targets=None, loss_mask=None):
    # ... compute logits ...
    
    if loss_mask is not None:
        # Compute per-token loss
        per_token_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply mask and normalize
        denom = loss_mask.sum()
        loss = (per_token_loss * loss_mask).sum() / denom
    else:
        # Standard loss
        loss = F.cross_entropy(logits, targets)
    
    return logits, loss
```

### 3. Data Loader Changes

`GPTDataLoader` now:

- Accepts `use_loss_mask` parameter in constructor
- Looks for `*_mask.bin` shard files alongside token shards
- Returns `(X, Y, mask)` if masks are available, otherwise `(X, Y)`

### 4. Training Loop

The training loop automatically detects batch format:

```python
batch = data_loader.get_batch('train')

if isinstance(batch, (tuple, list)) and len(batch) == 3:
    xb, yb, loss_mask = batch
    _, loss = model(xb, yb, loss_mask=loss_mask)
else:
    xb, yb = batch
    _, loss = model(xb, yb)
```

No explicit checks for `config.use_loss_mask` in the training loop—the data loader handles it.

---

## Usage

### Method 1: Using SFT Config Presets

The easiest way is to use pre-configured SFT presets:

```bash
# View available SFT configs
ls configs/*sft.json

# tiny_sft.json
# 150M_chat_sft.json
# 200M_chat_sft.json
# 250M_1024_chat_sft.json
```

**Example:**

```bash
python train.py \
    --config_file configs/200M_chat_sft.json \
    --model_name my_chat_model \
    --dataset_dir data/chat_dataset \
    --max_iters 10000
```

### Method 2: Custom Config

Create your own config with `use_loss_mask: true`:

```json
{
  "name": "My-Custom-SFT",
  "batch_size": 32,
  "block_size": 256,
  "vocab_size": 50304,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 16,
  "dropout": 0.1,
  "bias": false,
  "use_loss_mask": true
}
```

### Method 3: Two-Phase Training (Recommended)

**Phase 1:** Pre-train on general corpus (no masking)

```bash
python train.py \
    --config_file configs/200M.json \
    --model_name base_200M \
    --dataset_dir data/general_corpus \
    --max_iters 50000
```

**Phase 2:** Fine-tune with SFT (with masking)

```bash
python train.py \
    --config_file configs/200M_chat_sft.json \
    --init_from_model base_200M \
    --model_name chat_200M \
    --dataset_dir data/chat_sft \
    --max_iters 10000 \
    --learning_rate 3e-5  # Lower LR for fine-tuning
```

**What happens during fine-tuning:**

- ✅ **Weights are kept** from the base model
- ✅ **Architecture params** (n_layer, n_embd, n_head) are kept from base model
- ✅ **Training params** (use_loss_mask, dropout, batch_size) are **updated** from SFT config
- ✅ Dropout layers are updated to match new dropout value

This is the **recommended approach** used by OpenAI, Anthropic, etc.:

1. Base model learns general language understanding
2. SFT aligns it to assistant behavior

---

## Creating Masked Datasets

### Dataset Structure

For loss masking to work, your dataset directory must contain mask shards:

```
data/chat_dataset/
├── train/
│   ├── shard_000.bin       # Token IDs
│   ├── shard_000_mask.bin  # Loss masks (aligned)
│   ├── shard_001.bin
│   ├── shard_001_mask.bin
│   └── ...
├── val/
│   ├── shard_000.bin
│   ├── shard_000_mask.bin
│   └── ...
├── dataset_metadata.json
└── tokenizer_state.json
```

### Mask File Format

Mask files are binary files (`.bin`) with `uint8` values:

- `0` = Ignore position (don't compute loss)
- `1` = Compute loss (assistant tokens)

**Shape:** Same as corresponding token shard `(N,)` where `N` is number of tokens.

### Creating Masks Manually

If you have a conversation dataset in JSONL format:

```jsonl
{"role": "user", "content": "What is 2+2?"}
{"role": "assistant", "content": "2+2 equals 4."}
{"role": "user", "content": "What about 3+3?"}
{"role": "assistant", "content": "3+3 equals 6."}
```

**Pseudocode:**

```python
import numpy as np
from tokenizer import Tokenizer

tokenizer = Tokenizer(config, 'gpt2')
tokens = []
masks = []

for message in conversation:
    content_tokens = tokenizer.encode(message['content'])
    tokens.extend(content_tokens)
    
    # Mask: 1 for assistant, 0 for user
    if message['role'] == 'assistant':
        masks.extend([1] * len(content_tokens))
    else:
        masks.extend([0] * len(content_tokens))

# Save to binary
np.array(tokens, dtype=np.uint32).tofile('shard_000.bin')
np.array(masks, dtype=np.uint8).tofile('shard_000_mask.bin')
```

### Using `prepare_dataset.py` (Future Feature)

We plan to add mask generation to `scripts/prepare_dataset.py`:

```bash
# Future feature
python scripts/prepare_dataset.py \
    --input_files chat_data.jsonl \
    --out_dir data/chat_sft \
    --tokenization gpt2 \
    --format chat \
    --mask_role assistant
```

This will automatically create aligned token and mask shards.

---

## Best Practices

### 1. Use Lower Dropout for SFT

SFT configs use `dropout: 0.1` instead of the standard `0.2`:

```json
{
  "dropout": 0.1  // Less aggressive regularization for fine-tuning
}
```

### 2. Lower Learning Rate

When fine-tuning from a base model, use a lower learning rate:

```bash
--learning_rate 3e-5  # vs 3e-4 for base training
```

This prevents catastrophic forgetting of base knowledge.

### 3. Fewer Iterations

SFT typically requires 5-10x fewer iterations than base training:

- Base training: 50,000-100,000 iterations
- SFT: 5,000-20,000 iterations

### 4. Monitor Validation Loss

Watch for overfitting on small SFT datasets:

```
step 0: train loss 2.4312, val loss 2.4405
step 50: train loss 1.8234, val loss 1.9102  ← Good
step 100: train loss 1.2001, val loss 1.8934  ← Val not improving
step 150: train loss 0.8123, val loss 2.0211  ← Overfitting! Stop here
```

If validation loss stops improving or increases, stop training.

### 5. Align Mask Boundaries Carefully

Make sure mask transitions align with actual role boundaries:

```
Tokens: [USER_TAG, "Hi", ASST_TAG, "Hello", "!"]
Mask:   [   0    ,  0  ,    0    ,    1   , 1 ]
                         ↑ Mask starts AFTER assistant tag
```

If you include special tokens in the mask, the model will learn to generate them.

### 6. Validate Masks

Sanity check your masks:

```python
# Check that masks align with data
assert len(tokens) == len(masks), "Length mismatch!"
assert 0.1 < masks.mean() < 0.9, "Mask too sparse or too dense!"

# For chat data, typically 30-50% of tokens should be masked
# (assuming balanced user/assistant turns)
```

### 7. Use Pretrained Base Models

**Don't** train SFT models from scratch! Always start with a base model:

```bash
# ❌ Bad: SFT from scratch
python train.py --config_file configs/200M_chat_sft.json ...

# ✅ Good: SFT from base model
python train.py --config_file configs/200M_chat_sft.json \
                --init_from_model base_200M ...
```

The base model provides language understanding; SFT just aligns it.

---

## Examples

### Example 1: Simple Chat SFT

**Data:**

```
<User>What's the weather?</User>
<Assistant>I don't have access to weather data.</Assistant>
```

**Tokens:** `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`

**Mask:** `[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]`

**Training:**

```bash
python train.py \
    --config_file configs/tiny_sft.json \
    --model_name weather_chat \
    --dataset_dir data/weather_chat \
    --max_iters 1000
```

### Example 2: Multi-Turn Conversation

**Data:**

```
<User>Hi</User>
<Assistant>Hello! How can I help?</Assistant>
<User>What's 2+2?</User>
<Assistant>2+2 equals 4.</Assistant>
```

**Mask pattern:**

```
[USER tokens] → mask = 0
[ASST tokens] → mask = 1
[USER tokens] → mask = 0
[ASST tokens] → mask = 1
```

The model learns from the entire conversation context but only computes loss on assistant turns.

### Example 3: Instruction Following

**Data format (Alpaca-style):**

```
Below is an instruction...

### Instruction:
Summarize the following text.

### Input:
[Long text here]

### Response:
[Summary here] ← Only this part masked
```

**Mask:**

```python
tokens = encode(prompt + instruction + input_text + response)
masks = [0] * len(encode(prompt + instruction + input_text)) + \
        [1] * len(encode(response))
```

---

## Troubleshooting

### Issue 1: "No mask shards found"

**Error:**

```
⚠️ Warning: use_loss_mask=True but mask shards not found or incomplete
Expected 10 mask shards, found 0
```

**Solution:**

Your dataset directory is missing `*_mask.bin` files. Either:

1. Create mask files (see [Creating Masked Datasets](#creating-masked-datasets))
2. Set `use_loss_mask: false` in config

### Issue 2: Mask/Token Length Mismatch

**Error:**

```
RuntimeError: The size of tensor a (256) must match the size of tensor b (128)
```

**Solution:**

Mask file length doesn't match token file. Regenerate masks:

```python
assert len(tokens) == len(masks), f"Mismatch: {len(tokens)} vs {len(masks)}"
```

### Issue 3: Loss is NaN or Inf

**Symptoms:**

```
step 0: train loss nan, val loss nan
```

**Causes:**

1. **All masks are 0:** No positions to compute loss on
2. **Learning rate too high:** Gradients explode

**Solutions:**

```bash
# Check mask density
python -c "import numpy as np; m=np.memmap('train/shard_000_mask.bin', dtype='uint8'); print(m.mean())"
# Should be 0.2-0.8

# Reduce learning rate
--learning_rate 1e-5
```

### Issue 4: Model Only Generates Special Tokens

**Symptom:**

```
>>> generate_text(model, "Hello", max_tokens=50)
"<Assistant><Assistant><Assistant>..."
```

**Cause:** Special tokens (`<Assistant>`, etc.) were included in the mask.

**Solution:** Ensure mask starts **after** special tokens:

```python
# ❌ Wrong
tokens = [ASST_TAG, "Hello"]
masks  = [   1    ,    1   ]

# ✅ Correct
tokens = [ASST_TAG, "Hello"]
masks  = [   0    ,    1   ]
```

### Issue 5: No Improvement During SFT

**Symptom:**

```
step 0: train loss 2.4312, val loss 2.4405
step 1000: train loss 2.4201, val loss 2.4389
```

**Possible causes:**

1. Learning rate too low
2. Frozen layers (not applicable here, but common issue)
3. Dataset too small
4. Base model already aligned (no room for improvement)

**Solutions:**

```bash
# Try higher learning rate
--learning_rate 1e-4

# Check dataset size
wc -l data/chat_sft/train/*.bin  # Should have 10K+ examples

# Verify masks are being used
# Add debug print in forward() to check loss_mask is not None
```

---

## Summary

**Loss masking (SFT) in MyPT:**

✅ **Enabled via config:** `use_loss_mask: true`  
✅ **No CLI flags needed:** Everything in config  
✅ **Automatic detection:** Training loop adapts to batch format  
✅ **Backward compatible:** Works with non-masked datasets  
✅ **Standard practice:** Used by GPT-4, Claude, LLaMA instruction models  

**Key benefits:**

- Focus training on high-quality assistant responses
- Prevent model from learning to generate user-style text
- Efficient use of gradient updates
- Better instruction-following and chat behavior

**When to use:**

- ✅ Chat/assistant training
- ✅ Instruction fine-tuning
- ✅ RLHF preparation (before reward modeling)
- ❌ General language modeling (use standard configs)

**Two-phase approach (recommended):**

1. Base pre-training: General corpus, no masking, high learning rate
2. SFT: Chat/instruction data, with masking, low learning rate

This mirrors the training process used by modern LLMs like GPT-4 and Claude.

---

## Related Documentation

- [Fine-Tuning Guide](../README.md#fine-tuning--transfer-learning) - General fine-tuning workflows
- [Config Presets](../configs/README.md#sft-configs-supervised-fine-tuning-💬) - SFT config presets
- [Large Dataset Training](LARGE_DATASET_TRAINING.md) - Sharded dataset creation
- [Tokenization Guide](TOKENIZATION_COMPARISON.md) - GPT-2 vs character-level

---

**Questions or issues?** Open an issue on GitHub or consult the documentation index in `docs/README.md`.

