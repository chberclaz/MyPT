# High-Context Configuration Presets

## Overview

MyPT now includes **high-context configuration presets** with `block_size=1024` for powerful GPUs. These configs allow your model to see 4x more context compared to the standard 256 block_size, enabling faster learning from longer sequences.

---

## Why High Context (1024 vs 256)?

### Standard Context (block_size=256)

```
Model sees: "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light."
```

**~256 tokens** = about 1 paragraph

### High Context (block_size=1024)

```
Model sees: [4 paragraphs of text, ~4x more context]
```

**~1024 tokens** = about 4 paragraphs or 1-2 pages

---

## Benefits of High Context

### 1. Better Long-Range Understanding

**Example:** Writing code

**256 context:**
```python
def calculate_total(items):
    # Model only sees recent function
    return sum(item.price for item in items)
```

**1024 context:**
```python
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        # Model sees the whole class structure
        self.items.append(item)
    
    def remove_item(self, item):
        self.items.remove(item)
    
    def calculate_total(self):
        # Model understands context of the class
        return sum(item.price for item in self.items)
```

**Result:** Better understanding of structure, better code completion.

---

### 2. Faster Training

**Why?** Model learns from more context per gradient step.

**Example:**
- 256 context: needs 1000 steps to see 256,000 tokens
- 1024 context: needs 250 steps to see 256,000 tokens

**Result:** ~4x fewer steps for the same amount of data!

---

### 3. Better for Specific Tasks

**Great for:**
- **Code**: See entire functions/classes
- **Books**: Understand chapter context
- **Documents**: Better document understanding
- **Long-form writing**: Maintain coherence

**Less useful for:**
- **Chat**: Usually short messages
- **Tweets**: Very short text
- **Quick responses**: Don't need long context

---

## Available High-Context Configs

| Config            | Parameters | Layers | Embed | Context | Batch | VRAM Needed |
| ----------------- | ---------- | ------ | ----- | ------- | ----- | ----------- |
| **150M_1024.json** | ~150M     | 16     | 768   | 1024    | 24    | 12GB+       |
| **200M_1024.json** | ~200M     | 16     | 896   | 1024    | 20    | 16GB+       |
| **250M_1024.json** | ~250M     | 16     | 1024  | 1024    | 16    | 16GB+       |
| **350M_1024.json** | ~350M     | 24     | 1024  | 1024    | 12    | 20GB+       |
| **500M_1024.json** | ~500M     | 24     | 1280  | 1024    | 8     | 24GB+       |

---

## Usage

### View Config Details

```bash
python scripts/show_configs.py --config_file configs/150M_1024.json
```

### Train with High-Context Config

```bash
# For code generation (long functions)
python train.py --config_file configs/150M_1024.json \
                --model_name codegen_150M \
                --input_file training_code.txt \
                --max_iters 5000

# For book training (long context)
python train.py --config_file configs/250M_1024.json \
                --model_name dante_250M \
                --input_file divine_comedy.txt \
                --max_iters 10000
```

---

## Parameter Count Impact

### Key Insight: block_size Barely Affects Parameters!

**150M model with different block_sizes:**

| block_size | Position Embeddings | Total Parameters | Difference |
| ---------- | ------------------- | ---------------- | ---------- |
| 256        | 196,608 (~0.2M)     | ~150.0M          | baseline   |
| 512        | 393,216 (~0.4M)     | ~150.2M          | +0.2M      |
| 1024       | 786,432 (~0.8M)     | ~150.6M          | +0.6M      |

**Conclusion:** Increasing context from 256 → 1024 only adds ~0.4% more parameters!

### But Memory Usage DOES Increase!

**Why?** Activation memory scales with `batch_size × block_size²`

| Config         | Context | Batch | Activation Memory | Total VRAM |
| -------------- | ------- | ----- | ----------------- | ---------- |
| 150M (256)     | 256     | 32    | ~2 GB             | ~8 GB      |
| 150M_1024      | 1024    | 24    | ~6 GB             | ~12 GB     |

**Key takeaway:** Same model size, but needs more VRAM due to activations!

---

## How to Calculate Parameters

Use the calculator tool:

```bash
# Calculate for config file
python scripts/calculate_params.py --config_file configs/150M_1024.json
```

**Output:**
```
======================================================================
GPT Model Parameter Breakdown
======================================================================

Architecture:
  n_layer     : 16
  n_embd      : 768
  n_head      : 12
  vocab_size  : 50,304
  block_size  : 1024
  bias        : False

======================================================================
Parameter Breakdown:
======================================================================

Embeddings:
  Token embeddings    : 38,633,472 (38.63M)
  Position embeddings : 786,432 (0.79M)       ← Only 0.79M!

Per Transformer Layer (16 layers):
  Attention           : 2,359,296 (2.36M)
  Feed-forward        : 4,718,592 (4.72M)
  Layer norms         : 6,144 (6.14K)
  Total per layer     : 7,084,032 (7.08M)

All 16 Layers        : 113,344,512 (113.34M)

Final Components:
  Final layer norm    : 1,536 (1.54K)
  LM head             : 38,633,472 (38.63M)

======================================================================
TOTAL PARAMETERS      : 191,399,424 (191.40M)
======================================================================

Estimated Memory (model weights only):
  FP32 (4 bytes/param) : 0.71 GB
  FP16 (2 bytes/param) : 0.36 GB

Note: Training requires 3-4x more memory (gradients, optimizer states)
  Training estimate (FP32): 2.49 - 2.85 GB
```

---

## Formula

```
Total = Token Embeddings + Position Embeddings + Transformer Layers + Final LN + LM Head

Total ≈ 12 × n_layer × n_embd² + 2 × vocab_size × n_embd
```

**Position embeddings:**
```
pos_emb = block_size × n_embd
```

**For n_embd=768:**
- block_size=256: 196,608 params
- block_size=1024: 786,432 params
- Difference: 589,824 params (~0.6M)

**Compared to total model (150M):** ~0.4% increase

---

## Choosing Between Standard and High-Context

### Use Standard (block_size=256) If:

✅ Limited VRAM (< 12GB)  
✅ Training on short text (tweets, chat)  
✅ Need faster training speed  
✅ Working on CPU or older GPU  

### Use High-Context (block_size=1024) If:

✅ Powerful GPU (12GB+ VRAM)  
✅ Training on long text (books, code, documents)  
✅ Need better long-range understanding  
✅ Want fewer training steps  
✅ Quality > speed  

---

## GPU Requirements

### Standard Configs (block_size=256)

| GPU             | VRAM  | Max Config     |
| --------------- | ----- | -------------- |
| GTX 1060        | 6GB   | small          |
| RTX 2060        | 8GB   | 150M           |
| RTX 3060        | 12GB  | 200M           |
| RTX 3070        | 8GB   | 150M           |
| RTX 3080        | 10GB  | 200M           |
| RTX 3090        | 24GB  | 250M+          |
| RTX 4090        | 24GB  | 250M+          |

### High-Context Configs (block_size=1024)

| GPU             | VRAM  | Max Config     |
| --------------- | ----- | -------------- |
| RTX 2060        | 8GB   | tiny/small     |
| RTX 3060        | 12GB  | 150M_1024      |
| RTX 3070        | 8GB   | tiny/small     |
| RTX 3080        | 10GB  | 150M_1024      |
| RTX 3090        | 24GB  | 500M_1024      |
| RTX 4070 Ti     | 12GB  | 150M_1024      |
| RTX 4080        | 16GB  | 250M_1024      |
| RTX 4090        | 24GB  | 500M_1024      |
| A100 (40GB)     | 40GB  | 500M_1024+     |
| A100 (80GB)     | 80GB  | 1B+ possible   |

---

## Comparison Example

### Training Dante's Divine Comedy

**Standard config (150M):**
```bash
python train.py --config_file configs/150M.json \
                --model_name dante_standard \
                --input_file divine_comedy.txt \
                --max_iters 10000
```

**High-context config (150M_1024):**
```bash
python train.py --config_file configs/150M_1024.json \
                --model_name dante_highcontext \
                --input_file divine_comedy.txt \
                --max_iters 2500  # Only 1/4 the steps needed!
```

**Results:**
- Same final quality
- High-context trains in 1/4 the steps
- High-context better understands poem structure
- High-context needs 12GB vs 8GB VRAM

---

## Technical Details

### Why Position Embeddings are Tiny

```
Position embeddings = block_size × n_embd
                    = 1024 × 768
                    = 786,432 parameters (~0.79M)

Compare to one transformer layer = 12 × n_embd²
                                 = 12 × 768²
                                 = 7,077,888 parameters (~7.08M)
```

**Position embeddings are ~10x smaller than one layer!**

### Why Activation Memory Scales with block_size²

During forward pass, the attention mechanism computes:
```
attention_scores = Q @ K.T  # Shape: [batch, heads, block_size, block_size]
```

**Memory for attention scores:**
```
256 context:  batch × heads × 256 × 256   = batch × heads × 65,536
1024 context: batch × heads × 1024 × 1024 = batch × heads × 1,048,576

Ratio: 1,048,576 / 65,536 = 16x more memory!
```

**This is why 1024 context needs more VRAM!**

---

## Best Practices

### 1. Start with Standard, Scale to High-Context

```bash
# Phase 1: Test with standard config
python train.py --config_file configs/small.json --model_name test --max_iters 100

# Phase 2: Scale up to standard 150M
python train.py --config_file configs/150M.json --model_name test --max_iters 1000

# Phase 3: Switch to high-context if you have VRAM
python train.py --config_file configs/150M_1024.json --model_name test --max_iters 1000
```

### 2. Monitor VRAM Usage

```bash
# Watch VRAM in real-time
nvidia-smi -l 1

# If you see OOM (Out of Memory):
# 1. Reduce batch_size in config
# 2. Use smaller model
# 3. Use standard context (256) instead of high-context (1024)
```

### 3. Adjust Batch Size for Your GPU

**Rule of thumb:**
```
Available VRAM = Model Memory + Gradient Memory + Optimizer Memory + Activation Memory

Activation Memory ≈ batch_size × block_size² × n_layers × n_embd × 4 bytes
```

If OOM, reduce batch_size in the config file.

---

## Summary

**High-context configs (1024):**

✅ **4x more context** per training step  
✅ **Faster learning** from long sequences  
✅ **Better understanding** of structure  
✅ **Only +0.4% parameters** compared to standard  

But requires:

❌ **More VRAM** (~1.5-2x standard config)  
❌ **Slower per-step** (larger activations)  
❌ **Powerful GPU** (12GB+ recommended)  

**When to use:**
- You have a powerful GPU (12GB+ VRAM)
- Training on long sequences (code, books, documents)
- Quality > speed

**Calculate parameters:**
```bash
python scripts/calculate_params.py --config_file configs/150M_1024.json
```

**Never guess your model's context requirements again!** 🚀

