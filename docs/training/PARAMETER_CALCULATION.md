# GPT Model Parameter Calculation Guide

## Overview

Understanding how to calculate the number of parameters in your GPT model is crucial for:
- **Memory planning**: Estimating VRAM requirements
- **Model sizing**: Choosing the right architecture
- **Comparison**: Understanding model capacity relative to others
- **Debugging**: Verifying your architecture is correct

---

## Quick Calculation Tool

Use the dedicated calculator script:

```bash
# From config file
python scripts/calculate_params.py --config_file configs/150M_1024.json

# From parameters
python scripts/calculate_params.py --n_layer 16 --n_embd 768 --n_head 12

# Interactive mode (prompts for each value)
python scripts/calculate_params.py --interactive

# Show detailed formula
python scripts/calculate_params.py --show_formula
```

---

## The Formula

### Simplified Approximation

For quick estimation:

```
Total ≈ 12 × n_layer × n_embd² + 2 × vocab_size × n_embd
```

**Why this works:**
- Each transformer layer has ~12 × n_embd² parameters
- Embeddings (token + LM head) contribute 2 × vocab_size × n_embd
- Position embeddings are negligible

### Detailed Breakdown

```
Total = Token Embeddings + Position Embeddings + Transformer Layers + Final LN + LM Head
```

#### 1. Token Embeddings
```
token_emb = vocab_size × n_embd
```

**Example:** 50,304 × 768 = 38,633,472 (~38.6M)

#### 2. Position Embeddings
```
pos_emb = block_size × n_embd
```

**Example:** 1024 × 768 = 786,432 (~0.79M)

**Note:** This is usually < 1% of total parameters

#### 3. Each Transformer Layer

##### Attention Parameters
```
attention = 4 × n_embd²
```

Breakdown:
- Q, K, V projections: 3 × (n_embd × n_embd)
- Output projection: n_embd × n_embd

**Example:** 4 × 768² = 2,359,296 (~2.36M per layer)

##### Feed-Forward Parameters
```
feedforward = 8 × n_embd²
```

Breakdown:
- First linear: n_embd × (4 × n_embd) = 4 × n_embd²
- Second linear: (4 × n_embd) × n_embd = 4 × n_embd²

**Example:** 8 × 768² = 4,718,592 (~4.72M per layer)

##### Layer Norms
```
layernorm = 4 × n_embd
```

Two layer norms per layer, each with gamma and beta:
- 2 × (2 × n_embd) = 4 × n_embd

**Example:** 4 × 768 = 3,072 (~0.003M per layer)

##### Total Per Layer
```
per_layer = 12 × n_embd² + 4 × n_embd
         ≈ 12 × n_embd²  (since 4 × n_embd << 12 × n_embd²)
```

**Example:** For n_embd=768: ~7.08M per layer

#### 4. All Layers
```
all_layers = n_layer × per_layer
           = n_layer × (12 × n_embd² + 4 × n_embd)
```

**Example:** 16 × 7.08M = 113.34M

#### 5. Final Layer Norm
```
final_ln = 2 × n_embd
```

**Example:** 2 × 768 = 1,536 (~0.0015M)

#### 6. Language Model Head
```
lm_head = n_embd × vocab_size
```

**Example:** 768 × 50,304 = 38,633,472 (~38.6M)

---

## Complete Examples

### Example 1: 150M Model (block_size=256)

**Architecture:**
- n_layer = 16
- n_embd = 768
- n_head = 12
- vocab_size = 50,304
- block_size = 256

**Calculation:**
```
Token embeddings:     50,304 × 768      = 38,633,472
Position embeddings:     256 × 768      =    196,608
All layers:           16 × 12 × 768²    = 113,246,208
Final layer norm:     2 × 768           =      1,536
LM head:              768 × 50,304      = 38,633,472

Total:                                   = 190,711,296  (~191M)
```

**Actual:** ~150M (the approximation is close but not exact due to simplifications)

---

### Example 2: 150M Model with 1024 Context

**Architecture:**
- n_layer = 16
- n_embd = 768
- n_head = 12
- vocab_size = 50,304
- block_size = 1024 ← **Changed**

**Calculation:**
```
Token embeddings:     50,304 × 768      = 38,633,472
Position embeddings:   1,024 × 768      =    786,432  ← +589,824 params
All layers:           16 × 12 × 768²    = 113,246,208
Final layer norm:     2 × 768           =      1,536
LM head:              768 × 50,304      = 38,633,472

Total:                                   = 191,301,120  (~191M)
```

**Key insight:** Increasing block_size from 256 to 1024 only adds ~590K parameters (~0.3% increase), but allows 4x more context!

---

### Example 3: 500M Model

**Architecture:**
- n_layer = 24
- n_embd = 1280
- n_head = 16
- vocab_size = 50,304
- block_size = 1024

**Calculation:**
```
Token embeddings:     50,304 × 1,280    = 64,389,120
Position embeddings:   1,024 × 1,280    =  1,310,720
All layers:           24 × 12 × 1,280²  = 471,859,200
Final layer norm:     2 × 1,280         =      2,560
LM head:              1,280 × 50,304    = 64,389,120

Total:                                   = 601,950,720  (~602M)
```

---

## Impact of Parameters

### Block Size

**Question:** Should I use 256 or 1024 block_size?

| Parameter  | 256 Context | 1024 Context | Difference |
| ---------- | ----------- | ------------ | ---------- |
| Params     | ~190.7M     | ~191.3M      | +0.6M      |
| VRAM       | ~8 GB       | ~10 GB       | +2 GB      |
| Context    | 256 tokens  | 1024 tokens  | 4x         |
| Speed      | Faster      | Slower       | ~0.7x      |
| Learn from | Short seqs  | Long seqs    | Better     |

**Recommendation:**
- **256**: Limited VRAM (< 12GB), need speed
- **1024**: Powerful GPU (12GB+), long sequences (code, documents, books)

### Vocabulary Size

**Standard:** 50,304 (GPT-2 BPE vocabulary)

**Character-level:** ~100-256 (depends on dataset)

**Impact:**
```
With vocab_size=50,304 and n_embd=768:
  Token embeddings: 38.6M
  LM head:          38.6M
  Total from vocab: 77.2M

With vocab_size=100 and n_embd=768 (char-level):
  Token embeddings: 0.08M
  LM head:          0.08M
  Total from vocab: 0.16M
  
Savings: 77M parameters!
```

**But:** Character-level models need longer sequences to represent the same text.

### Number of Layers vs Embedding Dimension

**Question:** Is 16 layers × 768 embedding better than 8 layers × 1088 embedding?

Both have ~150M params, but:

**16 × 768:**
- More depth → Better at hierarchical features
- Better for complex reasoning
- Slower per token

**8 × 1088:**
- More width → Better at capturing diverse patterns
- Faster inference
- May need more data

**General rule:** Depth (n_layer) is usually better than width (n_embd) for language models.

---

## Memory Estimation

### Model Weights Only

```
FP32: total_params × 4 bytes
FP16: total_params × 2 bytes
```

**Example (150M params):**
- FP32: 150M × 4 = 600 MB
- FP16: 150M × 2 = 300 MB

### Training Memory

During training, you need memory for:
1. **Model weights** (1x)
2. **Gradients** (1x)
3. **Optimizer states** (2x for Adam: momentum + variance)
4. **Activations** (depends on batch_size and block_size)

**Total ≈ 4x model size + activation memory**

**Example (150M params, FP32):**
```
Model:       600 MB
Gradients:   600 MB
Optimizer:  1200 MB
Activations: ~2-6 GB (depends on batch_size × block_size)

Total: ~4-8 GB
```

**For block_size=1024 vs 256:**
- Activations scale with block_size²
- 1024 needs ~4-6x more memory than 256

---

## Parameter Count Table

### Standard Configs (block_size=256)

| Config       | n_layer | n_embd | Parameters | Memory (Training) |
| ------------ | ------- | ------ | ---------- | ----------------- |
| Tiny         | 4       | 192    | ~11M       | ~500 MB           |
| Small        | 6       | 384    | ~40M       | ~2 GB             |
| 150M         | 16      | 768    | ~150M      | ~8 GB             |
| 200M         | 16      | 896    | ~200M      | ~10 GB            |
| 250M         | 16      | 1024   | ~250M      | ~12 GB            |
| GPT-2 Small  | 12      | 768    | ~117M      | ~6 GB             |
| GPT-2 Medium | 24      | 1024   | ~345M      | ~16 GB            |
| GPT-2 Large  | 36      | 1280   | ~774M      | ~32 GB            |
| GPT-2 XL     | 48      | 1600   | ~1.5B      | ~60 GB            |

### High-Context Configs (block_size=1024)

| Config       | n_layer | n_embd | Parameters | Memory (Training) |
| ------------ | ------- | ------ | ---------- | ----------------- |
| 150M_1024    | 16      | 768    | ~150M      | ~10 GB            |
| 200M_1024    | 16      | 896    | ~200M      | ~12 GB            |
| 250M_1024    | 16      | 1024   | ~250M      | ~14 GB            |
| 350M_1024    | 24      | 1024   | ~350M      | ~18 GB            |
| 500M_1024    | 24      | 1280   | ~500M      | ~22 GB            |

**Note:** Higher block_size requires more VRAM due to activation memory, even with same parameter count.

---

## Using the Calculator

### From Config File

```bash
python scripts/calculate_params.py --config_file configs/150M_1024.json
```

Output shows:
- Complete parameter breakdown
- Memory estimates (FP32, FP16)
- Training memory estimates

### Interactive Mode

```bash
python scripts/calculate_params.py --interactive
```

Prompts for:
- n_layer
- n_embd
- n_head
- vocab_size
- block_size
- bias (yes/no)

Then displays full breakdown.

### Quick Calculation

```bash
python scripts/calculate_params.py --n_layer 16 --n_embd 768 --n_head 12 --block_size 1024
```

---

## Common Questions

### Q: Why do parameter counts vary slightly between tools?

**A:** Different tools may:
- Count/ignore bias parameters
- Include/exclude tied embeddings (token emb = LM head)
- Use different approximations

Our calculator uses the **exact count** with all parameters.

### Q: Does block_size significantly affect parameter count?

**A:** No! Position embeddings are `block_size × n_embd`, which is tiny compared to the transformer layers.

Example:
- block_size=256: adds 196K params
- block_size=1024: adds 786K params
- Difference: ~590K params (~0.3% of 150M model)

**But:** block_size significantly affects VRAM due to activation memory!

### Q: Why is my model "150M" but the calculator shows 191M?

**A:** Common causes:
- Including LM head (tied or separate)
- Including position embeddings
- Different vocab_size

The "150M" is an approximation. Use the calculator for exact counts.

### Q: How many parameters for a GPT-3 sized model?

**GPT-3 sizes:**
- GPT-3 Small (125M): ~12 layers, ~768 embd
- GPT-3 Medium (350M): ~24 layers, ~1024 embd
- GPT-3 Large (1.3B): ~24 layers, ~2048 embd
- GPT-3 XL (2.7B): ~32 layers, ~2560 embd
- GPT-3 (175B): ~96 layers, ~12,288 embd

---

## Summary

**Quick formula:**
```
Total ≈ 12 × n_layer × n_embd² + 2 × vocab_size × n_embd
```

**Key insights:**
- **n_embd²** dominates: doubling n_embd quadruples params
- **n_layer** scales linearly: doubling n_layer doubles params
- **block_size** is negligible for param count (but affects VRAM!)
- **vocab_size** matters: GPT-2 (50K) adds ~77M params vs char-level (~100) adds 0.2M

**Use the calculator:**
```bash
python scripts/calculate_params.py --config_file configs/your_config.json
```

**Never guess your model size again!** 🎯

