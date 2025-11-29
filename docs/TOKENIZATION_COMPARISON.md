# GPT-2 BPE vs Character-Level Tokenization

## Complete Comparison

This guide explains the differences between GPT-2 BPE and character-level tokenization and their impact on model size, training, and generation.

---

## Vocabulary Size Impact

### Small Model (6 layers, 384 embedding)

| Tokenization | vocab_size | Token Emb | LM Head | Other | **Total** |
|--------------|-----------|-----------|---------|-------|-----------|
| **GPT-2 BPE** | 50,304 | 19.3M | 19.3M | 10.7M | **~40M** |
| **Char-level** | ~256 | 0.1M | 0.1M | 10.7M | **~22M** |

**Difference: ~18M parameters (45% reduction for char-level!)**

---

### Parameter Breakdown

```
Small Model Architecture:
- n_layer: 6
- n_embd: 384
- n_head: 6
- block_size: 256

GPT-2 BPE (vocab_size=50,304):
├── Token embeddings:    50,304 × 384 = 19,316,736  (38.6M total)
├── Position embeddings:    256 × 384 =     98,304
├── Transformer layers: 6 × 12 × 384² = 10,616,832
├── Final layer norm:            2 × 384 =        768
└── LM head:            384 × 50,304 = 19,316,736
    Total:                                ~40.0M parameters

Character-level (vocab_size=256):
├── Token embeddings:       256 × 384 =     98,304  (0.2M total)
├── Position embeddings:    256 × 384 =     98,304
├── Transformer layers: 6 × 12 × 384² = 10,616,832
├── Final layer norm:            2 × 384 =        768
└── LM head:            384 × 256     =     98,304
    Total:                                ~22.0M parameters
```

**The transformer layers are identical! Only embeddings differ.**

---

## Sequence Length Comparison

### Same Text, Different Tokenization

**Text:** "Hello world, this is a test."

**GPT-2 BPE tokenization:**
```
Tokens: ["Hello", " world", ",", " this", " is", " a", " test", "."]
Length: 8 tokens
```

**Character-level tokenization:**
```
Tokens: ['H','e','l','l','o',' ','w','o','r','l','d',',',' ','t','h','i','s',' ','i','s',' ','a',' ','t','e','s','t','.']
Length: 28 characters
```

**Ratio: 3.5x longer with character-level!**

---

## Trade-offs

### GPT-2 BPE

**Pros:**
- ✅ More efficient (shorter sequences)
- ✅ Industry standard
- ✅ Faster training per token
- ✅ Faster generation per token
- ✅ Better for large datasets

**Cons:**
- ❌ Larger model (~40M vs ~22M for same architecture)
- ❌ Fixed vocabulary (can't handle all text)
- ❌ Requires tiktoken library

**Best for:**
- Large-scale training
- Production models
- English text
- Code generation
- When efficiency matters

---

### Character-Level

**Pros:**
- ✅ Smaller model (~22M vs ~40M for same architecture)
- ✅ Universal (works with any language/symbols)
- ✅ Simple (no external libraries)
- ✅ Never encounters unknown tokens
- ✅ Good for learning/understanding

**Cons:**
- ❌ Longer sequences (3-4x more tokens)
- ❌ Slower training overall (more tokens to process)
- ❌ Slower generation (more tokens to generate)
- ❌ Higher memory for activations (longer sequences)

**Best for:**
- Learning/experiments
- Small datasets
- Non-English text
- Special characters/symbols
- When simplicity matters

---

## Memory Comparison

### Model Weights

**Small model:**
- GPT-2 BPE: 40M params × 4 bytes = 160 MB
- Char-level: 22M params × 4 bytes = 88 MB
- **Savings: 72 MB (45%)**

---

### Training Memory

**Total training memory = weights + gradients + optimizer + activations**

| Tokenization | Weights | Gradients | Optimizer | Activations | **Total** |
|--------------|---------|-----------|-----------|-------------|-----------|
| GPT-2 BPE | 160 MB | 160 MB | 320 MB | ~1.5 GB | **~2.1 GB** |
| Char-level | 88 MB | 88 MB | 176 MB | ~1.5 GB | **~1.9 GB** |

**Savings: ~200 MB**

**Note:** Activations are similar because they depend on model architecture (layers, embedding), not vocab_size.

---

### Sequence Length Impact

For same text with block_size=256:

**GPT-2 BPE:**
- 256 tokens ≈ 1024 characters (4:1 ratio)
- Sees ~4x more text per batch

**Char-level:**
- 256 tokens = 256 characters
- Sees less text per batch

**Result:** Character-level needs ~4x more iterations to see the same amount of text!

---

## Training Time Comparison

### Same Dataset (1MB text, ~1M tokens GPT-2 / ~1M chars)

**GPT-2 BPE:**
```bash
python train.py \
    --config_file configs/small.json \
    --tokenization gpt2 \
    --max_iters 1000

Dataset: ~250K GPT-2 tokens
Coverage: ~33x (1000 iters × 32 batch × 256 block / 250K)
Time: ~15 minutes
```

**Character-level:**
```bash
python train.py \
    --config_file configs/small_char.json \
    --tokenization char \
    --max_iters 1000

Dataset: ~1M characters
Coverage: ~8x (1000 iters × 32 batch × 256 block / 1M)
Time: ~15 minutes (same iterations, but less text coverage)
```

**For same coverage, char-level needs ~4x more iterations!**

---

## Which to Choose?

### Use GPT-2 BPE If:

✅ Training on large datasets (100M+ tokens)  
✅ Production models  
✅ English or common languages  
✅ Want industry-standard approach  
✅ Efficiency matters  

**Example:**
```bash
python train.py \
    --config_file configs/150M.json \
    --tokenization gpt2 \
    ...
```

---

### Use Character-Level If:

✅ Learning/experimenting  
✅ Small datasets (< 10M tokens)  
✅ Non-English languages (rare scripts)  
✅ Want smaller model  
✅ Simplicity matters (no external deps)  

**Example:**
```bash
python train.py \
    --config_file configs/small_char.json \
    --tokenization char \
    ...
```

---

## Available Configs

### GPT-2 BPE Configs (vocab_size=50304)

| Config | Params | Best For |
|--------|--------|----------|
| `tiny.json` | ~11M | Quick experiments |
| `small.json` | ~40M | Development |
| `150M.json` | ~150M | Production |
| `200M.json` | ~200M | Production |
| `250M.json` | ~250M | Production |
| `150M_1024.json` | ~150M | High-context |
| `200M_1024.json` | ~200M | High-context |
| `250M_1024.json` | ~250M | High-context |
| `350M_1024.json` | ~350M | High-context |
| `500M_1024.json` | ~500M | High-context |

### Character-Level Configs (vocab_size=256)

| Config | Params | Best For |
|--------|--------|----------|
| `tiny_char.json` | ~3M | Quick char experiments |
| `small_char.json` | ~22M | Char development |

**Want more char-level configs?** Easy to create! Just copy a GPT-2 config and change `vocab_size: 256`.

---

## Creating Custom Char-Level Configs

### Method 1: Copy and Modify

```bash
# Copy existing config
cp configs/150M.json configs/150M_char.json

# Edit 150M_char.json
# Change: "vocab_size": 50304 → "vocab_size": 256
```

**Result:** ~150M param model becomes ~132M param model!

### Method 2: Calculate Parameters

```bash
# See how many parameters you'll get
python scripts/calculate_params.py \
    --n_layer 16 \
    --n_embd 768 \
    --n_head 12 \
    --vocab_size 256  # Char-level

# Output: ~132M params (vs ~150M for GPT-2 BPE)
```

---

## Summary

**Key insights:**

1. **vocab_size has huge impact on parameter count** (~20-40M difference)
2. **Config files assume GPT-2 BPE** (vocab_size=50304)
3. **Training auto-corrects vocab_size** based on tokenizer
4. **Calculator needs manual override** for char-level

**Solutions:**

✅ Use char-level configs: `tiny_char.json`, `small_char.json`  
✅ Override in calculator: `--vocab_size 256`  
✅ Check actual model: `inspect_model.py`  

**GPT-2 BPE vs Char-level:**
- **GPT-2:** Larger model, shorter sequences, more efficient
- **Char:** Smaller model, longer sequences, simpler

**Your choice depends on your use case!** 🎯

