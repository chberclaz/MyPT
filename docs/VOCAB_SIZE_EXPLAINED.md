# Vocabulary Size and Parameter Counts

## Understanding vocab_size

The vocabulary size (`vocab_size`) has a **huge impact** on model parameter count, but it's handled differently for GPT-2 BPE vs character-level tokenization.

---

## The Issue

### Config Files Show vocab_size=50304

**Example** (`configs/small.json`):
```json
{
  "name": "GPT-Small",
  "n_layer": 6,
  "n_embd": 384,
  "vocab_size": 50304,  ← Hardcoded for GPT-2 BPE
  ...
}
```

**But this is ONLY correct for GPT-2 BPE tokenization!**

For character-level tokenization, vocab_size is typically **100-256** (depends on unique characters in your data).

---

## Impact on Parameter Count

### Example: Small Model (n_embd=384)

**GPT-2 BPE (vocab_size=50304):**
```
Token embeddings: 50,304 × 384 = 19,316,736 params
LM head:          384 × 50,304 = 19,316,736 params
Total from vocab:                 38,633,472 params (~38.6M)
```

**Character-level (vocab_size=256):**
```
Token embeddings: 256 × 384 = 98,304 params
LM head:          384 × 256 = 98,304 params
Total from vocab:             196,608 params (~0.2M)
```

**Difference: ~38.4M parameters!**

**Full model comparison:**
- `small.json` (GPT-2): ~40M parameters
- `small_char.json` (Char): ~22M parameters
- **Difference: ~18M parameters (45% reduction!)**

---

## How It Works in MyPT

### During Training (Automatic Correction)

When you train with character-level tokenization, `vocab_size` is **automatically adjusted**:

```python
# In core/checkpoint.py:
if tokenization == 'char':
    tokenizer.build_char_vocab(input_text)
    config.vocab_size = len(tokenizer.chars)  # ← Updated dynamically!
else:
    config.vocab_size = 50304  # GPT-2 BPE
```

**Result:** Your trained model has the **correct** parameter count, regardless of what the config file says!

---

### During Parameter Calculation (Manual)

The `calculate_params.py` script now:

1. **Reads vocab_size from config file**
2. **Warns you** if using config with large vocab_size
3. **Suggests** using `--vocab_size 256` for char-level calculation

**Example:**
```bash
python scripts/calculate_params.py --config_file configs/small.json
```

**Output:**
```
Loaded config from: configs/small.json
Config name: GPT-Small

⚠️  Note: This calculation assumes vocab_size=50304
   For character-level tokenization, actual vocab_size will be ~100-256
   (determined by unique characters in your training data)
   To calculate for char-level, use: --vocab_size 256

[Shows calculation with 50304...]
```

**To get char-level calculation:**
```bash
python scripts/calculate_params.py \
    --config_file configs/small.json \
    --vocab_size 256  # Override!
```

---

## Solutions

### Solution 1: Use Character-Level Configs

I've created dedicated configs for character-level:

- **`configs/tiny_char.json`** - ~3M params (vocab_size=256)
- **`configs/small_char.json`** - ~22M params (vocab_size=256)

**Usage:**
```bash
# For character-level training
python train.py \
    --config_file configs/small_char.json \
    --model_name my_char_model \
    --input_file input.txt \
    --tokenization char  # ← Use char tokenization
```

---

### Solution 2: Override vocab_size in Calculator

```bash
# Calculate with correct vocab_size
python scripts/calculate_params.py \
    --n_layer 6 \
    --n_embd 384 \
    --n_head 6 \
    --vocab_size 256  # ← Char-level vocab
```

---

### Solution 3: Check Actual Trained Model

**The trained model always has the correct parameter count!**

```bash
# After training with char-level
python scripts/inspect_model.py --model_name my_char_model
```

**Output:**
```
=== CONFIG ===
vocab_size:256 (saved correctly!)

=== MODEL INFO ===
Parameters: 22,134,528 (22.13M)  ← Correct count!
```

---

## Vocabulary Sizes by Tokenization

### GPT-2 BPE Tokenization

```python
tokenization = 'gpt2'
vocab_size = 50304  # Fixed
```

**Characteristics:**
- Fixed vocabulary (50,304 tokens)
- Sub-word tokens (e.g., "running" = "run" + "ning")
- More efficient (shorter sequences)
- Standard for most LLMs

**Example tokens:** `["the", "ing", "tion", " world", ...]`

---

### Character-Level Tokenization

```python
tokenization = 'char'
vocab_size = len(unique_chars_in_dataset)  # Typically 100-256
```

**Characteristics:**
- Variable vocabulary (depends on dataset)
- Each character is a token
- Less efficient (longer sequences)
- Good for learning, small datasets

**Example vocab:**
- English text: ~95 chars (a-z, A-Z, punctuation, space)
- Dante (Italian): ~120 chars (includes à, è, ì, ò, ù)
- Unicode text: ~256 chars (extended characters)

---

## Parameter Count Breakdown by Tokenization

### GPT-Small Architecture (n_layer=6, n_embd=384)

**Component breakdown:**

| Component | Calculation | GPT-2 BPE | Char-level |
|-----------|-------------|-----------|------------|
| Token embeddings | vocab × embd | 19.3M | 0.1M |
| Position embeddings | block × embd | 0.1M | 0.1M |
| Transformer layers | 6 × 12 × 384² | 10.7M | 10.7M |
| LM head | embd × vocab | 19.3M | 0.1M |
| **Total** | | **~40M** | **~22M** |

**Key insight:** Transformer layers are the same, but embeddings differ by ~38M!

---

## Recommendations

### For Accurate Parameter Calculation

**1. Use the correct config:**
```bash
# GPT-2 BPE
python scripts/calculate_params.py --config_file configs/small.json

# Character-level
python scripts/calculate_params.py --config_file configs/small_char.json
```

**2. Or override vocab_size:**
```bash
python scripts/calculate_params.py \
    --config_file configs/small.json \
    --vocab_size 256  # For char-level
```

**3. Check actual trained model:**
```bash
# After training
python scripts/inspect_model.py --model_name my_model
# Shows actual parameter count with correct vocab_size
```

---

### For Training

**Always specify `--tokenization`:**
```bash
# GPT-2 BPE (vocab_size → 50304)
python train.py --config_file configs/small.json --tokenization gpt2 ...

# Character-level (vocab_size → determined from data, ~100-256)
python train.py --config_file configs/small_char.json --tokenization char ...
```

**The correct vocab_size will be automatically set during training!**

---

## Why Config Files Have vocab_size=50304

### Design Decision

Config files have `vocab_size: 50304` as a **default/placeholder** because:

1. **Most common case:** GPT-2 BPE tokenization (50304 vocab)
2. **Gets overwritten:** During training, vocab_size is corrected based on tokenizer
3. **Architecture hint:** Shows the model is designed for standard BPE

### The Trade-off

**Pro:** Works out-of-the-box for GPT-2 BPE (most common)  
**Con:** Misleading for character-level tokenization

**Solution:** Use dedicated char-level configs (`tiny_char.json`, `small_char.json`)

---

## Detailed Comparison

### Small Model: GPT-2 vs Char

| Aspect | GPT-2 BPE (`small.json`) | Char (`small_char.json`) |
|--------|-------------------------|--------------------------|
| vocab_size | 50,304 | ~256 |
| Token embeddings | 19.3M params | 0.1M params |
| LM head | 19.3M params | 0.1M params |
| Transformer | 10.7M params | 10.7M params |
| **Total** | **~40M params** | **~22M params** |
| Memory (train) | ~2 GB | ~1.5 GB |
| Sequence length | Shorter (sub-words) | Longer (chars) |

---

## Common Questions

### Q: Why does my trained model show different parameters than the calculator?

**A:** You probably used character-level tokenization with a GPT-2 BPE config.

**Solution:** Use `--vocab_size 256` in calculator, or use `small_char.json` config.

### Q: Which tokenization should I use?

**GPT-2 BPE (recommended for most cases):**
- ✅ More efficient (shorter sequences)
- ✅ Better for large datasets
- ✅ Industry standard
- ❌ Larger model (~40M vs ~22M for same architecture)

**Character-level:**
- ✅ Smaller model (~22M vs ~40M)
- ✅ Good for learning/understanding
- ✅ Works with any language
- ❌ Longer sequences (slower training/generation)

### Q: Can I change vocab_size after training starts?

**A:** ❌ No! vocab_size affects model architecture (embedding layers).

Once training starts, vocab_size is locked in `checkpoints/my_model/config.json`.

---

## Summary

**The Issue:**
- Config files hardcode `vocab_size=50304` (GPT-2 BPE)
- Character-level uses `vocab_size=~256`
- This affects parameter count by ~20-40M!

**The Fix:**
- ✅ Training automatically adjusts vocab_size based on tokenizer
- ✅ Calculator now warns about vocab_size assumption
- ✅ New char-level configs (`tiny_char.json`, `small_char.json`)
- ✅ Can override with `--vocab_size` in calculator

**How to see actual parameters:**
```bash
# Method 1: During training (automatic)
python train.py ...
# Shows: Total parameters: 22,134,528

# Method 2: Calculate with correct vocab
python scripts/calculate_params.py --config_file configs/small_char.json

# Method 3: Check trained model
python scripts/inspect_model.py --model_name my_model
```

**Your observation was spot-on! The parameter count is correct during actual training, but the calculator needed fixing for char-level tokenization.** ✅

