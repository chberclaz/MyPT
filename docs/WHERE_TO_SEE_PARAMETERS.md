# Where to See Model Parameter Counts

## Quick Answer

**During training** - Parameter count is automatically shown:

```bash
python train.py --config_file configs/small.json --model_name test --input_file input.txt
```

**Output:**

```
========== Model Configuration ==========
Architecture: 6 layers × 6 heads
Embedding dimension: 384
Context length: 256
Vocabulary size: 50304           ← Actual vocab used!
Total parameters: 40,982,784     ← Exact parameter count!
Device: cuda
```

---

## All Methods to See Parameters

### 1. **During Training (Automatic) ✅ RECOMMENDED**

**Shows: Actual parameter count with correct vocab_size**

```bash
python train.py --config_file configs/small.json --model_name my_model --input_file input.txt
```

**Output location:** After model initialization, before training starts

```
========== Model Configuration ==========
...
Total parameters: 40,982,784     ← HERE!
```

**Why this is best:**

- ✅ Shows **actual** parameter count
- ✅ Uses **correct** vocab_size (adjusted for char-level)
- ✅ No extra commands needed
- ✅ Always accurate

---

### 2. **Inspect Trained Model**

**Shows: Actual parameter count of saved model**

```bash
python scripts/inspect_model.py --model_name my_model
```

**Output:**

```
=== CONFIG ===
vocab_size: 256              ← Actual vocab (e.g., char-level)

=== MODEL INFO ===
Total parameters: 22,134,528  ← Actual count!
```

---

### 3. **View All Configs**

**Shows: Approximate counts for all presets**

```bash
python scripts/show_configs.py
```

**Output:**

```
File                 Name            Params       Layers   Heads    Embed
-------------------- --------------- ------------ -------- -------- --------
150M.json            GPT-150M        150.26M      16       12       768
small.json           GPT-Small       40.98M       6        6        384
small_char.json      GPT-Small-Char  22.13M       6        6        384  ← Char version
tiny.json            GPT-Tiny        10.74M       4        4        192
tiny_char.json       GPT-Tiny-Char   2.98M        4        4        192  ← Char version
```

---

### 4. **Calculate from Config File (With Warning)**

**Shows: Detailed breakdown, but may assume wrong vocab_size**

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

**For char-level, override:**

```bash
python scripts/calculate_params.py --config_file configs/small.json --vocab_size 256
```

**Or use char config:**

```bash
python scripts/calculate_params.py --config_file configs/small_char.json
```

---

### 5. **Calculate from Parameters**

**Shows: Detailed breakdown with custom values**

```bash
python scripts/calculate_params.py \
    --n_layer 6 \
    --n_embd 384 \
    --n_head 6 \
    --vocab_size 256 \  # ← Specify char-level vocab
    --block_size 256
```

**Output:** Full breakdown with correct vocab_size

---

### 6. **Interactive Calculator**

**Shows: Guided calculation with prompts**

```bash
python scripts/calculate_params.py --interactive
```

**Prompts:**

```
Number of layers (e.g., 16): 6
Embedding dimension (e.g., 768): 384
Number of attention heads (e.g., 12): 6
Vocabulary size (50304 for GPT-2, or custom): 256  ← Enter char vocab
Block size / context length (e.g., 256 or 1024): 256
Use bias? (y/n, default n): n

[Shows full breakdown with vocab_size=256]
```

---

### 7. **Programmatically (Python API)**

**Shows: Parameter count in your own code**

```python
from core import GPTConfig, GPT

# Create config
config = GPTConfig(
    n_layer=6,
    n_embd=384,
    n_head=6,
    vocab_size=256,  # Char-level
)

# Create model
model = GPT(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total: {total_params:,}")
# Output: Total: 22,134,528
```

---

## Important: vocab_size Correction During Training

### Config File Says One Thing

**`configs/small.json`:**

```json
{
  "vocab_size": 50304,  ← Config file value
  ...
}
```

### Training Corrects It

```bash
python train.py \
    --config_file configs/small.json \
    --tokenization char \  ← Char-level specified
    --input_file input.txt
```

**What happens:**

```python
# In core/checkpoint.py:
if tokenization == 'char':
    tokenizer.build_char_vocab(input_text)
    config.vocab_size = len(tokenizer.chars)  # ← Overwritten! (~256)
else:
    config.vocab_size = 50304  # GPT-2 BPE
```

**Result:** Model uses **correct** vocab_size regardless of config file!

---

## Comparison Table

| Method                           | Shows Actual?           | When to Use                             |
| -------------------------------- | ----------------------- | --------------------------------------- |
| **During training**              | ✅ YES (corrected)      | Always (automatic)                      |
| **inspect_model.py**             | ✅ YES (saved)          | Check trained model                     |
| **show_configs.py**              | ⚠️ Approximate          | Compare presets                         |
| **calculate_params.py (config)** | ⚠️ From config          | Before training (may be wrong for char) |
| **calculate_params.py (manual)** | ✅ YES (if you specify) | Custom calculation                      |
| **Interactive**                  | ✅ YES (if you specify) | Guided calculation                      |
| **Python API**                   | ✅ YES (if you specify) | Programmatic                            |

---

## Best Practices

### 1. Always Check During Training

The most reliable parameter count is shown **during training**:

```bash
python train.py ...

# Look for:
# Total parameters: 22,134,528  ← This is THE actual count!
```

---

### 2. Use Correct Configs for Tokenization

**GPT-2 BPE:** Use standard configs

```bash
python train.py --config_file configs/small.json --tokenization gpt2 ...
```

**Character-level:** Use char configs

```bash
python train.py --config_file configs/small_char.json --tokenization char ...
```

---

### 3. Override vocab_size in Calculator for Char

```bash
# Wrong (assumes GPT-2 vocab)
python scripts/calculate_params.py --config_file configs/small.json

# Right (overrides to char vocab)
python scripts/calculate_params.py --config_file configs/small.json --vocab_size 256
```

---

### 4. Inspect Saved Models

After training, check the actual count:

```bash
python scripts/inspect_model.py --model_name my_model
```

Shows the **exact** parameter count as saved.

---

## Examples

### Example 1: Small Model with GPT-2 BPE

```bash
python train.py --config_file configs/small.json --tokenization gpt2 --input_file input.txt
```

**Shows:**

```
Vocabulary size: 50304
Total parameters: 40,982,784  (~41M)
```

---

### Example 2: Small Model with Char-Level

```bash
python train.py --config_file configs/small.json --tokenization char --input_file input.txt
```

**Shows:**

```
Vocabulary size: 95              ← Corrected from 50304!
Total parameters: 22,134,528     ← Correct count!
```

**Even though config says 50304, training corrects it to 95!**

---

### Example 3: Calculate Before Training

**For GPT-2 BPE:**

```bash
python scripts/calculate_params.py --config_file configs/small.json
# Total: 40,982,784 (correct for GPT-2)
```

**For char-level:**

```bash
python scripts/calculate_params.py --config_file configs/small_char.json
# Total: 22,134,528 (correct for char)

# Or override:
python scripts/calculate_params.py --config_file configs/small.json --vocab_size 256
# Total: 22,134,528 (correct for char)
```

---

## Summary

**Where parameter count is shown:**

| Location            | How to See It                                                         |
| ------------------- | --------------------------------------------------------------------- |
| **Training output** | Automatic when you run `train.py`                                     |
| **Inspect script**  | `python scripts/inspect_model.py --model_name my_model`               |
| **Config viewer**   | `python scripts/show_configs.py`                                      |
| **Calculator**      | `python scripts/calculate_params.py --config_file configs/small.json` |
| **Python code**     | `sum(p.numel() for p in model.parameters())`                          |

**Most reliable:** Training output (always shows correct count)
