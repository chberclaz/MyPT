# Model Configuration Presets

This folder contains predefined model architecture configurations for different sizes. Use these to easily create models without specifying all parameters individually.

---

## Available Configurations

### Quick Reference

| Config                 | Parameters | Layers | Heads | Embedding | Block Size | Vocab | Use Case                     |
| ---------------------- | ---------- | ------ | ----- | --------- | ---------- | ----- | ---------------------------- |
| **tiny.json**          | ~11M       | 4      | 4     | 192       | 128        | GPT-2 | Quick experiments            |
| **tiny_char.json** 📝  | ~3M        | 4      | 4     | 192       | 128        | Char  | Quick char-level experiments |
| **small.json**         | ~40M       | 6      | 6     | 384       | 256        | GPT-2 | Default, testing             |
| **small_char.json** 📝 | ~22M       | 6      | 6     | 384       | 256        | Char  | Char-level, testing          |
| **150M.json**          | ~150M      | 16     | 12    | 768       | 256        | GPT-2 | Production, small            |
| **200M.json**          | ~200M      | 16     | 14    | 896       | 256        | GPT-2 | Production, medium           |
| **250M.json**          | ~250M      | 16     | 16    | 1024      | 256        | GPT-2 | Production, large            |
| **150M_1024.json** 🚀  | ~150M      | 16     | 12    | 768       | 1024       | GPT-2 | High context, powerful GPUs  |
| **200M_1024.json** 🚀  | ~200M      | 16     | 14    | 896       | 1024       | GPT-2 | High context, powerful GPUs  |
| **250M_1024.json** 🚀  | ~250M      | 16     | 16    | 1024      | 1024       | GPT-2 | High context, powerful GPUs  |
| **350M_1024.json** 🚀  | ~350M      | 24     | 16    | 1024      | 1024       | GPT-2 | High context, 16GB+ VRAM     |
| **500M_1024.json** 🚀  | ~500M      | 24     | 16    | 1280      | 1024       | GPT-2 | High context, 24GB+ VRAM     |

**Note:** Parameter counts are approximate. Character-level models (📝) use smaller vocab (~256) vs GPT-2 BPE (~50K), resulting in ~20M fewer parameters for same architecture.

---

## Standard Configs (block_size=256)

These configs use block_size=256 for balanced performance on most GPUs.

### `tiny.json` - ~11M parameters

**Best for:** Quick experiments, testing code changes

```json
{
  "name": "GPT-Tiny",
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 192,
  "block_size": 128,
  "batch_size": 16
}
```

**Training time:** ~5 minutes on GPU for 1000 iterations  
**Memory (Training):** ~500 MB  
**Memory (Generation):** ~100 MB

---

### `tiny_char.json` - ~3M parameters

**Best for:** Quick experiments with character-level tokenization

```json
{
  "name": "GPT-Tiny-Char",
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 192,
  "block_size": 128,
  "batch_size": 16,
  "vocab_size": 256
}
```

**Tokenization:** Character-level (vocab ~256)  
**Training time:** ~3 minutes on GPU for 1000 iterations  
**Memory (Training):** ~400 MB  
**Memory (Generation):** ~50 MB

**Parameter difference vs tiny.json:** ~8M fewer parameters (no large vocab embeddings)

---

### `small.json` - ~40M parameters (Default)

**Best for:** Most experiments, development

```json
{
  "name": "GPT-Small",
  "n_layer": 6,
  "n_head": 6,
  "n_embd": 384,
  "block_size": 256,
  "batch_size": 32
}
```

**Training time:** ~15 minutes on GPU for 1000 iterations  
**Memory (Training):** ~2 GB  
**Memory (Generation):** ~300 MB

---

### `small_char.json` - ~22M parameters

**Best for:** Most experiments with character-level tokenization

```json
{
  "name": "GPT-Small-Char",
  "n_layer": 6,
  "n_head": 6,
  "n_embd": 384,
  "block_size": 256,
  "batch_size": 32,
  "vocab_size": 256
}
```

**Tokenization:** Character-level (vocab ~256)  
**Training time:** ~10 minutes on GPU for 1000 iterations  
**Memory (Training):** ~1.5 GB  
**Memory (Generation):** ~200 MB

**Parameter difference vs small.json:** ~18M fewer parameters (smaller vocab)

---

### `150M.json` - ~150M parameters

**Best for:** Production models, serious training

```json
{
  "name": "GPT-150M",
  "n_layer": 16,
  "n_head": 12,
  "n_embd": 768,
  "block_size": 256,
  "batch_size": 32
}
```

**Training time:** ~1 hour on GPU for 1000 iterations  
**Memory (Training):** ~8 GB  
**Memory (Generation):** ~1 GB

---

### `200M.json` - ~200M parameters

**Best for:** Larger production models

```json
{
  "name": "GPT-200M",
  "n_layer": 16,
  "n_head": 14,
  "n_embd": 896,
  "block_size": 256,
  "batch_size": 32
}
```

**Training time:** ~1.5 hours on GPU for 1000 iterations  
**Memory (Training):** ~10 GB  
**Memory (Generation):** ~1.2 GB

---

### `250M.json` - ~250M parameters

**Best for:** Large production models

```json
{
  "name": "GPT-250M",
  "n_layer": 16,
  "n_head": 16,
  "n_embd": 1024,
  "block_size": 256,
  "batch_size": 32
}
```

**Training time:** ~2 hours on GPU for 1000 iterations  
**Memory (Training):** ~12 GB  
**Memory (Generation):** ~1.5 GB

---

## High-Context Configs (block_size=1024) 🚀

These configs use **block_size=1024** for powerful GPUs, allowing the model to see 4x more context. This enables faster learning from longer sequences but requires more VRAM.

**Benefits of 1024 context:**

- Learn from longer sequences
- Better understanding of long-range dependencies
- More efficient training (fewer gradient steps needed)
- Better for code, documents, long-form text

**Requirements:**

- Minimum 12GB VRAM (for 150M_1024)
- 16GB+ VRAM recommended (for 200M-350M_1024)
- 24GB+ VRAM for 500M_1024

---

### `150M_1024.json` - ~150M parameters

**Best for:** High-context training on 12GB+ GPUs

```json
{
  "name": "GPT-150M-1024",
  "n_layer": 16,
  "n_head": 12,
  "n_embd": 768,
  "block_size": 1024,
  "batch_size": 24
}
```

**Training time:** ~1.5 hours on GPU for 1000 iterations  
**Memory (Training):** ~10 GB  
**Memory (Generation):** ~1.5 GB

---

### `200M_1024.json` - ~200M parameters

**Best for:** High-context training on 16GB+ GPUs

```json
{
  "name": "GPT-200M-1024",
  "n_layer": 16,
  "n_head": 14,
  "n_embd": 896,
  "block_size": 1024,
  "batch_size": 20
}
```

**Training time:** ~2 hours on GPU for 1000 iterations  
**Memory (Training):** ~12 GB  
**Memory (Generation):** ~2 GB

---

### `250M_1024.json` - ~250M parameters

**Best for:** High-context training on 16GB+ GPUs

```json
{
  "name": "GPT-250M-1024",
  "n_layer": 16,
  "n_head": 16,
  "n_embd": 1024,
  "block_size": 1024,
  "batch_size": 16
}
```

**Training time:** ~2.5 hours on GPU for 1000 iterations  
**Memory (Training):** ~14 GB  
**Memory (Generation):** ~2.5 GB

---

### `350M_1024.json` - ~350M parameters

**Best for:** High-context training on 20GB+ GPUs

```json
{
  "name": "GPT-350M-1024",
  "n_layer": 24,
  "n_head": 16,
  "n_embd": 1024,
  "block_size": 1024,
  "batch_size": 12
}
```

**Training time:** ~3.5 hours on GPU for 1000 iterations  
**Memory (Training):** ~18 GB  
**Memory (Generation):** ~3 GB

---

### `500M_1024.json` - ~500M parameters

**Best for:** High-context training on 24GB+ GPUs (RTX 3090, RTX 4090, A100)

```json
{
  "name": "GPT-500M-1024",
  "n_layer": 24,
  "n_head": 16,
  "n_embd": 1280,
  "block_size": 1024,
  "batch_size": 8
}
```

**Training time:** ~5 hours on GPU for 1000 iterations  
**Memory (Training):** ~22 GB  
**Memory (Generation):** ~4 GB

---

## Usage

### Using a Config File

**Basic usage:**

```bash
python train.py --config_file configs/150M.json \
                --model_name dante_150M \
                --input_file input_dante.txt
```

**With custom training params:**

```bash
python train.py --config_file configs/200M.json \
                --model_name shakespeare_200M \
                --input_file input.txt \
                --max_iters 5000 \
                --learning_rate 1e-4
```

**Config file overrides individual architecture params**, but you can still customize training params!

---

### View All Available Configs

```bash
python scripts/show_configs.py
```

**Output:**

```
Available Model Configurations
======================================================================

File                 Name            Params       Layers   Heads    Embed
-------------------- --------------- ------------ -------- -------- --------
150M.json            GPT-150M        150.26M      16       12       768
200M.json            GPT-200M        199.85M      16       14       896
250M.json            GPT-250M        251.66M      16       16       1024
small.json           GPT-Small       40.98M       6        6        384
tiny.json            GPT-Tiny        10.74M       4        4        192
```

---

### View Specific Config Details

```bash
python scripts/show_configs.py --config_file configs/150M.json
```

**Output:**

```
======================================================================
Config: GPT-150M
File: configs/150M.json
======================================================================
Description: 150 million parameter GPT model

Architecture:
  Layers      : 16
  Heads       : 12
  Embedding   : 768
  Block size  : 256
  Vocab size  : 50304
  Dropout     : 0.2
  Bias        : False

Parameters:
  Total       : 150,260,736 (150.26M)
  Batch size  : 32

Usage:
  python train.py --config_file configs/150M.json --model_name my_model --input_file input.txt
======================================================================
```

---

## Creating Custom Configs

### Option 1: Copy and Modify

```bash
# Copy existing config
copy configs\small.json configs\my_custom.json

# Edit my_custom.json
# Change n_layer, n_embd, etc.
```

### Option 2: Create from Scratch

Create `configs/my_custom.json`:

```json
{
  "name": "GPT-Custom",
  "description": "My custom configuration",
  "batch_size": 32,
  "block_size": 512,
  "vocab_size": 50304,
  "n_embd": 512,
  "n_head": 8,
  "n_layer": 8,
  "dropout": 0.2,
  "bias": false
}
```

**Required fields:**

- `batch_size` - Batch size
- `block_size` - Context length
- `vocab_size` - Vocabulary size (50304 for GPT-2, varies for char)
- `n_embd` - Embedding dimension
- `n_head` - Number of attention heads
- `n_layer` - Number of transformer layers
- `dropout` - Dropout rate
- `bias` - Use bias in layers

**Optional fields:**

- `name` - Display name
- `description` - Description
- `device` - Device (auto-detected if not specified)

**Constraint:** `n_embd` must be divisible by `n_head`!

---

## Best Practices

### Choosing Model Size

**For learning/experimentation:**

- Use `tiny.json` or `small.json`
- Fast iteration, low memory

**For production:**

- Use `150M.json` or larger
- Better quality, but slower training

**For GPUs:**

- Check VRAM: `nvidia-smi`
- 8 GB VRAM → up to ~150M params
- 16 GB VRAM → up to ~250M params
- 24 GB VRAM → up to ~400M params

### Naming Convention

Suggested naming for your custom configs:

- `configs/<size>.json` - By parameter count (e.g., 100M.json)
- `configs/<use_case>.json` - By use case (e.g., fast.json, quality.json)
- `configs/<dataset>.json` - By dataset (e.g., dante.json, code.json)

---

## Parameter Count Guide

### How to Calculate Model Parameters

Use the dedicated calculator script:

```bash
# Calculate from config file
python scripts/calculate_params.py --config_file configs/150M_1024.json

# Calculate from parameters
python scripts/calculate_params.py --n_layer 16 --n_embd 768 --n_head 12 --block_size 1024

# Interactive mode
python scripts/calculate_params.py --interactive

# Show formula
python scripts/calculate_params.py --show_formula
```

### Formula

Approximate parameters for a GPT model:

```
Total ≈ 12 × n_layer × n_embd² + 2 × vocab_size × n_embd

Breakdown:
- Token embeddings: vocab_size × n_embd
- Position embeddings: block_size × n_embd (usually negligible)
- Each layer: ~12 × n_embd² (attention + feed-forward)
- LM head: n_embd × vocab_size
```

**Note:** block_size affects position embeddings, but this is usually a tiny fraction of total parameters.

### Examples

| Config | n_layer | n_embd | Params | Memory (training) |
| ------ | ------- | ------ | ------ | ----------------- |
| Tiny   | 4       | 192    | ~11M   | ~500 MB           |
| Small  | 6       | 384    | ~41M   | ~2 GB             |
| Medium | 12      | 512    | ~85M   | ~4 GB             |
| Large  | 16      | 768    | ~150M  | ~8 GB             |
| XL     | 24      | 1024   | ~350M  | ~16 GB            |

---

## Config vs CLI Arguments

### Precedence

When using `--config_file`:

1. **Architecture params** come from config file
2. **Training params** come from CLI arguments
3. CLI architecture params are **ignored**

**Example:**

```bash
python train.py --config_file configs/150M.json \
                --n_layer 8 \           # ❌ Ignored! (uses 16 from config)
                --max_iters 5000 \      # ✅ Used!
                --learning_rate 1e-4    # ✅ Used!
```

**Why?** Config file is meant to define a complete, tested architecture. Training params can vary.

---

## Tips

### 1. Start Small

Begin with `tiny.json` or `small.json` to verify your data and training loop work.

### 2. Scale Up Gradually

```bash
# Test with tiny
python train.py --config_file configs/tiny.json --model_name test --max_iters 100

# If it works, scale up
python train.py --config_file configs/small.json --model_name test --max_iters 1000

# Production training
python train.py --config_file configs/150M.json --model_name production --max_iters 10000
```

### 3. Monitor Memory

Watch VRAM usage:

```bash
nvidia-smi -l 1  # Update every second
```

If out of memory:

- Reduce `batch_size` in config file
- Use a smaller config
- Use gradient accumulation (future feature)

### 4. Save Your Configs

Commit your custom configs to version control so you can reproduce results later!

---

## Future Enhancements

Potential additions:

- [ ] Gradient accumulation steps in config
- [ ] Mixed precision training settings
- [ ] Learning rate schedule config
- [ ] More predefined configs (400M, 1B, etc.)
- [ ] Config validation tool
- [ ] Config optimizer (find best config for your GPU)

---

## Summary

**Model config presets:**

- ✅ Easy to use: `--config_file configs/150M.json`
- ✅ Pre-calculated sizes: 150M, 200M, 250M
- ✅ View all configs: `python scripts/show_configs.py`
- ✅ Parameter counts: Automatically calculated
- ✅ Custom configs: Easy to create

**Never forget what architecture you used again!** 🎯
