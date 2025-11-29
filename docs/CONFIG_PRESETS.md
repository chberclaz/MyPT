# Configuration Presets System

## Overview

MyPT now supports **JSON-based configuration presets** for easy model creation. Instead of specifying individual architecture parameters every time, you can use predefined configs for different model sizes.

---

## The Problem This Solves

### Before (Hard to Remember)

```bash
# What were those exact parameters for 150M?
python train.py --model_name dante \
                --n_layer 16 \
                --n_head 12 \
                --n_embd 768 \
                --block_size 256 \
                --batch_size 32 \
                --dropout 0.2 \
                ...
```

**Issues:**

- ❌ Hard to remember exact parameter combinations
- ❌ Easy to make mistakes
- ❌ Difficult to reproduce results
- ❌ Can't easily compare different sizes

### After (Easy to Use)

```bash
# Clear, memorable, reproducible
python train.py --config_file configs/150M.json \
                --model_name dante \
                --input_file input.txt
```

**Benefits:**

- ✅ Easy to remember: just the config file name
- ✅ No mistakes: tested parameter combinations
- ✅ Reproducible: same config every time
- ✅ Easy to compare: swap config file

---

## Available Presets

### Quick Reference Table

| Config       | Parameters | Layers | Heads | Embed | Block Size | Best For    |
| ------------ | ---------- | ------ | ----- | ----- | ---------- | ----------- |
| `tiny.json`  | ~11M       | 4      | 4     | 192   | 128        | Quick tests |
| `small.json` | ~40M       | 6      | 6     | 384   | 256        | Development |
| `150M.json`  | ~150M      | 16     | 12    | 768   | 256        | Production  |
| `200M.json`  | ~200M      | 16     | 14    | 896   | 256        | Production  |
| `250M.json`  | ~250M      | 16     | 16    | 1024  | 256        | Production  |

---

## Usage Examples

### 1. View All Available Configs

```bash
python scripts/show_configs.py
```

**Output:**

```
======================================================================
Available Model Configurations
======================================================================

File                 Name            Params       Layers   Heads    Embed
-------------------- --------------- ------------ -------- -------- --------
150M.json            GPT-150M        150.26M      16       12       768
200M.json            GPT-200M        199.85M      16       14       896
250M.json            GPT-250M        251.66M      16       16       1024
small.json           GPT-Small       40.98M       6        6        384
tiny.json            GPT-Tiny        10.74M       4        4        192

======================================================================

To use a config:
  python train.py --config_file configs/<filename> --model_name <name> --input_file <data>

To view details:
  python scripts/show_configs.py --config_file configs/<filename>
======================================================================
```

---

### 2. View Specific Config Details

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

### 3. Train with a Config Preset

```bash
python train.py --config_file configs/150M.json \
                --model_name dante_150M \
                --input_file input_dante.txt \
                --max_iters 5000
```

**What happens:**

1. Loads architecture from `configs/150M.json`
2. Uses `n_layer=16, n_head=12, n_embd=768` from config
3. Uses `max_iters=5000` from CLI argument
4. Trains the model

---

### 4. Compare Different Sizes

```bash
# Train tiny model for testing
python train.py --config_file configs/tiny.json \
                --model_name dante_tiny \
                --input_file input.txt \
                --max_iters 1000

# Train production model
python train.py --config_file configs/150M.json \
                --model_name dante_150M \
                --input_file input.txt \
                --max_iters 10000

# Compare quality
python generate.py --model_name dante_tiny --prompt "Test"
python generate.py --model_name dante_150M --prompt "Test"
```

---

## Config File Format

### Structure

```json
{
  "name": "GPT-150M",
  "description": "150 million parameter GPT model",
  "batch_size": 32,
  "block_size": 256,
  "vocab_size": 50304,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 16,
  "dropout": 0.2,
  "bias": false
}
```

### Required Fields

| Field        | Type  | Description               | Example |
| ------------ | ----- | ------------------------- | ------- |
| `batch_size` | int   | Batch size for training   | `32`    |
| `block_size` | int   | Context length            | `256`   |
| `vocab_size` | int   | Vocabulary size           | `50304` |
| `n_embd`     | int   | Embedding dimension       | `768`   |
| `n_head`     | int   | Number of attention heads | `12`    |
| `n_layer`    | int   | Number of layers          | `16`    |
| `dropout`    | float | Dropout rate              | `0.2`   |
| `bias`       | bool  | Use bias in layers        | `false` |

**Constraint:** `n_embd` must be divisible by `n_head`!

### Optional Fields

| Field         | Type   | Description                       | Example         |
| ------------- | ------ | --------------------------------- | --------------- |
| `name`        | string | Display name                      | `"GPT-150M"`    |
| `description` | string | Description                       | `"150M params"` |
| `device`      | string | Device (auto-detected if omitted) | `"cuda"`        |

---

## Creating Custom Configs

### Method 1: Copy and Modify

```bash
# Copy existing config
cp configs/small.json configs/my_custom.json

# Edit my_custom.json
# Change parameters as needed
```

### Method 2: Create from Scratch

**File: `configs/my_custom.json`**

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

**Test it:**

```bash
# Check parameter count
python scripts/show_configs.py --config_file configs/my_custom.json

# Train with it
python train.py --config_file configs/my_custom.json \
                --model_name test \
                --input_file input.txt
```

---

## Precedence Rules

### Config File vs CLI Arguments

When using `--config_file`:

**Architecture parameters** (from config):

- `n_layer`, `n_head`, `n_embd`, `block_size`, `batch_size`, `dropout`, `bias`
- **CLI arguments ignored** if config file is specified

**Training parameters** (from CLI):

- `max_iters`, `eval_interval`, `eval_iters`, `learning_rate`
- **CLI arguments used** even with config file

### Example

```bash
python train.py --config_file configs/150M.json \
                --n_layer 8 \           # ❌ IGNORED (uses 16 from config)
                --max_iters 5000 \      # ✅ USED
                --learning_rate 1e-4    # ✅ USED
```

**Why?** Config file defines a complete, tested architecture. Training params can vary per run.

---

## Best Practices

### 1. Start Small, Scale Up

```bash
# Phase 1: Test with tiny (fast iteration)
python train.py --config_file configs/tiny.json \
                --model_name test \
                --max_iters 100

# Phase 2: Verify with small
python train.py --config_file configs/small.json \
                --model_name test \
                --max_iters 1000

# Phase 3: Production training
python train.py --config_file configs/150M.json \
                --model_name production \
                --max_iters 10000
```

### 2. Name Your Models by Config

```bash
# Clear naming convention
python train.py --config_file configs/150M.json --model_name dante_150M --input_file input.txt
python train.py --config_file configs/250M.json --model_name dante_250M --input_file input.txt

# Now you remember what size each model is!
```

### 3. Monitor GPU Memory

```bash
# Check VRAM before training
nvidia-smi

# 8 GB VRAM  → Use up to 150M.json
# 16 GB VRAM → Use up to 250M.json
# 24 GB VRAM → Create custom larger configs
```

If out of memory:

- Reduce `batch_size` in config
- Use a smaller config
- Enable gradient checkpointing (future feature)

### 4. Version Control Your Custom Configs

```bash
# Commit custom configs to git
git add configs/my_custom.json
git commit -m "Add custom 100M config for code generation"

# Now you can reproduce this model any time!
```

---

## Parameter Count Estimation

### Formula

```python
# Approximate calculation
params ≈ 12 × n_layer × n_embd² + vocab_size × n_embd

# Breakdown:
# - Token embeddings: vocab_size × n_embd
# - Position embeddings: block_size × n_embd
# - Each transformer layer: ~12 × n_embd²
#   - Attention (Q, K, V, output): ~4 × n_embd²
#   - Feed-forward (2 layers, 4x expansion): ~8 × n_embd²
#   - Layer norms: negligible
# - LM head: n_embd × vocab_size
```

### Quick Reference

| n_layer | n_embd | Approximate Params |
| ------- | ------ | ------------------ |
| 4       | 192    | ~11M               |
| 6       | 384    | ~40M               |
| 12      | 512    | ~85M               |
| 16      | 768    | ~150M              |
| 16      | 896    | ~200M              |
| 16      | 1024   | ~250M              |
| 24      | 1024   | ~350M              |
| 32      | 1280   | ~600M              |

---

## Common Patterns

### Pattern 1: Size Series

Train the same dataset with different sizes to find the best quality/speed tradeoff:

```bash
python train.py --config_file configs/tiny.json --model_name dante_tiny --input_file input.txt --max_iters 1000
python train.py --config_file configs/small.json --model_name dante_small --input_file input.txt --max_iters 1000
python train.py --config_file configs/150M.json --model_name dante_150M --input_file input.txt --max_iters 1000

# Compare
python generate.py --model_name dante_tiny --prompt "Test"
python generate.py --model_name dante_small --prompt "Test"
python generate.py --model_name dante_150M --prompt "Test"
```

### Pattern 2: Dataset-Specific Configs

Create configs optimized for specific datasets:

**`configs/code.json`** (for code generation):

```json
{
  "name": "GPT-Code",
  "description": "Optimized for code generation",
  "block_size": 512,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 16,
  "dropout": 0.1,
  "bias": false
}
```

**`configs/shakespeare.json`** (for Shakespeare):

```json
{
  "name": "GPT-Shakespeare",
  "description": "Optimized for Shakespeare text",
  "block_size": 256,
  "n_embd": 384,
  "n_head": 6,
  "n_layer": 8,
  "dropout": 0.2,
  "bias": false
}
```

### Pattern 3: Experimentation Series

Test architecture variations:

```bash
# More layers, smaller embedding
python train.py --config_file configs/deep_narrow.json ...

# Fewer layers, larger embedding
python train.py --config_file configs/shallow_wide.json ...

# Balanced
python train.py --config_file configs/balanced.json ...
```

---

## Integration with Checkpoint System

Configs are automatically saved with your trained model:

```
checkpoints/dante_150M/
├── model.pt                  # Weights
├── config.json              # ← Architecture saved here
├── tokenizer.json           # Tokenizer
├── training_state.json      # Training progress
└── optimizer.pt             # Optimizer state
```

When you resume training, the saved `config.json` is loaded (not the config file!):

```bash
# First training
python train.py --config_file configs/150M.json --model_name dante --input_file input.txt

# Resume - uses saved config.json, NOT configs/150M.json
python train.py --model_name dante --max_iters 2000
```

**Important:** Once training starts, the architecture is locked in `checkpoints/dante/config.json`.

---

## Summary

**Configuration presets provide:**

✅ **Easy to remember:** `configs/150M.json` instead of 8 parameters  
✅ **Tested combinations:** Pre-validated architectures  
✅ **Reproducible:** Same config every time  
✅ **Flexible:** Create custom configs easily  
✅ **Clear:** See parameter counts with `show_configs.py`

**Usage:**

```bash
# View configs
python scripts/show_configs.py

# Use a config
python train.py --config_file configs/150M.json --model_name my_model --input_file data.txt
```

**Never forget model parameters again!** 🎯
