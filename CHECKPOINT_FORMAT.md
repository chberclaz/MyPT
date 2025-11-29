# Checkpoint Format - JSON-Based Structure

## Overview

**Important Change**: Checkpoints are now stored as separate files instead of a single `.pt` file. This makes them more flexible, easier to inspect, and resilient to config changes.

---

## New Checkpoint Structure

```
checkpoints/dante/
├── model.pt              # Model weights only (PyTorch state_dict)
├── config.json           # Model architecture configuration
├── tokenizer.json        # Tokenizer state and vocabulary
├── training_state.json   # Training progress metadata
└── optimizer.pt          # Optimizer state (for resuming training)
```

### Why Separate Files?

**Problem with old format:**
- Single `.pt` file contained everything bundled together
- Adding new fields to `GPTConfig` broke old checkpoints
- Hard to inspect config without loading entire model
- Config changes required re-training from scratch

**Benefits of new format:**
- ✅ **Robust**: Adding new config fields doesn't break old checkpoints
- ✅ **Inspectable**: Can view config/tokenizer without loading weights
- ✅ **Flexible**: Can modify config.json manually if needed
- ✅ **Standard**: Follows patterns used by Hugging Face, etc.
- ✅ **Smaller**: Weights file is ~2x smaller (no embedded config/optimizer)

---

## File Details

### 1. `model.pt` - Model Weights Only

Contains: PyTorch `state_dict` (just the neural network parameters)

**Size**: ~10-100 MB depending on model size

```python
# What's inside:
torch.load("model.pt")
# Returns:
{
    'token_embedding_table.weight': tensor([[...]]),
    'position_embedding_table.weight': tensor([[...]]),
    'blocks.0.sa.heads.0.key.weight': tensor([[...]]),
    # ... all layer weights ...
}
```

**No config, no tokenizer, no training state** - just pure weights!

---

### 2. `config.json` - Architecture Configuration

Contains: All hyperparameters defining the model architecture

**Size**: < 1 KB

```json
{
  "batch_size": 32,
  "block_size": 256,
  "vocab_size": 50304,
  "n_embd": 384,
  "n_head": 6,
  "n_layer": 6,
  "dropout": 0.2,
  "bias": false,
  "device": "cuda"
}
```

**Human-readable and editable!** You can:
- Inspect architecture without Python
- Share config separately
- Compare configs across models
- Document model architecture

---

### 3. `tokenizer.json` - Vocabulary and Tokenizer State

Contains: Tokenizer type and vocabulary

**Size**: Varies (GPT-2: < 1 KB, char-level: depends on corpus)

**For GPT-2 BPE:**
```json
{
  "token_kind": "gpt2"
}
```

**For char-level:**
```json
{
  "token_kind": "char",
  "chars": ["\n", " ", "!", "\"", "$", "&", "'", ",", "-", ".", "0", "1", ...]
}
```

**Critical for correctness**: The vocabulary must match exactly what the model was trained on!

---

### 4. `training_state.json` - Training Progress

Contains: Metadata about training progress

**Size**: < 1 KB

```json
{
  "step": 1000,
  "optimizer_file": "optimizer.pt"
}
```

**Optional**: Only needed if you want to resume training.

---

### 5. `optimizer.pt` - Optimizer State (Optional)

Contains: Adam optimizer momentum, learning rate schedule, etc.

**Size**: ~2x model size (stores momentum for each parameter)

**Only needed for resuming training** - not needed for inference!

---

## Usage

### Saving Checkpoints

**During training** (automatic):
```python
model.fit(
    data_loader=data_loader,
    optimizer=optimizer,
    max_iters=1000,
    checkpoint_dir="checkpoints/dante"  # All files saved here
)
```

**Manual save** (if needed):
```python
# Save everything
model.save_checkpoint_bundle(
    checkpoint_dir="checkpoints/dante",
    step=500,
    optimizer_state=optimizer.state_dict()
)

# Or save just the model weights
model.save("checkpoints/dante")  # Saves only model.pt
```

---

### Loading Checkpoints

**For inference** (automatic format detection):
```python
from core.checkpoint import CheckpointManager

# Automatically detects new JSON format or legacy format
model = CheckpointManager.load_for_inference("dante")
```

**For training** (automatic resume/init/fresh):
```python
from core.checkpoint import CheckpointManager

ckpt_manager = CheckpointManager("dante")
model, optimizer, start_step = ckpt_manager.initialize_for_training(
    config, "gpt2", text, learning_rate=3e-4
)

# If checkpoint exists:
#   - Loads from checkpoint
#   - Returns start_step to continue from
# If not:
#   - Creates fresh model
#   - Returns start_step=0
```

---

## Backwards Compatibility

**Your old checkpoints still work!**

The system automatically detects the format:

```python
# Tries new format first
if exists("checkpoints/dante/config.json"):
    load_new_format()

# Falls back to legacy
elif exists("checkpoints/dante/latest.pt"):
    load_legacy_format()
```

**Migration**: Old checkpoints will be **automatically converted** to new format on next save:
1. Run training with `--model_name dante`
2. System loads old `latest.pt` (legacy format)
3. On next checkpoint save, uses new format
4. Old `.pt` file is preserved (safe!)

---

## Migration Guide

### Option 1: Automatic (Recommended)

Just resume training - the system handles everything:

```bash
# Your old checkpoint: checkpoints/dante/latest.pt
python train.py --model_name dante --max_iters 1100

# System:
# 1. Detects legacy format
# 2. Loads from latest.pt
# 3. Saves next checkpoint in new format
# 4. Both formats coexist safely
```

### Option 2: Manual Conversion Script

Convert all your old checkpoints at once:

```python
# convert_checkpoints.py
import os
import torch
from core.checkpoint import CheckpointManager
from core.model import GPT

def convert_checkpoint(model_name, legacy_file="latest.pt"):
    """Convert a legacy checkpoint to new JSON format"""
    ckpt_manager = CheckpointManager(model_name)
    
    # Load legacy format
    legacy_path = ckpt_manager.get_path(legacy_file)
    if not os.path.exists(legacy_path):
        print(f"No legacy checkpoint found: {legacy_path}")
        return
    
    print(f"Converting {model_name}/{legacy_file}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, step, optim_state = GPT.load_legacy(legacy_path, map_location=device)
    
    # Save in new format
    model.save_checkpoint_bundle(
        ckpt_manager.checkpoint_dir,
        step=step,
        optimizer_state=optim_state
    )
    
    print(f"✅ Converted to new format at {ckpt_manager.checkpoint_dir}")

# Convert all your models
convert_checkpoint("dante", "final.pt")
convert_checkpoint("dante", "latest.pt")
convert_checkpoint("shakespeare", "final.pt")
```

### Option 3: Keep Using Legacy

The system supports both formats indefinitely. You can specify a legacy checkpoint:

```bash
# Load specific legacy file for generation
python generate.py --model_name dante --legacy_checkpoint final.pt

# Inspect legacy checkpoint
python inspect_model.py --model_name dante --legacy_checkpoint final.pt
```

---

## Inspecting Checkpoints

### New Format (JSON)

**View config without Python:**
```bash
# Windows
type checkpoints\dante\config.json

# Linux/Mac
cat checkpoints/dante/config.json
```

**View tokenizer:**
```bash
type checkpoints\dante\tokenizer.json
```

**View training state:**
```bash
type checkpoints\dante\training_state.json
```

### Using inspect_model.py

```bash
# Auto-detects format
python inspect_model.py --model_name dante

# Output:
# === FORMAT ===
# New JSON-based format
# Location: checkpoints/dante
#
# === CONFIG ===
# batch_size:32, block_size:256, vocab_size:50304, ...
#
# === TOKENIZER ===
# kind      : gpt2
# vocab     : GPT-2 BPE (fixed ~50k tokens)
#
# === TRAINING STATE ===
# Last recorded training step: 1000
```

---

## File Size Comparison

### Example Model (6 layers, 384 embedding)

**Old format (single .pt file):**
```
checkpoints/dante/latest.pt:  ~150 MB
  ├─ Model weights:            50 MB
  ├─ Optimizer state:          100 MB
  └─ Config + tokenizer:       < 1 MB
```

**New format (separate files):**
```
checkpoints/dante/
  ├─ model.pt:                 50 MB   (weights only)
  ├─ optimizer.pt:             100 MB  (optimizer only)
  ├─ config.json:              < 1 KB
  ├─ tokenizer.json:           < 1 KB
  └─ training_state.json:      < 1 KB
Total:                         ~150 MB (same)
```

**For inference only** (don't need optimizer):
```
checkpoints/dante/
  ├─ model.pt:                 50 MB
  ├─ config.json:              < 1 KB
  └─ tokenizer.json:           < 1 KB
Total:                         ~50 MB (3x smaller!)
```

---

## FAQ

### Q: Do I need to re-train my models?

**No!** Old checkpoints work perfectly. They'll be converted automatically on next save.

### Q: Can I delete old .pt files after conversion?

**Yes**, but keep a backup first. Once you've verified the new format works:
```bash
# Backup first!
copy checkpoints\dante\latest.pt checkpoints\dante\latest.pt.backup

# Test new format
python generate.py --model_name dante

# If it works, you can delete the backup
```

### Q: Can I share just the model without optimizer?

**Yes!** Just share these files:
```
- model.pt
- config.json
- tokenizer.json
```

Skip `optimizer.pt` and `training_state.json` (only needed for continuing training).

### Q: What if I edit config.json manually?

Be careful! If you change architecture params (`n_layer`, `n_embd`, etc.), the weights won't match. Only safe to change:
- `batch_size` (doesn't affect model)
- `device` (doesn't affect model)
- `dropout` (only matters for training)

### Q: Can I use the same config for multiple models?

**Yes!** You can copy `config.json` to create a new model with the same architecture:

```bash
# Copy config from dante to shakespeare
copy checkpoints\dante\config.json my_config.json

# Create new model (edit config if needed)
python train.py --model_name shakespeare ...
# (System will use default config, but you can manually copy my_config.json)
```

### Q: How do I see what's in model.pt?

Use Python:
```python
import torch
state_dict = torch.load("checkpoints/dante/model.pt")
print(f"Number of tensors: {len(state_dict)}")
print(f"Parameter names: {list(state_dict.keys())[:5]}...")
```

Or use `inspect_model.py` which shows everything.

---

## Summary

**New checkpoint format:**
- ✅ Separate files for weights, config, tokenizer, training state
- ✅ More robust (config changes don't break checkpoints)
- ✅ Human-readable JSON files
- ✅ Fully backwards compatible with old format
- ✅ Automatic format detection and conversion

**You don't need to do anything** - the system handles both formats seamlessly!

