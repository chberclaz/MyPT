# Migration to JSON-Based Checkpoints - Summary

## What Changed?

Checkpoints are now stored as **separate files** instead of one large `.pt` file. This makes them more robust and flexible.

---

## Before vs After

### Old Format (Legacy)

```
checkpoints/dante/
└── latest.pt          # Everything bundled together (150 MB)
    ├─ Model weights
    ├─ Config
    ├─ Tokenizer
    ├─ Step
    └─ Optimizer state
```

**Problem**: Changing `GPTConfig` (adding fields, changing defaults) breaks old checkpoints!

### New Format (JSON-Based)

```
checkpoints/dante/
├── model.pt              # Model weights only (50 MB)
├── config.json           # Architecture config (< 1 KB)
├── tokenizer.json        # Vocabulary (< 1 KB)
├── training_state.json   # Step, metadata (< 1 KB)
└── optimizer.pt          # Optimizer state (100 MB, optional)
```

**Benefits**:

- ✅ **Robust**: Config changes don't break checkpoints
- ✅ **Inspectable**: View config without loading model
- ✅ **Flexible**: Can edit JSON files manually
- ✅ **Smaller**: For inference, skip optimizer (50 MB vs 150 MB)

---

## What You Need to Do

### Short Answer: **Nothing!**

The system automatically handles both formats. Your old checkpoints work perfectly.

### Long Answer: **Optional Migration**

You can convert old checkpoints to new format for better organization:

**Option 1: Automatic (recommended)**

```bash
# Just resume training - it converts automatically
python train.py --model_name dante --max_iters 1100
```

**Option 2: Bulk conversion**

```bash
# Convert all legacy checkpoints at once
python convert_legacy_checkpoints.py --all

# Or convert specific model
python convert_legacy_checkpoints.py --model_name dante
```

---

## Example: Training with New Format

```bash
python train.py --model_name my_model --input_file input.txt --max_iters 1000
```

**Output** (every eval_interval):

```
Saved model weights to checkpoints/my_model/model.pt
Saved config to checkpoints/my_model/config.json
Saved tokenizer to checkpoints/my_model/tokenizer.json
Saved optimizer state to checkpoints/my_model/optimizer.pt
Saved training state to checkpoints/my_model/training_state.json
```

**Result**:

```
checkpoints/my_model/
├── model.pt              ✨ Weights only
├── config.json           ✨ Human-readable
├── tokenizer.json        ✨ Easy to inspect
├── training_state.json   ✨ Metadata
└── optimizer.pt          ✨ For resuming
```

---

## Example: Viewing Config Without Python

**New format** (JSON):

```bash
# Windows
type checkpoints\dante\config.json

# Output:
# {
#   "batch_size": 32,
#   "block_size": 256,
#   "vocab_size": 50304,
#   "n_embd": 384,
#   "n_head": 6,
#   "n_layer": 6,
#   "dropout": 0.2,
#   "bias": false,
#   "device": "cuda"
# }
```

**Old format** (legacy):

```bash
# Can't easily view - need Python to load the .pt file
```

---

## Backwards Compatibility

### Loading Checkpoints

**The system auto-detects format:**

```python
from core.checkpoint import CheckpointManager

# Works with both old and new formats!
model = CheckpointManager.load_for_inference("dante")
```

**Detection logic:**

1. Check for `config.json` + `model.pt` → Load new format
2. Otherwise, check for `latest.pt` or `final.pt` → Load legacy format
3. If neither found → Error

### Training/Generation

**All commands work the same:**

```bash
# Training
python train.py --model_name dante --input_file input_dante.txt

# Generation
python generate.py --model_name dante --prompt "Hello"

# Inspection
python inspect_model.py --model_name dante
```

No changes needed to your workflow!

---

## Technical Details

### Config Robustness

**Old way** (brittle):

```python
# If you add a new field to GPTConfig:
@dataclass
class GPTConfig:
    batch_size: int = 32
    new_field: int = 42  # ← Added this

# Old checkpoints break because they don't have "new_field"!
checkpoint = torch.load("old_checkpoint.pt")
config = GPTConfig(**checkpoint["config"])  # ❌ Error: missing new_field
```

**New way** (robust):

```python
# Config is in JSON, separate from code:
{
  "batch_size": 32,
  "block_size": 256,
  ...
}

# When loading:
config = GPTConfig.load_json("config.json")
# If new_field is not in JSON, uses default value (42) ✅
```

### Saving Process

**Old**: Single method bundled everything

```python
model.save("checkpoint.pt", step=100, optimizer_state=optimizer.state_dict())
# Saved: weights + config + tokenizer + step + optimizer in ONE file
```

**New**: Modular saving

```python
model.save_checkpoint_bundle(
    checkpoint_dir="checkpoints/dante",
    step=100,
    optimizer_state=optimizer.state_dict()
)
# Saved to SEPARATE files:
# - model.pt (weights)
# - config.json (config)
# - tokenizer.json (tokenizer)
# - training_state.json (step)
# - optimizer.pt (optimizer)
```

### Loading Process

**Old**: Load everything from one file

```python
checkpoint = torch.load("checkpoint.pt")
config = GPTConfig(**checkpoint["config"])
model = GPT(config)
model.load_state_dict(checkpoint["model_state_dict"])
# ... extract tokenizer, step, optimizer ...
```

**New**: Load from directory

```python
# Load config first
config = GPTConfig.load_json("checkpoints/dante/config.json")

# Create model with config
model = GPT(config)

# Load weights
model.load_state_dict(torch.load("checkpoints/dante/model.pt"))

# Load tokenizer (optional)
tokenizer_state = json.load(open("checkpoints/dante/tokenizer.json"))

# Load training state (optional)
training_state = json.load(open("checkpoints/dante/training_state.json"))
```

---

## Code Changes Summary

### Files Modified

1. **`core/model.py`**

   - Added: `GPTConfig.save_json()`, `GPTConfig.load_json()`
   - Changed: `GPT.save()` → Now saves only weights
   - Added: `GPT.save_checkpoint_bundle()` → Saves everything separately
   - Changed: `GPT.load()` → Now loads from directory
   - Added: `GPT.load_legacy()` → For backwards compatibility

2. **`core/checkpoint.py`**

   - Updated: `CheckpointManager.exists()` → Checks both formats
   - Updated: `CheckpointManager.initialize_for_training()` → Handles both formats
   - Updated: `CheckpointManager.load_for_inference()` → Auto-detects format

3. **`train.py`**

   - No API changes! Uses new `save_checkpoint_bundle()` internally

4. **`generate.py`**

   - No API changes! Auto-detects format

5. **`inspect_model.py`**

   - Enhanced to show which format is being used

6. **New files**:
   - `CHECKPOINT_FORMAT.md` - Comprehensive documentation
   - `convert_legacy_checkpoints.py` - Migration script
   - `JSON_CHECKPOINT_MIGRATION.md` - This file

---

## Migration Checklist

- [ ] **Read** `CHECKPOINT_FORMAT.md` for details
- [ ] **Test** loading your existing models (should work automatically)
- [ ] **Optional**: Convert legacy checkpoints using:
  ```bash
  python convert_legacy_checkpoints.py --all
  ```
- [ ] **Optional**: Verify new format:
  ```bash
  python inspect_model.py --model_name your_model
  ```
- [ ] **Optional**: Test training/generation still works
- [ ] **Done!** New checkpoints will use JSON format automatically

---

## FAQ

**Q: Will my old checkpoints stop working?**  
A: No! Both formats are fully supported.

**Q: Do I need to convert my checkpoints?**  
A: No, but it's recommended for better organization and robustness.

**Q: Can I mix formats?**  
A: Yes! You can have some models in old format and some in new format.

**Q: What if conversion fails?**  
A: Your original checkpoint is never modified. Conversion creates new files.

**Q: Can I delete old .pt files after conversion?**  
A: Yes, but keep a backup first. The script asks before deleting.

**Q: Will this affect my workflow?**  
A: No! All commands (`train.py`, `generate.py`, etc.) work exactly the same.

**Q: Can I edit config.json manually?**  
A: Yes, but be careful! Only change `batch_size`, `device`, or `dropout`. Don't change architecture params like `n_layer` or `n_embd` (weights won't match).

**Q: Is the new format compatible with PyTorch?**  
A: Yes! `model.pt` is a standard PyTorch `state_dict`.

---

## Summary

**What changed:**

- Checkpoints are now separate files (model.pt + config.json + tokenizer.json + etc.)

**Why it's better:**

- Config changes don't break old checkpoints
- Human-readable JSON files
- Smaller download size (skip optimizer for inference)

**What you need to do:**

- Nothing! Old checkpoints work automatically
- Optionally convert for better organization

**Next steps:**

1. Test that your models still load: `python generate.py --model_name your_model`
2. Optionally convert: `python convert_legacy_checkpoints.py --all`
3. Continue using MyPT as normal!

---

## Need Help?

See `CHECKPOINT_FORMAT.md` for detailed documentation, or ask for help!
