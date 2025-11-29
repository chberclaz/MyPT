# CLI Scripts Refactoring - Using Convenience Functions

## Overview

The CLI scripts (`train.py` and `generate.py`) have been refactored to use the new convenience functions from `core/__init__.py`, making them cleaner, more readable, and demonstrating best practices for using the MyPT API.

---

## Changes Summary

### `generate.py` - Simplified Model Loading

**Before:**
```python
from core.checkpoint import CheckpointManager

model = CheckpointManager.load_for_inference(
    args.model_name, 
    legacy_filename=args.legacy_checkpoint
)
```

**After:**
```python
from core import load_model, get_model_info

# Simple case (most common)
model = load_model(args.model_name)

# With legacy checkpoint (if needed)
if args.legacy_checkpoint:
    from core.checkpoint import CheckpointManager
    model = CheckpointManager.load_for_inference(
        args.model_name,
        legacy_filename=args.legacy_checkpoint
    )
```

**Benefits:**
- ✅ Cleaner imports: `from core import load_model`
- ✅ Simpler API for common case
- ✅ Fallback to detailed API when needed
- ✅ Added `--show_info` flag to preview model without loading

---

### `train.py` - Enhanced with Model Info

**Before:**
```python
from core.model import GPT, GPTConfig
from core.tokenizer import Tokenizer
from core.data_loader import GPTDataLoader
from core.checkpoint import CheckpointManager

# Multiple imports from submodules
```

**After:**
```python
from core import GPTConfig, GPTDataLoader, CheckpointManager, get_model_info

# Cleaner imports from single entry point
```

**New Features:**
1. **Pre-training model info**: Shows existing model info before resuming
2. **Better output formatting**: Prettier, more informative messages
3. **Token count estimates**: Shows approximate token counts for data
4. **Parameter count**: Displays total model parameters
5. **Next steps**: Suggests commands to run after training

---

## Detailed Changes

### `generate.py`

#### 1. Cleaner Imports

**Before:**
```python
from core.checkpoint import CheckpointManager
from generator import Generator
```

**After:**
```python
from core import load_model, get_model_info
from generator import Generator
```

#### 2. New `--show_info` Flag

Shows model information before loading (useful for quick inspection):

```bash
python generate.py --model_name dante --show_info
```

Output:
```
========== Model Info ==========
Format: json
Layers: 6
Embedding dim: 384
Vocab size: 50304
Tokenizer: gpt2
Training step: 1000
```

#### 3. Simplified Model Loading

**Before:**
```python
model = CheckpointManager.load_for_inference(
    args.model_name, 
    legacy_filename=args.legacy_checkpoint
)
```

**After:**
```python
# Default case - simple and clean
model = load_model(args.model_name)

# Legacy checkpoint case - use detailed API
if args.legacy_checkpoint:
    from core.checkpoint import CheckpointManager
    model = CheckpointManager.load_for_inference(...)
```

#### 4. Enhanced Output

**Before:**
```
Model loaded successfully!
Tokenizer: gpt2
Vocab size: 50304
```

**After:**
```
✅ Model loaded successfully!
Tokenizer: gpt2
Vocab size: 50304
Device: cuda
```

---

### `train.py`

#### 1. Cleaner Imports

**Before:**
```python
from core.model import GPT, GPTConfig
from core.tokenizer import Tokenizer
from core.data_loader import GPTDataLoader
from core.checkpoint import CheckpointManager
```

**After:**
```python
from core import GPTConfig, GPTDataLoader, CheckpointManager, get_model_info
```

Only imports what's needed, all from the clean public API.

#### 2. Pre-Training Model Detection

Now shows information about existing models before resuming:

```python
if ckpt_manager.exists():
    print("========== Existing Model Detected ==========")
    info = get_model_info(args.model_name)
    print(f"Last training step: {info['training_state']['step']}")
    print(f"Architecture: {info['config']['n_layer']} layers")
```

**Output:**
```
========== Existing Model Detected ==========
Will resume training from existing checkpoint
Format: json
Last training step: 1000
Architecture: 6 layers, 384 embd dim
```

#### 3. Enhanced Data Loading Output

**Before:**
```
Loaded 1000000 characters
```

**After:**
```
Loaded 1,000,000 characters
Approximate tokens (char-level): 1,000,000
Approximate tokens (GPT-2): ~250,000
```

#### 4. Better Model Configuration Display

**Before:**
```
Model configuration:
batch_size:32, block_size:256, vocab_size:50304, n_embd:384, ...
```

**After:**
```
========== Model Configuration ==========
Architecture: 6 layers × 6 heads
Embedding dimension: 384
Context length: 256
Vocabulary size: 50,304
Total parameters: 10,788,929
Device: cuda
Starting from step: 0
```

#### 5. Enhanced Training Output

**Before:**
```
========== Starting Training ==========
Checkpoints will be saved to: checkpoints/dante
Format: model.pt + config.json + tokenizer.json + training_state.json
```

**After:**
```
========== Starting Training ==========
Training from step 0 to 1000
Checkpoints: checkpoints/dante
Format: JSON-based (model.pt + config.json + tokenizer.json + ...)
Evaluation every 50 steps
```

#### 6. Post-Training Next Steps

**New:**
```
========== Training Complete ==========
✅ Model saved to: checkpoints/dante

Checkpoint files:
  📄 model.pt           - Model weights
  📄 config.json        - Architecture configuration
  📄 tokenizer.json     - Vocabulary
  📄 training_state.json - Training progress
  📄 optimizer.pt       - Optimizer state

Next steps:
  Generate: python generate.py --model_name dante --prompt 'Your prompt'
  Inspect:  python inspect_model.py --model_name dante
```

---

## Usage Examples

### generate.py

**Basic generation:**
```bash
python generate.py --model_name dante --prompt "Nel mezzo"
```

**With model info preview:**
```bash
python generate.py --model_name dante --prompt "Hello" --show_info
```

**Q&A mode:**
```bash
python generate.py --model_name dante --prompt "What is love?" --mode qa
```

**Legacy checkpoint:**
```bash
python generate.py --model_name dante --legacy_checkpoint final.pt
```

---

### train.py

**Basic training:**
```bash
python train.py --model_name my_model --input_file input.txt
```

**Resume training:**
```bash
# Just run again - automatically detects and resumes
python train.py --model_name my_model --max_iters 2000
```

**Fine-tune from another model:**
```bash
python train.py --model_name dante_v2 \
                --init_from_model dante \
                --input_file new_data.txt
```

**Custom architecture:**
```bash
python train.py --model_name big_model \
                --n_layer 12 \
                --n_head 12 \
                --n_embd 768 \
                --input_file input.txt
```

---

## Benefits of Refactoring

### For Users

✅ **Cleaner output**: Better formatted, more informative messages  
✅ **Progress visibility**: See model info before training/generation  
✅ **Helpful hints**: "Next steps" suggestions after training  
✅ **Better UX**: Emoji indicators (✅, 📄) for visual clarity  

### For Developers

✅ **Cleaner imports**: Single entry point (`from core import ...`)  
✅ **API consistency**: Uses same convenience functions as programmatic usage  
✅ **Less boilerplate**: Simplified common operations  
✅ **Better examples**: CLI scripts demonstrate best practices  

### For API Adoption

✅ **Shows the way**: CLI scripts model how to use the API  
✅ **Gradual complexity**: Simple case is simple, complex case still possible  
✅ **Discoverability**: Users see convenience functions in action  

---

## Comparison: Before vs After

### Import Complexity

**Before:**
```python
# 4 imports from different submodules
from core.model import GPT, GPTConfig
from core.tokenizer import Tokenizer
from core.data_loader import GPTDataLoader
from core.checkpoint import CheckpointManager
```

**After:**
```python
# 1 import from public API (only what's needed)
from core import GPTConfig, GPTDataLoader, CheckpointManager, get_model_info
```

### Model Loading (generate.py)

**Before:**
```python
model = CheckpointManager.load_for_inference(
    args.model_name, 
    legacy_filename=args.legacy_checkpoint
)
```

**After:**
```python
# Simple case (90% of usage)
model = load_model(args.model_name)

# Complex case (10% of usage, when needed)
if args.legacy_checkpoint:
    model = CheckpointManager.load_for_inference(...)
```

### Output Quality

**Before:**
```
Loading model 'dante' from latest.pt...
Model loaded successfully!
Tokenizer: gpt2
Vocab size: 50304

Prompt: Hello

[generated text]

!!! Finished !!!
```

**After:**
```
========== Generation Configuration ==========
Model: dante
Mode: basic
Max new tokens: 100

Loading model 'dante'...
✅ Model loaded successfully!
Tokenizer: gpt2
Vocab size: 50,304
Device: cuda

========== Generating ==========
Prompt: Hello

[generated text]

========== Generation Complete ==========
```

---

## Migration Guide

### If You're Using the CLI Scripts

**No changes required!** All existing commands work exactly the same:

```bash
# Still works
python train.py --model_name dante --input_file input.txt
python generate.py --model_name dante --prompt "Hello"
```

### If You're Importing from Scripts

**Old way** (before refactoring):
```python
# Direct import from script
from train import main as train_main
```

**New way** (recommended):
```python
# Use the public API instead
from core import create_model, load_model, GPTDataLoader

# Better: Use the API, don't import from scripts
```

---

## Future Enhancements

### Potential Additions

1. **Verbose mode**: `--verbose` flag for more detailed output
2. **Quiet mode**: `--quiet` flag for minimal output
3. **JSON output**: `--json` flag for machine-readable output
4. **Progress bars**: Add tqdm for training progress
5. **Logging**: Proper logging instead of print statements
6. **Configuration files**: Load settings from YAML/JSON

### Example Future Usage

```bash
# Verbose mode
python train.py --model_name dante --verbose

# JSON output for scripting
python generate.py --model_name dante --json > output.json

# Config file
python train.py --config training_config.yaml
```

---

## Summary

The CLI scripts have been refactored to:

1. ✅ Use convenience functions from `core/__init__.py`
2. ✅ Provide cleaner, more informative output
3. ✅ Demonstrate best practices for API usage
4. ✅ Maintain full backwards compatibility
5. ✅ Add helpful features (model info preview, next steps)

**Result**: The CLI scripts are now cleaner, more user-friendly, and serve as good examples of how to use the MyPT API!

