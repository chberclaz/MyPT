# MyPT Project Structure

## Overview

MyPT is organized with a clean, professional structure that separates main scripts, core package, utilities, examples, and documentation.

---

## Directory Layout

```
MyPT/
├── 📄 README.md                # Project overview and quick start
├── 📄 pyproject.toml           # Package configuration (PEP 518/621)
├── 📄 requirements.txt         # Python dependencies
│
├── 🚀 train.py                 # Main training script (user-facing)
├── 🚀 generate.py              # Main generation script (user-facing)
│
├── 📦 core/                    # Core package (importable API)
│   ├── __init__.py             # Public API exports
│   ├── model.py                # GPT model with training methods
│   ├── tokenizer.py            # GPT-2 BPE and char-level tokenizers
│   ├── data_loader.py          # Data loading and batching
│   ├── checkpoint.py           # Checkpoint management
│   └── generator.py            # Text generation strategies
│
├── 🛠️ scripts/                 # Utility scripts
│   ├── README.md               # Scripts documentation
│   ├── inspect_model.py        # Inspect model checkpoints
│   └── convert_legacy_checkpoints.py  # Convert old checkpoints
│
├── 📚 examples/                # Example code and tutorials
│   ├── README.md               # Examples documentation
│   ├── example_usage.py        # API usage examples
│   └── helper_selfaggregation.py  # Educational self-attention code
│
├── 📖 docs/                    # Documentation
│   ├── README.md               # Documentation index
│   ├── INSTALL.md              # Installation guide
│   ├── CHECKPOINT_FORMAT.md    # Checkpoint system
│   ├── JSON_CHECKPOINT_MIGRATION.md  # Migration guide
│   ├── PACKAGING_SUMMARY.md    # Package details
│   ├── CLI_REFACTORING.md      # CLI enhancements
│   ├── REFACTORING_SUMMARY.md  # Architecture overview
│   ├── PYTORCH_SECURITY_FIX.md # Security improvements
│   ├── VERIFICATION.md         # Testing and QA
│   ├── FINAL_SUMMARY.md        # Complete project overview
│   └── PROJECT_STRUCTURE.md    # This file
│
├── 💾 checkpoints/             # Model checkpoints (gitignored)
│   ├── dante/
│   │   ├── model.pt
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   └── ...
│
└── 🗑️ .gitignore               # Git ignore patterns
```

---

## Design Principles

### 1. **Main Scripts in Root**
User-facing scripts stay in root for easy access:
- `train.py` - Most common operation (training)
- `generate.py` - Most common operation (generation)

**Why?** Users can quickly run `python train.py` or `python generate.py` without navigating folders.

### 2. **Utility Scripts in `scripts/`**
Less frequently used utilities organized separately:
- `inspect_model.py` - Debugging/inspection
- `convert_legacy_checkpoints.py` - One-time migration

**Why?** Keeps root clean while still providing easy access to utilities.

### 3. **Core Package Organization**
All importable code in `core/` package:
```python
from core import GPT, GPTConfig, Tokenizer, Generator
```

**Why?** Clean import structure, professional package design.

### 4. **Examples Separate from Core**
Educational and example code in `examples/`:
- Not part of the production API
- Easy to explore and learn from
- Can be run independently

**Why?** Separates learning resources from production code.

### 5. **Documentation Organization**
All docs in `docs/` except main README:
- Main README stays visible on GitHub
- All other docs organized in docs/
- Comprehensive documentation index

**Why?** Clean root, easy to find documentation.

---

## File Count by Directory

| Directory | Files | Description |
|-----------|-------|-------------|
| **Root** | 5 | README, config, main scripts |
| **core/** | 6 | Core package modules |
| **scripts/** | 3 | Utility scripts + README |
| **examples/** | 3 | Example code + README |
| **docs/** | 11 | Documentation files |
| **Total** | 28 | Clean, organized structure |

---

## Usage Patterns

### For End Users

**Most common operations** (easy access in root):
```bash
# Training
python train.py --model_name my_model --input_file input.txt

# Generation
python generate.py --model_name my_model --prompt "Hello"
```

**Occasional operations** (in scripts/):
```bash
# Inspect model
python scripts/inspect_model.py --model_name my_model

# Convert checkpoints
python scripts/convert_legacy_checkpoints.py --all
```

### For Developers

**Programmatic usage** (import from core):
```python
from core import create_model, load_model, Generator

# Create model
model = create_model(n_layer=6, n_head=6, n_embd=384)

# Load model
model = load_model("my_model")

# Generate
output = model.generate("Hello", max_new_tokens=100)
```

**Learning** (examples folder):
```bash
# See API examples
python examples/example_usage.py

# Learn about self-attention
python examples/helper_selfaggregation.py
```

---

## Import Structure

### Public API (core/)

All imports go through `core/__init__.py`:

```python
from core import (
    # Models & Config
    GPT,
    GPTConfig,
    
    # Tokenization
    Tokenizer,
    
    # Data
    GPTDataLoader,
    
    # Checkpoints
    CheckpointManager,
    
    # Generation
    Generator,
    
    # Convenience functions
    create_model,
    load_model,
    get_model_info,
)
```

**Why?** Single import point, clean namespace, easy to discover API.

### Internal Imports

Scripts import from core:
```python
# generate.py
from core import load_model, get_model_info, Generator

# train.py
from core import GPTConfig, GPTDataLoader, CheckpointManager
```

---

## Benefits of This Structure

### Clarity
✅ Main operations obvious (in root)  
✅ Clear separation of concerns  
✅ Easy to navigate  

### Maintainability
✅ Modular design  
✅ Clear boundaries between components  
✅ Easy to extend  

### Professionalism
✅ Follows Python best practices  
✅ Clean root directory  
✅ Professional package structure  

### Usability
✅ Common operations easy to find  
✅ Examples separate from production code  
✅ Documentation organized  

---

## Comparison: Before vs After

### Before (Cluttered)
```
MyPT/
├── README.md
├── train.py
├── generate.py
├── generator.py                    # ❌ Should be in core
├── inspect_model.py                # ❌ Not main operation
├── convert_legacy_checkpoints.py  # ❌ Not main operation
├── example_usage.py                # ❌ Not main script
├── helper_selfaggregation.py      # ❌ Educational content
├── INSTALL.md                      # ❌ Documentation
├── CHECKPOINT_FORMAT.md            # ❌ Documentation
├── ... 8 more .md files ...        # ❌ Too many files
├── core/
└── checkpoints/
```
**Problem**: 18+ files in root, hard to find what you need

### After (Organized)
```
MyPT/
├── README.md                       # ✅ Main overview
├── pyproject.toml                  # ✅ Package config
├── requirements.txt                # ✅ Dependencies
├── train.py                        # ✅ Main script
├── generate.py                     # ✅ Main script
├── core/                          # ✅ Core package
├── scripts/                       # ✅ Utilities organized
├── examples/                      # ✅ Examples organized
├── docs/                          # ✅ Docs organized
└── checkpoints/                   # ✅ Data
```
**Solution**: 5 files + 5 folders in root, everything has its place

---

## Future Scalability

This structure makes it easy to add:

### More Scripts
Add to `scripts/` folder:
- `scripts/benchmark.py`
- `scripts/export_onnx.py`
- `scripts/quantize_model.py`

### More Examples
Add to `examples/` folder:
- `examples/fine_tuning.py`
- `examples/custom_tokenizer.py`
- `examples/beam_search.py`

### More Documentation
Add to `docs/` folder:
- `docs/API_REFERENCE.md`
- `docs/TROUBLESHOOTING.md`
- `docs/CONTRIBUTING.md`

### More Core Modules
Add to `core/` package:
- `core/losses.py`
- `core/metrics.py`
- `core/schedulers.py`

**All without cluttering the root directory!**

---

## Summary

The MyPT project structure is:

✅ **Clean** - Only 5 files in root  
✅ **Organized** - Everything has its place  
✅ **Professional** - Follows best practices  
✅ **Scalable** - Easy to extend  
✅ **User-friendly** - Main operations obvious  
✅ **Developer-friendly** - Clear import structure  

**Perfect balance of simplicity and organization!** 📁✨

