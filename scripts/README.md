# MyPT Scripts

Utility scripts for model management and inspection.

## Available Scripts

### `inspect_model.py`
Inspect model checkpoints and display configuration details.

**Usage:**
```bash
python scripts/inspect_model.py --model_name dante
```

**Features:**
- Shows model configuration (layers, embedding size, etc.)
- Displays tokenizer information
- Shows training progress (last step)
- Auto-detects JSON or legacy checkpoint format

**Arguments:**
- `--model_name`: Name of the model to inspect (required)
- `--legacy_checkpoint`: Specific legacy checkpoint file (optional)

---

### `convert_legacy_checkpoints.py`
Convert old single-file checkpoints to new JSON-based format.

**Usage:**
```bash
# Convert all legacy checkpoints
python scripts/convert_legacy_checkpoints.py --all

# Convert specific model
python scripts/convert_legacy_checkpoints.py --model_name dante

# Dry run (preview without converting)
python scripts/convert_legacy_checkpoints.py --all --dry-run
```

**Features:**
- Finds all legacy checkpoints automatically
- Converts to JSON-based format
- Keeps backup of original files
- Validates conversion

**Arguments:**
- `--model_name`: Specific model to convert
- `--checkpoint`: Legacy filename (default: latest.pt)
- `--all`: Convert all found legacy checkpoints
- `--dry-run`: Show what would be converted without converting

---

## See Also

- **Main scripts** (in root): `train.py`, `generate.py`
- **Examples**: See `examples/` folder
- **Documentation**: See `docs/` folder

