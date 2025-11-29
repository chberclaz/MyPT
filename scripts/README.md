# MyPT Scripts

Utility scripts for model management, inspection, and dataset preparation.

## Available Scripts

### `prepare_dataset.py`
Prepare sharded datasets for large-scale training.

**Purpose:** Convert text files into binary shards for training on large datasets (100M+ tokens) without loading everything into RAM.

**Usage:**
```bash
# Basic usage
python scripts/prepare_dataset.py \
    --input_files data.txt \
    --out_dir data/my_dataset

# Multiple sources
python scripts/prepare_dataset.py \
    --input_files wiki.txt books.txt news.txt \
    --out_dir data/large_corpus \
    --tokenization gpt2 \
    --tokens_per_shard 10000000

# Character-level
python scripts/prepare_dataset.py \
    --input_files input.txt \
    --out_dir data/char_dataset \
    --tokenization char
```

**Features:**
- Streams text from multiple files (low memory usage)
- Cleans and normalizes text
- Deduplicates lines (optional)
- Tokenizes incrementally
- Writes binary shards (~10M tokens each, ~40MB)
- Splits into train/val directories

**Arguments:**
- `--input_files`: List of text files (required)
- `--out_dir`: Output directory (required)
- `--tokenization`: gpt2 or char (default: gpt2)
- `--tokens_per_shard`: Tokens per shard (default: 10M)
- `--val_fraction`: Validation fraction (default: 0.1)
- `--no_normalize`: Skip text normalization
- `--no_filter`: Skip line filtering
- `--no_dedup`: Skip deduplication

---

### `calculate_params.py`
Calculate the number of parameters in a GPT model.

**Usage:**
```bash
# From config file
python scripts/calculate_params.py --config_file configs/150M_1024.json

# From parameters
python scripts/calculate_params.py --n_layer 16 --n_embd 768 --n_head 12

# Interactive mode
python scripts/calculate_params.py --interactive

# Show formula
python scripts/calculate_params.py --show_formula
```

**Features:**
- Detailed parameter breakdown by component
- Memory estimates (FP32, FP16)
- Training memory estimates
- Works with config files or manual input

**Arguments:**
- `--config_file`: Path to config JSON file
- `--n_layer`: Number of layers
- `--n_embd`: Embedding dimension
- `--n_head`: Number of attention heads
- `--vocab_size`: Vocabulary size (default: 50304)
- `--block_size`: Context length (default: 256)
- `--bias`: Use bias in layers
- `--interactive`: Interactive mode
- `--show_formula`: Display calculation formula

---

### `show_configs.py`
Display available model configuration presets.

**Usage:**
```bash
# List all configs
python scripts/show_configs.py

# Show details for specific config
python scripts/show_configs.py --config_file configs/150M_1024.json
```

**Features:**
- Lists all available configs
- Shows parameter counts
- Displays architecture details
- Provides usage examples

**Arguments:**
- `--config_file`: Show details for specific config (optional)
- `--configs_dir`: Directory containing configs (default: configs/)

---

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

