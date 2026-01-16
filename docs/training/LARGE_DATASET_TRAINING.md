# Large Dataset Training Guide

## Overview

MyPT now supports training on **very large datasets** (100M+ tokens, even billions) without loading everything into RAM. This guide shows you how to prepare and train on large datasets efficiently.

---

## The Problem

### Current In-Memory Approach (For Small Datasets)

```python
# train.py (legacy mode)
text = read_text("input.txt")           # Load entire file into RAM
tokens = tokenizer.encode(text)         # All tokens in RAM
data = torch.tensor(tokens)             # 8 bytes per token!
```

**For 500M tokens:**
- Text string: ~2-3 GB
- Python list (tokens): ~4 GB
- Torch tensor (int64): ~4 GB (500M × 8 bytes)
- **Total: 8-10 GB just for data!**

Plus model weights, gradients, optimizer → **Easily crashes on 16GB machines**

---

## The Solution: Sharded Datasets

### New Approach (For Large Datasets)

```python
# 1. Prepare dataset once (offline)
python scripts/prepare_dataset.py --input_files data/*.txt --out_dir data/my_dataset

# 2. Train (streams from disk, minimal RAM)
python train.py --dataset_dir data/my_dataset --model_name my_model
```

**Benefits:**
- ✅ **Minimal RAM usage** (~100-500 MB for data, regardless of dataset size)
- ✅ **Handle billions of tokens** (limited only by disk space)
- ✅ **Faster startup** (no tokenization wait)
- ✅ **Easy to add more data** (just add more shards)
- ✅ **Memory-mapped** (OS handles caching intelligently)

---

## Step-by-Step Guide

### Step 1: Prepare Your Dataset

Create binary shards from your text files:

```bash
python scripts/prepare_dataset.py \
    --input_files wiki.txt books.txt news.txt \
    --out_dir data/large_corpus \
    --tokenization gpt2 \
    --tokens_per_shard 10000000
```

**What this does:**
1. Streams text from all input files (low memory)
2. Cleans and normalizes text
3. Deduplicates lines (optional)
4. Tokenizes in chunks
5. Writes binary shards (~10M tokens each, ~40MB per shard)
6. Splits into train/val directories

**Output structure:**
```
data/large_corpus/
├── dataset_metadata.json
├── tokenizer_state.json
├── train/
│   ├── shard_00000.bin
│   ├── shard_00001.bin
│   ├── ...
│   └── shard_00045.bin  (~450M tokens)
└── val/
    ├── shard_00046.bin
    ├── ...
    └── shard_00050.bin  (~50M tokens)
```

---

### Step 2: Train with Sharded Dataset

```bash
python train.py \
    --dataset_dir data/large_corpus \
    --model_name my_large_model \
    --config_file configs/150M.json \
    --max_iters 10000
```

**What this does:**
- Loads shard metadata (tiny JSON files)
- For each batch:
  - Randomly picks a shard
  - Memory-maps it (doesn't load into RAM)
  - Samples sequences from that shard
  - Trains on those sequences

**RAM usage:** Only ~100-500 MB for data (vs 8-10 GB for in-memory!)

---

## Detailed Examples

### Example 1: Wikipedia + Books (500M tokens)

```bash
# Step 1: Prepare dataset
python scripts/prepare_dataset.py \
    --input_files wikipedia_en.txt gutenberg_books.txt \
    --out_dir data/wiki_books \
    --tokenization gpt2 \
    --tokens_per_shard 10000000 \
    --val_fraction 0.1

# Output:
#   Total tokens: 500,000,000
#   Total shards: 50
#   Train shards: 45 in data/wiki_books/train
#   Val shards: 5 in data/wiki_books/val

# Step 2: Train
python train.py \
    --dataset_dir data/wiki_books \
    --model_name wiki_gpt \
    --config_file configs/150M_1024.json \
    --max_iters 50000 \
    --learning_rate 3e-4
```

**Result:** Trains on 500M tokens using ~12GB VRAM (model + optimizer), minimal system RAM!

---

### Example 2: Multilingual Dataset

```bash
# Prepare multilingual dataset
python scripts/prepare_dataset.py \
    --input_files wiki_en.txt wiki_de.txt europarl.txt news_combined.txt \
    --out_dir data/multilingual \
    --tokenization gpt2 \
    --tokens_per_shard 20000000  # Larger shards
```

---

### Example 3: Character-Level

```bash
# Step 1: Prepare character-level dataset
python scripts/prepare_dataset.py \
    --input_files shakespeare.txt dante.txt \
    --out_dir data/literature_char \
    --tokenization char \
    --tokens_per_shard 5000000  # Smaller shards for char-level

# Step 2: Train with character-level tokenization
python train.py \
    --dataset_dir data/literature_char \
    --tokenization char \  # Important: must match dataset!
    --model_name char_model \
    --config_file configs/small_char.json \
    --max_iters 5000
```

**Important:** When using character-level tokenization (`--tokenization char`), the tokenizer vocabulary is automatically loaded from `tokenizer_state.json` in the dataset directory. This ensures the model uses the exact same character vocabulary that was used during dataset preparation.

---

## Preparation Script Options

### Basic Arguments

```bash
python scripts/prepare_dataset.py \
    --input_files file1.txt file2.txt file3.txt \  # Required: input text files
    --out_dir data/my_dataset \                    # Required: output directory
    --tokenization gpt2 \                          # gpt2 or char
    --tokens_per_shard 10000000 \                  # Tokens per shard (10M default)
    --val_fraction 0.1                             # Validation fraction (0.1 = 10%)
```

### Advanced Options

```bash
# Skip text normalization (keep original case/whitespace)
--no_normalize

# Skip line filtering (keep all lines, even short ones)
--no_filter

# Skip deduplication (faster, but may have duplicates)
--no_dedup
```

### Full Example

```bash
python scripts/prepare_dataset.py \
    --input_files data/source1.txt data/source2.txt data/source3.txt \
    --out_dir data/my_large_dataset \
    --tokenization gpt2 \
    --tokens_per_shard 10000000 \
    --val_fraction 0.15 \
    --no_dedup  # Skip deduplication for speed
```

---

## Training Script Changes

### Legacy Mode (In-Memory, Small Datasets)

Still works for small datasets (<100M tokens):

```bash
python train.py \
    --model_name my_model \
    --input_file input.txt \
    --max_iters 1000
```

### New Mode (Sharded, Large Datasets)

For large datasets (100M+ tokens):

```bash
python train.py \
    --model_name my_model \
    --dataset_dir data/my_dataset \
    --max_iters 10000
```

**Key difference:** `--input_file` vs `--dataset_dir`

---

## Memory Comparison

### In-Memory Mode (Legacy)

**500M tokens:**
- Text file: ~2-3 GB RAM
- Token list: ~4 GB RAM
- Torch tensor: ~4 GB RAM
- **Total data: ~10 GB RAM**
- Plus model/gradients/optimizer: **~18-20 GB total**

**Result:** Crashes on 16GB machines!

### Sharded Mode (New)

**500M tokens (50 shards):**
- Memory-mapped shards: ~100-500 MB RAM (OS caching)
- Current batch: ~50 MB RAM
- **Total data: ~500 MB RAM**
- Plus model/gradients/optimizer: **~8-10 GB total**

**Result:** Works comfortably on 16GB machines!

---

## How Sharding Works

### Shard Format

Each shard is a binary file containing token IDs as `uint32`:

```python
# shard_00000.bin structure:
# [token_id_0, token_id_1, ..., token_id_9999999]
# Each token: 4 bytes (uint32)
# Total size: 10M tokens × 4 bytes = 40 MB
```

### Memory Mapping

Python's `numpy.memmap` allows treating files as arrays without loading them:

```python
# This doesn't load the entire file into RAM!
data = np.memmap("shard_00000.bin", dtype=np.uint32, mode='r')

# Only accessed parts are loaded into RAM (OS manages this)
batch = data[1000:1256]  # Only loads this small portion
```

### Batch Sampling

Each batch:
1. Randomly select a shard (e.g., `shard_00023.bin`)
2. Memory-map that shard (lazy, doesn't load all)
3. Sample `batch_size` random positions from shard
4. Extract sequences (only these are loaded into RAM)
5. Return as PyTorch tensors

**RAM usage per batch:** ~50 MB (not ~4 GB!)

---

## Best Practices

### 1. Choose Shard Size

**General rule:**
- **10M tokens/shard** (default): Good for most cases (~40 MB per shard)
- **5M tokens/shard**: For character-level or limited disk I/O
- **20M tokens/shard**: For very large datasets with fast SSDs

### 2. Optimize for Your Hardware

**HDD (slow disk):**
```bash
--tokens_per_shard 20000000  # Larger shards, fewer seeks
```

**SSD (fast disk):**
```bash
--tokens_per_shard 10000000  # Default, good balance
```

**NVMe SSD (very fast):**
```bash
--tokens_per_shard 5000000   # Smaller shards, more variety per epoch
```

### 3. Validation Split

**Default (10%):**
```bash
--val_fraction 0.1  # 10% for validation
```

**For very large datasets:**
```bash
--val_fraction 0.05  # 5% is enough when you have billions of tokens
```

### 4. Deduplication

**Enable for:**
- Web scraping (lots of duplicates)
- Multiple source files that may overlap

**Disable for:**
```bash
--no_dedup  # Skip dedup for speed
```

Use when:
- Single source file (no duplicates expected)
- Very large datasets where dedup would use too much RAM
- Speed is more important than data quality

---

## Troubleshooting

### Issue: "No training shards found"

**Cause:** Dataset not prepared or wrong path

**Solution:**
```bash
# Check directory structure
ls -R data/my_dataset

# Should see:
# data/my_dataset/train/*.bin
# data/my_dataset/val/*.bin
# data/my_dataset/dataset_metadata.json
```

### Issue: "Out of memory" during preparation

**Cause:** Deduplication uses too much RAM for very large datasets

**Solution:**
```bash
# Skip deduplication
python scripts/prepare_dataset.py ... --no_dedup
```

### Issue: Slow training

**Cause:** Disk I/O bottleneck (especially HDD)

**Solutions:**
1. Use larger shards: `--tokens_per_shard 20000000`
2. Move dataset to SSD if available
3. Increase OS file cache (more RAM for caching)

### Issue: Want to add more data

**Solution:** Prepare a new dataset and merge:

```bash
# Prepare new data
python scripts/prepare_dataset.py --input_files new_data.txt --out_dir data/new_shards

# Move shards to existing dataset
mv data/new_shards/train/*.bin data/my_dataset/train/
mv data/new_shards/val/*.bin data/my_dataset/val/

# Update metadata (optional, for records)
```

---

## Performance Comparison

### Small Dataset (10M tokens, ~40 MB)

| Mode | RAM Usage | Startup Time | Training Speed |
|------|-----------|--------------|----------------|
| In-memory | ~500 MB | 5 seconds | Fast |
| Sharded | ~100 MB | 1 second | Fast |

**Recommendation:** Use in-memory (simpler)

### Medium Dataset (100M tokens, ~400 MB)

| Mode | RAM Usage | Startup Time | Training Speed |
|------|-----------|--------------|----------------|
| In-memory | ~4 GB | 30 seconds | Fast |
| Sharded | ~200 MB | 2 seconds | Fast |

**Recommendation:** Use sharded (more flexible)

### Large Dataset (500M tokens, ~2 GB)

| Mode | RAM Usage | Startup Time | Training Speed |
|------|-----------|--------------|----------------|
| In-memory | ~18 GB | 2-3 minutes | Fast |
| Sharded | ~500 MB | 2 seconds | Fast |

**Recommendation:** Use sharded (only option for most machines)

### Very Large Dataset (10B tokens, ~40 GB)

| Mode | RAM Usage | Startup Time | Training Speed |
|------|-----------|--------------|----------------|
| In-memory | **Crashes** | N/A | N/A |
| Sharded | ~1 GB | 2 seconds | Fast |

**Recommendation:** Sharded is the only option

---

## Troubleshooting

### Issue: Gibberish output with character-level tokenization

**Symptom:** After training with `--tokenization char`, generation produces random characters instead of coherent text.

**Cause:** This was an issue where the tokenizer vocabulary wasn't being loaded correctly from the dataset directory.

**Solution:** This has been fixed! The training script now automatically loads `tokenizer_state.json` from the dataset directory, ensuring the correct character vocabulary is used.

**Verify the fix:**
```bash
# Check training output - vocab_size should match your character set (e.g., ~95-256)
python train.py --dataset_dir data/my_char_dataset --tokenization char ...

# Should show:
# Loading tokenizer state from data/my_char_dataset...
# Vocabulary size: 95  ← Correct!

# Generation should now work correctly
python generate.py --model_name my_char_model --prompt "Hello"
```

**See also:** `docs/SHARDED_TOKENIZER_FIX.md` for technical details about this fix.

---

---

## Domain Adaptation Training (Phase 2)

For domain adaptation, you typically want to:
1. Train on domain-specific data
2. Evaluate on **both** domain validation AND general validation (to detect catastrophic forgetting)

### Dual Eval Setup

```bash
# Phase 2 domain training with dual evaluation
python train.py \
    --dataset_dir data/domain_corpus \
    --model_name domain_model \
    --config_file configs/pretrain/750M_1024_domain_adapt.json \
    --init_from_model checkpoints/base_model \
    --eval_dataset_dir data/general_corpus \
    --max_iters 65500
```

**What this does:**
- Trains on `data/domain_corpus/train/*.bin`
- Evaluates on `data/domain_corpus/val/*.bin` (domain val loss)
- Evaluates on `data/general_corpus/val/*.bin` (general val loss)
- Logs both: `step 100: val 2.15 | eval_general 2.89`

### Programmatic Dual Eval

```python
from core import GPTDataLoader, Tokenizer

# Domain loader (train + val)
domain_loader = GPTDataLoader(config, tokenizer, dataset_dir="data/domain_corpus")

# General eval loader (val only - no training data loaded)
general_loader = GPTDataLoader(
    config, tokenizer,
    dataset_dir="data/general_corpus",
    eval_only=True  # Only loads val shards
)

# Train with dual evaluation
model.fit(
    data_loader=domain_loader,
    optimizer=optimizer,
    eval_data_loaders={"general": general_loader}
)
```

### Monitoring Catastrophic Forgetting

During domain adaptation, watch both metrics:

| Metric | Healthy | Warning |
|--------|---------|---------|
| `val` (domain) | ↓ Decreasing | Expected to improve |
| `eval_general` | Stable or slight ↑ | Large ↑ = forgetting |

**If general loss increases significantly:**
- Reduce learning rate (try 5e-5 instead of 3e-4)
- Reduce epochs (3-5 epochs max for domain adaptation)
- Increase warmup iterations

---

## Summary

**When to use in-memory mode:**
- ✅ Small datasets (< 100M tokens)
- ✅ Simple single-file training
- ✅ Quick experiments

**When to use sharded mode:**
- ✅ Large datasets (100M+ tokens)
- ✅ Multiple data sources
- ✅ Limited RAM
- ✅ Want to add data incrementally
- ✅ Production training

**When to use dual eval mode:**
- ✅ Domain adaptation (Phase 2 training)
- ✅ Monitoring catastrophic forgetting
- ✅ Multi-domain evaluation

**Quick start:**

```bash
# 1. Prepare dataset (once)
python scripts/prepare_dataset.py \
    --input_files data/*.txt \
    --out_dir data/my_dataset

# 2. Train (many times)
python train.py \
    --dataset_dir data/my_dataset \
    --model_name my_model \
    --config_file configs/150M.json

# 3. Domain adaptation with dual eval
python train.py \
    --dataset_dir data/domain_corpus \
    --init_from_model checkpoints/base_model \
    --eval_dataset_dir data/general_eval \
    --model_name domain_adapted
```

**Train on billions of tokens without running out of RAM!** 🚀

