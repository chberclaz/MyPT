# Data Persistence and Reusability

This guide explains how prepared datasets (shards) and RAG indexes persist on disk and can be reused across multiple training runs.

## Overview

| Data Type | Prepare Once? | Reuse Across Models? | Constraint |
|-----------|---------------|----------------------|------------|
| **RAG Index** | ✅ | ✅ Any model | None |
| **Training Shards** | ✅ | ✅ Same tokenization | Must match `gpt2` or `char` |
| **SFT Shards + Masks** | ✅ | ✅ Same tokenization | Must match base model |

---

## RAG Index (Model-Independent)

The RAG embedding index is **completely independent** of your trained LLM.

### Why?

- Uses `LocalEmbedder` (hashed n-grams), not your model
- Stores document chunks and their vector representations
- Any trained model can use the same index

### Location

```
workspace/index/
├── embeddings.npy      # Chunk vectors [N, 256]
├── meta.jsonl          # Chunk text + source info
└── config.json         # Embedder settings
```

### Reuse Example

```bash
# Build index once
python scripts/build_rag_index.py --docs_dir workspace/docs --out_dir workspace/index/v1

# Use with different models (same index!)
python scripts/workspace_chat.py --model_name model_A --index_dir workspace/index/v1
python scripts/workspace_chat.py --model_name model_B --index_dir workspace/index/v1
python scripts/workspace_chat.py --model_name model_C --index_dir workspace/index/v1
```

### When to Rebuild

| Change | Rebuild Index? |
|--------|---------------|
| Add/remove documents | ✅ Yes |
| Change chunk size/overlap | ✅ Yes |
| Change embedding dimension | ✅ Yes |
| Train a new model | ❌ No |
| Fine-tune existing model | ❌ No |

---

## Training Shards (Tokenization-Dependent)

Tokenized training data is **persistent** but **tokenization-specific**.

### Why the Constraint?

Shards store token IDs (integers), not text. A token ID like `15339` means:
- GPT-2: "hello" 
- Char: Invalid (vocab is only 0-255)

Using wrong shards = garbage output.

### Location

```
data/my_dataset/
├── train/
│   ├── shard_00000.bin     # Token IDs (uint16)
│   └── shard_00001.bin
├── val/
│   └── shard_00000.bin
├── tokenizer_state.json    # MUST match model
└── dataset_metadata.json
```

### Valid Reuse

```bash
# Create shards with GPT-2 tokenization
python scripts/prepare_dataset.py --input corpus.txt --output_dir data/corpus_gpt2 --tokenization gpt2

# ✅ Train different architectures (same tokenization)
python train.py --model_name small_model --dataset_dir data/corpus_gpt2 --config_file configs/pretrain/small.json
python train.py --model_name large_model --dataset_dir data/corpus_gpt2 --config_file configs/pretrain/150M.json

# ✅ Resume training
python train.py --model_name small_model --dataset_dir data/corpus_gpt2 --max_iters 10000

# ❌ WRONG: Different tokenization
python train.py --model_name char_model --dataset_dir data/corpus_gpt2 --tokenization char
# This will produce garbage!
```

### When to Regenerate

| Change | Regenerate Shards? |
|--------|-------------------|
| New/modified input text | ✅ Yes |
| Change tokenization (gpt2↔char) | ✅ Yes |
| Change shard size | ✅ Yes |
| Train different model size | ❌ No |
| Train longer (more iters) | ❌ No |
| Fine-tune with SFT | ❌ No |

---

## SFT Shards with Loss Masks

Same rules as training shards, plus:
- Includes `*_mask.bin` files (which tokens to train on)
- Must match base model's tokenization

### Location

```
data/chat_sft/
├── train/
│   ├── shard_00000.bin        # Token IDs
│   └── shard_00000_mask.bin   # Loss mask (0 or 1)
├── val/
│   ├── shard_00000.bin
│   └── shard_00000_mask.bin
├── tokenizer_state.json
└── dataset_metadata.json
```

### Valid Reuse

```bash
# Create SFT dataset once
python scripts/prepare_chat_sft.py --input conversations.jsonl --output_dir data/chat_sft

# ✅ Fine-tune different base models (same SFT data)
python train.py --model_name sft_A --init_from_model base_A --dataset_dir data/chat_sft
python train.py --model_name sft_B --init_from_model base_B --dataset_dir data/chat_sft

# ✅ Continue fine-tuning
python train.py --model_name sft_A --dataset_dir data/chat_sft --max_iters 20000
```

---

## Recommended Workflow

```bash
# ============================================================
# PHASE 1: ONE-TIME DATA PREPARATION
# ============================================================

# Pretrain data (do once per corpus)
python scripts/prepare_dataset.py \
    --input data/large_corpus.txt \
    --output_dir data/pretrain_gpt2 \
    --tokenization gpt2

# SFT data (do once per conversation set)
python scripts/prepare_chat_sft.py \
    --input data/conversations.jsonl \
    --output_dir data/sft_chat

# RAG index (do once, update when docs change)
python scripts/build_rag_index.py \
    --docs_dir workspace/docs \
    --out_dir workspace/index/v1

# ============================================================
# PHASE 2: EXPERIMENTATION (reuse all prepared data)
# ============================================================

# Experiment 1: Small model
python train.py --model_name exp1 --dataset_dir data/pretrain_gpt2 \
    --config_file configs/pretrain/small.json --max_iters 5000

# Experiment 2: Larger model (SAME shards)
python train.py --model_name exp2 --dataset_dir data/pretrain_gpt2 \
    --config_file configs/pretrain/150M.json --max_iters 10000

# Experiment 3: Fine-tune with SFT (SAME SFT shards)
python train.py --model_name exp3 --init_from_model exp2 \
    --dataset_dir data/sft_chat --config_file configs/sft1/small.json

# ============================================================
# PHASE 3: INFERENCE (reuse RAG index with any model)
# ============================================================

python scripts/workspace_chat.py --model_name exp3 --index_dir workspace/index/v1
```

---

## Storage Locations Summary

```
project/
├── data/                           # Tokenized training data
│   ├── pretrain_gpt2/              # GPT-2 tokenized corpus
│   │   ├── train/*.bin
│   │   ├── val/*.bin
│   │   └── tokenizer_state.json
│   ├── pretrain_char/              # Char tokenized corpus (separate!)
│   │   └── ...
│   └── sft_chat/                   # SFT with loss masks
│       ├── train/*.bin
│       ├── train/*_mask.bin
│       └── ...
│
├── workspace/                      # RAG data (model-independent)
│   ├── docs/                       # Source documents
│   └── index/                      # Embedding indexes
│       ├── v1/                     # Version 1
│       └── v2/                     # After adding docs
│
└── checkpoints/                    # Trained models
    ├── exp1/
    ├── exp2/
    └── exp3/
```

---

## Troubleshooting

### "Model outputs garbage after loading shards"

**Cause:** Tokenization mismatch between shards and model.

**Fix:** Check `tokenizer_state.json` in both locations:
```bash
cat data/my_shards/tokenizer_state.json    # What shards expect
cat checkpoints/my_model/tokenizer.json    # What model has
```

Both must have same `tokenization` value (`gpt2` or `char`).

### "Want to try different tokenization"

Create separate shard directories:
```bash
python scripts/prepare_dataset.py --output_dir data/corpus_gpt2 --tokenization gpt2
python scripts/prepare_dataset.py --output_dir data/corpus_char --tokenization char
```

### "RAG returns wrong results after adding documents"

Rebuild the index:
```bash
python scripts/build_rag_index.py --docs_dir workspace/docs --out_dir workspace/index/v2
```

Then reload in chat or restart with new `--index_dir`.

---

## See Also

- [Large Dataset Training](LARGE_DATASET_TRAINING.md) - Sharded dataset details
- [Tokenization Comparison](TOKENIZATION_COMPARISON.md) - GPT-2 vs char
- [Chat SFT with Context](chat_sft_with_context.md) - SFT dataset creation

