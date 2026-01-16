# MyPT Quick Reference

A cheat sheet of common commands and configurations.

---

## Training Commands

### Basic Training (Small Dataset)

```bash
python train.py \
    --config_file configs/pretrain/small.json \
    --model_name my_model \
    --input_file data/corpus.txt \
    --max_iters 10000
```

### Large Dataset Training (Sharded)

```bash
# Step 1: Prepare shards
python scripts/prepare_weighted_dataset.py \
    --source corpus:data/my_corpus/*.txt \
    --total_tokens 100000000 \
    --out_dir data/my_shards

# Step 2: Train
python train.py \
    --config_file configs/pretrain/150M.json \
    --model_name my_model \
    --dataset_dir data/my_shards \
    --max_iters 50000
```

### Continue Training (Resume)

```bash
python train.py \
    --model_name my_model \
    --dataset_dir data/my_shards \
    --max_iters 100000  # Will resume from last checkpoint
```

### Domain Adaptation (Phase 2)

```bash
python train.py \
    --config_file configs/pretrain/750M_1024_domain_adapt.json \
    --model_name domain_model \
    --dataset_dir data/domain_corpus \
    --init_from_model checkpoints/base_model \
    --eval_dataset_dir data/general_eval \
    --learning_rate 5e-5 \
    --max_iters 45000
```

### SFT (Supervised Fine-Tuning)

```bash
python train.py \
    --config_file configs/sft/750M_chat_sft.json \
    --model_name chat_model \
    --dataset_dir data/sft_episodes \
    --init_from_model checkpoints/domain_model \
    --max_iters 5000
```

---

## Generation Commands

### Basic Generation

```bash
python generate.py \
    --model_name my_model \
    --prompt "Once upon a time" \
    --max_new_tokens 200
```

### With Custom Parameters

```bash
python generate.py \
    --model_name my_model \
    --prompt "The meaning of life is" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95
```

### Interactive Mode

```bash
python generate.py --model_name my_model --interactive
```

### Deterministic Output

```bash
python generate.py \
    --model_name my_model \
    --prompt "Hello" \
    --temperature 0.0
```

---

## Data Preparation Commands

### Simple Dataset (Single File)

```bash
python scripts/prepare_dataset.py \
    --input_file data/corpus.txt \
    --output_dir data/shards \
    --tokenization gpt2
```

### Weighted Multi-Source Dataset

```bash
python scripts/prepare_weighted_dataset.py \
    --source wiki_en:data/wikipedia_en/*.txt:0.5 \
    --source wiki_de:data/wikipedia_de/*.txt:0.4 \
    --source news:data/news/*.txt:0.1 \
    --total_tokens 500000000 \
    --out_dir data/multilingual
```

### Mix Tokenized Datasets (Replay)

```bash
python scripts/mix_tokenized_datasets.py \
    --domain_dir data/domain_tokenized \
    --replay_dir data/general_tokenized \
    --output_dir data/mixed_20pct \
    --replay_ratio 0.20
```

### Append to Existing Dataset

```bash
python scripts/append_to_dataset.py \
    --dataset_dir data/existing_shards \
    --source new_data:data/new_corpus.txt \
    --target_tokens 50000000
```

---

## Utility Commands

### List Configurations

```bash
python scripts/show_configs.py
```

### Inspect Model

```bash
python scripts/inspect_model.py --model_name my_model
```

### Calculate Parameters

```bash
python scripts/calculate_params.py --config_file configs/pretrain/150M.json
```

### Count Tokens in File

```bash
python scripts/count_tokens.py --input_file data/corpus.txt
```

### Build RAG Index

```bash
python scripts/build_rag_index.py \
    --docs_dir workspace/docs \
    --out_dir workspace/index/latest
```

---

## Web Application

### Launch Web UI

```bash
python -m webapp.main --host 0.0.0.0 --port 8000
```

### With Specific Model

```bash
python -m webapp.main --model_name my_model --port 8000
```

---

## Configuration Quick Reference

### Model Sizes

| Config | Params | Context | VRAM | Use Case |
|--------|--------|---------|------|----------|
| `tiny.json` | ~11M | 128 | 2 GB | Testing |
| `small.json` | ~40M | 256 | 6 GB | Development |
| `150M.json` | ~150M | 256 | 12 GB | Production |
| `150M_1024.json` | ~150M | 1024 | 14 GB | High context |
| `350M_1024.json` | ~350M | 1024 | 24 GB | Large |
| `750M_1024.json` | ~750M | 1024 | 40 GB | Maximum |

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | from config | Learning rate (e.g., 3e-4, 5e-5) |
| `--max_iters` | from config | Training iterations |
| `--batch_size` | from config | Samples per step |
| `--eval_interval` | 500 | Steps between evaluations |
| `--warmup_iters` | from config | LR warmup steps |

### Key Generation Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--temperature` | 0.8 | 0.0-2.0 | Creativity (higher = more random) |
| `--top_k` | 50 | 0-vocab | Consider only top K tokens |
| `--top_p` | 0.95 | 0.0-1.0 | Nucleus sampling threshold |
| `--repetition_penalty` | 1.1 | 1.0-2.0 | Penalize repeated tokens |

---

## File Locations

| What | Where |
|------|-------|
| Model checkpoints | `checkpoints/<model_name>/` |
| Config presets | `configs/pretrain/` |
| SFT configs | `configs/sft/` |
| Training data | `data/` |
| Sharded datasets | `data/<dataset_name>/train/`, `val/` |
| RAG documents | `workspace/docs/` |
| RAG index | `workspace/index/` |
| Logs | `logs/` |
| Audit logs | `logs/audit/` |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Select GPU (e.g., `0`, `0,1`) |
| `PYTORCH_CUDA_ALLOC_CONF` | Memory allocation config |

```bash
# Use GPU 0 only
CUDA_VISIBLE_DEVICES=0 python train.py ...

# Use CPU only
CUDA_VISIBLE_DEVICES="" python train.py ...
```

---

## Iteration Calculations

**Tokens per iteration:**
```
tokens_per_iter = batch_size × block_size
```

**Iterations for N epochs:**
```
iters = (total_tokens × epochs) / (batch_size × block_size)
```

**Example (750M config, 200M tokens, 5 epochs):**
```
iters = (200M × 5) / (12 × 1024) = 81,380 iterations
```

---

## Common Patterns

### Phase 1 → Phase 2 → Phase 3 Pipeline

```bash
# Phase 1: Base pretraining
python train.py --config_file configs/pretrain/750M_1024.json \
    --model_name base_750M --dataset_dir data/wiki --max_iters 128000

# Phase 2: Domain adaptation  
python train.py --config_file configs/pretrain/750M_1024_domain_adapt.json \
    --model_name domain_750M --dataset_dir data/domain \
    --init_from_model checkpoints/base_750M --max_iters 45000

# Phase 3: Chat SFT
python train.py --config_file configs/sft/750M_chat_sft.json \
    --model_name chat_750M --dataset_dir data/sft_episodes \
    --init_from_model checkpoints/domain_750M --max_iters 5000
```

### Two-Stage Domain Adaptation

```bash
# Stage 1: Foundation (30% replay, low LR)
python train.py --init_from_model checkpoints/base \
    --model_name domain_v3 --dataset_dir data/mixed_30pct \
    --learning_rate 3e-5 --max_iters 45000

# Stage 2: Specialization (20% replay, higher LR)
python train.py --init_from_model checkpoints/domain_v3 \
    --model_name domain_v5 --dataset_dir data/mixed_20pct \
    --learning_rate 5e-5 --max_iters 45000
```

---

*Keep this reference handy for quick command lookups!*

