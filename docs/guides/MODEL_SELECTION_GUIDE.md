# Model Selection Guide

How to choose the right model size, context length, and configuration for your use case.

---

## Quick Decision Tree

```
What's your primary use case?
│
├─► Testing/Learning → tiny.json or small.json
│
├─► Development/Prototyping → 150M.json
│
├─► Production (short context) → 200M.json or 350M.json
│
└─► Production (long context) → 350M_1024.json or 750M_1024.json
```

---

## Available Configurations

### Standard Context (256 tokens)

| Config | Parameters | VRAM | Training Time* | Use Case |
|--------|------------|------|----------------|----------|
| `tiny.json` | ~11M | 2 GB | Minutes | Testing, debugging |
| `small.json` | ~40M | 6 GB | 30 min | Learning, quick experiments |
| `150M.json` | ~150M | 12 GB | 2-4 hours | Production (limited hardware) |
| `200M.json` | ~200M | 16 GB | 4-6 hours | Production |

### High Context (1024 tokens)

| Config | Parameters | VRAM | Training Time* | Use Case |
|--------|------------|------|----------------|----------|
| `150M_1024.json` | ~150M | 14 GB | 3-5 hours | Documents, code |
| `350M_1024.json` | ~350M | 24 GB | 8-12 hours | Advanced applications |
| `750M_1024.json` | ~750M | 40 GB | 24-48 hours | Maximum capability |

*Training times are estimates for 10,000 iterations on RTX 3090

---

## Context Length: 256 vs 1024

### What is Context Length?

Context length (`block_size`) is how many tokens the model can "see" at once.

| block_size | Approximate Text | Good For |
|------------|------------------|----------|
| 128 | ~100 words | Testing only |
| 256 | ~200 words | Short text, Q&A |
| 512 | ~400 words | Paragraphs, simple docs |
| 1024 | ~800 words | Long documents, code |

### When to Use 256 (Standard)

✅ **Choose 256 when:**
- Hardware is limited (< 16 GB VRAM)
- Working with short texts
- Need faster training
- Batch size is more important than context

### When to Use 1024 (High Context)

✅ **Choose 1024 when:**
- Working with long documents
- Processing code files
- Need better coherence over long text
- Have sufficient VRAM (16+ GB)

### Memory Impact

Context length has a **quadratic** impact on memory:

| block_size | Relative Memory |
|------------|-----------------|
| 256 | 1x |
| 512 | ~2x |
| 1024 | ~4x |

---

## Model Size Selection

### By Hardware

| Your GPU VRAM | Recommended Config |
|---------------|-------------------|
| 4-6 GB | `small.json` (40M) |
| 8-12 GB | `150M.json` |
| 12-16 GB | `150M_1024.json` or `200M.json` |
| 16-24 GB | `350M_1024.json` |
| 24-40 GB | `750M_1024.json` |
| 40+ GB (A100) | `750M_1024.json` with larger batch |

### By Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Testing/Learning** | `small.json` | Fast, works on any hardware |
| **Prototype** | `150M.json` | Good quality, reasonable speed |
| **Chat/Q&A** | `200M.json` | Good for short conversations |
| **Document RAG** | `350M_1024.json` | Long context for documents |
| **Code Generation** | `750M_1024.json` | Best coherence, long context |
| **Domain Expert** | `750M_1024.json` | Capacity for specialized knowledge |

### By Dataset Size

| Dataset Size | Minimum Model | Recommended |
|--------------|---------------|-------------|
| < 10 MB | `tiny.json` | `small.json` |
| 10-100 MB | `small.json` | `150M.json` |
| 100 MB - 1 GB | `150M.json` | `350M_1024.json` |
| 1-10 GB | `350M_1024.json` | `750M_1024.json` |
| 10+ GB | `750M_1024.json` | `750M_1024.json` |

**Rule of thumb:** Model parameters should be roughly 10-20% of your token count for good learning.

---

## Quality vs Speed Trade-offs

### Training Speed

| Factor | Faster ← → Slower |
|--------|-------------------|
| Model size | Smaller → Larger |
| Context length | 256 → 1024 |
| Batch size | Larger → Smaller |
| Dataset size | Smaller → Larger |

### Quality

| Factor | Lower ← → Higher |
|--------|------------------|
| Model size | Smaller → Larger |
| Training iterations | Fewer → More |
| Dataset quality | Noisy → Clean |
| Dataset size | Smaller → Larger |

### Sweet Spots

| Priority | Configuration |
|----------|---------------|
| **Speed over quality** | `small.json`, 256 context, 2000 iters |
| **Balanced** | `150M.json`, 256 context, 10000 iters |
| **Quality over speed** | `750M_1024.json`, 1024 context, 100000+ iters |

---

## Domain Adaptation Considerations

For Phase 2 (domain adaptation), consider:

### Model Size

| Base Model | Domain Data | Notes |
|------------|-------------|-------|
| 150M | 10-50M tokens | May underfit if domain is complex |
| 350M | 50-200M tokens | Good balance |
| 750M | 100-500M tokens | Best for specialized domains |

### Learning Rate Adjustments

| Scenario | Learning Rate |
|----------|---------------|
| From scratch | 3e-4 |
| Domain adaptation (cautious) | 3e-5 |
| Domain adaptation (balanced) | 5e-5 |
| Domain adaptation (aggressive) | 7e-5 |
| Fine-tuning (SFT) | 1e-5 to 3e-5 |

---

## Practical Examples

### Example 1: Personal Project (Limited Hardware)

**Scenario:** GTX 1660 (6GB), want to train on blog posts

**Recommendation:**
```bash
python train.py \
    --config_file configs/pretrain/small.json \
    --model_name my_blog_model \
    --input_file data/blog_posts.txt \
    --max_iters 5000
```

---

### Example 2: Company RAG System

**Scenario:** RTX 3090 (24GB), internal documents, need good quality

**Recommendation:**
```bash
# Prepare data
python scripts/prepare_weighted_dataset.py \
    --source docs:data/company_docs/*.txt \
    --out_dir data/company_shards

# Train with high context
python train.py \
    --config_file configs/pretrain/350M_1024.json \
    --model_name company_model \
    --dataset_dir data/company_shards \
    --max_iters 50000
```

---

### Example 3: Specialized Domain Expert

**Scenario:** A100 (40GB), IT security domain, need best quality

**Recommendation:**
```bash
# Phase 1: Base model on general data
python train.py \
    --config_file configs/pretrain/750M_1024.json \
    --model_name base_750M \
    --dataset_dir data/wiki_goldset \
    --max_iters 128000

# Phase 2: Domain adaptation
python train.py \
    --config_file configs/pretrain/750M_1024_domain_adapt.json \
    --model_name security_expert \
    --dataset_dir data/security_corpus \
    --init_from_model checkpoints/base_750M \
    --learning_rate 5e-5 \
    --max_iters 45000
```

---

## Configuration Parameters Explained

### Architecture Parameters

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `n_layer` | Deeper understanding, more compute |
| `n_embd` | Richer representations, more memory |
| `n_head` | More attention patterns, more compute |
| `block_size` | Longer context, quadratic memory |

### Training Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `batch_size` | Samples per step | 8-64 |
| `learning_rate` | Step size | 1e-5 to 3e-4 |
| `dropout` | Regularization | 0.0-0.2 |
| `warmup_iters` | LR warmup | 100-2000 |

### Recommended Combinations

| Model Size | batch_size | learning_rate | dropout |
|------------|------------|---------------|---------|
| tiny/small | 32-64 | 3e-4 | 0.1 |
| 150M | 24-32 | 3e-4 | 0.1 |
| 350M | 12-16 | 2e-4 | 0.1 |
| 750M | 8-12 | 1e-4 to 3e-4 | 0.1 |

---

## Testing Your Choice

Before committing to a long training run:

1. **Quick validation run:**
   ```bash
   python train.py \
       --config_file configs/pretrain/YOUR_CHOICE.json \
       --model_name test_run \
       --input_file data/sample.txt \
       --max_iters 100
   ```

2. **Check memory usage:**
   ```bash
   nvidia-smi -l 1  # Monitor GPU memory
   ```

3. **Verify loss decreases:**
   - Loss should drop noticeably in first 100 steps
   - If flat or erratic, adjust learning rate

---

## Summary Decision Matrix

| Question | Small (40M) | Medium (150-200M) | Large (350-750M) |
|----------|-------------|-------------------|------------------|
| VRAM < 8GB? | ✅ | ❌ | ❌ |
| Quick experiments? | ✅ | ⚠️ | ❌ |
| Production quality? | ❌ | ✅ | ✅ |
| Long documents? | ❌ | ⚠️ | ✅ |
| Domain expertise? | ❌ | ⚠️ | ✅ |
| Code generation? | ❌ | ⚠️ | ✅ |

---

*When in doubt, start with `150M.json` and scale up as needed.*

