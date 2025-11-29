# Memory Requirements Guide

## Understanding Memory Usage: Training vs Generation

When working with GPT models, memory requirements differ significantly between training and generation (inference). This guide explains why and helps you plan accordingly.

---

## Quick Reference

### Standard Configs (block_size=256)

| Config       | Training Memory | Generation Memory | Ratio  |
| ------------ | --------------- | ----------------- | ------ |
| **tiny**     | ~500 MB         | ~100 MB           | 5:1    |
| **small**    | ~2 GB           | ~300 MB           | 6.7:1  |
| **150M**     | ~8 GB           | ~1 GB             | 8:1    |
| **200M**     | ~10 GB          | ~1.2 GB           | 8.3:1  |
| **250M**     | ~12 GB          | ~1.5 GB           | 8:1    |

### High-Context Configs (block_size=1024)

| Config            | Training Memory | Generation Memory | Ratio  |
| ----------------- | --------------- | ----------------- | ------ |
| **150M_1024**     | ~10 GB          | ~1.5 GB           | 6.7:1  |
| **200M_1024**     | ~12 GB          | ~2 GB             | 6:1    |
| **250M_1024**     | ~14 GB          | ~2.5 GB           | 5.6:1  |
| **350M_1024**     | ~18 GB          | ~3 GB             | 6:1    |
| **500M_1024**     | ~22 GB          | ~4 GB             | 5.5:1  |

**Key insight:** Training uses **5-8x more memory** than generation!

---

## Why the Difference?

### Training Memory Components

When training, you need memory for:

1. **Model Weights** (1x)
   - All parameters in the neural network
   - Example: 150M params × 4 bytes (FP32) = 600 MB

2. **Gradients** (1x)
   - Derivative of loss w.r.t each parameter
   - Same size as model weights
   - Example: 150M params × 4 bytes = 600 MB

3. **Optimizer States** (2x for Adam)
   - Momentum: 1x model size
   - Variance: 1x model size
   - Example: 150M params × 4 bytes × 2 = 1200 MB

4. **Activations** (varies with batch_size and block_size)
   - Intermediate values during forward pass
   - Needed for backpropagation
   - Scales with: batch_size × block_size² × n_layers × n_embd
   - Example for 150M (batch=32, block=256): ~4-6 GB

**Total Training ≈ Model + Gradients + Optimizer + Activations**  
**≈ 1x + 1x + 2x + (large) = 4x + activations**

---

### Generation Memory Components

When generating text, you only need:

1. **Model Weights** (1x)
   - All parameters in the neural network
   - Example: 150M params × 4 bytes (FP32) = 600 MB

2. **KV Cache** (small, grows with generation length)
   - Cached key/value tensors for efficient generation
   - Much smaller than training activations
   - Example: ~200-400 MB for typical generation

3. **Small Working Memory**
   - Current tokens, attention outputs
   - Usually < 100 MB

**Total Generation ≈ Model + KV Cache + Working Memory**  
**≈ 1x + small overhead**

---

## Detailed Breakdown

### Example: 150M Model (block_size=256)

#### Training (8 GB total)

```
Model weights:        600 MB  (150M params × 4 bytes)
Gradients:            600 MB  (150M params × 4 bytes)
Optimizer (Adam):   1,200 MB  (150M params × 4 bytes × 2)
Activations:      ~5,600 MB  (batch=32, block=256, layers=16)

Total:            ~8,000 MB  (8 GB)
```

#### Generation (1 GB total)

```
Model weights:        600 MB  (150M params × 4 bytes)
KV cache:            ~300 MB  (generation length up to 1024 tokens)
Working memory:      ~100 MB  (current batch processing)

Total:            ~1,000 MB  (1 GB)
```

**Savings: 8 GB → 1 GB (87% reduction!)**

---

### Example: 150M Model with High Context (block_size=1024)

#### Training (10 GB total)

```
Model weights:        600 MB  (same as before)
Gradients:            600 MB  (same as before)
Optimizer (Adam):   1,200 MB  (same as before)
Activations:      ~7,600 MB  (batch=24, block=1024) ← larger!

Total:           ~10,000 MB  (10 GB)
```

**Why more activations?**
- Attention scores: batch × heads × block_size × block_size
- 1024² = 1,048,576 vs 256² = 65,536 (16x more!)

#### Generation (1.5 GB total)

```
Model weights:        600 MB  (same as before)
KV cache:            ~700 MB  (can cache up to 1024 tokens) ← larger!
Working memory:      ~200 MB  (larger context)

Total:            ~1,500 MB  (1.5 GB)
```

**Still 85% reduction compared to training!**

---

## Practical Implications

### You Can Generate on Smaller GPUs!

**Training a 150M model:**
- Needs: RTX 3060 (12GB) or better
- Uses: ~8 GB VRAM

**Generating with same model:**
- Works on: GTX 1070 (8GB), RTX 2060 (8GB)
- Uses: ~1 GB VRAM

**Example workflow:**
```bash
# Train on powerful GPU (RTX 3090 with 24GB)
python train.py --config_file configs/250M.json \
                --model_name my_model \
                --input_file data.txt

# Generate on modest GPU (RTX 2060 with 8GB)
python generate.py --model_name my_model \
                   --prompt "Your prompt"
```

---

### Batch Size Affects Training Memory (Not Generation)

**Training with different batch sizes (150M model):**

| Batch Size | Training Memory | Generation Memory |
| ---------- | --------------- | ----------------- |
| 8          | ~4 GB           | ~1 GB             |
| 16         | ~6 GB           | ~1 GB             |
| 32         | ~8 GB           | ~1 GB             |
| 64         | ~14 GB          | ~1 GB             |

**Generation always uses ~1 GB regardless of training batch size!**

---

### Block Size Affects Both (But Generation Less)

**150M model with different block sizes:**

| Block Size | Training Memory | Generation Memory | Training/Gen Ratio |
| ---------- | --------------- | ----------------- | ------------------ |
| 128        | ~6 GB           | ~700 MB           | 8.6:1              |
| 256        | ~8 GB           | ~1 GB             | 8:1                |
| 512        | ~12 GB          | ~1.3 GB           | 9.2:1              |
| 1024       | ~20 GB          | ~1.5 GB           | 13.3:1             |

**Key insight:** Higher block_size increases training memory much more than generation!

---

## Memory Optimization Tips

### For Training

1. **Reduce batch_size in config file**
   ```json
   {
     "batch_size": 16  // Instead of 32
   }
   ```

2. **Use gradient accumulation** (future feature)
   - Simulate larger batches without extra memory
   - Trade-off: slower training

3. **Use mixed precision (FP16)**
   - Reduces memory by ~50%
   - Slightly faster on modern GPUs
   - May need careful tuning

4. **Use smaller block_size**
   - 256 instead of 1024 if you don't need long context
   - Saves activation memory

### For Generation

1. **Generation is already efficient!**
   - Usually doesn't need optimization

2. **If needed, reduce generation length**
   ```bash
   python generate.py --model_name my_model \
                      --max_new_tokens 100  # Instead of 500
   ```

3. **Use FP16 for inference** (future feature)
   - Can reduce generation memory by ~50%
   - Minimal quality impact

---

## GPU Selection Guide

### For Training

| Your Goal          | Minimum GPU              | Recommended GPU     |
| ------------------ | ------------------------ | ------------------- |
| Experiments        | RTX 2060 (8GB)           | RTX 3060 (12GB)     |
| Small models       | RTX 3060 (12GB)          | RTX 3070 (8GB)      |
| 150M standard      | RTX 3060 (12GB)          | RTX 3070 Ti (8GB)   |
| 150M high-context  | RTX 3060 (12GB)          | RTX 3080 (10GB)     |
| 250M standard      | RTX 3070 Ti (8GB)        | RTX 3080 (10GB)     |
| 250M high-context  | RTX 3080 (10GB)          | RTX 3090 (24GB)     |
| 500M high-context  | RTX 3090 (24GB)          | RTX 4090 (24GB)     |
| Very large models  | RTX 4090 (24GB)          | A100 (40GB/80GB)    |

### For Generation

| Model Size       | Minimum GPU       | Comfortable GPU  |
| ---------------- | ----------------- | ---------------- |
| Tiny/Small       | Any GPU (2GB+)    | GTX 1060 (6GB)   |
| 150M             | GTX 1070 (8GB)    | RTX 2060 (8GB)   |
| 250M             | RTX 2060 (8GB)    | RTX 3060 (12GB)  |
| 500M             | RTX 3060 (12GB)   | RTX 3070 (8GB)   |
| 500M high-context| RTX 3070 (8GB)    | RTX 3080 (10GB)  |

**Key takeaway:** You can run inference on much smaller GPUs than needed for training!

---

## Checking Your Memory Usage

### On Windows

```powershell
# Check VRAM usage in real-time
nvidia-smi -l 1

# One-time check
nvidia-smi
```

### On Linux/Mac

```bash
# Watch VRAM usage
watch -n 1 nvidia-smi

# One-time check
nvidia-smi
```

### In Your Code

```python
import torch

# Check allocated memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Check reserved memory
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# Check max memory used
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

---

## Summary

**Training vs Generation Memory:**

| Aspect       | Training                           | Generation              |
| ------------ | ---------------------------------- | ----------------------- |
| Memory needs | **4-8x model size**                | **~1-1.5x model size**  |
| Components   | Weights + Gradients + Optimizer + Activations | Weights + KV cache |
| Example      | 150M model = 8 GB                  | 150M model = 1 GB       |
| GPU needed   | Powerful (RTX 3060+)               | Modest (GTX 1070+)      |
| Bottleneck   | Activation memory (scales with batch × block²) | Model weights  |

**Key insights:**

1. ✅ **Training needs 5-8x more memory than generation**
2. ✅ **You can generate on much smaller GPUs**
3. ✅ **block_size affects training much more than generation**
4. ✅ **batch_size only affects training, not generation**
5. ✅ **Activations are the biggest training memory user**

**Plan accordingly:**
- **Train:** Choose GPU based on training memory needs
- **Deploy/Generate:** Can use much smaller GPU
- **Share models:** Others can use them with modest hardware

**Never confuse training and generation memory requirements again!** 🎯

