# Dataset Coverage Analysis

## Overview

MyPT now includes an automatic **dataset coverage analysis** that calculates how many times your model will see the entire dataset during training. This helps you:
- ✅ Avoid underfitting (model doesn't see enough data)
- ✅ Avoid overfitting (model sees data too many times)
- ✅ Set optimal `max_iters` for your dataset
- ✅ Understand training progress in terms of "epochs"

---

## What is Dataset Coverage?

**Coverage** = How many times the model sees the entire dataset during training.

**Formula:**
```
Tokens viewed per iteration = batch_size × block_size
Total tokens viewed = max_iters × batch_size × block_size
Coverage ratio = Total tokens viewed / Dataset tokens
```

**Optimal range:** 2-5x (model sees dataset 2-5 times)

---

## Automatic Analysis

When you run `train.py`, you'll see:

```
======================================================================
Dataset Coverage Analysis
======================================================================
Dataset size:           1,234,567 tokens
Tokens per iteration:   8,192 tokens
Total iterations:       1,000
Total tokens to view:   8,192,000 tokens

Dataset coverage:       6.64x (663.8%)
Full passes:            6x + 63.8% of dataset

✅ GOOD: Coverage is in optimal range (2-5x)
   Model will see dataset multiple times for good learning.

Recommendations:
  Minimum (2x):         --max_iters 302
  Optimal (3.5x):       --max_iters 528
  Maximum (5x):         --max_iters 754
======================================================================
```

---

## Coverage Ranges

### < 0.5x (Very Low)
```
⚠️  WARNING: Very low coverage!
   Your model will only see a small fraction of the data.
   Risk: Underfitting, poor generalization
```

**Action:** Significantly increase `max_iters`

---

### 0.5-1.0x (Low)
```
⚠️  WARNING: Low coverage
   Your model won't see the entire dataset even once.
   Risk: Underfitting
```

**Action:** Increase `max_iters` to at least recommended minimum

**Interactive prompt:** Training will ask for confirmation before continuing

---

### 1.0-2.0x (Below Optimal)
```
ℹ️  Note: Below recommended coverage
   Model will see dataset ~1x. Consider training longer.
```

**Action:** Consider increasing `max_iters` for better results

---

### 2.0-5.0x (Optimal) ✅
```
✅ GOOD: Coverage is in optimal range (2-5x)
   Model will see dataset multiple times for good learning.
```

**Action:** Proceed with training

---

### 5.0-10.0x (High)
```
ℹ️  Note: High coverage
   Model will see dataset many times. May overfit on small datasets.
```

**Action:** Consider reducing `max_iters` or using a larger dataset

---

### > 10.0x (Very High)
```
⚠️  WARNING: Very high coverage
   Model will see dataset many times. Risk of overfitting.
   Consider: Reducing max_iters, or use a larger dataset.
```

**Action:** Reduce `max_iters` or use a larger/more diverse dataset

---

## Examples

### Example 1: Too Few Iterations

```bash
python train.py \
    --config_file configs/small.json \
    --model_name test \
    --input_file input.txt \
    --max_iters 100
```

**Output:**
```
Dataset Coverage Analysis
======================================================================
Dataset size:           1,000,000 tokens
Tokens per iteration:   8,192 tokens (32 batch × 256 block)
Total iterations:       100
Total tokens to view:   819,200 tokens

Dataset coverage:       0.82x (81.9%)
Progress:               [████████████████████████████████████████░░░░░░░░░░]

⚠️  WARNING: Low coverage
   Your model won't see the entire dataset even once.
   Risk: Underfitting

Recommendations:
  Minimum (2x):         --max_iters 245
  Optimal (3.5x):       --max_iters 428
  Maximum (5x):         --max_iters 611

💡 Suggestion:
   Increase max_iters to at least 245
   for 2x coverage (current: 100)
======================================================================

⚠️  Your model will not see the entire dataset!
Continue anyway? (y/n):
```

**Recommendation:** Increase to at least 245 iterations

---

### Example 2: Optimal Coverage

```bash
python train.py \
    --config_file configs/small.json \
    --model_name test \
    --input_file input.txt \
    --max_iters 500
```

**Output:**
```
Dataset Coverage Analysis
======================================================================
Dataset size:           1,000,000 tokens
Tokens per iteration:   8,192 tokens
Total iterations:       500
Total tokens to view:   4,096,000 tokens

Dataset coverage:       4.10x (409.6%)
Full passes:            4x + 9.6% of dataset

✅ GOOD: Coverage is in optimal range (2-5x)
   Model will see dataset multiple times for good learning.

Recommendations:
  Minimum (2x):         --max_iters 245
  Optimal (3.5x):       --max_iters 428
  Maximum (5x):         --max_iters 611
======================================================================
```

**Recommendation:** Proceed with training! 🎯

---

### Example 3: Too Many Iterations

```bash
python train.py \
    --config_file configs/small.json \
    --model_name test \
    --input_file input.txt \
    --max_iters 5000
```

**Output:**
```
Dataset Coverage Analysis
======================================================================
Dataset size:           1,000,000 tokens
Tokens per iteration:   8,192 tokens
Total iterations:       5,000
Total tokens to view:   40,960,000 tokens

Dataset coverage:       40.96x (4096.0%)
Full passes:            40x + 96.0% of dataset

⚠️  WARNING: Very high coverage
   Model will see dataset many times. Risk of overfitting.
   Consider: Reducing max_iters, or use a larger dataset.

Recommendations:
  Minimum (2x):         --max_iters 245
  Optimal (3.5x):       --max_iters 428
  Maximum (5x):         --max_iters 611

💡 Suggestion:
   Reduce max_iters to 611 for 5x coverage
   OR use a larger dataset to avoid overfitting
======================================================================
```

**Recommendation:** Reduce to ~400-600 iterations or use more data

---

## Calculating Optimal max_iters

### Manual Calculation

```python
# 1. Count your tokens
dataset_tokens = 1,000,000

# 2. Calculate tokens per iteration
tokens_per_iter = batch_size × block_size
                = 32 × 256
                = 8,192

# 3. For 3x coverage (recommended)
optimal_max_iters = (3.0 × dataset_tokens) / tokens_per_iter
                  = (3.0 × 1,000,000) / 8,192
                  = 366 iterations
```

### Using the Calculator

```python
from core import calculate_dataset_coverage

coverage = calculate_dataset_coverage(
    max_iters=1000,
    batch_size=32,
    block_size=256,
    total_tokens=1_000_000
)

print(f"Optimal: {coverage['recommended_optimal_iters']} iterations")
# Output: Optimal: 428 iterations
```

---

## Impact of Parameters

### Batch Size

**Larger batch_size** = More tokens per iteration = Fewer iterations needed

| Batch Size | Tokens/Iter | Iters for 3x | Training Time |
|------------|-------------|--------------|---------------|
| 16         | 4,096       | 732          | Longer        |
| 32         | 8,192       | 366          | Medium        |
| 64         | 16,384      | 183          | Shorter       |

**Trade-off:** Larger batches need more VRAM

---

### Block Size

**Larger block_size** = More tokens per iteration = Fewer iterations needed

| Block Size | Tokens/Iter | Iters for 3x | Context |
|------------|-------------|--------------|---------|
| 128        | 4,096       | 732          | Short   |
| 256        | 8,192       | 366          | Medium  |
| 512        | 16,384      | 183          | Long    |
| 1024       | 32,768      | 92           | Very long |

**Trade-off:** Larger context needs more VRAM

---

## Special Cases

### Resuming Training

When resuming, coverage analysis shows **total** coverage (including previous training):

```
Dataset Coverage Analysis
======================================================================
Dataset size:           1,000,000 tokens
Tokens per iteration:   8,192 tokens
Total iterations:       1,000  (starting from step 500)
Total tokens to view:   8,192,000 tokens

Dataset coverage:       8.19x (819.2%)
Full passes:            8x + 19.2% of dataset

Note: Starting from step 500, will train to step 1000 (500 more iterations)
Cumulative coverage after this run: ~12.3x
======================================================================
```

---

### Sharded Datasets

Works the same way! Total tokens are read from `dataset_metadata.json`:

```bash
python train.py \
    --dataset_dir data/my_large_dataset \
    --model_name my_model \
    --max_iters 10000
```

```
Dataset Coverage Analysis
======================================================================
Dataset size:           500,000,000 tokens  (from metadata)
Tokens per iteration:   8,192 tokens
Total iterations:       10,000
Total tokens to view:   81,920,000 tokens

Dataset coverage:       0.16x (16.4%)
Progress:               [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]

⚠️  WARNING: Very low coverage!
   Your model will only see a small fraction of the data.
   Risk: Underfitting, poor generalization

Recommendations:
  Minimum (2x):         --max_iters 122,100
  Optimal (3.5x):       --max_iters 213,675
  Maximum (5x):         --max_iters 305,250

💡 Suggestion:
   Increase max_iters to at least 122,100
   for 2x coverage (current: 10,000)
======================================================================
```

For 500M tokens, you need 120K+ iterations for 2x coverage!

---

## Best Practices

### 1. Start with Optimal Coverage (3-4x)

```bash
# Calculate optimal iterations
python -c "
dataset_tokens = 1_000_000  # Your dataset size
batch_size = 32
block_size = 256
tokens_per_iter = batch_size * block_size
optimal_iters = int(3.5 * dataset_tokens / tokens_per_iter)
print(f'Use: --max_iters {optimal_iters}')
"
```

### 2. Adjust Based on Results

**If validation loss plateaus early:**
- Model may be overfitting
- Reduce `max_iters` or add more data

**If validation loss still decreasing:**
- Model may benefit from more training
- Increase `max_iters`

### 3. Consider Dataset Size

**Small datasets (< 10M tokens):**
- Use 3-5x coverage
- Watch for overfitting

**Medium datasets (10-100M tokens):**
- Use 2-4x coverage
- Balanced training

**Large datasets (100M+ tokens):**
- Use 1-3x coverage
- Even 1x may be sufficient

### 4. Factor in Model Size

**Larger models need more data:**

| Model Size | Recommended Coverage | Reason |
|------------|---------------------|---------|
| Tiny (11M) | 3-5x | Small capacity, won't overfit easily |
| Small (40M) | 2-4x | Medium capacity |
| 150M+ | 1-3x | Large capacity, overfits faster |

---

## Disabling the Check

If you want to skip the coverage analysis and confirmation prompt:

**Option 1:** Set max_iters high enough (>= 1.0x coverage) to avoid the warning

**Option 2:** For automated training, pipe "yes":
```bash
echo "y" | python train.py --model_name test --input_file input.txt --max_iters 100
```

**Option 3:** Future enhancement - add `--skip_coverage_check` flag

---

## Summary

**Coverage analysis helps you:**
- ✅ Set optimal `max_iters` automatically
- ✅ Avoid wasting compute on under/over-training
- ✅ Understand training in terms of "epochs"
- ✅ Get recommendations tailored to your dataset

**Optimal coverage: 2-5x**
- **Less than 1x:** Model won't see all data (underfitting risk)
- **1-2x:** Below optimal, but may work
- **2-5x:** Optimal range ✅
- **5-10x:** High, may overfit on small datasets
- **10x+:** Very high, likely overfitting

**Quick formula:**
```
optimal_max_iters = (3.5 × dataset_tokens) / (batch_size × block_size)
```

**Never guess your training iterations again!** 🎯

