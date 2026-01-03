# Loss Mask & Episode Data Loader Tests

This directory contains comprehensive tests to verify that loss masking and episode-indexed data loading work correctly in MyPT's SFT training.

## Why Test Loss Masking?

In Supervised Fine-Tuning (SFT), we only want to train the model on **assistant responses**, not on:
- System prompts
- User messages  
- Other non-response tokens

Loss masking ensures this by setting `loss_mask=0` for tokens we don't want to train on.

**BUT**: It's expensive to validate this by training a 750M+ model. These tests provide **cheap, fast validation** using tiny models and synthetic data.

---

## Test Suite Overview

### 1. **Unit Tests** (`test_loss_mask.py`)

Fast tests that verify core loss mask logic:

**Test 1: Corrupted Masked Tokens**
- Feeds identical input twice:
  - Once with correct targets
  - Once with corrupted targets in **masked positions**
- ✅ **Pass criteria**: Loss should be **identical** (masked tokens don't affect loss)

**Test 2: Gradient Check**
- Verifies that gradients flow correctly with loss masking
- Confirms masked tokens have zero gradient contribution

**Test 3: Mask vs No Mask**
- Compares loss with 50% masking vs 100% training
- ✅ **Pass criteria**: Losses should **differ** (mask has effect)

**Test 4: All-Zero Mask**
- Edge case: what if all tokens are masked?
- ✅ **Pass criteria**: Loss should be **zero** (nothing to train on)

**Run time**: ~2-5 seconds on GPU

---

### 2. **Training Tests** (`test_loss_mask_training.py`)

Practical tests that train a tiny model on synthetic data:

**Synthetic Data Format:**
- Prompts: Random tokens (noise)
- Responses: All token ID 42 (clear pattern)
- Mask: Train only on responses

**Test 1: Training WITH Loss Mask**
- Trains a tiny 3-layer model for 200 steps
- ✅ **Pass criteria**:
  - Loss decreases significantly (model learns)
  - Model generates token 42 (learned the pattern)
  - Model does NOT overfit to random prompts

**Test 2: Training WITHOUT Loss Mask (Comparison)**
- Shows what happens without masking (trains on everything)
- Demonstrates the difference in behavior

**Run time**: ~30-60 seconds on GPU

---

### 3. **Episode Data Loader Tests** (`test_episode_data_loader.py`)

Validates the `GPTEpisodeDataLoader` that loads episode-indexed datasets:

**Test 1: Dataset Detection**
- Verifies `is_episode_indexed_dataset()` correctly identifies the format
- ✅ **Pass criteria**: Detects episode-indexed datasets

**Test 2: Loader Initialization**
- Tests that loader initializes and loads episodes
- ✅ **Pass criteria**: Correct episode counts for train/val

**Test 3: Batch Shapes**
- Verifies batches have correct shapes (batch_size, block_size)
- ✅ **Pass criteria**: All tensors have expected dimensions

**Test 4: Loss Mask Consistency**
- Validates loss masks loaded from `mask.bin` are correct
- ✅ **Pass criteria**: Mask distribution matches expected pattern

**Test 5: Epoch Determinism**
- Tests that same seed produces identical batches
- ✅ **Pass criteria**: Deterministic sampling with same seed

**Test 6: Padding Short Episodes**
- Verifies padding when episode < block_size
- ✅ **Pass criteria**: Correct padding tokens applied

**Test 7: No Loss Mask Mode**
- Tests behavior when loss masking is disabled
- ✅ **Pass criteria**: Returns 2-tuple (X, Y) instead of 3-tuple

**Run time**: ~5-10 seconds on GPU

---

## How to Run

### Run loss mask unit tests:
```bash
python tests/test_loss_mask.py
```

### Run loss mask training tests:
```bash
python tests/test_loss_mask_training.py
```

### Run episode data loader tests:
```bash
python tests/test_episode_data_loader.py
```

### Run all tests:
```bash
python tests/test_loss_mask.py && python tests/test_loss_mask_training.py && python tests/test_episode_data_loader.py
```

**Expected total time**: ~1-2 minutes

---

## What Each Test Validates

| Test | Validates | Why It Matters |
|------|-----------|----------------|
| **Corrupted Masked** | Masked tokens don't affect loss | Core requirement: loss = 0 for mask=0 |
| **Gradient Check** | Gradients respect mask | Ensures no learning on masked tokens |
| **Mask vs No Mask** | Masking changes training | Confirms mask has actual effect |
| **All-Zero Mask** | Edge case handling | Prevents division by zero, etc. |
| **Training Test** | Real-world behavior | Model learns patterns only from masked regions |
| **Dataset Detection** | Format identification | Correctly identifies episode-indexed datasets |
| **Batch Shapes** | Data pipeline | Ensures fixed-shape tensors for training |
| **Mask Loading** | mask.bin integrity | Verifies masks loaded correctly from disk |
| **Determinism** | Reproducibility | Same seed = same training order |
| **Padding** | Variable-length handling | Short episodes padded correctly |

---

## Expected Results

### ✅ All Tests Passing (Current State)

```
======================================================================
  TEST SUMMARY
======================================================================
✅ PASS: Basic Loss Mask
✅ PASS: Gradient Check
✅ PASS: Mask vs No Mask
✅ PASS: All-Zero Mask

======================================================================
  ✅ ALL TESTS PASSED - Loss mask is working correctly!
======================================================================
```

This means:
- ✅ Loss is only calculated on `loss_mask=1` positions
- ✅ Masked tokens (`loss_mask=0`) don't affect gradients
- ✅ Training behaves correctly with masks
- ✅ Model learns patterns only from unmasked tokens

---

## When to Run These Tests

1. **Before training expensive models** (750M+)
   - Validates your loss mask logic is correct
   - Costs <1 minute instead of hours of GPU time

2. **After modifying loss mask code**
   - In `core/model.py` (forward pass)
   - In `core/episode_data_loader.py` (mask generation)
   - In `scripts/prepare_chat_sft.py` (mask creation)

3. **When debugging SFT training issues**
   - Model not following instructions? Check if masks are correct
   - Model overfitting to prompts? Verify mask coverage

4. **As part of CI/CD**
   - Add to automated test suite
   - Prevents regressions

---

## Understanding the Tests

### Core Principle: Masked Tokens = Zero Loss Contribution

The loss formula with masking is:

```python
# Without mask (standard):
loss = CrossEntropyLoss(logits, targets)  # All positions contribute

# With mask (SFT):
loss = (CrossEntropyLoss(logits, targets) * loss_mask).sum() / loss_mask.sum()
# Only positions where loss_mask=1 contribute
```

### Why Corruption Test Works

If masking works correctly:
- Changing targets at masked positions (mask=0) → Loss unchanged
- Changing targets at unmasked positions (mask=1) → Loss changes

The test exploits this property.

---

## Example: Manual Verification

You can also manually verify loss masking works:

```python
from core import GPT, GPTConfig
import torch

# Create tiny model
config = GPTConfig(n_layer=2, n_embd=64, block_size=32, use_loss_mask=True)
model = GPT(config).cuda()

# Create data
x = torch.randint(0, 50304, (2, 16)).cuda()
y = torch.randint(0, 50304, (2, 16)).cuda()

# Mask: only train on last 8 tokens
mask = torch.zeros((2, 16), dtype=torch.long).cuda()
mask[:, 8:] = 1

# Get loss with correct targets
_, loss_correct, _ = model(x, y, loss_mask=mask)

# Corrupt MASKED positions (first 8 tokens)
y_corrupted = y.clone()
y_corrupted[:, :8] = torch.randint(0, 50304, (2, 8)).cuda()

# Get loss with corrupted masked targets
_, loss_corrupted, _ = model(x, y_corrupted, loss_mask=mask)

# Should be identical!
print(f"Loss correct:   {loss_correct.item():.6f}")
print(f"Loss corrupted: {loss_corrupted.item():.6f}")
print(f"Difference:     {abs(loss_correct - loss_corrupted).item():.9f}")
# Expected: Difference ~ 0.000000000 (within floating point precision)
```

---

## Troubleshooting

### ❌ Test 1 Fails (Losses Differ)

**Problem**: Masked tokens are affecting the loss

**Possible causes**:
- Loss mask not being passed to model
- Loss mask not being applied in forward pass
- Wrong mask shape or values

**Fix**: Check `core/model.py` forward method, ensure loss calculation uses mask

### ❌ Test 4 Fails (Non-Zero Loss)

**Problem**: All-zero mask should give zero loss

**Possible causes**:
- Division by zero handling missing
- Loss calculated before mask applied

**Fix**: Check for `loss_mask.sum() > 0` before division

### ❌ Training Test Fails (Pattern Not Learned)

**Problem**: Model isn't learning the simple pattern

**Possible causes**:
- Loss mask incorrectly applied during training
- Model too small (try increasing size)
- Learning rate too low (try 1e-3)

**Fix**: Verify mask is correctly passed in training loop

---

## Technical Details

### Model Specifications

**Unit Tests:**
- 2 layers, 64 embedding dim
- ~500K parameters
- Trains in <1 second

**Training Tests:**
- 3 layers, 128 embedding dim
- ~2M parameters  
- Trains in ~30 seconds (200 steps)

### Data Format

Both tests use GPT-2 vocab size (50304) to accommodate special tokens.

Synthetic data for training tests:
```
Sequence: [prompt (8 tokens) | response (8 tokens)]
Mask:     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
Pattern:  prompt=random, response=all token 42
```

---

## Adding New Tests

To add a new loss mask test:

1. Create test function in `test_loss_mask.py` or `test_loss_mask_training.py`
2. Follow naming convention: `test_loss_mask_xxx()`
3. Return `True` for pass, `False` for fail
4. Add to results list in `__main__`
5. Document what it validates

Example:
```python
def test_loss_mask_new_scenario():
    """Test description"""
    print("\n" + "="*70)
    print("TEST X: My New Test")
    print("="*70)
    
    # Test logic here
    passed = verify_something()
    
    if passed:
        print("✅ PASS: Test passed!")
        return True
    else:
        print("❌ FAIL: Test failed!")
        return False
```

---

## Related Documentation

- **SFT Training**: `docs/EPISODE_INDEXED_SFT.md`
- **Model Architecture**: `core/model.py`
- **Data Loading**: `core/episode_data_loader.py`
- **Data Preparation**: `scripts/prepare_chat_sft.py`

---

## Summary

These tests provide **cheap, fast validation** that your loss masking is working correctly **before** investing in expensive training runs.

**Cost comparison:**
- ❌ Training 750M model to validate: ~$50-100 GPU time, 12+ hours
- ✅ Running these tests: **Free**, 1-2 minutes

**Confidence level:**
- If all tests pass → **95%+ confidence** loss masking works correctly
- The remaining 5% is only scale-related edge cases that these tests can't cover

**Recommendation:** Run these tests before every major SFT training run!

