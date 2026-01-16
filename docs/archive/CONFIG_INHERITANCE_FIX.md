# Config Inheritance Fix for Fine-Tuning

## The Problem

When fine-tuning a model with `--init_from_model`, the base model's config was being loaded and the new config was ignored. This meant that settings like `use_loss_mask: true` in SFT configs were not being applied.

### Example of the Bug

```bash
# Base model (use_loss_mask: false)
python train.py \
    --config_file configs/200M.json \
    --model_name base_200M \
    --dataset_dir data/corpus \
    --max_iters 50000

# Try to fine-tune with SFT config (use_loss_mask: true)
python train.py \
    --config_file configs/200M_chat_sft.json \  # Has use_loss_mask: true
    --init_from_model base_200M \
    --model_name chat_200M \
    --dataset_dir data/chat_sft \
    --max_iters 10000

# BUG: Model would still have use_loss_mask: false!
# The SFT config was being ignored during fine-tuning
```

---

## The Solution

Modified `initialize_for_training()` in `core/checkpoint.py` to:

1. **Load weights and architecture** from the base model
2. **Update training parameters** from the new config

### What Gets Updated

When fine-tuning with `--init_from_model`, the following happens:

#### ✅ Kept from Base Model (Architecture)

These parameters are **fixed** because they define the model's architecture (weights have specific shapes):

- `n_layer` - Number of transformer layers
- `n_embd` - Embedding dimension
- `n_head` - Number of attention heads
- `vocab_size` - Vocabulary size (unless char tokenization)
- `bias` - Whether to use bias in linear layers

#### ✅ Updated from New Config (Training Params)

These parameters are **mutable** and control training behavior:

- `use_loss_mask` - Enable loss masking for SFT
- `dropout` - Dropout probability
- `batch_size` - Training batch size
- `device` - Device to use (CPU/GPU)

**Note:** When `dropout` is changed, the actual dropout layers in the model are also updated.

---

## Code Changes

**File: `core/checkpoint.py`**

```python
# After loading base model...
# Update model config with new training parameters
print(f"Updating model config with new training parameters...")
model.config.batch_size = config.batch_size
model.config.dropout = config.dropout
model.config.use_loss_mask = config.use_loss_mask
model.config.device = config.device

# Update dropout in model layers if it changed
if hasattr(model, 'blocks'):
    for block in model.blocks:
        if hasattr(block, 'sa') and hasattr(block.sa, 'dropout'):
            block.sa.dropout.p = config.dropout
        if hasattr(block, 'fwd') and hasattr(block.fwd, 'net'):
            for layer in block.fwd.net:
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = config.dropout

print(f"  use_loss_mask: {model.config.use_loss_mask}")
print(f"  dropout: {model.config.dropout}")
print(f"  batch_size: {model.config.batch_size}")
```

---

## Usage Examples

### Example 1: SFT Fine-Tuning

```bash
# Base model (standard config, no loss masking)
python train.py \
    --config_file configs/200M.json \
    --model_name base_200M \
    --dataset_dir data/general_corpus \
    --max_iters 50000

# SFT fine-tuning (enables loss masking)
python train.py \
    --config_file configs/200M_chat_sft.json \  # use_loss_mask: true, dropout: 0.1
    --init_from_model base_200M \
    --model_name chat_200M \
    --dataset_dir data/chat_sft \
    --max_iters 10000 \
    --learning_rate 3e-5

# ✅ Now correctly applies:
#    - use_loss_mask: true (from SFT config)
#    - dropout: 0.1 (from SFT config, down from 0.2)
#    - Keeps architecture (n_layer: 16, n_embd: 896, etc.) from base model
```

### Example 2: Changing Dropout

```bash
# Base model with dropout 0.2
python train.py \
    --config_file configs/150M.json \  # dropout: 0.2
    --model_name base_150M \
    --input_file data.txt \
    --max_iters 10000

# Fine-tune with lower dropout
python train.py \
    --config_file configs/150M_chat_sft.json \  # dropout: 0.1
    --init_from_model base_150M \
    --model_name finetuned_150M \
    --input_file specialized_data.txt \
    --max_iters 5000

# ✅ Dropout is updated to 0.1 in both config and model layers
```

### Example 3: Changing Batch Size

```bash
# Base model with batch_size: 32
python train.py \
    --config_file configs/250M.json \  # batch_size: 32
    --model_name base_250M \
    --dataset_dir data/corpus \
    --max_iters 50000

# Fine-tune with smaller batch size (lower VRAM usage)
# Create custom config: 250M_lowmem.json with batch_size: 16
python train.py \
    --config_file configs/250M_lowmem.json \  # batch_size: 16
    --init_from_model base_250M \
    --model_name finetuned_250M \
    --dataset_dir data/specialized \
    --max_iters 10000

# ✅ Batch size is updated to 16
```

---

## Verification

You can verify the config is correctly applied by checking the training output:

```
Initializing from base model 'base_200M' (new format)
Using base model's char vocabulary (size: 256)
Updating model config with new training parameters...
  use_loss_mask: True      ← ✅ Correctly updated
  dropout: 0.1             ← ✅ Correctly updated
  batch_size: 32
Using config: batch_size:32, block_size:256, vocab_size:256, n_embd:896, 
              n_head:14, n_layer:16, dropout:0.1, bias:False, device:cuda, 
              use_loss_mask:True   ← ✅ Shows in final config
```

Or add debug prints to `train.py`:

```python
print(f"Using loss masking: {model.config.use_loss_mask}")
print(f"Dropout: {model.config.dropout}")
```

---

## Why This Matters

### For SFT Training

This fix is **critical** for SFT workflows:

1. **Base pre-training:** Train on general corpus with `use_loss_mask: false`
2. **SFT fine-tuning:** Fine-tune on chat data with `use_loss_mask: true`

Without this fix, step 2 would train without loss masking, defeating the purpose of SFT.

### For Dropout Tuning

Lower dropout during fine-tuning is a common practice:

- Base training: `dropout: 0.2` (more regularization)
- Fine-tuning: `dropout: 0.1` (less aggressive, preserve base knowledge)

Without this fix, dropout would stay at 0.2.

---

## Technical Details

### Why Not Update All Config Fields?

**Architecture fields (n_layer, n_embd, etc.) cannot change** because:

- The loaded weights have specific shapes
- Changing n_embd from 768 to 896 would require reshaping embedding matrices
- Changing n_layer from 12 to 16 would require adding new layer weights

**Training fields (use_loss_mask, dropout, etc.) can change** because:

- They control training behavior, not architecture
- Dropout is just a probability, can be changed
- Batch size doesn't affect model structure
- use_loss_mask is a data loading parameter

### Dropout Update Details

When dropout changes, we update:

1. **Config value:** `model.config.dropout = new_dropout`
2. **Attention dropout:** `block.sa.dropout.p = new_dropout`
3. **Feedforward dropout:** `layer.p = new_dropout` for all Dropout layers

This ensures the actual dropout layers use the new probability.

---

## Related Documentation

- [SFT Loss Masking Guide](SFT_LOSS_MASKING.md) - Complete SFT documentation
- [Fine-Tuning Guide](../README.md#fine-tuning--transfer-learning) - General fine-tuning
- [Config Presets](../configs/README.md) - All available configs

---

## Summary

**The fix:**

✅ Preserves base model's architecture (weights, layers, dimensions)  
✅ Updates training parameters (use_loss_mask, dropout, batch_size)  
✅ Enables proper SFT workflows  
✅ Allows dropout tuning during fine-tuning  
✅ Provides clear output of what was updated  

**When fine-tuning:**

```
Loaded from base model:  Architecture (n_layer, n_embd, n_head, vocab_size, bias)
Updated from new config: Training params (use_loss_mask, dropout, batch_size, device)
```

This gives you the best of both worlds: leverage pre-trained knowledge while customizing training behavior.

