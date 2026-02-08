# Training Configuration Storage

## Overview

MyPT now saves **ALL configuration** used for training, separating concerns into appropriate files:

---

## Configuration Files

### 1. `config.json` - Model Architecture
**Contains:** GPTConfig parameters (affects model structure)

```json
{
  "batch_size": 32,
  "block_size": 256,
  "vocab_size": 50304,
  "n_embd": 384,
  "n_head": 6,
  "n_layer": 6,
  "dropout": 0.2,
  "bias": false,
  "device": "cuda"
}
```

**Why separate?** 
- These define the neural network architecture
- Cannot be changed after model is created
- Must match exactly when loading weights

---

### 2. `training_state.json` - Training Progress & Hyperparameters
**Contains:** Training progress and hyperparameters

```json
{
  "step": 1000,
  "optimizer_file": "optimizer.pt",
  "training_config": {
    "max_iters": 2000,
    "eval_interval": 50,
    "eval_iters": 200,
    "learning_rate": 0.0003,
    "start_step": 0
  }
}
```

**Why separate?**
- Can be changed between training runs
- Useful for resuming with same settings
- Doesn't affect model architecture

---

## What Gets Saved Where

### Model Architecture (config.json)
**Cannot change after training starts:**
- `batch_size` - Batch size for training
- `block_size` - Context length (max sequence length)
- `vocab_size` - Vocabulary size
- `n_embd` - Embedding dimension
- `n_head` - Number of attention heads
- `n_layer` - Number of transformer layers
- `dropout` - Dropout rate
- `bias` - Use bias in layers

### Training State (training_state.json)
**Can change between runs:**
- `step` - Current training iteration
- `optimizer_file` - Reference to optimizer.pt
- `training_config`:
  - `max_iters` - Total iterations to train for
  - `eval_interval` - Evaluate every N steps
  - `eval_iters` - Iterations to average for evaluation
  - `learning_rate` - Learning rate for optimizer
  - `start_step` - Step training started from (useful for fine-tuning)

---

## Why This Separation?

### Problem: What if we bundled everything in config.json?

```json
{
  "n_layer": 6,        // Architecture - can't change
  "n_embd": 384,       // Architecture - can't change
  "max_iters": 1000,   // Training param - CAN change
  "learning_rate": 0.0003  // Training param - CAN change
}
```

**Issue:** Mixing concerns! Architecture params are fixed, training params can vary.

### Solution: Separate files

**config.json** (fixed architecture):
```json
{
  "n_layer": 6,
  "n_embd": 384,
  ...
}
```

**training_state.json** (mutable training settings):
```json
{
  "step": 1000,
  "training_config": {
    "max_iters": 2000,
    "learning_rate": 0.0003,
    ...
  }
}
```

**Benefits:**
- ✅ Clear separation of concerns
- ✅ Can change training params without affecting model
- ✅ Can resume with different max_iters or learning_rate
- ✅ Architecture is immutable and clear

---

## Example: Training Configuration Lifecycle

### First Training Run

```bash
python train.py --model_name dante \
                --input_file input.txt \
                --n_layer 6 \
                --max_iters 1000 \
                --learning_rate 3e-4
```

**Saves:**

`checkpoints/dante/config.json`:
```json
{
  "batch_size": 32,
  "block_size": 256,
  "vocab_size": 50304,
  "n_embd": 384,
  "n_head": 6,
  "n_layer": 6,
  "dropout": 0.2,
  "bias": false,
  "device": "cuda"
}
```

`checkpoints/dante/training_state.json`:
```json
{
  "step": 1000,
  "optimizer_file": "optimizer.pt",
  "training_config": {
    "max_iters": 1000,
    "eval_interval": 50,
    "eval_iters": 200,
    "learning_rate": 0.0003,
    "start_step": 0
  }
}
```

---

### Resume Training (Can Change Training Params)

```bash
# Continue training with MORE iterations and LOWER learning rate
python train.py --model_name dante \
                --max_iters 2000 \
                --learning_rate 1e-4
```

**What happens:**
1. Loads `config.json` → Model architecture (unchanged)
2. Loads `training_state.json` → Sees we're at step 1000
3. **Uses NEW max_iters (2000) and learning_rate (1e-4)** from CLI
4. Continues from step 1000 to 2000 with new learning rate

**Updated training_state.json:**
```json
{
  "step": 2000,
  "optimizer_file": "optimizer.pt",
  "training_config": {
    "max_iters": 2000,           // ✅ Updated
    "eval_interval": 50,
    "eval_iters": 200,
    "learning_rate": 0.0001,     // ✅ Updated
    "start_step": 1000            // ✅ Shows we resumed from 1000
  }
}
```

---

## Inspecting Training Configuration

### View with Cat/Type

```bash
# Windows
type checkpoints\dante\training_state.json

# Linux/Mac  
cat checkpoints/dante/training_state.json
```

### View with inspect_model.py

```bash
python scripts/inspect_model.py --model_name dante
```

Output now includes:
```
=== TRAINING STATE ===
Last recorded training step: 2000
Training configuration:
  max_iters: 2000
  eval_interval: 50
  eval_iters: 200
  learning_rate: 0.0001
  start_step: 1000
```

### Programmatic Access

```python
from core import get_model_info

info = get_model_info("dante")
training_config = info['training_state']['training_config']

print(f"Model was trained for {training_config['max_iters']} iterations")
print(f"Learning rate: {training_config['learning_rate']}")
```

---

## Use Cases

### 1. Reproducing Results

Someone asks: "How did you train this model?"

**Answer:** Just check the checkpoint files!
```bash
type checkpoints\dante\config.json          # Architecture
type checkpoints\dante\training_state.json  # Training hyperparameters
```

Full training configuration is saved and reproducible.

### 2. Resuming Training

Want to continue training:
```bash
# Original training
python train.py --model_name dante --max_iters 1000

# Resume with same settings (auto-detected from training_state.json)
python train.py --model_name dante --max_iters 2000

# Or override settings
python train.py --model_name dante --max_iters 2000 --learning_rate 1e-4
```

### 3. Comparing Training Runs

Compare different models:
```bash
# Model A
type checkpoints\modelA\training_state.json
# max_iters: 1000, learning_rate: 3e-4

# Model B
type checkpoints\modelB\training_state.json
# max_iters: 2000, learning_rate: 1e-4
```

Easy to see what settings were used!

### 4. Fine-Tuning

When fine-tuning, you can see the base model's training:
```bash
# Base model training settings
type checkpoints\dante_base\training_state.json

# Fine-tuned model training settings
type checkpoints\dante_purgatorio\training_state.json
```

Track the entire training lineage!

---

## What Gets Saved in Each File

### Complete Checkpoint Bundle

```
checkpoints/my_model/
├── model.pt                    # Neural network weights (~50 MB)
│   └── Contains: state_dict with all layer parameters
│
├── config.json                 # Model architecture (< 1 KB)
│   └── Contains: GPTConfig (n_layer, n_embd, n_head, etc.)
│
├── tokenizer.json              # Vocabulary (< 1 KB for GPT-2, varies for char)
│   └── Contains: token_kind, chars (for char-level)
│
├── training_state.json         # Training metadata (< 1 KB)
│   ├── step: Current iteration number
│   ├── optimizer_file: Reference to optimizer.pt
│   └── training_config:
│       ├── max_iters: Target iteration count
│       ├── eval_interval: Evaluation frequency
│       ├── eval_iters: Evaluation iterations
│       ├── learning_rate: Learning rate
│       └── start_step: Starting step (for resume tracking)
│
└── optimizer.pt                # Optimizer state (~100 MB)
    └── Contains: Adam momentum, variance, etc.
```

---

## Benefits

### Reproducibility
✅ All settings saved - can reproduce exact training  
✅ Easy to inspect - just view JSON files  
✅ Version control friendly - JSON diffs are readable  

### Flexibility
✅ Can change training params when resuming  
✅ Architecture is protected (immutable)  
✅ Training history is tracked (start_step)  

### Debugging
✅ Easy to compare models  
✅ Clear what settings were used  
✅ Can inspect without loading model  

---

---

## Gold Checkpoint (Best-of-Run)

Training automatically saves a **Gold checkpoint** -- the best model by validation loss -- alongside the regular (latest) checkpoint. This ensures you always have the optimal model from a run, even if training continues past the sweet spot into overfitting. Works for both base training and SFT.

### Checkpoint Layout After Training

```
checkpoints/
├── my_model/          # Last step (resume training)
├── my_model_gold/     # Best val loss step (evaluation / deployment)
└── my_model_fp16/     # Last step in fp16 (lightweight deployment)
```

| Directory | Contents | Purpose |
|-----------|----------|---------|
| `<model_name>/` | Last step | Resume training |
| `<model_name>_gold/` | Best val loss step | Evaluation / deployment |
| `<model_name>_fp16/` | Last step in fp16 | Lightweight deployment |

### Overfit Guards

Gold is **not** blindly saved on every new val-loss minimum. Two guards prevent degenerate gold saves where an overfitting model flukes a low val loss:

1. **Overfit ratio guard** -- If `val_loss / train_loss > 5x`, the model is memorizing, not generalizing. Gold is blocked.
2. **Trend guard (anti-flapping)** -- If val loss has been rising for 2+ consecutive evals before a dip, we've passed the sweet spot. The dip is likely noise, not genuine improvement. Gold is blocked.

Both guards are checked independently. Either one blocking is sufficient to prevent a gold save.

### Training Log Output

During training:
```
step 200: val 0.6326 | lr 7.00e-05
  🏆 New GOLD checkpoint! val_loss=0.6326 at step 200

step 300: val 0.5173 | lr 7.00e-05
  🏆 New GOLD checkpoint! val_loss=0.5173 at step 300

step 600: val 0.4900 | lr 7.00e-05
  ⚠️  GOLD blocked: val 0.4900 < best 0.5173 but overfit (val/train=490.0x, threshold=5x)
```

End-of-training summary:
```
Training finished!
  Resume training from: checkpoints/my_model
  Deploy inference from: checkpoints/my_model_fp16
  🏆 Best (GOLD) checkpoint: checkpoints/my_model_gold (step 300, val_loss=0.5173)
```

### When to Use the Gold Checkpoint

- **Always evaluate gold**, not just the final checkpoint
- If gold was never blocked → gold = final (healthy training, no overfitting detected)
- If gold was blocked early → LR is too high or dataset coverage is too high -- reduce and re-run
- Gold includes optimizer state, so you can resume training from the best point if needed

### Configuration

The guard thresholds are defined in `core/model.py` inside the `fit()` method:

| Constant | Default | Meaning |
|----------|---------|---------|
| `GOLD_OVERFIT_RATIO` | 5.0 | Max val/train ratio before gold is blocked |
| `GOLD_CONSEC_RISES` | 2 | Consecutive val increases before trend guard triggers |

---

## Dual Evaluation for Domain Adaptation

When performing domain adaptation (Phase 2 training), you can evaluate on multiple datasets:

### CLI Usage

```bash
python train.py \
    --dataset_dir data/domain_corpus \
    --eval_dataset_dir data/general_eval \
    --init_from_model checkpoints/base_model \
    --model_name domain_adapted
```

### Programmatic Usage

```python
from core import GPTDataLoader

# Primary loader (train + val)
domain_loader = GPTDataLoader(config, tokenizer, dataset_dir="data/domain_corpus")

# Additional eval loader (val only)
general_loader = GPTDataLoader(
    config, tokenizer,
    dataset_dir="data/general_eval",
    eval_only=True  # Only loads val shards
)

# Train with dual evaluation
model.fit(
    data_loader=domain_loader,
    optimizer=optimizer,
    eval_data_loaders={"general": general_loader}
)
```

### Output Format

Training logs show both metrics:
```
step 1000: val 2.45 | eval_general 2.89
step 2000: val 2.31 | eval_general 2.91
```

JSONL log entries include all eval losses:
```json
{"iter": 1000, "val_loss": 2.45, "eval_general": 2.89, "train_loss": 2.52}
```

---

## Summary

**Current storage:**

| File | Contains | Can Change? | Size |
|------|----------|-------------|------|
| `config.json` | Model architecture | ❌ No (fixed) | < 1 KB |
| `model.pt` | Neural net weights | ❌ No (trained) | ~50 MB |
| `tokenizer.json` | Vocabulary | ❌ No (fixed) | < 1 KB |
| `training_state.json` | Progress + training hyperparameters | ✅ Yes (flexible) | < 1 KB |
| `optimizer.pt` | Optimizer state | ✅ Yes (can reset) | ~100 MB |

**Everything you need for full reproducibility and flexibility!** 🎯

