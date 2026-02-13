# Training Configuration Storage

## Overview

MyPT now saves **ALL configuration** used for training, separating concerns into appropriate files:

---

## Configuration Files

### 1. `config.json` - Model Architecture

**Contains:** GPTConfig parameters (affects model structure)

```json
{
  "batch_size": 12,
  "block_size": 1024,
  "vocab_size": 50304,
  "n_embd": 1280,
  "n_head": 20,
  "n_layer": 32,
  "dropout": 0.1,
  "bias": false,
  "device": "cuda",
  "pos_encoding": "rope",
  "mlp_type": "swiglu",
  "norm_type": "rmsnorm",
  "rope_theta": 10000.0
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
- `pos_encoding` - Position encoding type: `"learned"` (GPT-2) or `"rope"` (LLaMA-2)
- `mlp_type` - MLP activation: `"gelu"` (GPT-2) or `"swiglu"` (LLaMA-2)
- `norm_type` - Normalization: `"layernorm"` (GPT-2) or `"rmsnorm"` (LLaMA-2)
- `rope_theta` - RoPE base frequency (default: 10000.0, only when `pos_encoding="rope"`)

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
  - `grad_accum_steps` - Gradient accumulation steps (effective batch = batch_size \* grad_accum_steps)

---

## Why This Separation?

### Problem: What if we bundled everything in config.json?

```json
{
  "n_layer": 6, // Architecture - can't change
  "n_embd": 384, // Architecture - can't change
  "max_iters": 1000, // Training param - CAN change
  "learning_rate": 0.0003 // Training param - CAN change
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
    "max_iters": 2000, // ✅ Updated
    "eval_interval": 50,
    "eval_iters": 200,
    "learning_rate": 0.0001, // ✅ Updated
    "start_step": 1000 // ✅ Shows we resumed from 1000
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
├── model.pt                    # Neural network weights (~1.4 GB for 750M)
│   └── Contains: state_dict with all layer parameters
│
├── config.json                 # Model architecture (< 1 KB)
│   └── Contains: GPTConfig (n_layer, n_embd, n_head, pos_encoding, mlp_type, norm_type, etc.)
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
│       ├── start_step: Starting step (for resume tracking)
│       └── grad_accum_steps: Gradient accumulation factor
│
└── optimizer.pt                # Optimizer state (~2.8 GB for 750M)
    └── Contains: AdamW momentum, variance, etc.

logs/train/
└── <model_name>_eval.jsonl     # Structured training log (see JSONL Training Log section)
    └── Contains: training_start, eval, phase_switch, gold_checkpoint, gold_blocked, training_complete events
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

| Directory            | Contents           | Purpose                 |
| -------------------- | ------------------ | ----------------------- |
| `<model_name>/`      | Last step          | Resume training         |
| `<model_name>_gold/` | Best val loss step | Evaluation / deployment |
| `<model_name>_fp16/` | Last step in fp16  | Lightweight deployment  |

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

| Constant             | Default | Meaning                                               |
| -------------------- | ------- | ----------------------------------------------------- |
| `GOLD_OVERFIT_RATIO` | 5.0     | Max val/train ratio before gold is blocked            |
| `GOLD_CONSEC_RISES`  | 2       | Consecutive val increases before trend guard triggers |

---

## JSONL Training Log

Training produces a structured JSONL log file (one JSON object per line) at the path specified by `--log_file` (default: `logs/train/<model_name>_eval.jsonl`). This log is the primary record of a training run -- it captures everything needed to analyze training dynamics after the fact.

### Log File Location

```bash
# Automatically created at:
logs/train/<model_name>_eval.jsonl

# Or specified explicitly:
python train.py --model_name my_model --log_file logs/custom_log.jsonl
```

### Event Types

The log contains 5 event types, written in chronological order:

#### 1. `training_start` -- Written once at the beginning

```json
{
  "event": "training_start",
  "timestamp": "2026-02-09T14:30:00.123456",
  "config": {
    "max_iters": 75000,
    "start_step": 0,
    "learning_rate": 0.00015,
    "warmup_iters": 3750,
    "grad_clip": 1.0,
    "grad_accum_steps": 8,
    "batch_size": 12,
    "block_size": 1024,
    "tokens_per_step": 98304,
    "eval_interval": 350,
    "eval_iters": 200,
    "use_amp": true,
    "amp_dtype": "bf16",
    "config_file": "configs/base/750M_unified_v1.json",
    "dataset_dir": "data/unified_phase1_circuit"
  },
  "model": {
    "total_params": 699000000,
    "n_layer": 32,
    "n_head": 20,
    "n_embd": 1280,
    "pos_encoding": "rope",
    "mlp_type": "swiglu",
    "norm_type": "rmsnorm"
  }
}
```

#### 2. `eval` -- Written every `eval_interval` steps

```json
{
  "event": "eval",
  "iter": 350,
  "timestamp": "2026-02-09T14:45:12.654321",
  "elapsed_s": 912.5,
  "tokens_processed": 34406400,
  "progress_pct": 0.47,
  "lr": 0.000014,
  "phase": "circuit_formation",
  "val_loss": 4.123456,
  "eval_general": 4.234567,
  "eval_domain": 4.345678,
  "eval_code": 3.987654,
  "eval_retrieval": 4.56789,
  "eval_duration_s": 12.3
}
```

Fields:

- `iter` -- optimizer step number
- `timestamp` -- wall-clock ISO timestamp
- `elapsed_s` -- seconds since training started
- `tokens_processed` -- total tokens processed so far
- `progress_pct` -- percentage of training complete
- `lr` -- current learning rate
- `phase` -- current curriculum phase name
- `val_loss` -- validation loss on the primary dataset
- `eval_<name>` -- loss on each additional eval set (one field per eval loader)
- `eval_duration_s` -- how long this eval took (useful for ETA estimation)

All loss values are rounded to 6 decimal places.

#### 3. `phase_switch` -- Written when the curriculum changes data loaders

```json
{
  "event": "phase_switch",
  "iter": 9125,
  "timestamp": "2026-02-09T18:22:33.111222",
  "elapsed_s": 13953.0,
  "tokens_processed": 897024000,
  "new_phase": "balanced"
}
```

#### 4. `gold_checkpoint` / `gold_blocked` -- Written on GOLD checkpoint decisions

When a new best val loss is saved:

```json
{
  "event": "gold_checkpoint",
  "iter": 1050,
  "timestamp": "2026-02-09T15:12:44.333444",
  "val_loss": 3.876543,
  "prev_best": 3.912345
}
```

When a potential GOLD save is blocked by a guard:

```json
{
  "event": "gold_blocked",
  "iter": 50000,
  "timestamp": "2026-02-10T02:15:55.666777",
  "val_loss": 3.654321,
  "best_val_loss": 3.7,
  "reason": "overfit: val/train=6.2x, threshold=5.0x"
}
```

Possible `reason` values:

- `"overfit: val/train=Xx, threshold=5.0x"` -- model is memorizing
- `"trend: N consecutive val rises"` -- val loss has been climbing, dip is likely noise
- `"eval regression: <name> X.XXXX→Y.YYYY (+Z%, threshold=20%)"` -- an eval set degraded too much

#### 5. `training_complete` -- Written once at the end

```json
{
  "event": "training_complete",
  "timestamp": "2026-02-10T08:30:00.999888",
  "started_at": "2026-02-09T14:30:00.123456",
  "total_elapsed_s": 64800.0,
  "total_elapsed_h": 18.0,
  "total_steps": 75000,
  "total_tokens": 7372800000,
  "steps_per_sec": 1.16,
  "tokens_per_sec": 113778,
  "final_train_loss": 2.987654,
  "final_val_loss": 3.123456,
  "best_val_loss": 3.1,
  "gold_step": 72000,
  "config_file": "configs/base/750M_unified_v1.json",
  "dataset_dir": "data/unified_phase1_circuit"
}
```

### Console Output

During training, eval steps print a rich two-line summary:

```
step 350/75000 (0.5%) | 14:45:12 | 15m | ETA 52.3h | 34M tokens
  val 4.1235 | lr 1.40e-05 | phase: circuit_formation | general 4.2346 | domain 4.3457 | code 3.9877 | retrieval 4.5679
```

Line 1: progress, wall-clock time, elapsed, ETA, tokens processed.
Line 2: val loss, learning rate, curriculum phase, all eval set losses.

### Analyzing Log Files

The JSONL format is easy to parse. Each line is a self-contained JSON object:

```python
import json

with open("logs/train/unified_v1_eval.jsonl") as f:
    events = [json.loads(line) for line in f]

# Filter to eval events
evals = [e for e in events if e["event"] == "eval"]

# Plot val_loss over tokens
tokens = [e["tokens_processed"] for e in evals]
losses = [e["val_loss"] for e in evals]

# Find the GOLD checkpoint
gold = [e for e in events if e["event"] == "gold_checkpoint"]
print(f"Best checkpoint at step {gold[-1]['iter']}, val_loss={gold[-1]['val_loss']}")
```

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

Training logs show both metrics with full context:

```
step 1000/75000 (1.3%) | 15:22:04 | 28m | ETA 35.2h | 98M tokens
  val 2.4500 | lr 4.00e-05 | phase: balanced | general 2.8900 | domain 3.1200
```

JSONL log entries include all eval losses plus timestamps, progress, and phase (see **JSONL Training Log** section above for the full schema):

```json
{
  "event": "eval",
  "iter": 1000,
  "timestamp": "2026-02-09T15:22:04",
  "elapsed_s": 1680.0,
  "tokens_processed": 98304000,
  "progress_pct": 1.33,
  "lr": 4e-5,
  "phase": "balanced",
  "val_loss": 2.45,
  "eval_general": 2.89,
  "eval_domain": 3.12,
  "eval_duration_s": 8.5
}
```

---

## Summary

**Current storage:**

| File                  | Contains                                                     | Can Change?     | Size (750M model)     |
| --------------------- | ------------------------------------------------------------ | --------------- | --------------------- |
| `config.json`         | Model architecture (incl. pos_encoding, mlp_type, norm_type) | No (fixed)      | < 1 KB                |
| `model.pt`            | Neural net weights                                           | No (trained)    | ~1.4 GB               |
| `tokenizer.json`      | Vocabulary                                                   | No (fixed)      | < 1 KB                |
| `training_state.json` | Progress + training hyperparameters                          | Yes (flexible)  | < 1 KB                |
| `optimizer.pt`        | Optimizer state (AdamW)                                      | Yes (can reset) | ~2.8 GB               |
| `<model>_eval.jsonl`  | Full training log (timestamped events)                       | Append-only     | Grows during training |
