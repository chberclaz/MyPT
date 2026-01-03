# Episode-Indexed SFT Data Loader

## Overview

The Episode-Indexed SFT (Supervised Fine-Tuning) data loader is designed for training on conversation datasets where each **episode** (conversation) must be treated as an atomic unit. Unlike the standard token-stream loader that samples random windows across all data, this loader:

- **Never crosses episode boundaries** during sampling
- **Returns fixed-shape tensors** padded to `block_size`
- **Supports deterministic epoch-based sampling** for full dataset coverage
- **Enables loss masking** for assistant-only training
- **Provides full audit traceability** of training order

## When to Use

Use the episode-indexed loader when:

1. **Fine-tuning on conversations** (Chat SFT, Tool-calling SFT)
2. **Each sample must remain intact** (no cross-conversation contamination)
3. **Reproducibility is critical** (same seed = same training order)
4. **Compliance requires traceability** (audit logs show exact episode order)

## Dataset Format

Episode-indexed datasets are prepared by `prepare_chat_sft.py` or `prepare_tool_sft.py` and have this structure:

```
dataset_dir/
├── dataset_metadata.json     # Schema info, episode counts
├── tokenizer_state.json      # Tokenizer configuration
├── train/
│   └── shard_00000/          # Multi-shard support
│       ├── tokens.bin        # Concatenated token IDs (uint16)
│       ├── mask.bin          # Loss mask (uint8, 0/1)
│       └── episodes.idx      # Episode index (start, length pairs)
└── val/
    └── shard_00000/
        ├── tokens.bin
        ├── mask.bin
        └── episodes.idx
```

### File Formats

| File           | Format            | Description                               |
| -------------- | ----------------- | ----------------------------------------- |
| `tokens.bin`   | `np.uint16`       | Concatenated token IDs                    |
| `mask.bin`     | `np.uint8`        | Loss mask (1 = compute loss, 0 = ignore)  |
| `episodes.idx` | `np.uint64` pairs | `(start_offset, length)` for each episode |

### Auto-Detection

The loader automatically detects episode-indexed datasets by checking for:

- `train/shard_*/episodes.idx` files, or
- `train/episodes.idx` file

## Configuration

All parameters are configured via `GPTConfig` (in JSON config files):

```json
{
  "batch_size": 8,
  "block_size": 512,
  "use_loss_mask": true,
  "batch_sampling_mode": "epoch",
  "epoch_seed": 1337,
  "epoch_shuffle": true,
  "epoch_drop_last": true,
  "pad_token_id": null,
  "episode_min_tokens": 2
}
```

### Configuration Parameters

| Parameter             | Type      | Default   | Description                                                           |
| --------------------- | --------- | --------- | --------------------------------------------------------------------- |
| `batch_sampling_mode` | str       | `"epoch"` | `"epoch"` for deterministic coverage, `"random"` for uniform sampling |
| `epoch_seed`          | int       | `1337`    | **Base seed for reproducibility** - same seed = same training order   |
| `epoch_shuffle`       | bool      | `true`    | Shuffle episode order each epoch                                      |
| `epoch_drop_last`     | bool      | `true`    | Drop incomplete final batch                                           |
| `pad_token_id`        | int\|null | `null`    | Token ID for padding (null = use EOT token)                           |
| `episode_min_tokens`  | int       | `2`       | Minimum episode length (shorter episodes skipped)                     |
| `use_loss_mask`       | bool      | `false`   | Enable assistant-only loss masking                                    |

## Reproducibility

The implementation is **fully deterministic**:

1. **Same seed produces same episode order** across runs
2. **Epoch-based shuffling**: `effective_seed = epoch_seed + epoch_number`
3. **Config is single source of truth** - no hardcoded defaults

### Reconstructing Training Order

Given the seed and epoch, you can reconstruct the exact episode order:

```python
import numpy as np

def get_episode_order(epoch_seed: int, epoch: int, num_episodes: int) -> np.ndarray:
    """Reconstruct the exact episode order for any epoch."""
    effective_seed = epoch_seed + epoch
    rng = np.random.RandomState(effective_seed)
    return rng.permutation(num_episodes)

# Example: Get order for epoch 0 with seed 1337
order = get_episode_order(1337, 0, 504)
print(f"First 10 episodes: {order[:10]}")
```

## Audit Traceability

All training events are logged to `logs/audit/audit_*.log`:

### Events Logged

| Event            | Description    | Key Fields                                             |
| ---------------- | -------------- | ------------------------------------------------------ |
| `dataset_load`   | Dataset loaded | `epoch_seed`, `epoch_shuffle`, `num_episodes`          |
| `epoch_start`    | Epoch begins   | `seed`, `first_episode_ids` (first 10), `num_episodes` |
| `epoch_complete` | Epoch ends     | `seed_used`, `episodes_seen`                           |

### Example Audit Entries

```
2026-01-03T16:44:19.628Z | TRAINING | INFO | action=dataset_load | epoch_seed=42 | epoch_shuffle=true | num_train_episodes=504 | ...
2026-01-03T16:44:19.629Z | TRAINING | INFO | action=epoch_start | epoch=0 | seed=42 | first_episode_ids="[173, 274, 489, 72, 305, 76, 475, 140, 469, 498]" | ...
```

## Usage

### Preparing a Dataset

```bash
# Chat SFT dataset
python scripts/prepare_chat_sft.py \
    --input data/conversations.jsonl \
    --output_dir data/chat_sft \
    --val_split 0.1

# Tool-calling SFT dataset
python scripts/prepare_tool_sft.py \
    --input data/tool_conversations.jsonl \
    --output_dir data/tool_sft \
    --val_split 0.1
```

### Training

```bash
# Train with episode-indexed dataset (auto-detected)
python train.py \
    --model_name my_sft_model \
    --config_file configs/sft1/150M.json \
    --dataset_dir data/chat_sft \
    --max_iters 5000
```

### Custom Seed for Reproducibility

Create a config file `configs/sft1/my_experiment.json`:

```json
{
  "inherits": "configs/sft1/150M.json",
  "epoch_seed": 42,
  "batch_sampling_mode": "epoch"
}
```

Then train:

```bash
python train.py \
    --model_name experiment_seed42 \
    --config_file configs/sft1/my_experiment.json \
    --dataset_dir data/chat_sft
```

## Sampling Modes

### Epoch Mode (Default)

- Each episode seen exactly once per epoch
- Deterministic order based on `epoch_seed + epoch_number`
- Full dataset coverage guaranteed
- Best for SFT where every conversation matters

### Random Mode

- Episodes sampled uniformly with replacement
- No epoch boundaries
- Some episodes may be seen multiple times, others not at all
- Useful for very large datasets or continuous training

## Loss Masking

When `use_loss_mask=true`, loss is computed only on assistant tokens:

```
System: You are a helpful assistant.
User: What is 2+2?                    ← mask=0 (no loss)
Assistant: The answer is 4.           ← mask=1 (compute loss)
```

This ensures the model learns to generate assistant responses, not copy user inputs.

## Multi-Shard Support

For very large datasets, the loader supports multiple shards:

```
train/
├── shard_00000/
│   ├── tokens.bin
│   ├── mask.bin
│   └── episodes.idx
├── shard_00001/
│   ├── tokens.bin
│   ├── mask.bin
│   └── episodes.idx
└── ...
```

Episodes are globally indexed across shards, maintaining deterministic sampling.

## Related Documentation

- [SFT Loss Masking](SFT_LOSS_MASKING.md) - Loss masking details
- [Tool-call SFT](toolcall_sft.md) - Training tool-using agents
- [Audit & Compliance](AUDIT_COMPLIANCE.md) - Audit logging details
- [Configuration Presets](CONFIG_PRESETS.md) - Model configurations
