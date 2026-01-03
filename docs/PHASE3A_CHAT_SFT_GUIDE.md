# Phase 3a: Chat SFT Training Guide

This guide covers phase 3a training: fine-tuning your 750M reasoning model on gold conversation episodes using the episode-indexed SFT data loader.

---

## Overview

**Phase 3a Goal**: Teach the 750M reasoning model to follow conversational patterns and instructions using high-quality gold episodes.

**Key Features**:
- ✅ **Loss masking**: Train only on assistant responses
- ✅ **Episode-indexed**: Deterministic, reproducible training
- ✅ **Continuation training**: Builds on your existing `750M_gold_2.22` checkpoint
- ✅ **Small, high-quality dataset**: ~Gold episodes optimized for quality over quantity

---

## Configuration Parameters Explained

### Architecture (Unchanged from Base Model)

```json
{
  "n_layer": 32,        // Keep same depth as pretrained model
  "n_embd": 1280,       // Keep same embedding dimension
  "n_head": 20,         // Keep same attention heads
  "block_size": 1024,   // Keep long context (good for conversations)
  "dropout": 0.15       // ↑ Slightly increased from 0.1 to prevent overfitting on small dataset
}
```

**Why keep architecture?** You're continuing from a pretrained checkpoint, so architecture must match.

---

### Training Parameters (Optimized for SFT)

| Parameter | Pretrain (Phase 1-2) | SFT (Phase 3a) | Reason |
|-----------|---------------------|----------------|---------|
| **learning_rate** | 1.5e-4 | **5e-5** | **Lower LR for fine-tuning** - prevents catastrophic forgetting of pretrained knowledge |
| **batch_size** | 12 | **10** | Slightly smaller for SFT (more stable with loss masking) |
| **max_iters** | 315,000 | **5,000** | **Much fewer iterations** - gold dataset is small, high-quality. ~10-20 epochs. |
| **warmup_iters** | 5% (15,750) | **3% (150)** | Shorter warmup for fine-tuning |
| **eval_interval** | 3,000 | **500** | **More frequent eval** - catch overfitting early on small dataset |
| **save_every** | 10,000 | **1,000** | Save more frequently to track progress |
| **weight_decay** | 0.1 | **0.05** | Moderate regularization for SFT |

---

### Loss Masking (SFT-Specific)

```json
{
  "use_loss_mask": true  // CRITICAL: Train only on assistant responses
}
```

**What it does**: 
- ❌ **Don't train** on system prompts, user messages
- ✅ **Only train** on assistant responses

This ensures the model learns to generate appropriate responses, not memorize prompts.

---

### Episode-Indexed Settings (Reproducibility)

```json
{
  "batch_sampling_mode": "epoch",  // Deterministic epoch-based sampling
  "epoch_seed": 3301,              // Fixed seed for reproducibility (phase 3a = 3301)
  "epoch_shuffle": true,           // Shuffle episodes each epoch
  "epoch_drop_last": true,         // Drop incomplete final batch
  "episode_min_tokens": 10         // Skip episodes shorter than 10 tokens
}
```

**Benefits**:
- ✅ **Reproducible**: Same seed = same training order
- ✅ **Auditable**: Logs show which episode trained when
- ✅ **Full coverage**: Every episode trained exactly once per epoch

---

## Dataset Preparation

### Step 1: Prepare the Gold Dataset

```bash
cd D:\coding\MyPT

# Prepare gold conversation episodes in episode-indexed format
python scripts/prepare_chat_sft.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_prepared \
  --tokenization gpt2 \
  --val_split 0.1 \
  --verbose
```

**Expected output**:
```
Dataset preparation complete!
  Format: episode_indexed_sft_v1
  Train episodes: ~90% of gold set
  Val episodes:   ~10% of gold set
  Output: data/sft_conversation_goldset_prepared/
```

---

### Step 2: Verify Dataset Quality

Run the episode data loader tests to ensure your dataset is correct:

```bash
python tests/test_episode_data_loader.py
```

**Expected**: ✅ All 7 tests pass

---

## Training

### Step 3: Start Phase 3a Training

```bash
python train.py \
  --model 750M_gold_2.22 \
  --config configs/sft1/750M_1024_chat_sft_phase3a.json \
  --dataset data/sft_conversation_goldset_prepared \
  --output checkpoints/750M_phase3a_chat_sft \
  --tokenization gpt2
```

**What happens**:
1. ✅ Loads your pretrained `750M_gold_2.22` checkpoint
2. ✅ Detects episode-indexed dataset → uses `GPTEpisodeDataLoader`
3. ✅ Applies loss masking (trains only on assistant responses)
4. ✅ Trains for 5,000 iterations (~10-20 epochs depending on dataset size)
5. ✅ Saves checkpoints every 1,000 steps

---

### Training Time Estimate

**On A100 GPU:**
- ~2-3 hours for 5,000 iterations
- Depends on:
  - Dataset size (gold episodes)
  - Batch size (10)
  - Block size (1024)

**Cost**: ~$5-10 on cloud GPU rental

---

### Step 4: Monitor Training

**Loss expectations:**

```
Step 0:    train=~8-10,  val=~8-10   (Initial loss, model adapting)
Step 500:  train=~4-6,   val=~4-6    (Learning conversation patterns)
Step 1000: train=~3-4,   val=~3-4.5  (Converging)
Step 2000: train=~2-3,   val=~3-3.5  (Good fit)
Step 5000: train=~1.5-2, val=~2.5-3  (Target: low train loss, reasonable val loss)
```

**Watch for**:
- ❌ **Overfitting**: Val loss > train loss by > 1.0 → stop early or increase dropout
- ✅ **Good fit**: Val loss ≈ train loss + 0.5
- ❌ **Underfitting**: Both losses plateau high → increase max_iters

---

## Training via WebUI

Alternatively, use the WebUI for easier monitoring:

1. **Start WebUI**:
   ```bash
   python -m webapp.main
   ```

2. **Navigate to Training Tab**

3. **Select**:
   - **Training Type**: Chat SFT
   - **Config**: `750M_1024_chat_sft_phase3a.json`
   - **Dataset**: `data/sft_conversation_goldset_prepared`
   - **Init from**: `750M_gold_2.22`
   - **Output**: `750M_phase3a_chat_sft`

4. **Click "Start Training"**

5. **Monitor**: Real-time loss curves and training progress

---

## Validation

### Step 5: Test the Fine-Tuned Model

After training completes, test your phase 3a model:

```bash
python generate.py \
  --model checkpoints/750M_phase3a_chat_sft \
  --prompt "<myPT_system>You are a helpful assistant.</myPT_system>
<myPT_user>What is 2+2?</myPT_user>
<myPT_assistant>" \
  --max_new_tokens 50
```

**Expected**: Model generates coherent, on-topic assistant response.

---

### Step 6: Test in WebUI Chat

1. **Reload models** (or restart WebUI)
2. **Select**: `750M_phase3a_chat_sft`
3. **Chat**: Ask questions and evaluate response quality

**Compare**:
- Base model (`750M_gold_2.22`) vs
- Phase 3a model (`750M_phase3a_chat_sft`)

Phase 3a should:
- ✅ Follow instructions better
- ✅ Generate more conversational responses
- ✅ Stay on topic
- ✅ Use appropriate tone

---

## Troubleshooting

### Issue 1: "Out of Memory" Error

**Symptoms**: CUDA OOM during training

**Solutions**:
1. Reduce `batch_size` from 10 → 8 or 6
2. Reduce `block_size` from 1024 → 768 or 512 (but this changes architecture!)
3. Use gradient accumulation:
   ```json
   "gradient_accumulation_steps": 2  // Effective batch_size = 10 * 2 = 20
   ```

### Issue 2: Val Loss Much Higher Than Train Loss

**Symptoms**: 
```
Step 3000: train=1.5, val=4.0  // Gap > 2.0 = overfitting
```

**Solutions**:
1. Increase `dropout` from 0.15 → 0.2
2. Increase `weight_decay` from 0.05 → 0.1
3. Stop training earlier (e.g., 3000 steps instead of 5000)
4. Add more validation data

### Issue 3: Loss Not Decreasing

**Symptoms**: Loss stays high (~8-10) for >1000 steps

**Solutions**:
1. Check dataset format: Run `python tests/test_episode_data_loader.py`
2. Verify loss mask is applied: Check logs for "Using loss masking"
3. Increase `learning_rate` from 5e-5 → 7.5e-5
4. Check if model is frozen (shouldn't be)

### Issue 4: Model Generates Nonsense

**Symptoms**: After training, model outputs gibberish

**Possible causes**:
1. **Learning rate too high**: Caused catastrophic forgetting
   - **Fix**: Lower LR to 3e-5, retrain
2. **Loss mask not applied**: Trained on prompts too
   - **Fix**: Verify `use_loss_mask: true` in config
3. **Wrong tokenizer**: Mismatch between pretrain and SFT
   - **Fix**: Use same tokenization as pretraining

---

## Next Steps: Phase 3b (Tool Calling SFT)

After phase 3a completes successfully:

1. **Prepare tool-calling dataset**:
   ```bash
   python scripts/prepare_tool_sft.py \
     --input data/sft_tool_goldset \
     --output data/sft_tool_goldset_prepared
   ```

2. **Create phase 3b config** (similar to 3a, but for tool calling)

3. **Train phase 3b**:
   ```bash
   python train.py \
     --model checkpoints/750M_phase3a_chat_sft \
     --config configs/sft1/750M_1024_tool_sft_phase3b.json \
     --dataset data/sft_tool_goldset_prepared \
     --output checkpoints/750M_phase3b_tool_sft
   ```

---

## Key Takeaways

✅ **Use lower learning rate** (5e-5) for fine-tuning vs pretraining (1.5e-4)  
✅ **Enable loss masking** (`use_loss_mask: true`) for SFT  
✅ **Use episode-indexed data** for determinism and auditability  
✅ **Monitor validation loss** to catch overfitting early  
✅ **Test thoroughly** before deploying to production  

**Estimated cost**: $5-10 for phase 3a training on cloud GPU

**Expected quality**: Significantly improved conversational ability vs base pretrained model!

---

## Summary Commands

```bash
# 1. Prepare dataset
python scripts/prepare_chat_sft.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_prepared \
  --tokenization gpt2 \
  --val_split 0.1

# 2. Validate dataset (optional but recommended)
python tests/test_episode_data_loader.py

# 3. Train phase 3a
python train.py \
  --model 750M_gold_2.22 \
  --config configs/sft1/750M_1024_chat_sft_phase3a.json \
  --dataset data/sft_conversation_goldset_prepared \
  --output checkpoints/750M_phase3a_chat_sft

# 4. Test model
python generate.py \
  --model checkpoints/750M_phase3a_chat_sft \
  --prompt "<myPT_system>You are helpful.</myPT_system><myPT_user>Hello!</myPT_user><myPT_assistant>"
```

Good luck with your phase 3a training! 🚀

