# Phase 3a Configuration: Chat SFT

This directory contains the configuration for **Phase 3a** training: fine-tuning the 750M reasoning model on gold conversation episodes.

---

## Quick Start

```bash
# 1. Prepare your gold conversation dataset
python scripts/prepare_chat_sft.py \
  --input data/sft_conversation_goldset \
  --output data/sft_conversation_goldset_prepared \
  --tokenization gpt2 \
  --val_split 0.1

# 2. Train phase 3a
python train.py \
  --model 750M_gold_2.22 \
  --config configs/sft1/750M_1024_chat_sft_phase3a.json \
  --dataset data/sft_conversation_goldset_prepared \
  --output checkpoints/750M_phase3a_chat_sft
```

---

## Configuration File

**`750M_1024_chat_sft_phase3a.json`**

### Key Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| **learning_rate** | 5e-5 | Lower than pretraining (1.5e-4) to prevent catastrophic forgetting |
| **max_iters** | 5,000 | Small dataset, high quality → fewer iterations needed |
| **batch_size** | 10 | Slightly smaller for SFT stability |
| **dropout** | 0.15 | Increased from 0.1 to prevent overfitting on small dataset |
| **use_loss_mask** | true | **Critical**: Train only on assistant responses |
| **epoch_seed** | 3301 | Phase 3a seed for reproducibility |

### Expected Training Time

- **A100 GPU**: ~2-3 hours
- **Cost**: ~$5-10 on cloud rental
- **Iterations**: 5,000 (~10-20 epochs depending on dataset size)

### Expected Loss Trajectory

```
Step 0:    train=~8-10,  val=~8-10   (Initial)
Step 1000: train=~3-4,   val=~3-4.5  (Learning)
Step 3000: train=~2-3,   val=~3-3.5  (Converging)
Step 5000: train=~1.5-2, val=~2.5-3  (Target)
```

---

## What This Training Does

### Before (750M_gold_2.22)
- Strong reasoning ability
- Good at multi-step logic
- **But**: Not optimized for conversational patterns

### After (750M_phase3a_chat_sft)
- ✅ Maintains reasoning ability
- ✅ **Improved**: Follows instructions better
- ✅ **Improved**: More conversational tone
- ✅ **Improved**: Stays on topic
- ✅ **Improved**: Better context awareness

---

## Validation

After training, test your model:

```bash
# CLI test
python generate.py \
  --model checkpoints/750M_phase3a_chat_sft \
  --prompt "<myPT_system>You are helpful.</myPT_system><myPT_user>What is 2+2?</myPT_user><myPT_assistant>"

# WebUI test
python -m webapp.main
# Then select 750M_phase3a_chat_sft in Chat tab
```

---

## Troubleshooting

### Loss Not Decreasing

**Check**:
1. Dataset format: `python tests/test_episode_data_loader.py`
2. Loss masking enabled: Look for "Using loss masking" in logs
3. Learning rate: Try 7.5e-5 if stuck

### Overfitting (Val >> Train Loss)

**Solutions**:
1. Increase `dropout` to 0.2
2. Stop training earlier (3000 steps)
3. Increase `weight_decay` to 0.1

### Out of Memory

**Solutions**:
1. Reduce `batch_size` to 8 or 6
2. Use gradient accumulation
3. Reduce `block_size` (but this changes architecture!)

---

## Next Steps

After phase 3a completes:

1. **Test thoroughly** in production scenarios
2. **Compare** with base model (`750M_gold_2.22`)
3. **Proceed to Phase 3b**: Tool-calling SFT
4. **Deploy** if quality meets requirements

---

## Full Documentation

See [Phase 3a Chat SFT Guide](../../docs/PHASE3A_CHAT_SFT_GUIDE.md) for complete details.

---

## Summary

✅ **Config**: `750M_1024_chat_sft_phase3a.json`  
✅ **Base model**: `750M_gold_2.22` (your pretrained reasoning model)  
✅ **Dataset**: Gold conversation episodes (episode-indexed format)  
✅ **Training time**: ~2-3 hours on A100  
✅ **Cost**: ~$5-10  
✅ **Result**: Conversationally-tuned 750M model  

**Ready to train!** 🚀

