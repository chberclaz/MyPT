# MyPT Troubleshooting Guide

Solutions to common issues encountered when using MyPT.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Training Issues](#training-issues)
3. [Generation Issues](#generation-issues)
4. [Memory Issues](#memory-issues)
5. [Data Issues](#data-issues)
6. [Checkpoint Issues](#checkpoint-issues)
7. [Web Application Issues](#web-application-issues)

---

## Installation Issues

### "ModuleNotFoundError: No module named 'core'"

**Cause:** MyPT not installed in development mode.

**Solution:**
```bash
pip install -e .
```

---

### "ModuleNotFoundError: No module named 'tiktoken'"

**Cause:** Missing dependency.

**Solution:**
```bash
pip install tiktoken>=0.5.0
```

---

### PyTorch CUDA version mismatch

**Symptom:** `CUDA error: no kernel image is available for execution on the device`

**Solution:** Reinstall PyTorch with correct CUDA version:
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### "torch.cuda.is_available() returns False"

**Possible causes:**
1. No NVIDIA GPU
2. CUDA drivers not installed
3. PyTorch CPU-only version installed

**Solutions:**

```bash
# Check if NVIDIA driver is working
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Training Issues

### "CUDA out of memory"

**Solutions (in order of preference):**

1. **Reduce batch size:**
   ```bash
   python train.py --batch_size 8 ...  # or even 4, 2, 1
   ```

2. **Use a smaller model:**
   ```bash
   python train.py --config_file configs/pretrain/small.json ...
   ```

3. **Use gradient accumulation** (same effective batch size, less memory):
   ```bash
   python train.py --batch_size 4 --gradient_accumulation_steps 4 ...
   # Effective batch size = 4 × 4 = 16
   ```

4. **Reduce context length:**
   ```bash
   python train.py --block_size 512 ...  # instead of 1024
   ```

5. **Use mixed precision** (if not already):
   - Ensure `"use_amp": true` in config

---

### Loss is NaN or Inf

**Causes:**
- Learning rate too high
- Data contains invalid values
- Numerical instability

**Solutions:**

1. **Lower learning rate:**
   ```bash
   python train.py --learning_rate 1e-4 ...  # or 5e-5
   ```

2. **Enable gradient clipping:**
   - Ensure `"grad_clip": 1.0` in config

3. **Check your data:**
   ```bash
   python -c "
   with open('data/input.txt', 'r', encoding='utf-8') as f:
       text = f.read()
       print(f'Length: {len(text)}')
       print(f'Has NUL: {chr(0) in text}')
   "
   ```

---

### Loss not decreasing

**Possible causes and solutions:**

| Cause | Solution |
|-------|----------|
| Learning rate too low | Increase to 3e-4 or 1e-3 |
| Learning rate too high | Decrease to 1e-4 or 5e-5 |
| Not enough data | Need at least ~100KB |
| Not enough iterations | Try 10x more iterations |
| Data quality issues | Check for duplicates, noise |

**Diagnostic:** Watch the first 100 iterations:
- Loss should drop noticeably in first 100 steps
- If flat, learning rate is too low
- If erratic/increasing, learning rate is too high

---

### Training is very slow

**Solutions:**

1. **Enable GPU:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   # Should print True
   ```

2. **Use mixed precision:**
   - Add `"use_amp": true, "amp_dtype": "bf16"` to config

3. **Increase batch size** (if GPU memory allows):
   ```bash
   python train.py --batch_size 32 ...
   ```

4. **Use sharded datasets** for large data:
   ```bash
   python scripts/prepare_weighted_dataset.py ...
   python train.py --dataset_dir data/shards ...
   ```

---

### "RuntimeError: expected scalar type BFloat16 but found Float"

**Cause:** Dtype mismatch in mixed precision training (usually on A100/H100).

**Solution:** This was fixed in the codebase. If you see it:
1. Pull latest code
2. Or set `"amp_dtype": "fp16"` instead of `"bf16"` in config

---

## Generation Issues

### Generated text is gibberish

**Possible causes:**

1. **Model undertrained:**
   - Train for more iterations
   - Check that val loss decreased during training

2. **Wrong tokenizer:**
   ```bash
   # Check tokenizer matches
   python scripts/inspect_model.py --model_name my_model
   ```

3. **Temperature too high:**
   ```bash
   python generate.py --model_name my_model --prompt "Hello" --temperature 0.7
   ```

---

### Generated text is repetitive

**Solutions:**

1. **Increase temperature:**
   ```bash
   python generate.py --temperature 1.0 ...
   ```

2. **Increase repetition penalty:**
   ```bash
   python generate.py --repetition_penalty 1.3 ...
   ```

3. **Use top-p sampling:**
   ```bash
   python generate.py --top_p 0.9 --top_k 0 ...
   ```

---

### Generation is very slow

**Solutions:**

1. **Use GPU:**
   ```bash
   # Verify GPU is being used
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **Reduce max_new_tokens:**
   ```bash
   python generate.py --max_new_tokens 100 ...
   ```

3. **Use smaller model for testing

---

## Memory Issues

### System RAM full (not GPU)

**Cause:** Loading large text files into memory.

**Solution:** Use sharded datasets:
```bash
# Prepare shards (streams data, low memory)
python scripts/prepare_weighted_dataset.py \
    --source corpus:data/large_corpus/*.txt \
    --out_dir data/shards

# Train on shards (memory-mapped, low RAM)
python train.py --dataset_dir data/shards ...
```

---

### GPU memory slowly increasing during training

**Cause:** Memory leak or accumulating tensors.

**Solution:** 
1. Pull latest code (fixes applied)
2. Reduce eval frequency:
   ```bash
   python train.py --eval_interval 1000 ...
   ```

---

## Data Issues

### "UnicodeDecodeError" when loading data

**Cause:** File not UTF-8 encoded.

**Solutions:**

1. **Convert to UTF-8:**
   ```bash
   # Check encoding
   file data/input.txt
   
   # Convert (Linux/macOS)
   iconv -f ISO-8859-1 -t UTF-8 data/input.txt > data/input_utf8.txt
   ```

2. **Specify encoding in Python:**
   ```python
   with open('data/input.txt', 'r', encoding='utf-8', errors='ignore') as f:
       text = f.read()
   ```

---

### Tokenization mismatch between training and inference

**Symptom:** Model generates gibberish despite good training loss.

**Cause:** Different tokenizer state during training vs. inference.

**Solution:** Ensure tokenizer state is saved and loaded:
```bash
# Check saved tokenizer
ls checkpoints/my_model/tokenizer.json
```

For sharded datasets, the tokenizer is saved in the dataset:
```bash
ls data/my_shards/tokenizer_state.json
```

---

### Dataset shows 0 tokens

**Cause:** Glob pattern not matching files.

**Solution:** Check your paths:
```bash
# Verify files exist
ls data/my_corpus/*.txt

# Use correct pattern
python scripts/prepare_weighted_dataset.py \
    --source corpus:data/my_corpus/*.txt ...
```

---

## Checkpoint Issues

### "KeyError" when loading old checkpoint

**Cause:** Config changed since checkpoint was saved.

**Solution:** New checkpoint format is backwards compatible. For very old checkpoints:
```bash
python scripts/migrate_checkpoint.py --model_name old_model
```

---

### Can't find model checkpoint

**Check these locations:**
```bash
ls checkpoints/
ls checkpoints/my_model/
ls checkpoints/my_model/model.pt
```

**Expected structure:**
```
checkpoints/my_model/
├── config.json
├── model.pt
├── tokenizer.json
└── training_state.json
```

---

### Checkpoint too large

**Cause:** Optimizer state saved with checkpoint.

**Solution:** For inference-only, you can remove optimizer:
```bash
# Just keep model.pt and config.json for inference
rm checkpoints/my_model/optimizer.pt
```

---

## Web Application Issues

### Web UI not loading

**Check:**
1. Server running: `python -m webapp.main`
2. Correct port: Default is 8000
3. Firewall not blocking

**Test:**
```bash
curl http://localhost:8000/health
```

---

### "Model not found" in web UI

**Solution:** Specify model path:
```bash
python -m webapp.main --model_name my_model
```

Or check checkpoints exist:
```bash
ls checkpoints/
```

---

### Authentication issues

**Default credentials:**
- Admin: `admin:admin`
- User: `user:user`

**Reset:**
```bash
# Edit users file (if it exists)
cat webapp/users.json
```

---

## Getting Help

If your issue isn't listed here:

1. **Check the logs:**
   ```bash
   ls logs/
   cat logs/train.log
   ```

2. **Enable debug mode:**
   ```bash
   python train.py --debug ...
   ```

3. **Search existing documentation:**
   - `docs/` folder contains detailed guides
   - Check `docs/archive/` for historical fixes

4. **Minimal reproduction:**
   - Try with smallest config (`tiny.json`)
   - Use sample data
   - Report exact error message

---

*Still stuck? Check the project issues or create a new one with your error details.*

