# Getting Started with MyPT

A complete beginner's guide to training your first model with MyPT — from installation to text generation.

## What You'll Achieve

By the end of this guide, you will have:
- ✅ MyPT installed and working
- ✅ Your first trained model (~40M parameters)
- ✅ Generated text from your model
- ✅ Understanding of the basic workflow

**Time required:** ~30-60 minutes (depending on hardware)

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.10 or 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU works) | NVIDIA with 4+ GB VRAM |
| Disk Space | 2 GB | 10 GB |

---

## Step 1: Installation (5 minutes)

### 1.1 Clone the Repository

```bash
git clone https://github.com/yourusername/mypt.git
cd mypt
```

### 1.2 Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 1.4 Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tiktoken; print('Tiktoken: OK')"
python -c "from core import GPT; print('MyPT: OK')"
```

**Expected output:**
```
PyTorch: 2.x.x
Tiktoken: OK
MyPT: OK
```

### 1.5 Check GPU (Optional but Recommended)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If `CUDA available: True`, you have GPU acceleration! 🚀

---

## Step 2: Prepare Training Data (5 minutes)

### 2.1 Option A: Use Sample Data (Quickest)

Create a simple text file with any content:

```bash
# Download a sample text (The Divine Comedy - public domain)
curl -o data/input.txt https://www.gutenberg.org/files/8800/8800-0.txt
```

Or create manually:
```bash
echo "Your training text goes here. Add more content for better results." > data/input.txt
```

### 2.2 Option B: Use Your Own Data

Place your text file(s) in the `data/` folder:
- Plain text (`.txt`)
- One file or multiple files
- Any language (GPT-2 tokenizer handles most)

**Recommended minimum:** ~1 MB of text for meaningful training

---

## Step 3: View Available Configurations (2 minutes)

```bash
python scripts/show_configs.py
```

**Output (partial):**
```
Available configurations:
  tiny.json       - ~11M params, 128 context (testing)
  small.json      - ~40M params, 256 context (development)
  150M.json       - ~150M params, 256 context (production)
  ...
```

For your first model, we'll use `small.json` (~40M parameters):
- Fast to train
- Works on CPU
- Good for learning

---

## Step 4: Train Your First Model (10-30 minutes)

### 4.1 Start Training

```bash
python train.py \
    --config_file configs/pretrain/small.json \
    --model_name my_first_model \
    --input_file data/input.txt \
    --max_iters 2000
```

**What the flags mean:**
| Flag | Purpose |
|------|---------|
| `--config_file` | Model architecture (size, layers, etc.) |
| `--model_name` | Name for your saved model |
| `--input_file` | Your training text |
| `--max_iters` | How many training steps |

### 4.2 Watch Training Progress

You'll see output like:
```
========== Model Configuration ==========
Architecture: 6 layers × 6 heads
Embedding dimension: 384
Total parameters: 40,982,784
Device: cuda

step 0: val 10.8234
step 100: val 6.4521
step 200: val 4.8923
...
step 2000: val 3.2156
Training complete! Model saved to checkpoints/my_first_model
```

**What to look for:**
- `val` (validation loss) should **decrease** over time
- Lower is better (2.5-4.0 is reasonable for small models)
- Training stops automatically at `max_iters`

### 4.3 Training Time Estimates

| Hardware | 2000 iterations |
|----------|-----------------|
| CPU (8 cores) | ~20-30 min |
| GTX 1650 | ~5-10 min |
| RTX 3060 | ~3-5 min |
| RTX 4090 | ~1-2 min |

---

## Step 5: Generate Text (2 minutes)

### 5.1 Basic Generation

```bash
python generate.py \
    --model_name my_first_model \
    --prompt "Once upon a time" \
    --max_new_tokens 100
```

**Example output:**
```
Once upon a time there was a king who had three sons. The eldest was...
```

### 5.2 Interactive Mode

```bash
python generate.py --model_name my_first_model --interactive
```

Type prompts and get responses in real-time!

### 5.3 Adjust Generation Quality

```bash
# More creative (higher temperature)
python generate.py --model_name my_first_model --prompt "The secret" --temperature 1.0

# More focused (lower temperature)
python generate.py --model_name my_first_model --prompt "The secret" --temperature 0.5

# Deterministic (same output every time)
python generate.py --model_name my_first_model --prompt "The secret" --temperature 0.0
```

---

## Step 6: Inspect Your Model (Optional)

### 6.1 View Model Info

```bash
python scripts/inspect_model.py --model_name my_first_model
```

**Output:**
```
Model: my_first_model
Parameters: 40,982,784
Layers: 6
Embedding: 384
Context: 256
Trained for: 2000 iterations
Final val loss: 3.2156
```

### 6.2 Check Saved Files

```bash
ls checkpoints/my_first_model/
```

```
config.json          # Model architecture
model.pt             # Trained weights
tokenizer.json       # Tokenizer state
training_state.json  # Training progress
```

---

## Congratulations! 🎉

You've successfully:
1. Installed MyPT
2. Prepared training data
3. Trained a ~40M parameter language model
4. Generated text

---

## What's Next?

### Train a Larger Model

```bash
# 150M parameters (needs ~12GB GPU VRAM)
python train.py \
    --config_file configs/pretrain/150M.json \
    --model_name my_150M_model \
    --input_file data/input.txt \
    --max_iters 10000
```

### Train on a Large Dataset

For datasets > 100MB, use sharded training:

```bash
# 1. Prepare sharded dataset
python scripts/prepare_dataset.py \
    --input_dir data/my_corpus/ \
    --output_dir data/my_shards \
    --tokenization gpt2

# 2. Train on shards
python train.py \
    --config_file configs/pretrain/150M.json \
    --model_name my_model \
    --dataset_dir data/my_shards \
    --max_iters 50000
```

### Try the Web Interface

```bash
python -m webapp.main --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### Domain Adaptation (Phase 2)

Want to specialize your model for a specific domain? See:
- [Domain Adaptation Guide](DOMAIN_ADAPTATION_GUIDE.md)

---

## Common Issues

### "CUDA out of memory"

**Solution:** Use a smaller config or reduce batch size:
```bash
python train.py --config_file configs/pretrain/small.json --batch_size 16 ...
```

### "ModuleNotFoundError"

**Solution:** Make sure you installed in editable mode:
```bash
pip install -e .
```

### Training loss not decreasing

**Possible causes:**
- Not enough data (need at least ~100KB)
- Learning rate too high/low
- Try more iterations

### Generated text is gibberish

**Possible causes:**
- Model undertrained (increase `max_iters`)
- Wrong tokenizer (should match training)
- Temperature too high (try 0.7)

---

## Quick Reference

| Task | Command |
|------|---------|
| Train model | `python train.py --config_file configs/pretrain/small.json --model_name NAME --input_file FILE` |
| Generate text | `python generate.py --model_name NAME --prompt "TEXT"` |
| List configs | `python scripts/show_configs.py` |
| Inspect model | `python scripts/inspect_model.py --model_name NAME` |
| Launch web UI | `python -m webapp.main` |

---

## Further Reading

| Guide | When to Read |
|-------|--------------|
| [Model Selection Guide](MODEL_SELECTION_GUIDE.md) | Choosing the right model size |
| [Troubleshooting](TROUBLESHOOTING.md) | When things go wrong |
| [Quick Reference](QUICK_REFERENCE.md) | Command cheat sheet |
| [Domain Adaptation](DOMAIN_ADAPTATION_GUIDE.md) | Phase 2 training |
| [Large Dataset Training](../training/LARGE_DATASET_TRAINING.md) | Handling big data |

---

*Welcome to MyPT! You're now ready to build AI systems.*

