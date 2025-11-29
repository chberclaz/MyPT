# Installation Guide for MyPT

## Quick Start

### 1. System Requirements

**Python Version:** 3.9 or higher

**Operating Systems:** Windows, Linux, macOS

**Hardware:**
- **Minimum**: CPU with 8GB RAM
- **Recommended**: NVIDIA GPU with CUDA support (for faster training)

---

## Installation Methods

### Method 1: Pip Install (Recommended for Users)

```bash
# Clone the repository
git clone https://github.com/yourusername/mypt.git
cd mypt

# Install dependencies
pip install -r requirements.txt
```

**With CUDA support** (recommended for NVIDIA GPUs):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install tiktoken>=0.5.0
```

**CPU only**:
```bash
pip install torch torchvision torchaudio
pip install tiktoken>=0.5.0
```

---

### Method 2: Development Install (for Contributors)

```bash
# Clone and navigate to directory
git clone https://github.com/yourusername/mypt.git
cd mypt

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements.txt

# Optional: Install dev tools
pip install pytest black isort flake8 mypy
```

---

### Method 3: Using pyproject.toml (Modern Python)

```bash
# Clone repository
git clone https://github.com/yourusername/mypt.git
cd mypt

# Install with pip (reads pyproject.toml automatically)
pip install .

# Or install in editable mode for development
pip install -e .

# Install with optional dev dependencies
pip install -e ".[dev]"
```

---

## Verify Installation

After installation, verify everything works:

```python
# Test import
python -c "from core import GPT, GPTConfig, load_model; print('✅ MyPT installed successfully!')"

# Test PyTorch CUDA (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test tiktoken
python -c "import tiktoken; print('✅ Tiktoken installed successfully!')"
```

---

## Quick Test

Create a simple test script `test_install.py`:

```python
from core import create_model, GPTConfig

# Create a small model
print("Creating model...")
model = create_model(n_layer=2, n_head=2, n_embd=128, block_size=64)

print(f"Model created successfully!")
print(f"Device: {model.config.device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test generation (random output since not trained)
output = model.generate("Hello", max_new_tokens=10)
print(f"Generated: {output}")
print("✅ All tests passed!")
```

Run it:
```bash
python test_install.py
```

Expected output:
```
Creating model...
Model created successfully!
Device: cuda  # or 'cpu'
Parameters: 1,234,567
Generated: Hello [random tokens]
✅ All tests passed!
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'core'`

**Solution**: Make sure you're in the project root directory, or install in editable mode:
```bash
pip install -e .
```

---

### Issue: `CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--batch_size 16` (default is 32)
2. Reduce model size: `--n_layer 4 --n_embd 256`
3. Use CPU: Edit config to set `device='cpu'`

---

### Issue: PyTorch not detecting GPU

**Check CUDA installation**:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Solution**: Reinstall PyTorch with correct CUDA version:
```bash
# Check your CUDA version first
nvidia-smi

# Install matching PyTorch (example for CUDA 11.8)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Issue: `ImportError: cannot import name 'tiktoken'`

**Solution**:
```bash
pip install tiktoken>=0.5.0
```

---

## Virtual Environment Setup (Recommended)

### Why use a virtual environment?

- Isolates project dependencies
- Avoids conflicts with other Python projects
- Easy to replicate on other machines

### Windows

```powershell
# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate
deactivate
```

### Linux/Mac

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate
deactivate
```

---

## Dependency Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | >=2.0.0 | Deep learning framework (neural networks, GPU support) |
| **tiktoken** | >=0.5.0 | OpenAI's tokenizer for GPT-2 BPE encoding |

### Development Dependencies (Optional)

| Package | Version | Purpose |
|---------|---------|---------|
| **pytest** | >=7.0.0 | Unit testing framework |
| **black** | >=23.0.0 | Code formatter |
| **isort** | >=5.12.0 | Import sorter |
| **flake8** | >=6.0.0 | Linter for code quality |
| **mypy** | >=1.0.0 | Static type checker |

---

## Platform-Specific Notes

### Windows

**PowerShell Execution Policy**: If you get an error activating the venv:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Long Path Support**: Enable long paths for Windows:
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Linux

**Install Python 3.9+** (if not already installed):
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip

# Fedora
sudo dnf install python39 python39-pip
```

### macOS

**Install Python via Homebrew** (recommended):
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.9

# Verify
python3 --version
```

---

## Docker Installation (Advanced)

For a completely isolated environment, use Docker:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Command to run
CMD ["python", "train.py", "--help"]
```

Build and run:
```bash
docker build -t mypt .
docker run -it mypt
```

---

## Upgrading

To upgrade MyPT to the latest version:

```bash
# Pull latest changes
git pull origin main

# Reinstall dependencies (in case they changed)
pip install -r requirements.txt --upgrade

# Or if installed in editable mode
pip install -e . --upgrade
```

---

## Uninstallation

To completely remove MyPT:

```bash
# If installed via pip
pip uninstall mypt

# Remove virtual environment
# Windows:
rmdir /s venv
# Linux/Mac:
rm -rf venv

# Remove downloaded files
cd ..
rm -rf mypt
```

---

## Next Steps

After successful installation:

1. **Read the README**: `README.md` for usage examples
2. **Train your first model**: 
   ```bash
   python train.py --model_name my_first_model --input_file input.txt --max_iters 100
   ```
3. **Generate text**:
   ```bash
   python generate.py --model_name my_first_model --prompt "Hello" --max_new_tokens 50
   ```
4. **Explore the API**:
   ```python
   from core import create_model, load_model
   # ... see README for examples
   ```

---

## Getting Help

- **Issues**: Report bugs at [GitHub Issues](https://github.com/yourusername/mypt/issues)
- **Documentation**: See `README.md`, `CHECKPOINT_FORMAT.md`, and `JSON_CHECKPOINT_MIGRATION.md`
- **Examples**: Check the `examples/` directory (if available)

---

## Summary

**Minimal install** (2 commands):
```bash
git clone https://github.com/yourusername/mypt.git && cd mypt
pip install -r requirements.txt
```

**Verify install** (1 command):
```bash
python -c "from core import GPT; print('✅ Success!')"
```

**Start training**:
```bash
python train.py --model_name test --input_file input.txt --max_iters 100
```

You're ready to go! 🚀

