# MyPT Dependencies

Complete dependency reference for offline installation and containerization.

---

## Quick Install

```bash
# Standard installation (uses PyPI)
pip install torch tiktoken numpy

# Or from requirements.txt
pip install -r requirements.txt
```

---

## Python Version

| Requirement     | Tested              |
| --------------- | ------------------- |
| **Minimum**     | Python 3.9          |
| **Recommended** | Python 3.10 or 3.11 |
| **Maximum**     | Python 3.12         |

---

## Core Dependencies

| Package      | Version  | Size                       | Purpose                      |
| ------------ | -------- | -------------------------- | ---------------------------- |
| **torch**    | >=2.0.0  | ~2GB (CPU) / ~2.5GB (CUDA) | Deep learning framework      |
| **tiktoken** | >=0.5.0  | ~2MB                       | GPT-2 BPE tokenization       |
| **numpy**    | >=1.21.0 | ~20MB                      | Array operations, embeddings |

### Total Size (approximate)

- **CPU only**: ~2.1 GB
- **With CUDA 11.8**: ~2.6 GB
- **With CUDA 12.1**: ~2.6 GB

---

## PyTorch Installation Options

### Option A: CPU Only (smallest)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Option B: CUDA 11.8 (most compatible)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Option C: CUDA 12.1 (latest)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Option D: Default (auto-detect)

```bash
pip install torch
```

---

## Offline Installation

### Step 1: Download wheels on internet-connected machine

```bash
# Create download directory
mkdir -p wheels

# Download PyTorch (choose your CUDA version)
pip download torch --dest wheels --index-url https://download.pytorch.org/whl/cu118

# Download other packages
pip download tiktoken numpy --dest wheels
```

### Step 2: Transfer `wheels/` folder to offline machine

### Step 3: Install from wheels

```bash
pip install --no-index --find-links=wheels/ torch tiktoken numpy
```

---

## Docker

### Minimal Dockerfile (CPU)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Install as package (enables mypt-* commands)
RUN pip install -e .

# Default command
CMD ["mypt-generate", "--help"]
```

### With CUDA Support

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir tiktoken numpy

# Copy code
COPY . .

# Install as package (enables mypt-* commands)
RUN pip install -e .

CMD ["mypt-generate", "--help"]
```

### Build

```bash
docker build -t mypt:latest .
```

### Run Commands

#### Training

```bash
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    mypt:latest mypt-train \
        --model_name my_model \
        --input_file data/train.txt \
        --config_file configs/pretrain/small.json
```

#### Generation

```bash
docker run --gpus all \
    -v $(pwd)/checkpoints:/app/checkpoints \
    mypt:latest mypt-generate \
        --model_name my_model \
        --prompt "Hello world"
```

#### Agentic RAG - Build Index

```bash
docker run \
    -v $(pwd)/workspace:/app/workspace \
    mypt:latest mypt-build-index \
        --docs_dir workspace/docs \
        --out_dir workspace/index/latest
```

#### Agentic RAG - Generate SFT Data

```bash
docker run \
    -v $(pwd)/workspace:/app/workspace \
    -v $(pwd)/data:/app/data \
    mypt:latest mypt-generate-sft \
        --docs_dir workspace/docs \
        --output data/agent_sft.jsonl \
        --num_examples 500
```

#### Agentic RAG - Workspace Chat (Interactive)

```bash
docker run -it --gpus all \
    -v $(pwd)/workspace:/app/workspace \
    -v $(pwd)/checkpoints:/app/checkpoints \
    mypt:latest mypt-workspace-chat \
        --model_name my_agent \
        --workspace_dir workspace/
```

#### Dataset Preparation

```bash
# Prepare sharded dataset
docker run \
    -v $(pwd)/data:/app/data \
    mypt:latest mypt-prepare-dataset \
        --input_files data/corpus.txt \
        --out_dir data/sharded

# Prepare tool SFT dataset
docker run \
    -v $(pwd)/data:/app/data \
    mypt:latest mypt-prepare-tool-sft \
        --input data/agent_sft.jsonl \
        --output_dir data/tool_sft
```

#### Utilities

```bash
# Show available configs
docker run mypt:latest mypt-show-configs

# Calculate model parameters
docker run mypt:latest mypt-calculate-params \
    --config_file configs/pretrain/150M.json
```

---

## Available Commands (Entry Points)

After installing with `pip install -e .`, these commands are available:

| Command                   | Script                                      | Description                   |
| ------------------------- | ------------------------------------------- | ----------------------------- |
| **Core**                  |                                             |                               |
| `mypt-train`              | `train.py`                                  | Train a model                 |
| `mypt-generate`           | `generate.py`                               | Generate text                 |
| **Agentic RAG**           |                                             |                               |
| `mypt-workspace-chat`     | `scripts/workspace_chat.py`                 | Interactive agent chat        |
| `mypt-build-index`        | `scripts/build_rag_index.py`                | Build RAG embedding index     |
| `mypt-generate-sft`       | `scripts/generate_agent_sft.py`             | Generate synthetic SFT data   |
| **Dataset Preparation**   |                                             |                               |
| `mypt-prepare-dataset`    | `scripts/prepare_dataset.py`                | Create sharded datasets       |
| `mypt-prepare-chat-sft`   | `scripts/prepare_chat_sft.py`               | Prepare chat SFT data         |
| `mypt-prepare-tool-sft`   | `scripts/prepare_tool_sft.py`               | Prepare tool SFT data         |
| `mypt-prepare-weighted`   | `scripts/prepare_weighted_dataset.py`       | Weighted multi-source dataset |
| `mypt-fetch-multilingual` | `scripts/fetch_and_prepare_multilingual.py` | Download multilingual corpora |
| **Utilities**             |                                             |                               |
| `mypt-show-configs`       | `scripts/show_configs.py`                   | List configuration presets    |
| `mypt-calculate-params`   | `scripts/calculate_params.py`               | Calculate model parameters    |
| `mypt-convert-legacy`     | `scripts/convert_legacy_checkpoints.py`     | Convert old checkpoints       |

---

## Dependency Tree

```
mypt
├── torch>=2.0.0
│   ├── filelock
│   ├── typing-extensions
│   ├── sympy
│   ├── networkx
│   ├── jinja2
│   ├── fsspec
│   └── nvidia-* (CUDA builds only)
├── tiktoken>=0.5.0
│   ├── regex
│   └── requests
└── numpy>=1.21.0
```

---

## Optional Development Dependencies

For testing and code quality (not required for running):

```bash
pip install pytest black isort flake8 mypy
```

| Package | Version  | Purpose         |
| ------- | -------- | --------------- |
| pytest  | >=7.0.0  | Testing         |
| black   | >=23.0.0 | Code formatting |
| isort   | >=5.12.0 | Import sorting  |
| flake8  | >=6.0.0  | Linting         |
| mypy    | >=1.0.0  | Type checking   |

---

## System Requirements

### Minimum (CPU training)

- RAM: 8 GB
- Storage: 5 GB free
- CPU: x86_64 or ARM64

### Recommended (GPU training)

- RAM: 16 GB
- Storage: 20 GB free
- GPU: NVIDIA with 8+ GB VRAM
- CUDA: 11.8 or 12.1
- cuDNN: 8.x

### For Large Models (200M+ params)

- RAM: 32 GB
- GPU: 16+ GB VRAM (RTX 3090, A100, etc.)

---

## Version Lock File

For reproducible builds, use exact versions:

```bash
# Generate lock file
pip freeze > requirements.lock

# Install from lock file
pip install -r requirements.lock
```

Example `requirements.lock`:

```
torch==2.1.2
tiktoken==0.5.2
numpy==1.26.2
# ... transitive dependencies
```

---

## Troubleshooting

### "No module named torch"

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "CUDA not available"

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "tiktoken requires Rust"

```bash
# On Linux/Mac
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
pip install tiktoken

# On Windows: install Rust from https://rustup.rs/
```

### Disk space issues

```bash
# Use --no-cache-dir
pip install --no-cache-dir torch tiktoken numpy

# Clear pip cache
pip cache purge
```

---

## Summary Table

| Component     | Package  | Version  | Required    |
| ------------- | -------- | -------- | ----------- |
| Deep Learning | torch    | >=2.0.0  | ✅ Yes      |
| Tokenizer     | tiktoken | >=0.5.0  | ✅ Yes      |
| Arrays        | numpy    | >=1.21.0 | ✅ Yes      |
| Testing       | pytest   | >=7.0.0  | ❌ Optional |
| Formatting    | black    | >=23.0.0 | ❌ Optional |

**Total required packages: 3**
