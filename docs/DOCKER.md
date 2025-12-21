# MyPT Docker Guide

Complete guide for running MyPT in Docker containers with GPU support.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Building the Image](#building-the-image)
- [Running with Docker Compose](#running-with-docker-compose)
- [Running with Docker CLI](#running-with-docker-cli)
- [Volume Management](#volume-management)
- [Environment Variables](#environment-variables)
- [Common Operations](#common-operations)
- [Offline Deployment](#offline-deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Build the image
docker-compose build

# Start webapp with GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up

# Access at http://localhost:8000
# Default credentials: admin:admin, user:user
```

---

## Prerequisites

### For GPU Support (Recommended)

1. **NVIDIA GPU** with CUDA Compute Capability 3.5+
2. **NVIDIA Driver** 470.57.02+ (for CUDA 11.8)
3. **Docker** 20.10+
4. **NVIDIA Container Toolkit**

**Install NVIDIA Container Toolkit (Linux):**

```bash
# Add NVIDIA GPG key
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

**Verify GPU access:**

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### For CPU-Only Mode

Only Docker 20.10+ is required.

---

## Building the Image

### Using Docker Compose (Recommended)

```bash
# Build with default settings
docker-compose build

# Build with no cache (for fresh builds)
docker-compose build --no-cache

# Build with specific target platform
docker-compose build --build-arg TARGETPLATFORM=linux/amd64
```

### Using Docker CLI

```bash
# Basic build
docker build -t mypt:latest .

# Build with custom tag
docker build -t mypt:v0.2.0 .

# Multi-platform build (requires buildx)
docker buildx build --platform linux/amd64,linux/arm64 -t mypt:latest .
```

---

## Running with Docker Compose

### Start Webapp with GPU

```bash
# Start in foreground
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up

# Start in background
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# View logs
docker-compose logs -f mypt

# Stop
docker-compose down
```

### Start Webapp CPU-Only (Testing)

```bash
docker-compose up
```

### Run Training

```bash
# With GPU
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mypt train \
    --model_name my_model \
    --config_file configs/pretrain/small.json \
    --dataset_dir data/my_dataset \
    --max_iters 5000

# Resume training
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mypt train \
    --model_name my_model \
    --max_iters 10000
```

### Run Generation

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mypt generate \
    --model_name my_model \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8
```

### Build RAG Index

```bash
docker-compose run --rm mypt build-index \
    --docs_dir workspace/docs \
    --out_dir workspace/index/latest
```

### Interactive Shell

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mypt shell
```

---

## Running with Docker CLI

### Webapp with GPU

```bash
docker run -d \
    --name mypt \
    --gpus all \
    -p 8000:8000 \
    -v mypt-checkpoints:/app/checkpoints \
    -v mypt-workspace:/app/workspace \
    -v mypt-data:/app/data \
    -v mypt-logs:/app/logs \
    -e MYPT_DEBUG=false \
    mypt:latest
```

### Training

```bash
docker run --rm \
    --gpus all \
    -v mypt-checkpoints:/app/checkpoints \
    -v mypt-data:/app/data \
    mypt:latest train \
    --model_name my_model \
    --config_file configs/pretrain/150M.json \
    --dataset_dir data/my_dataset \
    --max_iters 10000
```

### Generation

```bash
docker run --rm \
    --gpus all \
    -v mypt-checkpoints:/app/checkpoints \
    mypt:latest generate \
    --model_name my_model \
    --prompt "Hello, world!"
```

### Show Available Commands

```bash
docker run --rm mypt:latest help
```

---

## Volume Management

### Named Volumes (Default)

Docker Compose uses named volumes for data persistence:

| Volume | Container Path | Purpose |
|--------|----------------|---------|
| `mypt-checkpoints` | `/app/checkpoints` | Trained models |
| `mypt-workspace` | `/app/workspace` | Documents & RAG index |
| `mypt-data` | `/app/data` | Training datasets |
| `mypt-logs` | `/app/logs` | Audit & app logs |
| `mypt-configs` | `/app/configs` | Model configurations (customer-specific) |

> **Note:** The configs volume is automatically seeded with default configurations on first run.

**Inspect volume location:**

```bash
docker volume inspect mypt-checkpoints
```

**List all volumes:**

```bash
docker volume ls | grep mypt
```

**Copy configs to host for customization:**

```bash
# Create local configs directory from volume
docker run --rm -v mypt-configs:/configs -v $(pwd)/my-configs:/out alpine \
    cp -r /configs/* /out/

# Edit configs locally, then copy back
docker run --rm -v mypt-configs:/configs -v $(pwd)/my-configs:/in alpine \
    cp -r /in/* /configs/
```

### Using Bind Mounts

For easier access to files, use bind mounts instead. Create a `.env` file:

```env
MYPT_CHECKPOINTS_PATH=./checkpoints
MYPT_WORKSPACE_PATH=./workspace
MYPT_DATA_PATH=./data
MYPT_LOGS_PATH=./logs
```

Then modify `docker-compose.yml` volumes section:

```yaml
volumes:
  - ${MYPT_CHECKPOINTS_PATH:-./checkpoints}:/app/checkpoints
  - ${MYPT_WORKSPACE_PATH:-./workspace}:/app/workspace
  - ${MYPT_DATA_PATH:-./data}:/app/data
  - ${MYPT_LOGS_PATH:-./logs}:/app/logs
```

### Backup Volumes

```bash
# Backup checkpoints
docker run --rm \
    -v mypt-checkpoints:/data \
    -v $(pwd)/backup:/backup \
    alpine tar czf /backup/checkpoints.tar.gz -C /data .

# Restore checkpoints
docker run --rm \
    -v mypt-checkpoints:/data \
    -v $(pwd)/backup:/backup \
    alpine tar xzf /backup/checkpoints.tar.gz -C /data
```

### Clean Up Volumes

```bash
# Remove all MyPT volumes (WARNING: deletes all data!)
docker volume rm mypt-checkpoints mypt-workspace mypt-data mypt-logs mypt-configs

# Or using docker-compose
docker-compose down -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MYPT_HOST` | `0.0.0.0` | Webapp bind address |
| `MYPT_PORT` | `8000` | Webapp port |
| `MYPT_DEBUG` | `false` | Enable debug logging |
| `MYPT_SECRET_KEY` | (generated) | JWT signing key (set for production!) |
| `CUDA_VISIBLE_DEVICES` | `all` | Which GPUs to use (e.g., `0`, `0,1`) |

**Using environment file:**

Create `.env`:

```env
MYPT_DEBUG=true
MYPT_SECRET_KEY=your-secure-secret-key-here
CUDA_VISIBLE_DEVICES=0
```

Then:

```bash
docker-compose --env-file .env -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

## Common Operations

### Dataset Preparation

**Prepare sharded dataset:**

```bash
docker-compose run --rm mypt prepare-dataset \
    --input_files data/corpus.txt \
    --out_dir data/my_dataset \
    --tokenization gpt2 \
    --tokens_per_shard 10000000
```

**Prepare tool SFT dataset:**

```bash
docker-compose run --rm mypt prepare-tool-sft \
    --input data/agent_sft.jsonl \
    --output_dir data/tool_sft
```

### RAG Operations

**Build index:**

```bash
docker-compose run --rm mypt build-index \
    --docs_dir workspace/docs \
    --out_dir workspace/index/latest
```

**Interactive workspace chat:**

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run -it --rm mypt workspace-chat \
    --model_name my_agent \
    --workspace_dir workspace/ \
    --index_dir workspace/index/latest
```

### Model Inspection

**Show available configs:**

```bash
docker-compose run --rm mypt show-configs
```

**Calculate model parameters:**

```bash
docker-compose run --rm mypt calculate-params \
    --config_file configs/pretrain/150M.json
```

**Inspect trained model:**

```bash
docker-compose run --rm mypt inspect-model \
    --model_name my_model
```

---

## Offline Deployment

For air-gapped environments without internet access:

### 1. Build and Save Image (On Connected Machine)

```bash
# Build image
docker-compose build

# Save to tarball
docker save mypt:latest | gzip > mypt-docker-image.tar.gz

# Also save the compose files and configs
tar czf mypt-configs.tar.gz \
    docker-compose.yml \
    docker-compose.gpu.yml \
    configs/
```

### 2. Transfer to Offline Machine

Transfer `mypt-docker-image.tar.gz` and `mypt-configs.tar.gz` via USB, network share, etc.

### 3. Load and Run (On Offline Machine)

```bash
# Extract configs
tar xzf mypt-configs.tar.gz

# Load Docker image
gunzip -c mypt-docker-image.tar.gz | docker load

# Verify image
docker images | grep mypt

# Run
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

## Resource Requirements

### Minimum (CPU-only inference)

- CPU: 4 cores
- RAM: 8 GB
- Disk: 20 GB (image + models)

### Recommended (GPU training)

| Model Size | GPU VRAM | System RAM | Disk |
|------------|----------|------------|------|
| Tiny (11M) | 2 GB | 8 GB | 20 GB |
| Small (40M) | 4 GB | 12 GB | 30 GB |
| 150M | 8 GB | 16 GB | 50 GB |
| 350M | 16 GB | 24 GB | 100 GB |
| 750M | 24+ GB | 32 GB | 200 GB |

---

## Troubleshooting

### Container won't start

**Check logs:**

```bash
docker-compose logs mypt
```

**Check container status:**

```bash
docker-compose ps
```

### GPU not detected

**Verify NVIDIA Container Toolkit:**

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

**Check CUDA in container:**

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mypt shell
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory (OOM)

**Reduce batch size:**

```bash
docker-compose run --rm mypt train \
    --model_name my_model \
    --batch_size 16  # Reduce from default
```

**Use smaller model config:**

```bash
docker-compose run --rm mypt train \
    --config_file configs/pretrain/small.json  # Instead of 150M
```

### Permission denied on volumes

**Linux:** Ensure volumes have correct permissions:

```bash
sudo chown -R 1000:1000 ./checkpoints ./workspace ./data ./logs
```

### Port already in use

**Change port:**

```bash
MYPT_PORT=8080 docker-compose up
```

Or in `.env`:

```env
MYPT_PORT=8080
```

### Health check failing

**Check webapp is responding:**

```bash
docker exec mypt curl -f http://localhost:8000/health
```

**Check logs for errors:**

```bash
docker-compose logs --tail=100 mypt
```

---

## Updating

### Update to new version

```bash
# Stop running container
docker-compose down

# Pull/build new image
docker-compose build --no-cache

# Start with new image (volumes preserved)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Clean rebuild

```bash
# Remove everything except volumes
docker-compose down --rmi all

# Rebuild
docker-compose build --no-cache

# Start fresh
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

---

## Security Considerations

### Production Deployment

1. **Set a persistent secret key:**
   ```bash
   export MYPT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   ```

2. **Change default passwords** after first login

3. **Use HTTPS** via reverse proxy (nginx, Traefik, etc.)

4. **Restrict network access** to trusted networks

5. **Regular backups** of volumes

### Read-only rootfs (Advanced)

For enhanced security, run with read-only root filesystem:

```yaml
services:
  mypt:
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs
```

---

## See Also

- [Installation Guide](INSTALL.md) - Non-Docker installation
- [Web Application Guide](WEBAPP_GUIDE.md) - Webapp usage
- [Training Configuration](TRAINING_CONFIG.md) - Config options
- [Large Dataset Training](LARGE_DATASET_TRAINING.md) - Training at scale

