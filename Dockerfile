# =============================================================================
# MyPT - Unified Docker Image with CUDA Support
# =============================================================================
# A complete, self-contained AI platform for secure, offline environments.
# Supports: webapp, training, generation, dataset preparation, RAG indexing
#
# Build:
#   docker build -t mypt:latest .
#
# Run webapp:
#   docker run -p 8000:8000 --gpus all -v mypt-data:/app/data mypt:latest
#
# Run training:
#   docker run --gpus all -v mypt-data:/app/data mypt:latest train \
#       --model_name my_model --config_file configs/pretrain/small.json
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with CUDA + Python
# -----------------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Install system dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools (for some Python packages)
    build-essential \
    # Git (optional, for development)
    git \
    # Curl for health checks
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# -----------------------------------------------------------------------------
# Stage 3: Install Python dependencies
# -----------------------------------------------------------------------------
# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional document format support
RUN pip install --no-cache-dir PyMuPDF>=1.23.0 python-docx>=1.1.0 lxml>=4.9.0

# -----------------------------------------------------------------------------
# Stage 4: Copy application code
# -----------------------------------------------------------------------------
COPY pyproject.toml .
COPY train.py generate.py ./
COPY core/ ./core/
COPY webapp/ ./webapp/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Copy default configs to a separate location (will be copied to volume on first run)
COPY configs/ ./configs-default/

# Install package in editable mode (enables mypt-* commands)
RUN pip install -e .

# -----------------------------------------------------------------------------
# Stage 5: Create directories and set permissions
# -----------------------------------------------------------------------------
# Create directories that will be mounted as volumes
RUN mkdir -p /app/checkpoints \
             /app/workspace/docs \
             /app/workspace/index \
             /app/data \
             /app/logs/audit \
             /app/logs/app \
             /app/configs

# -----------------------------------------------------------------------------
# Stage 6: Copy entrypoint and set up container
# -----------------------------------------------------------------------------
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose webapp port
EXPOSE 8000

# Health check for webapp
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV MYPT_HOST=0.0.0.0 \
    MYPT_PORT=8000 \
    MYPT_DEBUG=false

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run webapp
CMD ["webapp"]

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
LABEL maintainer="MyPT Team" \
      version="0.2.0" \
      description="MyPT - Offline AI Platform for Secure Environments" \
      org.opencontainers.image.source="https://github.com/yourusername/mypt" \
      org.opencontainers.image.title="MyPT" \
      org.opencontainers.image.description="Complete offline AI platform with training, inference, and RAG"

