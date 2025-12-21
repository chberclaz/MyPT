#!/bin/bash
# =============================================================================
# MyPT Docker Entrypoint
# =============================================================================
# Flexible entrypoint supporting multiple run modes:
#   - webapp    : Start the web application (default)
#   - train     : Run training
#   - generate  : Generate text
#   - prepare   : Prepare datasets
#   - index     : Build RAG index
#   - chat      : Interactive workspace chat
#   - shell     : Drop into bash shell
#   - *         : Run any custom command
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Initialize configs volume with defaults if empty
# -----------------------------------------------------------------------------

init_configs() {
    # Check if configs directory is empty (mounted as volume)
    if [ -d "/app/configs-default" ] && [ -z "$(ls -A /app/configs 2>/dev/null)" ]; then
        echo "Initializing configs volume with default configurations..."
        cp -r /app/configs-default/* /app/configs/
        echo "Default configs copied to /app/configs"
    fi
}

# Initialize configs on startup
init_configs

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

print_banner() {
    echo "=============================================================="
    echo "  MyPT - Offline AI Platform"
    echo "  Your Data. Your Model. Your Control."
    echo "=============================================================="
}

print_cuda_info() {
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "CUDA Status:"
        python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'  GPU Count: {torch.cuda.device_count()}')" 2>/dev/null || echo "  Unable to detect CUDA"
        echo ""
    fi
}

show_help() {
    echo ""
    echo "Usage: docker run [options] mypt:latest <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  webapp              Start web application (default)"
    echo "  train               Train a model"
    echo "  generate            Generate text"
    echo "  prepare-dataset     Prepare sharded dataset"
    echo "  prepare-tool-sft    Prepare tool SFT dataset"
    echo "  prepare-chat-sft    Prepare chat SFT dataset"
    echo "  build-index         Build RAG index"
    echo "  workspace-chat      Interactive workspace chat"
    echo "  show-configs        Show available configurations"
    echo "  shell               Drop into bash shell"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Start webapp"
    echo "  docker run -p 8000:8000 --gpus all mypt:latest"
    echo ""
    echo "  # Train a model"
    echo "  docker run --gpus all -v ./checkpoints:/app/checkpoints \\"
    echo "      -v ./data:/app/data mypt:latest train \\"
    echo "      --model_name my_model --config_file configs/pretrain/small.json \\"
    echo "      --dataset_dir data/my_dataset"
    echo ""
    echo "  # Generate text"
    echo "  docker run --gpus all -v ./checkpoints:/app/checkpoints \\"
    echo "      mypt:latest generate --model_name my_model --prompt 'Hello'"
    echo ""
    echo "  # Build RAG index"
    echo "  docker run -v ./workspace:/app/workspace mypt:latest build-index \\"
    echo "      --docs_dir workspace/docs --out_dir workspace/index/latest"
    echo ""
}

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

print_banner

# Get the command (first argument)
COMMAND="${1:-webapp}"

case "$COMMAND" in
    # -------------------------------------------------------------------------
    # Web Application
    # -------------------------------------------------------------------------
    webapp|web|server)
        print_cuda_info
        echo "Starting MyPT Web Application..."
        echo "  Host: ${MYPT_HOST:-0.0.0.0}"
        echo "  Port: ${MYPT_PORT:-8000}"
        echo "  Debug: ${MYPT_DEBUG:-false}"
        echo ""
        shift || true
        exec python -m webapp.main \
            --host "${MYPT_HOST:-0.0.0.0}" \
            --port "${MYPT_PORT:-8000}" \
            ${MYPT_DEBUG:+--debug} \
            "$@"
        ;;

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    train|training)
        print_cuda_info
        echo "Starting training..."
        shift
        exec python train.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Text Generation
    # -------------------------------------------------------------------------
    generate|gen)
        print_cuda_info
        echo "Running generation..."
        shift
        exec python generate.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Dataset Preparation
    # -------------------------------------------------------------------------
    prepare-dataset|prepare)
        echo "Preparing dataset..."
        shift
        exec python scripts/prepare_dataset.py "$@"
        ;;

    prepare-tool-sft|tool-sft)
        echo "Preparing tool SFT dataset..."
        shift
        exec python scripts/prepare_tool_sft.py "$@"
        ;;

    prepare-chat-sft|chat-sft)
        echo "Preparing chat SFT dataset..."
        shift
        exec python scripts/prepare_chat_sft.py "$@"
        ;;

    generate-agent-sft|agent-sft)
        echo "Generating agent SFT data..."
        shift
        exec python scripts/generate_agent_sft.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # RAG & Workspace
    # -------------------------------------------------------------------------
    build-index|index)
        echo "Building RAG index..."
        shift
        exec python scripts/build_rag_index.py "$@"
        ;;

    workspace-chat|chat)
        print_cuda_info
        echo "Starting workspace chat..."
        shift
        exec python scripts/workspace_chat.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    show-configs|configs)
        echo "Available configurations:"
        shift || true
        exec python scripts/show_configs.py "$@"
        ;;

    calculate-params|params)
        echo "Calculating parameters..."
        shift
        exec python scripts/calculate_params.py "$@"
        ;;

    inspect-model|inspect)
        print_cuda_info
        echo "Inspecting model..."
        shift
        exec python scripts/inspect_model.py "$@"
        ;;

    convert-legacy)
        echo "Converting legacy checkpoints..."
        shift
        exec python scripts/convert_legacy_checkpoints.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Development & Debug
    # -------------------------------------------------------------------------
    shell|bash)
        print_cuda_info
        echo "Dropping into shell..."
        exec /bin/bash
        ;;

    python)
        print_cuda_info
        shift
        exec python "$@"
        ;;

    help|--help|-h)
        show_help
        exit 0
        ;;

    # -------------------------------------------------------------------------
    # Custom command
    # -------------------------------------------------------------------------
    *)
        # If it looks like a flag, assume webapp with flags
        if [[ "$COMMAND" == -* ]]; then
            print_cuda_info
            echo "Starting MyPT Web Application with custom flags..."
            exec python -m webapp.main "$@"
        fi
        
        # Otherwise, try to run the command directly
        echo "Running custom command: $@"
        exec "$@"
        ;;
esac

