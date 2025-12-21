#!/bin/bash
# =============================================================================
# Quick Docker Test Script for MyPT
# =============================================================================
# A simple bash script to test Docker build and basic functionality.
#
# Usage:
#   ./tests/test_docker.sh
#
# Or with GPU:
#   ./tests/test_docker.sh --gpu
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="mypt:test"
CONTAINER_NAME="mypt-quicktest"
USE_GPU=""

# Parse arguments
if [[ "$1" == "--gpu" ]]; then
    USE_GPU="--gpus all"
    echo -e "${BLUE}GPU mode enabled${NC}"
fi

# Helper functions
pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; exit 1; }
info() { echo -e "${BLUE}→${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

cleanup() {
    info "Cleaning up..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm -f $CONTAINER_NAME 2>/dev/null || true
    docker volume rm -f mypt-test-vol 2>/dev/null || true
}

# Ensure we're in project root
if [[ ! -f "Dockerfile" ]]; then
    fail "Dockerfile not found. Run from project root: cd /path/to/MyPT && ./tests/test_docker.sh"
fi

# Cleanup before starting
cleanup

header "Test 1: Build Docker Image"
info "Building image (this may take a few minutes)..."
if docker build -t $IMAGE_NAME . ; then
    pass "Docker build succeeded"
else
    fail "Docker build failed"
fi

header "Test 2: Test Help Command"
if docker run --rm $IMAGE_NAME help | grep -q "webapp"; then
    pass "Help command works"
else
    fail "Help command failed"
fi

header "Test 3: Test Show Configs"
if docker run --rm $IMAGE_NAME show-configs | grep -qi "tiny\|small"; then
    pass "Show configs works"
else
    fail "Show configs failed"
fi

header "Test 4: Test PyTorch Import"
if docker run --rm $IMAGE_NAME python -c "import torch; print(f'PyTorch {torch.__version__}')"; then
    pass "PyTorch imports successfully"
else
    fail "PyTorch import failed"
fi

header "Test 5: Test CUDA Detection"
if [[ -n "$USE_GPU" ]]; then
    CUDA_OUTPUT=$(docker run --rm $USE_GPU $IMAGE_NAME python -c "import torch; print(f'CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')")
    echo "  $CUDA_OUTPUT"
    if echo "$CUDA_OUTPUT" | grep -q "CUDA=True"; then
        pass "CUDA is available"
    else
        warn "CUDA not available (check nvidia-docker)"
    fi
else
    info "Skipping GPU test (use --gpu flag to enable)"
fi

header "Test 6: Test Core Imports"
IMPORT_CMD="from core import GPT, GPTConfig; from core.agent import AgentController; print('OK')"
if docker run --rm $IMAGE_NAME python -c "$IMPORT_CMD" | grep -q "OK"; then
    pass "Core module imports work"
else
    fail "Core module imports failed"
fi

header "Test 7: Test Configs Volume Initialization"
# Create fresh volume and test
docker volume rm -f mypt-test-vol 2>/dev/null || true
if docker run --rm -v mypt-test-vol:/app/configs $IMAGE_NAME ls /app/configs/pretrain/ | grep -q ".json"; then
    pass "Configs volume initialized with defaults"
else
    fail "Configs volume initialization failed"
fi
docker volume rm -f mypt-test-vol 2>/dev/null || true

header "Test 8: Test Webapp Startup"
info "Starting webapp container..."
docker run -d --name $CONTAINER_NAME -p 8765:8000 $USE_GPU $IMAGE_NAME webapp

info "Waiting for webapp to start..."
sleep 5

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    pass "Container is running"
else
    docker logs $CONTAINER_NAME
    fail "Container stopped unexpectedly"
fi

# Test health endpoint
info "Testing health endpoint..."
for i in {1..10}; do
    if curl -sf http://localhost:8765/health | grep -q "healthy"; then
        pass "Health endpoint responds"
        break
    fi
    if [[ $i -eq 10 ]]; then
        docker logs --tail 20 $CONTAINER_NAME
        fail "Health endpoint not responding"
    fi
    sleep 2
done

# Cleanup
header "Cleanup"
cleanup

# Summary
header "All Tests Passed! ✓"
echo ""
echo -e "${GREEN}Docker setup is working correctly.${NC}"
echo ""
echo "Quick start commands:"
echo "  # Build image"
echo "  docker-compose build"
echo ""
echo "  # Run webapp with GPU"
echo "  docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up"
echo ""
echo "  # Run training"
echo "  docker-compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm mypt train \\"
echo "      --model_name my_model --config_file configs/pretrain/small.json"
echo ""

