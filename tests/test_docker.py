#!/usr/bin/env python3
"""
Docker Build and Runtime Tests for MyPT

This script tests the Docker setup by:
1. Building the image
2. Testing container startup
3. Verifying CUDA availability
4. Testing webapp health endpoint
5. Testing volume initialization
6. Testing CLI commands

Usage:
    python tests/test_docker.py

Requirements:
    - Docker installed and running
    - NVIDIA Container Toolkit (for GPU tests)
    - Python 3.9+

Note: Run from the project root directory.
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path

# Test configuration
IMAGE_NAME = "mypt:test"
CONTAINER_NAME = "mypt-test"
WEBAPP_PORT = 8765  # Use non-standard port to avoid conflicts
TIMEOUT_SECONDS = 120


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def print_test(name: str, passed: bool, message: str = ""):
    """Print test result."""
    status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
    print(f"  {status} {name}")
    if message and not passed:
        print(f"         {Colors.YELLOW}{message}{Colors.RESET}")


def run_cmd(cmd: list, timeout: int = 60, capture: bool = True) -> tuple:
    """Run a command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def docker_available() -> bool:
    """Check if Docker is available."""
    success, _, _ = run_cmd(["docker", "version"])
    return success


def cleanup():
    """Clean up test containers and images."""
    print("Cleaning up previous test artifacts...")
    # Stop and remove container if exists
    run_cmd(["docker", "stop", CONTAINER_NAME], timeout=10)
    run_cmd(["docker", "rm", "-f", CONTAINER_NAME], timeout=10)
    # Remove test volumes
    run_cmd(["docker", "volume", "rm", "-f", "mypt-test-configs"], timeout=10)


def test_docker_build() -> bool:
    """Test Docker image build."""
    print_header("Test 1: Docker Build")
    
    # Check Dockerfile exists
    if not Path("Dockerfile").exists():
        print_test("Dockerfile exists", False, "Dockerfile not found in current directory")
        return False
    print_test("Dockerfile exists", True)
    
    # Build the image
    print("  Building image (this may take a few minutes)...")
    success, stdout, stderr = run_cmd(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        timeout=TIMEOUT_SECONDS * 3,
        capture=True
    )
    
    if not success:
        print_test("Docker build", False, stderr[:200] if stderr else "Build failed")
        return False
    
    print_test("Docker build", True)
    
    # Verify image exists
    success, stdout, _ = run_cmd(["docker", "images", "-q", IMAGE_NAME])
    if not success or not stdout.strip():
        print_test("Image created", False, "Image not found after build")
        return False
    
    print_test("Image created", True)
    return True


def test_container_help() -> bool:
    """Test container help command."""
    print_header("Test 2: Container Help Command")
    
    success, stdout, stderr = run_cmd(
        ["docker", "run", "--rm", IMAGE_NAME, "help"],
        timeout=30
    )
    
    if not success:
        print_test("Help command", False, stderr[:200] if stderr else "Command failed")
        return False
    
    # Check help output contains expected content
    checks = [
        ("Shows usage info", "Usage:" in stdout or "docker run" in stdout.lower()),
        ("Lists commands", "webapp" in stdout.lower()),
        ("Lists train command", "train" in stdout.lower()),
        ("Lists generate command", "generate" in stdout.lower()),
    ]
    
    all_passed = True
    for name, passed in checks:
        print_test(name, passed)
        if not passed:
            all_passed = False
    
    return all_passed


def test_show_configs() -> bool:
    """Test show-configs command."""
    print_header("Test 3: Show Configs Command")
    
    success, stdout, stderr = run_cmd(
        ["docker", "run", "--rm", IMAGE_NAME, "show-configs"],
        timeout=30
    )
    
    if not success:
        print_test("Show configs", False, stderr[:200] if stderr else "Command failed")
        return False
    
    print_test("Show configs runs", True)
    
    # Check for expected config names in output
    checks = [
        ("Lists tiny config", "tiny" in stdout.lower()),
        ("Lists small config", "small" in stdout.lower()),
        ("Shows parameters", "param" in stdout.lower() or "layer" in stdout.lower()),
    ]
    
    all_passed = True
    for name, passed in checks:
        print_test(name, passed)
        if not passed:
            all_passed = False
    
    return all_passed


def test_cuda_detection() -> bool:
    """Test CUDA detection in container."""
    print_header("Test 4: CUDA Detection")
    
    # Test without GPU
    cuda_check_cmd = [
        "docker", "run", "--rm", IMAGE_NAME, "python", "-c",
        "import torch; print(f'CUDA_AVAILABLE={torch.cuda.is_available()}')"
    ]
    
    success, stdout, stderr = run_cmd(cuda_check_cmd, timeout=30)
    
    if not success:
        print_test("PyTorch loads", False, stderr[:200] if stderr else "Failed to run Python")
        return False
    
    print_test("PyTorch loads", True)
    print_test("CUDA check runs", "CUDA_AVAILABLE=" in stdout)
    
    # Test with GPU (if available)
    gpu_check_cmd = [
        "docker", "run", "--rm", "--gpus", "all", IMAGE_NAME, "python", "-c",
        "import torch; print(f'CUDA_AVAILABLE={torch.cuda.is_available()}'); print(f'GPU_COUNT={torch.cuda.device_count()}')"
    ]
    
    success, stdout, stderr = run_cmd(gpu_check_cmd, timeout=30)
    
    if success and "CUDA_AVAILABLE=True" in stdout:
        print_test("GPU detected (with --gpus all)", True)
        # Extract GPU count
        for line in stdout.split("\n"):
            if "GPU_COUNT=" in line:
                print(f"         {Colors.GREEN}Found: {line}{Colors.RESET}")
    else:
        print_test("GPU detected (with --gpus all)", False, 
                   "No GPU available or NVIDIA Container Toolkit not installed")
        print(f"         {Colors.YELLOW}This is expected if no GPU is present{Colors.RESET}")
    
    return True  # Don't fail tests if no GPU


def test_configs_volume_init() -> bool:
    """Test that configs volume is initialized with defaults."""
    print_header("Test 5: Configs Volume Initialization")
    
    # Create a fresh volume and run container
    run_cmd(["docker", "volume", "rm", "-f", "mypt-test-configs"], timeout=10)
    
    # Run container with empty configs volume
    success, stdout, stderr = run_cmd([
        "docker", "run", "--rm",
        "-v", "mypt-test-configs:/app/configs",
        IMAGE_NAME,
        "ls", "-la", "/app/configs/"
    ], timeout=30)
    
    if not success:
        print_test("Container runs with volume", False, stderr[:200] if stderr else "Failed")
        return False
    
    print_test("Container runs with volume", True)
    
    # Check configs were copied
    checks = [
        ("pretrain/ exists", "pretrain" in stdout),
        ("sft1/ exists", "sft1" in stdout or "sft" in stdout.lower()),
    ]
    
    all_passed = True
    for name, passed in checks:
        print_test(name, passed)
        if not passed:
            all_passed = False
    
    # Check a specific config file
    success, stdout, _ = run_cmd([
        "docker", "run", "--rm",
        "-v", "mypt-test-configs:/app/configs",
        IMAGE_NAME,
        "cat", "/app/configs/pretrain/small.json"
    ], timeout=30)
    
    if success and "n_layer" in stdout:
        print_test("Config files valid JSON", True)
    else:
        print_test("Config files valid JSON", False)
        all_passed = False
    
    # Cleanup test volume
    run_cmd(["docker", "volume", "rm", "-f", "mypt-test-configs"], timeout=10)
    
    return all_passed


def test_webapp_startup() -> bool:
    """Test webapp starts and responds to health check."""
    print_header("Test 6: Webapp Startup")
    
    # Start container in background
    success, _, stderr = run_cmd([
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "-p", f"{WEBAPP_PORT}:8000",
        "-e", "MYPT_DEBUG=false",
        IMAGE_NAME, "webapp"
    ], timeout=30)
    
    if not success:
        print_test("Container starts", False, stderr[:200] if stderr else "Failed to start")
        return False
    
    print_test("Container starts", True)
    
    # Wait for webapp to be ready
    print("  Waiting for webapp to start...")
    max_attempts = 30
    for i in range(max_attempts):
        time.sleep(2)
        
        # Check container is still running
        success, stdout, _ = run_cmd([
            "docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"
        ])
        
        if not stdout.strip():
            # Container stopped, get logs
            _, logs, _ = run_cmd(["docker", "logs", CONTAINER_NAME])
            print_test("Container stays running", False, f"Container stopped. Logs: {logs[:300]}")
            return False
        
        # Try health check
        try:
            import urllib.request
            req = urllib.request.Request(f"http://localhost:{WEBAPP_PORT}/health")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                if data.get("status") == "healthy":
                    print_test("Health check responds", True)
                    print_test("Status is healthy", True)
                    
                    # Stop container
                    run_cmd(["docker", "stop", CONTAINER_NAME], timeout=10)
                    return True
        except Exception:
            pass  # Keep trying
    
    print_test("Health check responds", False, f"Timeout after {max_attempts * 2} seconds")
    
    # Get logs for debugging
    _, logs, _ = run_cmd(["docker", "logs", "--tail", "50", CONTAINER_NAME])
    print(f"\n  {Colors.YELLOW}Container logs:{Colors.RESET}")
    for line in logs.split("\n")[-10:]:
        print(f"    {line}")
    
    run_cmd(["docker", "stop", CONTAINER_NAME], timeout=10)
    return False


def test_python_imports() -> bool:
    """Test that all Python modules can be imported."""
    print_header("Test 7: Python Module Imports")
    
    modules_to_test = [
        ("core", "from core import GPT, GPTConfig"),
        ("core.model", "from core.model import GPT"),
        ("core.tokenizer", "from core.tokenizer import Tokenizer"),
        ("core.checkpoint", "from core.checkpoint import CheckpointManager"),
        ("core.agent", "from core.agent import AgentController"),
        ("core.rag", "from core.rag import Retriever"),
        ("webapp", "from webapp.main import app"),
    ]
    
    all_passed = True
    for name, import_stmt in modules_to_test:
        success, _, stderr = run_cmd([
            "docker", "run", "--rm", IMAGE_NAME, "python", "-c", import_stmt
        ], timeout=30)
        
        print_test(f"Import {name}", success, stderr[:100] if stderr and not success else "")
        if not success:
            all_passed = False
    
    return all_passed


def main():
    """Run all Docker tests."""
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("  MyPT Docker Test Suite")
    print(f"{'=' * 60}{Colors.RESET}")
    
    # Check we're in the right directory
    if not Path("Dockerfile").exists():
        print(f"\n{Colors.RED}Error: Dockerfile not found.{Colors.RESET}")
        print("Please run this script from the project root directory:")
        print("  cd /path/to/MyPT")
        print("  python tests/test_docker.py")
        sys.exit(1)
    
    # Check Docker is available
    if not docker_available():
        print(f"\n{Colors.RED}Error: Docker is not available.{Colors.RESET}")
        print("Please ensure Docker is installed and running.")
        sys.exit(1)
    
    print(f"\n{Colors.GREEN}Docker is available ✓{Colors.RESET}")
    
    # Cleanup before tests
    cleanup()
    
    # Run tests
    results = {}
    
    try:
        # Build test (required for other tests)
        results["build"] = test_docker_build()
        
        if results["build"]:
            # Run other tests only if build succeeded
            results["help"] = test_container_help()
            results["configs"] = test_show_configs()
            results["cuda"] = test_cuda_detection()
            results["volume_init"] = test_configs_volume_init()
            results["imports"] = test_python_imports()
            results["webapp"] = test_webapp_startup()
        else:
            print(f"\n{Colors.RED}Skipping remaining tests due to build failure{Colors.RESET}")
    
    finally:
        # Cleanup after tests
        print_header("Cleanup")
        cleanup()
        print("  Done.")
    
    # Summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {status} - {name}")
    
    print(f"\n  {Colors.BOLD}Total: {total} | Passed: {passed} | Failed: {failed}{Colors.RESET}")
    
    if failed > 0:
        print(f"\n{Colors.RED}Some tests failed!{Colors.RESET}")
        sys.exit(1)
    else:
        print(f"\n{Colors.GREEN}All tests passed!{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()

