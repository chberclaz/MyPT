"""Repo-relative paths for the GGUF export pipeline."""
from __future__ import annotations

from pathlib import Path

EXPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPORT_DIR.parent
LLAMA_SRC = REPO_ROOT / "third_party" / "llama.cpp"
LLAMA_BIN = REPO_ROOT / "third_party" / "llama.cpp-bin"
PIN_FILE = EXPORT_DIR / "LLAMACPP_PIN.txt"
SPECIAL_TOKENS_JSON = EXPORT_DIR / "special_tokens.json"
HF_TOKENIZER_DIR = EXPORT_DIR / "hf_tokenizer"
ARTIFACTS_DIR = EXPORT_DIR / "artifacts"
FIXTURES_DIR = EXPORT_DIR / "fixtures"
RESULTS_DIR = EXPORT_DIR / "results"
CHECKPOINT_GOLD = REPO_ROOT / "checkpoints" / "phase6_3_ground_gold"

MODEL_VOCAB_SIZE = 50304
BASE_VOCAB_SIZE = 50257


def llama_cli() -> Path:
    for name in ("llama-cli.exe", "llama-cli"):
        p = LLAMA_BIN / name
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-cli not found in {LLAMA_BIN}")


def llama_server() -> Path:
    for name in ("llama-server.exe", "llama-server"):
        p = LLAMA_BIN / name
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-server not found in {LLAMA_BIN}")


def llama_quantize() -> Path:
    for name in ("llama-quantize.exe", "llama-quantize"):
        p = LLAMA_BIN / name
        if p.exists():
            return p
    raise FileNotFoundError(f"llama-quantize not found in {LLAMA_BIN}")


def llama_tokenize() -> Path | None:
    for name in ("llama-tokenize.exe", "llama-tokenize"):
        p = LLAMA_BIN / name
        if p.exists():
            return p
    return None


def llama_perplexity() -> Path | None:
    for name in ("llama-perplexity.exe", "llama-perplexity"):
        p = LLAMA_BIN / name
        if p.exists():
            return p
    return None


def llama_imatrix() -> Path | None:
    for name in ("llama-imatrix.exe", "llama-imatrix"):
        p = LLAMA_BIN / name
        if p.exists():
            return p
    return None


def read_pin_sha() -> str:
    text = PIN_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line.lower()):
            return line
    raise ValueError(f"No SHA in {PIN_FILE}")
