#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python export/tiktoken_to_hf.py
python export/to_hf_llama.py
python export/convert_to_gguf.py
python export/verify_gguf.py
python export/verify_lineage.py checkpoints/phase6_3_ground_gold/mypt-f16.gguf
