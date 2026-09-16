#!/usr/bin/env python3
"""Standalone lineage printer for a MyPT GGUF (sidecar .lineage.json). Customer-runnable."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Print MyPT GGUF lineage sidecar")
    p.add_argument("gguf", type=Path)
    args = p.parse_args()
    side = args.gguf.with_suffix(".lineage.json")
    if not side.exists():
        print(f"NO_LINEAGE_SIDECAR {side}", file=sys.stderr)
        sys.exit(2)
    meta = json.loads(side.read_text(encoding="utf-8"))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    required = [
        "mypt.lineage.checkpoint_id",
        "mypt.lineage.llamacpp_commit",
        "mypt.special_token_ids",
    ]
    missing = [k for k in required if k not in meta]
    if missing:
        print("MISSING", missing, file=sys.stderr)
        sys.exit(1)
    print("LINEAGE_OK")


if __name__ == "__main__":
    main()
